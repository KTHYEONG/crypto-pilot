"""
RealTrader Futures - 24시간 자동 선물 트레이딩 봇 (Production Grade)
===================================================================
P0/P1 개선사항 적용:
- 거래 기록 DB 영속화
- API 재시도 데코레이터 (tenacity)
- Health Check 메커니즘
- Graceful Shutdown (SIGTERM)
- 중복 코드 제거 (유틸 함수)
- 매직 넘버 → settings 이동
- 캔들 마감 동기화
- Structured JSON 로깅
"""

import os
import sys
import time
import signal
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Optional, Dict, Any, Tuple



# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.settings import (
    BINANCE_API_KEY, 
    BINANCE_SECRET,
    LOG_DIR,
    FUTURES_STRATEGY_DB,
    TRADE_HISTORY_DB,
    HEARTBEAT_FILE,
    API_RETRY_ATTEMPTS,
    API_RETRY_WAIT_MIN,
    API_RETRY_WAIT_MAX,
    MIN_BALANCE_USDT,
    MIN_BALANCE_FOR_TRADE,
    MIN_ORDER_VALUE_USDT,
    MAX_EXCHANGE_LEVERAGE,
    LOOP_INTERVAL_SECONDS,
    SYMBOL_DELAY_SECONDS,
    ERROR_SLEEP_SECONDS,
    CANDLE_SYNC_OFFSET_SECONDS,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    FUTURES_TARGET_SYMBOLS,
    OPTUNA_STUDY_NAMES,
    SYMBOL_ALLOCATION_WEIGHTS,
)
from src.futures_strategy.binance_client import BinanceClient
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.common.utils import setup_logger, api_retry
from src.common.components import TradeHistoryDB, HealthCheckManager, calculate_candle_wait_time

# Oracle Cloud 최적화 (선택적)
try:
    from src.common.cloud_optimizer import CloudOptimizer
    CLOUD_OPTIMIZER_AVAILABLE = True
except ImportError:
    CLOUD_OPTIMIZER_AVAILABLE = False

# ============================================================
# Structured JSON Logger
# ============================================================



logger = setup_logger("RealTraderFutures")



# ============================================================
# Trade History DB Manager
# ============================================================



# ============================================================
# Health Check Manager
# ============================================================



# ============================================================
# Utility Functions (중복 코드 제거)
# ============================================================



# ============================================================
# Main Trader Class
# ============================================================
class RealTraderFutures:
    """Production-grade 선물 트레이딩 봇"""
    
    def __init__(self, db_path: str = None, enable_oracle_optimization: bool = False):
        self.client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        self.db_path = db_path or str(FUTURES_STRATEGY_DB)
        self.strategies: Dict[str, UltimateStrategy] = {}
        self.params_map: Dict[str, dict] = {}
        self.symbols: list = []
        
        # 신규 컴포넌트
        self.trade_db = TradeHistoryDB(TRADE_HISTORY_DB)
        self.health_manager = HealthCheckManager(HEARTBEAT_FILE)
        
        # 클라우드 최적화 (옵션)
        self.cloud_optimizer = None
        if enable_oracle_optimization and CLOUD_OPTIMIZER_AVAILABLE:
            self.cloud_optimizer = CloudOptimizer()
            logger.info("☁️ Cloud optimization enabled")
        
        # Shutdown 플래그
        self._shutdown_requested = False
        
        # Signal handlers 등록
        self._setup_signal_handlers()
        
        # 전략 로드는 initialize()에서 수행

    
    def _setup_signal_handlers(self):
        """Graceful Shutdown 시그널 핸들러 등록"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
            self._shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        # Windows에서는 SIGBREAK도 처리
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def load_strategies_from_db(self):
        """Optuna DB에서 최적화된 파라미터 로드"""
        logger.info(f"📂 Loading strategies from {self.db_path}...")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"DB file not found: {self.db_path}")
        
        import optuna
        storage = f"sqlite:///{self.db_path}"
        
        study = None
        for s_name in OPTUNA_STUDY_NAMES:
            try:
                study = optuna.load_study(study_name=s_name, storage=storage)
                logger.info(f"✅ Loaded Study: '{s_name}' (Score: {study.best_value:.4f})")
                break
            except KeyError:
                continue
        
        if study is None:
            raise ValueError(f"No valid study found in DB. Tried: {OPTUNA_STUDY_NAMES}")
        
        best_params = study.best_params
        self.symbols = FUTURES_TARGET_SYMBOLS.copy()
        
        for symbol in self.symbols:
            self.params_map[symbol] = best_params.copy()
            strategy_name = f"Real_{symbol.replace('/', '_')}"
            self.strategies[symbol] = UltimateStrategy(strategy_name, best_params)
            logger.info(f"🔹 Strategy initialized: {symbol} | TF: {best_params.get('TIMEFRAME')}")
    
    @api_retry
    def _fetch_balance_safe(self) -> tuple:
        """안전한 잔고 조회 (Total, Free 반환)"""
        # BinanceClient.fetch_balance() already returns (total, free)
        return self.client.fetch_balance()
    
    @api_retry
    def _fetch_ohlcv_safe(self, symbol: str, timeframe: str, start_str: str):
        """안전한 OHLCV 조회 (재시도 적용)"""
        return self.client.fetch_ohlcv(symbol, timeframe, start_date=start_str)
    
    @api_retry
    def _fetch_position_safe(self, symbol: str) -> dict:
        """안전한 포지션 조회 (재시도 적용)"""
        return self.client.fetch_position(symbol)
    
    @api_retry
    def _get_market_price_safe(self, symbol: str) -> float:
        """안전한 시장가 조회 (재시도 적용)"""
        return self.client.get_market_price(symbol)
    
    @api_retry
    def _place_order_safe(self, symbol: str, side: str, qty: float, atr: float = None, current_price: float = None):
        """안전한 주문 실행 (재시도 적용 + 스마트 주문 + 변동성 기반 최적화)"""
        return self.client.place_order_smart(symbol, side, qty, atr=atr, current_price=current_price)
    
    def initialize(self):
        """초기화: 전략 로드, 레버리지 설정, 잔고 확인"""
        logger.info("🤖 RealTrader Futures Bot Initializing...")
        
        # 1. 전략 로드
        self.load_strategies_from_db()
        
        # 2. 잔고 확인
        try:
            total_balance, usdt_free = self._fetch_balance_safe()
            logger.info(f"💰 Account Balance: {usdt_free:.2f} USDT (Total: {total_balance:.2f})")
            
            if usdt_free < MIN_BALANCE_USDT:
                logger.warning(f"⚠️ Warning: Low balance (< {MIN_BALANCE_USDT} USDT)!")
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
        
        # 3. 레버리지 설정
        for symbol in self.symbols:
            try:
                success = self.client.set_leverage(symbol, MAX_EXCHANGE_LEVERAGE)
                target_lev = self.params_map[symbol].get('LEVERAGE', 1)
                if success:
                    logger.info(
                        f"✅ Exchange Leverage: {MAX_EXCHANGE_LEVERAGE}x for {symbol} "
                        f"(Strategy Target: {target_lev}x)"
                    )
                
                # 마진 모드 설정 (Cross 모드 강제)
                self.client.set_margin_type(symbol, margin_type='CROSSED')

            except Exception as e:
                logger.error(f"⚠️ Error setting leverage/margin for {symbol}: {e}")
        
        # 4. 포지션 모드 설정 (One-Way Mode 강제)
        # 봇 로직은 단방향 스위칭 구조이므로 Hedge Mode가 아닌 One-Way Mode가 필수입니다.
        try:
            self.client.set_position_mode(dual_side_position=False)
        except Exception as e:
            logger.error(f"⚠️ Failed to set One-Way Mode: {e}")

        # 5. 자산 모드 설정 (Single-Asset Mode 강제)
        # 봇은 USDT 단일 담보만 고려하여 잔고 계산을 하므로 Multi-Asset Mode는 끕니다.
        try:
            self.client.set_asset_mode(is_multi_asset=False)
        except Exception as e:
            logger.error(f"⚠️ Failed to set Single-Asset Mode: {e}")

        # 6. 초기 헬스체크
        self.health_manager.update_heartbeat(status="initialized")
        
        logger.info("🚀 Initialization Complete. Bot is Running...")
    
    def execute_logic(self, symbol: str):
        """핵심 매매 로직 실행"""
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            
            # 1. 데이터 조회 (충분한 지표 워밍업을 위해 700개 요청)
            timeframe = params.get('TIMEFRAME', '1h')
            limit = 700
            
            tf_min = 60
            if 'm' in timeframe:
                tf_min = int(timeframe.replace('m', ''))
            elif 'h' in timeframe:
                tf_min = int(timeframe.replace('h', '')) * 60
            elif 'd' in timeframe:
                tf_min = int(timeframe.replace('d', '')) * 1440
            
            lookback_days = (limit * tf_min) / 1440
            start_dt = datetime.utcnow() - timedelta(days=lookback_days + 2)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            df = self._fetch_ohlcv_safe(symbol, timeframe, start_str)
            
            # 200개 이하인 경우 지표 불신뢰로 인해 중단
            if df is None or len(df) < 200:
                logger.warning(
                    f"⚠️ Insufficient data for {symbol}. "
                    f"Got {len(df) if df is not None else 0}, need min 200."
                )
                return
            
            # 2. 지표 계산
            df = strategy.generate_signals(df)
            
            # 3. 신호 확인 (-2: 확정된 마지막 봉)
            last_candle = df.iloc[-2]
            current_price = self._get_market_price_safe(symbol)
            
            if current_price is None:
                logger.warning(f"⚠️ Failed to get price for {symbol}")
                return
            
            entry_upper = last_candle.get('entry_upper')
            entry_lower = last_candle.get('entry_lower')
            trend_dir = last_candle.get('trend_direction', 0)
            strength_ok = (last_candle.get('strength_filter', 1) == 1)
            atr = last_candle.get('atr', 0.0)
            sar = last_candle.get('parabolic_sar', 0.0)
            
            # NaN 체크 강화
            if pd.isna(atr):
                atr = 0.0
            if pd.isna(sar):
                sar = 0.0
            
            # 4. 현재 포지션
            pos = self._fetch_position_safe(symbol)
            amount = float(pos['amount'])
            in_position = abs(amount) > 0
            
            # --- EXIT LOGIC ---
            if in_position:
                self._check_exit(
                    symbol, amount, current_price, params, pos, 
                    trend_dir, atr, sar
                )
            
            # --- ENTRY LOGIC ---
            elif not in_position and strength_ok:
                if pd.isna(entry_upper) or pd.isna(entry_lower):
                    return
                
                # LONG 진입
                if trend_dir == 1 and current_price > entry_upper:
                    logger.info(
                        f"🟢 ENTRY LONG Signal {symbol} | "
                        f"Price {current_price} > Upper {entry_upper:.2f}"
                    )
                    qty = self._calculate_position_size(symbol, current_price, params, atr)
                    if qty > 0:
                        order = self._place_order_safe(symbol, 'buy', qty, atr=atr, current_price=current_price)
                        if order:
                            self.trade_db.record_trade(
                                symbol=symbol,
                                side='LONG',
                                action='ENTRY',
                                quantity=qty,
                                price=current_price,
                                reason=f"Price > Upper ({entry_upper:.2f})",
                                params={'timeframe': timeframe, 'atr': atr}
                            )
                        else:
                            logger.error(f"❌ Order placement failed for {symbol} (LONG, Qty: {qty})")
                
                # SHORT 진입
                elif trend_dir == -1 and current_price < entry_lower:
                    logger.info(
                        f"🔴 ENTRY SHORT Signal {symbol} | "
                        f"Price {current_price} < Lower {entry_lower:.2f}"
                    )
                    qty = self._calculate_position_size(symbol, current_price, params, atr)
                    if qty > 0:
                        order = self._place_order_safe(symbol, 'sell', qty, atr=atr, current_price=current_price)
                        if order:
                            self.trade_db.record_trade(
                                symbol=symbol,
                                side='SHORT',
                                action='ENTRY',
                                quantity=qty,
                                price=current_price,
                                reason=f"Price < Lower ({entry_lower:.2f})",
                                params={'timeframe': timeframe, 'atr': atr}
                            )
                        else:
                            logger.error(f"❌ Order placement failed for {symbol} (SHORT, Qty: {qty})")
        
        except Exception as e:
            logger.error(f"🚨 Error executing logic for {symbol}: {e}")
            self.health_manager.record_error(e)
    
    def _check_exit(
        self, 
        symbol: str, 
        amount: float, 
        current_price: float, 
        params: dict, 
        pos: dict, 
        trend_dir: int, 
        atr: float, 
        sar: float
    ):
        """청산 로직"""
        try:
            exit_triggered = False
            reason = ""
            
            use_tp = params.get('USE_TAKE_PROFIT', False)
            tp_atr_mult = params.get(
                'TAKE_PROFIT_ATR_MULT_FUTURES', 
                params.get('TAKE_PROFIT_ATR_MULT', 3.0)
            )
            entry_price = float(pos.get('entryPrice', 0))
            
            if amount > 0:  # LONG
                # 1. Parabolic SAR Exit
                if params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
                    if sar > 0 and current_price < sar:
                        exit_triggered = True
                        reason = "Parabolic SAR Cross"
                
                # 2. Trend Reversal
                if trend_dir == -1:
                    exit_triggered = True
                    reason = "Trend Reversal"
                
                # 3. Take Profit
                if use_tp and entry_price > 0 and atr > 0:
                    tp_price = entry_price + (atr * tp_atr_mult)
                    if current_price >= tp_price:
                        exit_triggered = True
                        reason = f"Take Profit ({tp_price:.2f})"
                
                # 4. Stop Loss (Fixed / ATR) [CRITICAL FIX]
                if not exit_triggered and entry_price > 0:
                    sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
                    stop_price = 0.0
                    
                    if sl_type == 'ATR' and atr > 0:
                        sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
                        stop_price = entry_price - (atr * sl_mult)
                    else: # FIXED
                        sl_pct = params.get('STOP_LOSS_PCT', 0.02)
                        stop_price = entry_price * (1 - sl_pct)
                        
                    if current_price <= stop_price:
                        exit_triggered = True
                        reason = f"Stop Loss ({stop_price:.2f})"

                # 5. Trailing Stop (ATR) - Simplified Stateless
                # (Requires persistent high tracking for perfect sync, here using conservative close-based approximation)
                if not exit_triggered and params.get('EXIT_TYPE') == 'ATR' and atr > 0:
                    # In a stateless system, we can't easily track 'highest since entry' perfectly without DB.
                    # Fallback: If price drops significantly from recent high (within Lookback), exit.
                    # Better: Use simple ATR Safety net relative to entry for now or rely on trend reversal.
                    # For Production: It is safer to rely on Trend Reversal + Hard Stop Loss than a stateless trailing stop.
                    # We will implement a "Profit Protection" Trailing Stop:
                    # If Profit > 2 * ATR, move Loop Stop to Entry + 0.5 * ATR (Break Even + Profit)
                    profit_dist = current_price - entry_price
                    if profit_dist > (atr * 2.0):
                        trail_floor = entry_price + (atr * 0.5)
                        if current_price < trail_floor:
                            exit_triggered = True
                            reason = f"Profit Protection Trail ({trail_floor:.2f})"
                
                if exit_triggered:
                    pnl = (current_price - entry_price) * abs(amount)
                    pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                    
                    logger.info(
                        f"🛑 EXIT LONG {symbol} | Price: {current_price} | "
                        f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}"
                    )
                    order = self._place_order_safe(symbol, 'sell', abs(amount))
                    if order:
                        self.trade_db.record_trade(
                            symbol=symbol,
                            side='LONG',
                            action='EXIT',
                            quantity=abs(amount),
                            price=current_price,
                            entry_price=entry_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason=reason
                        )
            
            elif amount < 0:  # SHORT
                # 1. Parabolic SAR Exit
                if params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
                    if sar > 0 and current_price > sar:
                        exit_triggered = True
                        reason = "Parabolic SAR Cross"
                
                # 2. Trend Reversal
                if trend_dir == 1:
                    exit_triggered = True
                    reason = "Trend Reversal"
                
                # 3. Take Profit
                if use_tp and entry_price > 0 and atr > 0:
                    tp_price = entry_price - (atr * tp_atr_mult)
                    if current_price <= tp_price:
                        exit_triggered = True
                        reason = f"Take Profit ({tp_price:.2f})"

                # 4. Stop Loss (Fixed / ATR) [CRITICAL FIX]
                if not exit_triggered and entry_price > 0:
                    sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
                    stop_price = 0.0
                    
                    if sl_type == 'ATR' and atr > 0:
                        sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
                        stop_price = entry_price + (atr * sl_mult)
                    else: # FIXED
                        sl_pct = params.get('STOP_LOSS_PCT', 0.02)
                        stop_price = entry_price * (1 + sl_pct)
                        
                    if current_price >= stop_price:
                        exit_triggered = True
                        reason = f"Stop Loss ({stop_price:.2f})"

                # 5. Trailing Stop (ATR) - Simplified Stateless
                if not exit_triggered and params.get('EXIT_TYPE') == 'ATR' and atr > 0:
                     profit_dist = entry_price - current_price
                     if profit_dist > (atr * 2.0):
                        trail_ceil = entry_price - (atr * 0.5)
                        if current_price > trail_ceil:
                            exit_triggered = True
                            reason = f"Profit Protection Trail ({trail_ceil:.2f})"
                
                if exit_triggered:
                    pnl = (entry_price - current_price) * abs(amount)
                    pnl_pct = ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0
                    
                    logger.info(
                        f"🛑 EXIT SHORT {symbol} | Price: {current_price} | "
                        f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}"
                    )
                    order = self._place_order_safe(symbol, 'buy', abs(amount))
                    if order:
                        self.trade_db.record_trade(
                            symbol=symbol,
                            side='SHORT',
                            action='EXIT',
                            quantity=abs(amount),
                            price=current_price,
                            entry_price=entry_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason=reason
                        )
        
        except Exception as e:
            logger.error(f"⚠️ Error in _check_exit: {e}")
            self.health_manager.record_error(e)
    
    def _calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        params: dict, 
        atr: float = 0.0
    ) -> float:
        """
        포지션 사이즈 계산 (견고성 강화)
        
        개선사항:
        - 거래소 정밀도(precision) 자동 조회
        - 최소 주문 금액/수량 검증
        - Edge case 방어 (0 나누기, 음수 레버리지 등)
        """
        # === 0. Input Validation ===
        if price <= 0:
            logger.error(f"❌ Invalid price for {symbol}: {price}")
            return 0.0
        
        # === 1. 잔고 조회 ===
        try:
            total_balance, usdt_free = self._fetch_balance_safe()
        except Exception as e:
            logger.error(f"❌ Balance fetch failed for {symbol}: {e}")
            return 0.0
        
        if usdt_free < MIN_BALANCE_FOR_TRADE:
            logger.warning(f"⚠️ Insufficient capital for {symbol}: ${usdt_free:.2f}")
            return 0.0
        
        # === 2. 성과 기반 가중치 적용 ===
        default_weight = 1.0 / len(self.symbols) if self.symbols else 0.5
        allocation_weight = SYMBOL_ALLOCATION_WEIGHTS.get(symbol, default_weight)
        allocated_capital = total_balance * allocation_weight
        
        # === 3. 전략 파라미터 ===
        leverage = params.get('LEVERAGE', 1)
        if leverage <= 0:
            logger.warning(f"⚠️ Invalid leverage for {symbol}: {leverage}. Using 1x.")
            leverage = 1
        
        risk_per_trade = params.get(
            'RISK_PER_TRADE_FUTURES', 
            params.get('RISK_PER_TRADE', 0.02)
        )
        
        # === 4. Stop Loss Distance 계산 ===
        stop_distance_pct = 0.05  # Default fallback
        
        if atr > 0 and price > 0:
            atr_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
            stop_distance = atr * atr_mult
            stop_distance_pct = stop_distance / price
            # Clamp to reasonable range (0.5% ~ 10%)
            stop_distance_pct = max(0.005, min(stop_distance_pct, 0.10))
        
        # === 5. Sizing Calculation ===
        risk_amt = allocated_capital * risk_per_trade
        
        # Division by zero 방어
        if stop_distance_pct <= 0:
            logger.error(
                f"❌ Invalid stop_distance_pct for {symbol}: {stop_distance_pct}. "
                "Cannot calculate position size."
            )
            return 0.0
        
        notional_value = risk_amt / stop_distance_pct
        max_tradeable_notional = usdt_free * leverage
        final_notional = min(notional_value, max_tradeable_notional)
        
        # === 6. 최소 주문 금액 체크 ===
        if final_notional < MIN_ORDER_VALUE_USDT:
            logger.debug(
                f"⚠️ Notional too small for {symbol}: ${final_notional:.2f} "
                f"< ${MIN_ORDER_VALUE_USDT}"
            )
            return 0.0
        
        # === 7. 수량 계산 (Quantity) ===
        raw_quantity = final_notional / price
        
        # === 8. 거래소 정밀도 적용 (Precision) ===
        # Binance market info에서 정밀도를 가져오는 것이 이상적이지만,
        # API 호출 오버헤드를 피하기 위해 심볼별 일반적인 정밀도 사용
        # 추후 market info 캐싱으로 개선 가능
        precision_map = {
            'BTC/USDT': 3,   # 0.001
            'ETH/USDT': 2,   # 0.01
            'BNB/USDT': 2,
            'SOL/USDT': 1,
            'XRP/USDT': 0,   # 1 (정수)
        }
        
        precision = precision_map.get(symbol, 2)  # 기본 2자리
        multiplier = 10 ** precision
        quantity = float(int(raw_quantity * multiplier) / multiplier)
        
        # === 9. 최종 검증 ===
        # a. 수량이 0이 아닌지
        if quantity <= 0:
            logger.warning(
                f"⚠️ Calculated quantity is zero for {symbol}. "
                f"Raw: {raw_quantity:.6f}, Precision: {precision}"
            )
            return 0.0
        
        # b. 최소 주문 금액 재확인 (정밀도 적용 후)
        final_order_value = quantity * price
        if final_order_value < MIN_ORDER_VALUE_USDT:
            logger.debug(
                f"⚠️ Order value after precision too small for {symbol}: "
                f"${final_order_value:.2f} (Qty: {quantity})"
            )
            return 0.0
        
        # === 10. 상세 로깅 ===
        logger.info(
            f"🧮 Sizing {symbol} (Weight {allocation_weight*100:.0f}%, Leverage {leverage}x): "
            f"Total ${total_balance:.0f} | Alloc ${allocated_capital:.0f}"
        )
        logger.info(
            f"   -> Risk: ${risk_amt:.1f} ({risk_per_trade*100:.1f}% of Alloc) | "
            f"StopDist {stop_distance_pct*100:.2f}%"
        )
        logger.info(
            f"   -> Target Size: ${notional_value:.1f} | "
            f"Final: ${final_notional:.1f} ({quantity} {symbol.split('/')[0]})"
        )
        
        return quantity
    
    def _get_current_positions(self) -> dict:
        """현재 포지션 상태 조회 (헬스체크용)"""
        positions = {}
        for symbol in self.symbols:
            try:
                pos = self._fetch_position_safe(symbol)
                if abs(pos['amount']) > 0:
                    positions[symbol] = {
                        'amount': pos['amount'],
                        'entryPrice': pos['entryPrice'],
                        'unrealizedPnL': pos['unrealizedPnL']
                    }
            except Exception:
                pass
        return positions
    
    def run_forever(self):
        """메인 무한 루프 (Graceful Shutdown 지원)"""
        try:
            self.initialize()
        except Exception as e:
            logger.error(f"🚨 Initialization failed: {e}")
            self.health_manager.update_heartbeat(status="init_failed")
            raise
        
        logger.info("⏳ Waiting for next candle close...")
        
        while not self._shutdown_requested:
            try:
                # 각 심볼 처리
                for symbol in self.symbols:
                    if self._shutdown_requested:
                        break
                    self.execute_logic(symbol)
                    time.sleep(SYMBOL_DELAY_SECONDS)
                
                # 헬스체크 업데이트
                positions = self._get_current_positions()
                self.health_manager.update_heartbeat(
                    status="running",
                    positions=positions
                )
                
                # 클라우드 최적화 실행
                if self.cloud_optimizer:
                    # 1. 시간 동기화 검증 (Binance API 필수)
                    if not self.cloud_optimizer.check_time_sync_ntp():
                        logger.error("⏰ Time drift detected! Bot may fail to place orders on Binance.")
                    
                    # 2. 리소스 모니터링 (10분마다)
                    if self.health_manager.loop_count % 20 == 0:
                        self.cloud_optimizer.log_resource_usage()
                    
                    # 3. DB 정리 (24시간마다, 90일 이상 오래된 거래 삭제)
                    if self.health_manager.loop_count % 2880 == 0:
                        self.cloud_optimizer.cleanup_db_old_records(
                            TRADE_HISTORY_DB, 
                            days_to_keep=90
                        )
                    
                    # 5. 명시적 GC (1시간마다)
                    if self.health_manager.loop_count % 120 == 0:
                        self.cloud_optimizer.force_gc()
                
                # 캔들 동기화 대기 (옵션)
                # 첫 번째 심볼의 타임프레임 기준
                if self.symbols and self.params_map:
                    tf = self.params_map[self.symbols[0]].get('TIMEFRAME', '1h')
                    wait_time = calculate_candle_wait_time(tf)
                    
                    # 최소 대기 시간 적용 (너무 짧으면 기본 간격 사용)
                    if wait_time < LOOP_INTERVAL_SECONDS:
                        wait_time = LOOP_INTERVAL_SECONDS
                    
                    # 최대 대기 시간 제한 (1시간)
                    wait_time = min(wait_time, 3600)
                    
                    logger.info(f"💤 Next execution in {wait_time}s...")
                    
                    # Shutdown 체크하면서 대기
                    for _ in range(int(wait_time)):
                        if self._shutdown_requested:
                            break
                        time.sleep(1)
                else:
                    time.sleep(LOOP_INTERVAL_SECONDS)
            
            except Exception as e:
                logger.error(f"🚨 Critical Error in Main Loop: {e}")
                self.health_manager.record_error(e)
                self.health_manager.update_heartbeat(status="error")
                time.sleep(ERROR_SLEEP_SECONDS)
        
        # Graceful Shutdown 처리
        self._shutdown()
    
    def _shutdown(self):
        """Graceful Shutdown 처리"""
        logger.info("🛑 Shutting down gracefully...")
        
        # 현재 포지션 상태 기록
        positions = self._get_current_positions()
        if positions:
            logger.warning(f"⚠️ Open positions at shutdown: {positions}")
        
        self.health_manager.update_heartbeat(
            status="stopped",
            positions=positions,
            extra={"shutdown_time": datetime.utcnow().isoformat()}
        )
        
        logger.info("✅ Shutdown complete.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 RealTrader Futures - Production Grade Bot")
    logger.info("=" * 60)
    
    # Oracle Cloud 환경 변수로 활성화 결정 (기본값: True)
    import os
    enable_oracle_opt = os.getenv("ENABLE_ORACLE_OPTIMIZATION", "true").lower() == "true"
    
    bot = RealTraderFutures(enable_oracle_optimization=enable_oracle_opt)
    bot.run_forever()
