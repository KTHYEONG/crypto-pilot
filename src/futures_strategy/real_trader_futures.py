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
import gc
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
    FUTURES_STATE_FILE,
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
# State Manager (JSON 파일 기반 - Futures 진입 시간 추적)
# ============================================================
class StateManager:
    """거래 상태 관리 (진입 시간, 진입가 등 로컬 저장)"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """상태 파일이 없으면 생성"""
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save({})

    def _load(self) -> dict:
        """상태 로드"""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ State load error: {e}")
            return {}

    def _save(self, state: dict):
        """상태 저장"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"⚠️ State save error: {e}")

    def get_symbol_state(self, symbol: str) -> dict:
        """특정 심볼의 상태 조회"""
        state = self._load()
        return state.get(symbol, {})

    def update_symbol_state(self, symbol: str, data: dict):
        """특정 심볼의 상태 업데이트"""
        state = self._load()
        if symbol not in state:
            state[symbol] = {}
        state[symbol].update(data)
        self._save(state)

    def clear_symbol_state(self, symbol: str):
        """특정 심볼의 상태 초기화"""
        state = self._load()
        if symbol in state:
            state[symbol] = {}
            self._save(state)
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
        self.state_manager = StateManager(FUTURES_STATE_FILE)
        
        # 클라우드 최적화 (옵션)
        self.cloud_optimizer = None
        if enable_oracle_optimization and CLOUD_OPTIMIZER_AVAILABLE:
            self.cloud_optimizer = CloudOptimizer()
            logger.info("☁️ Cloud optimization enabled")
        
        # Shutdown 플래그
        self._shutdown_requested = False
        
        # [Optimization] 중복 계산 방지용 캐시
        self.last_calc_candle: Dict[str, str] = {}
        
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
        
        # [Lazy Loading] optuna is heavy, load only when needed
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

    @api_retry
    def _place_stop_loss_safe(self, symbol: str, side: str, qty: float, stop_price: float):
        """안전한 서버 사이드 Stop Loss 실행 (재시도 적용)"""
        return self.client.place_stop_market_order(symbol, side, qty, stop_price)

    @api_retry
    def _cancel_all_orders_safe(self, symbol: str):
        """안전한 모든 주문 취소 (재시도 적용)"""
        return self.client.cancel_all_orders(symbol)
    
    def initialize(self):
        """초기화: 전략 로드, 레버리지 설정, 잔고 확인"""
        logger.info("🤖 RealTrader Futures Bot Initializing...")
        
        # 1. 전략 로드
        self.load_strategies_from_db()
        gc.collect() # 전략 객체 생성 후 메모리 정리
        
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
        """핵심 매매 로직 실행 (메모리 최적화)"""
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            timeframe = params.get('TIMEFRAME', '1h')

            # 현재 포지션 확인 (가벼운 API)
            pos = self._fetch_position_safe(symbol)
            amount = float(pos['amount'])
            in_position = abs(amount) > 0
            
            # [Optimization] 중복 계산 방지 체크
            current_slot = self._get_candle_slot_id(timeframe)
            already_calculated = self.last_calc_candle.get(symbol) == current_slot
            
            # 진입 시점 확인
            is_entry_time = self._is_entry_time(timeframe)

            # --- Case 1: 진입 시점(정시) && 아직 계산 안 함 -> 무거운 데이터 로드 ---
            if is_entry_time and not already_calculated:
                logger.info(f"🔍 [{symbol}] Checking for entry signals ({timeframe} candle closure)...")
                # 전체 캔들 데이터 조회 (지표 계산용)
                tf_min = 60
                if 'm' in timeframe:
                    tf_min = int(timeframe.replace('m', ''))
                elif 'h' in timeframe:
                    tf_min = int(timeframe.replace('h', '')) * 60
                elif 'd' in timeframe:
                    tf_min = int(timeframe.replace('d', '')) * 1440
                
                limit = 700
                lookback_days = (limit * tf_min) / 1440
                start_dt = datetime.utcnow() - timedelta(days=lookback_days + 2)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                
                df = self._fetch_ohlcv_safe(symbol, timeframe, start_str)
                
                if df is None or len(df) < 200:
                    logger.warning(
                        f"⚠️ Insufficient data for {symbol}. "
                        f"Got {len(df) if df is not None else 0}, need min 200."
                    )
                    return
                
                # [Correction] Ensure float64 for TA-Lib compatibility
                float_cols = ['open', 'high', 'low', 'close', 'volume']
                df[float_cols] = df[float_cols].astype(np.float64)
                
                # 지표 계산
                df = strategy.generate_signals(df)
                last_candle = df.iloc[-2]
                
                # 지표값 추출 및 캐싱
                entry_upper = last_candle.get('entry_upper')
                entry_lower = last_candle.get('entry_lower')
                trend_dir = last_candle.get('trend_direction', 0)
                strength_ok = (last_candle.get('strength_filter', 1) == 1)
                atr = last_candle.get('atr', 0.0)
                sar = last_candle.get('parabolic_sar', 0.0)
                vol_ratio = last_candle.get('volume_ratio', 1.0)
                
                if pd.isna(atr): atr = 0.0
                if pd.isna(sar): sar = 0.0
                if pd.isna(trend_dir): trend_dir = 0
                if pd.isna(vol_ratio): vol_ratio = 1.0
                
                # [CACHE] 다음 4시간 동안 재사용할 지표값 저장
                self._cache_indicators(symbol, {
                    'trend_direction': int(trend_dir),
                    'atr': float(atr),
                    'parabolic_sar': float(sar),
                    'entry_upper': float(entry_upper) if entry_upper is not None else None,
                    'entry_lower': float(entry_lower) if entry_lower is not None else None,
                    'strength_filter': int(strength_ok),
                    'volume_ratio': float(vol_ratio),
                    'rsi': float(last_candle.get('rsi', 50.0)),
                    'hurst': float(last_candle.get('hurst', 0.5)),
                    'natr': float(last_candle.get('natr', 0.0)),
                    'cached_at': datetime.utcnow().isoformat()
                })
                
                # 계산 완료 기록
                self.last_calc_candle[symbol] = current_slot
                logger.debug(f"📊 Indicators calculated and cached for {symbol} (Slot: {current_slot})")
                
            # --- Case 2: 비진입 시점이거나 이미 계산됨 -> 캐시된 지표 사용 ---
            else:
                # 캐시된 지표값 로드 (4시간 전에 계산한 값)
                cached = self._get_cached_indicators(symbol)
                trend_dir = cached.get('trend_direction', 0)
                atr = cached.get('atr', 0.0)
                sar = cached.get('parabolic_sar', 0.0)
                
                # 진입 조건은 체크 안 함 (시간이 안 맞으므로)
                entry_upper = cached.get('entry_upper')
                entry_lower = cached.get('entry_lower')
                strength_ok = bool(cached.get('strength_filter', False))
            
            # 현재가 조회 (가벼운 API - 항상 필요)
            current_price = self._get_market_price_safe(symbol)
            if current_price is None:
                logger.warning(f"⚠️ Failed to get price for {symbol}")
                return
            
            # --- EXIT LOGIC (항상 실행) ---
            if in_position:
                self._check_exit(
                    symbol, amount, current_price, params, pos, 
                    trend_dir, atr, sar
                )
            
            # --- ENTRY LOGIC (진입 시점에만 실행) ---
            elif not in_position and is_entry_time and strength_ok:
                if pd.isna(entry_upper) or pd.isna(entry_lower):
                    return
                
                # Volume Filter Check
                cached = self._get_cached_indicators(symbol)
                vol_ratio = cached.get('volume_ratio', 1.0)
                use_vol_filter = params.get('USE_VOLUME_FILTER', False)
                vol_threshold = params.get('VOLUME_THRESHOLD_MULT', 1.0)
                
                vol_ok = (not use_vol_filter) or (vol_ratio >= vol_threshold)
                if not vol_ok:
                    logger.debug(f"⏭️ {symbol} - Volume filter not passed (Ratio: {vol_ratio:.2f} < {vol_threshold})")
                    return

                # LONG 진입
                if trend_dir == 1 and current_price > entry_upper:
                    logger.info(
                        f"🟢 ENTRY LONG Signal {symbol} | "
                        f"Price {current_price} > Upper {entry_upper:.2f}"
                    )

                    qty = self._calculate_position_size(
                        symbol, current_price, params, atr,
                        hurst=cached.get('hurst', 0.5), natr=cached.get('natr', 0.0)
                    )
                    if qty > 0:
                        order = self._place_order_safe(symbol, 'buy', qty, atr=atr, current_price=current_price)
                        if order:
                            # [HARD STOP LOSS] 서버 사이드 SL 즉시 설정
                            sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
                            stop_price = 0.0
                            if sl_type == 'ATR' and atr > 0:
                                sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
                                stop_price = current_price - (atr * sl_mult)
                            else:
                                sl_pct = params.get('STOP_LOSS_PCT', 0.02)
                                stop_price = current_price * (1 - sl_pct)
                            
                            # 주문 가격 정밀도 보정 (보통 가격 정밀도와 동일)
                            stop_price = round(stop_price, 2) if 'BTC' not in symbol else round(stop_price, 1)
                            
                            self._place_stop_loss_safe(symbol, 'sell', qty, stop_price)

                            self.trade_db.record_trade(
                                symbol=symbol,
                                side='LONG',
                                action='ENTRY',
                                quantity=qty,
                                price=current_price,
                                reason=f"Price > Upper ({entry_upper:.2f})",
                                params={'timeframe': timeframe, 'atr': atr, 'sl': stop_price}
                            )
                            
                            # [STATE] 진입 상태 저장 (Time Cut용)
                            self.state_manager.update_symbol_state(symbol, {
                                'entry_time': datetime.utcnow().isoformat(),
                                'entry_price': current_price,
                                'side': 'LONG'
                            })
                        else:
                            logger.error(f"❌ Order placement failed for {symbol} (LONG, Qty: {qty})")
                
                # SHORT 진입
                elif trend_dir == -1 and current_price < entry_lower:
                    logger.info(
                        f"🔴 ENTRY SHORT Signal {symbol} | "
                        f"Price {current_price} < Lower {entry_lower:.2f}"
                    )

                    qty = self._calculate_position_size(
                        symbol, current_price, params, atr,
                        hurst=cached.get('hurst', 0.5), natr=cached.get('natr', 0.0)
                    )
                    if qty > 0:
                        order = self._place_order_safe(symbol, 'sell', qty, atr=atr, current_price=current_price)
                        if order:
                            # [HARD STOP LOSS] 서버 사이드 SL 즉시 설정
                            sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
                            stop_price = 0.0
                            if sl_type == 'ATR' and atr > 0:
                                sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
                                stop_price = current_price + (atr * sl_mult)
                            else:
                                sl_pct = params.get('STOP_LOSS_PCT', 0.02)
                                stop_price = current_price * (1 + sl_pct)
                            
                            stop_price = round(stop_price, 2) if 'BTC' not in symbol else round(stop_price, 1)
                            
                            self._place_stop_loss_safe(symbol, 'buy', qty, stop_price)

                            self.trade_db.record_trade(
                                symbol=symbol,
                                side='SHORT',
                                action='ENTRY',
                                quantity=qty,
                                price=current_price,
                                reason=f"Price < Lower ({entry_lower:.2f})",
                                params={'timeframe': timeframe, 'atr': atr, 'sl': stop_price}
                            )
                            
                            # [STATE] 진입 상태 저장 (Time Cut용)
                            self.state_manager.update_symbol_state(symbol, {
                                'entry_time': datetime.utcnow().isoformat(),
                                'entry_price': current_price,
                                'side': 'SHORT'
                            })
                        else:
                            logger.error(f"❌ Order placement failed for {symbol} (SHORT, Qty: {qty})")
                
                # [Optimization] 대규모 데이터프레임 제거 및 메모리 강제 회수
                del df
                gc.collect()
        
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
        """청산 로직 (정밀화)"""
        try:
            exit_triggered = False
            reason = ""
            
            use_tp = params.get('USE_TAKE_PROFIT', False)
            tp_atr_mult = params.get('TAKE_PROFIT_ATR_MULT_FUTURES', params.get('TAKE_PROFIT_ATR_MULT', 3.0))
            entry_price = float(pos.get('entryPrice', 0))
            if entry_price <= 0: return

            # 1. 포지션별 기본 로깅 및 SL/TP/SAR/Trend 체크
            if amount > 0:  # LONG
                # [SL/TP/SAR]
                if params.get('EXIT_TYPE') == 'PARABOLIC_SAR' and sar > 0 and current_price < sar:
                    exit_triggered, reason = True, "Parabolic SAR Cross"
                elif trend_dir == -1:
                    exit_triggered, reason = True, "Trend Reversal"
                elif use_tp and current_price >= (entry_price + (atr * tp_atr_mult)):
                    exit_triggered, reason = True, f"Take Profit ({entry_price + (atr * tp_atr_mult):.2f})"
                
                # [Stop Loss]
                if not exit_triggered:
                    sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
                    stop_price = (entry_price - (atr * params.get('ATR_STOP_LOSS_MULT', 1.5))) if sl_type == 'ATR' else (entry_price * (1 - params.get('STOP_LOSS_PCT', 0.02)))
                    if current_price <= stop_price:
                        exit_triggered, reason = True, f"Stop Loss ({stop_price:.2f})"

            elif amount < 0:  # SHORT
                # [SL/TP/SAR]
                if params.get('EXIT_TYPE') == 'PARABOLIC_SAR' and sar > 0 and current_price > sar:
                    exit_triggered, reason = True, "Parabolic SAR Cross"
                elif trend_dir == 1:
                    exit_triggered, reason = True, "Trend Reversal"
                elif use_tp and current_price <= (entry_price - (atr * tp_atr_mult)):
                    exit_triggered, reason = True, f"Take Profit ({entry_price - (atr * tp_atr_mult):.2f})"

                # [Stop Loss]
                if not exit_triggered:
                    sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
                    stop_price = (entry_price + (atr * params.get('ATR_STOP_LOSS_MULT', 1.5))) if sl_type == 'ATR' else (entry_price * (1 + params.get('STOP_LOSS_PCT', 0.02)))
                    if current_price >= stop_price:
                        exit_triggered, reason = True, f"Stop Loss ({stop_price:.2f})"

            # 2. 공통 청산 로직 (Panic Exit & Time Cut)
            if not exit_triggered:
                # [Panic Exit]
                cached = self._get_cached_indicators(symbol)
                rsi = cached.get('rsi', 50.0)
                rsi_exit_thresh = params.get('RSI_EXIT_THRESHOLD', 80)
                if (rsi > rsi_exit_thresh and amount > 0) or (rsi < (100 - rsi_exit_thresh) and amount < 0):
                    exit_triggered, reason = True, f"Panic Exit (RSI {rsi:.1f})"
                
                # [Time Cut]
                if not exit_triggered:
                    max_holding_bars = params.get('MAX_HOLDING_BARS', 9999)
                    state = self.state_manager.get_symbol_state(symbol)
                    entry_time_str = state.get('entry_time')
                    if entry_time_str:
                        entry_dt = datetime.fromisoformat(entry_time_str)
                        tf = params.get('TIMEFRAME', '1h')
                        interval_min = 60
                        if tf.endswith('h'): interval_min = int(tf[:-1]) * 60
                        elif tf.endswith('m'): interval_min = int(tf[:-1])
                        elif tf.endswith('d'): interval_min = int(tf[:-1]) * 1440
                        
                        bars_held = ((datetime.utcnow() - entry_dt).total_seconds() / 60) / interval_min
                        if bars_held >= max_holding_bars:
                            pnl_pct = (((current_price / entry_price) - 1) * 100) if amount > 0 else (((entry_price / current_price) - 1) * 100)
                            if pnl_pct <= params.get('TIME_EXIT_PROFIT_THRESHOLD', 1.4):
                                exit_triggered, reason = True, f"Time Cut (Held {bars_held:.1f} bars, PnL {pnl_pct:.2f}%)"

            # 3. 실제 주문 실행
            if exit_triggered:
                pnl = ((current_price - entry_price) * abs(amount)) if amount > 0 else ((entry_price - current_price) * abs(amount))
                pnl_pct = (((current_price / entry_price) - 1) * 100) if amount > 0 else (((entry_price / current_price) - 1) * 100)
                
                side_str = "LONG" if amount > 0 else "SHORT"
                order_side = 'sell' if amount > 0 else 'buy'
                
                logger.info(f"🛑 EXIT {side_str} {symbol} | Price: {current_price} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}")
                
                if self._place_order_safe(symbol, order_side, abs(amount)):
                    self._cancel_all_orders_safe(symbol)
                    self.trade_db.record_trade(
                        symbol=symbol, side=side_str, action='EXIT', quantity=abs(amount),
                        price=current_price, entry_price=entry_price, pnl=pnl, pnl_pct=pnl_pct, reason=reason
                    )
                    self.state_manager.clear_symbol_state(symbol)
        
        except Exception as e:
            logger.error(f"⚠️ Error in _check_exit for {symbol}: {e}")
            self.health_manager.record_error(e)
    

    def _calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        params: dict, 
        atr: float = 0.0,
        hurst: float = 0.5,
        natr: float = 0.0
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
        
        # [NEW] Regime-based Multiplier (Dynamic Sizing)
        regime_mult = 1.0
        strong_hurst = params.get('STRONG_REGIME_HURST', 0.56)
        panic_natr = params.get('PANIC_REGIME_NATR', 6.0)
        strong_mult = params.get('STRONG_REGIME_MULTIPLIER', 1.4)
        panic_mult = params.get('PANIC_REGIME_MULTIPLIER', 0.25)
        
        if hurst > strong_hurst:
            regime_mult = strong_mult
            logger.info(f"💪 Strong Regime detected (Hurst {hurst:.2f}). Mult: {strong_mult}")
            
        if natr > panic_natr:
            regime_mult = panic_mult
            logger.info(f"😱 Panic Regime detected (NATR {natr:.2f}). Mult: {panic_mult}")
            
        allocation_weight *= regime_mult
        
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
            logger.warning(
                f"⚠️ Calculated size too small for {symbol}: ${final_notional:.2f} "
                f"< Min ${MIN_ORDER_VALUE_USDT}. Increase weight or balance."
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
            logger.warning(
                f"⚠️ Final order value too small for {symbol}: "
                f"${final_order_value:.2f} (Qty: {quantity}). "
                f"Min ${MIN_ORDER_VALUE_USDT} required."
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
    
    def _is_entry_time(self, timeframe: str) -> bool:
        """
        현재 시간이 타임프레임별 진입(봉 마감) 시점인지 확인
        예: 4h봉 -> 00, 04, 08, 12, 16, 20시의 0분~5분 사이인지
        """
        now = datetime.utcnow()
        minutes = now.minute

        # [TIGHT SYNC] 허용 범위: 정시 ~ +2분
        # 백테스트(시가 진입)와 최대한 일치시키기 위해 5분에서 2분으로 단축
        if minutes > 2:
            return False

        if timeframe.endswith('m'):
            interval = int(timeframe[:-1])
            return (now.minute % interval) <= 5
        
        elif timeframe.endswith('h'):
            interval = int(timeframe[:-1])
            return (now.hour % interval) == 0
        
        elif timeframe.endswith('d'):
            return now.hour == 0  # 00:00 UTC 기준
        
        return False
    
    def _cache_indicators(self, symbol: str, indicators: dict):
        """지표값을 메모리에 캐싱 (4시간 재사용)"""
        if not hasattr(self, '_indicator_cache'):
            self._indicator_cache = {}
        self._indicator_cache[symbol] = indicators
    
    def _get_cached_indicators(self, symbol: str) -> dict:
        """캐시된 지표값 조회"""
        if not hasattr(self, '_indicator_cache'):
            self._indicator_cache = {}
        return self._indicator_cache.get(symbol, {})

    def _get_candle_slot_id(self, timeframe: str) -> str:
        """현재 타임프레임 기준의 고유 캔들 ID 생성 (중복 계산 방지용)"""
        now = datetime.utcnow()
        if timeframe.endswith('m'):
            interval = int(timeframe[:-1])
            slot = (now.minute // interval) * interval
            return now.strftime(f"%Y%m%d%H{slot:02d}")
        elif timeframe.endswith('h'):
            interval = int(timeframe[:-1])
            slot = (now.hour // interval) * interval
            return now.strftime(f"%Y%m%d{slot:02d}00")
        elif timeframe.endswith('d'):
            return now.strftime("%Y%m%d0000")
        return now.strftime("%Y%m%d%H%M")

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
                    
                    # 1. 상태 및 파라미터 로드
                    params = self.params_map[symbol]
                    timeframe = params.get('TIMEFRAME', '1h')

                    # 2. 실행 (execute_logic 내부에서 진입 시점 체크)
                    try:
                        self.execute_logic(symbol)
                    except Exception as e:
                        logger.error(f"⚠️ Error processing {symbol}: {e}")

                    time.sleep(SYMBOL_DELAY_SECONDS)
                
                # 헬스체크 업데이트
                positions = self._get_current_positions()
                self.health_manager.update_heartbeat(
                    status="running",
                    positions=positions
                )
                
                # 클라우드 최적화 실행
                if self.cloud_optimizer:
                    # 1. 시간 동기화 검증
                    if not self.cloud_optimizer.check_time_sync_ntp():
                        logger.error("⏰ Time drift detected! Bot may fail to place orders on Binance.")
                    
                    # 2. 리소스 모니터링 (10분마다) & 메모리 보호 (AWS Free Tier)
                    if self.health_manager.loop_count % 60 == 0: # 10s * 60 = 10분
                        usage = self.cloud_optimizer.log_resource_usage()
                        if usage.get('memory_percent', 0) > 85.0:
                            logger.warning(f"⚠️ High Memory ({usage.get('memory_percent')}%) detected. Forcing GC...")
                            self.cloud_optimizer.force_gc()
                    
                    # 3. DB 정리 (24시간마다: 1440분)
                    if self.health_manager.loop_count % 1440 == 0:
                        self.cloud_optimizer.cleanup_db_old_records(
                            TRADE_HISTORY_DB, 
                            days_to_keep=90
                        )
                    
                    # 5. 명시적 GC (2시간마다: 120분)
                    if self.health_manager.loop_count % 120 == 0:
                        self.cloud_optimizer.force_gc()
                
                # [High-Frequency Monitoring] 청산 감시를 위해 10초(LOOP_INTERVAL_SECONDS)마다 반복
                # 진입은 _is_entry_time 로직에 의해 정시에만 수행됨
                wait_seconds = float(LOOP_INTERVAL_SECONDS)
                
                logger.debug(f"💤 Sleeping {wait_seconds:.1f}s until next monitoring cycle...")
                
                # Shutdown 체크하면서 대기
                start_wait = time.time()
                while time.time() - start_wait < wait_seconds:
                    if self._shutdown_requested:
                        break
                    time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"🚨 Critical Error in Main Loop: {e}")
                self.health_manager.record_error(e)
                self.health_manager.update_heartbeat(status="error")
                time.sleep(10) # 에러 시 10초 대기 후 재시도
        
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
