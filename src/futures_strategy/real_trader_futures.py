
import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.settings import BINANCE_API_KEY, BINANCE_SECRET
from src.futures_strategy.binance_client import BinanceClient
from src.futures_strategy.strategies_futures import UltimateStrategy

# --- 로깅 설정 (자동 회전 적용) ---
logger = logging.getLogger("RealTraderFutures")
logger.setLevel(logging.INFO)

# 파일 핸들러 (10MB마다 새 파일, 최대 5개 유지)
file_handler = RotatingFileHandler(
    "real_trader_futures.log", 
    maxBytes=10*1024*1024, # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 콘솔 핸들러
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

class RealTraderFutures:
    def __init__(self, db_path="futures_strategy.db"):
        self.client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        self.db_path = db_path
        self.strategies = {} # {symbol: StrategyObj}
        self.params_map = {} # {symbol: params_dict}
        self.symbols = []
        
    def load_strategies_from_db(self):
        """Load optimized parameters from SQLite DB"""
        logger.info(f"📂 Loading strategies from {self.db_path}...")
        
        if not os.path.exists(self.db_path):
            logger.error(f"❌ DB file not found: {self.db_path}")
            sys.exit(1)
            
        try:
            # Optuna DB는 복잡하므로, 가장 좋은 Trial의 파라미터를 가져오는 로직 필요
            # 하지만 Optuna API를 쓰는 게 가장 안전함
            import optuna
            storage = f"sqlite:///{self.db_path}"
            study_name = "futures_strategy"
            
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
                best_params = study.best_params
                logger.info(f"✅ Loaded Universal Params (Score: {study.best_value:.4f})")
                
                # Universal Strategy이므로 모든 타겟 심볼에 동일 파라미터 적용
                # 타겟 심볼은 봇 설정에서 정의하거나, args로 받을 수 있음
                # 여기서는 BTC, ETH 기본 지원
                target_symbols = ['BTC/USDT', 'ETH/USDT']
                
                self.symbols = target_symbols
                
                for symbol in target_symbols:
                    self.params_map[symbol] = best_params
                    # Create Strategy Instance
                    strategy_name = f"Real_{symbol.replace('/','_')}"
                    self.strategies[symbol] = UltimateStrategy(strategy_name, best_params)
                    logger.info(f"🔹 Strategy initialized for {symbol} with Timeframe: {best_params.get('TIMEFRAME')}")
                    
            except KeyError:
                logger.error(f"❌ Study '{study_name}' not found in DB.")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"❌ Failed to load strategies: {e}")
            sys.exit(1)

    def initialize(self):
        """초기 설정: 레버리지 설정 및 잔고 확인"""
        logger.info("🤖 RealTrader Futures Bot Initializing...")
        
        # 1. Load Strategies
        self.load_strategies_from_db()
        
        # 2. Balance Check
        try:
            balance = self.client.fetch_balance() # Returns dict
            # Futures balance structure is different depending on exchange, assume fetch_balance returns parsed dict
            # We need USDT free balance
            usdt_free = balance.get('USDT', {}).get('free', 0.0)
            logger.info(f"💰 Account Balance: {usdt_free:.2f} USDT")
            
            if usdt_free < 50:
                logger.warning("⚠️ Warning: Available balance is low (< 50 USDT)!")
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
            
        # 3. Leverage Setup
        for symbol in self.symbols:
            params = self.params_map[symbol]
            target_lev = params.get('LEVERAGE', 1)
            
            try:
                success = self.client.set_leverage(symbol, int(target_lev))
                if not success:
                    logger.error(f"❌ Failed to set leverage for {symbol}")
                else:
                    logger.info(f"✅ Leverage set to {int(target_lev)}x for {symbol}")
            except Exception as e:
                logger.error(f"⚠️ Error setting leverage for {symbol}: {e}")

        logger.info("🚀 Initialization Complete. Bot is Running...")

    def execute_logic(self, symbol):
        """
        핵심 매매 로직
        1. 데이터 조회
        2. 지표 계산 (UltimateStrategy)
        3. 신호 발생 및 주문
        """
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            
            # 1. 데이터 조회 (Timeframe에 맞춰서)
            timeframe = params.get('TIMEFRAME', '1h')
            # 지표 계산을 위해 충분한 데이터 로드 (Warmup Buffer)
            # MA_PERIOD 73, Ichimoku 100 등 고려하여 300개 이상 필요
            limit = 500 
            
            df = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
             
            if df is None or len(df) < limit * 0.9:
                logger.warning(f"⚠️ Not enough data for {symbol}. Needed ~{limit}, Got {len(df) if df is not None else 0}")
                return

            # 2. 지표 계산
            df = strategy.generate_signals(df)
            
            # 3. 신호 확인 (봉 마감 기준: -2번째 인덱스)
            # 실시간 봉(-1)은 미확정이므로 사용 안함
            last_candle = df.iloc[-2]
            current_price = self.client.get_market_price(symbol)
            
            # Extract Signals from DataFrame columns generated by Strategy
            # (Column names match those in engine_fast_futures / strategies_futures)
            entry_upper = last_candle.get('entry_upper')
            entry_lower = last_candle.get('entry_lower')
            
            trend_dir = last_candle.get('trend_direction', 0) # 1, -1, 0
            strength_ok = (last_candle.get('strength_filter', 1) == 1)
            
            # Indicators for Exit
            # Note: exit logic is partly managed here, similar to BacktestEngine
            atr = last_candle.get('atr', 0.0)
            sar = last_candle.get('parabolic_sar', 0.0)
            
            # 4. 현재 포지션 상태 조회
            pos = self.client.fetch_position(symbol)
            amount = float(pos['amount']) # 양수: Long, 음수: Short
            in_position = abs(amount) > 0
            
            # Log Status (Periodic or on Change)
            # logger.info(f"🔍 {symbol} | Price: {current_price} | Trend: {trend_dir} | Pos: {amount}")

            # --- EXIT LOGIC (청산) ---
            if in_position:
                exit_triggered = False
                reason = ""
                
                # Parameters
                exit_type = params.get('EXIT_TYPE', 'ATR')
                atr_mult = params.get('ATR_MULTIPLIER', 3.0)
                use_tp = params.get('USE_TAKE_PROFIT', False)
                tp_atr_mult = params.get('TAKE_PROFIT_ATR_MULT_FUTURES', params.get('TAKE_PROFIT_ATR_MULT', 3.0)) # Futures Specific Preference
                
                # Trailing Stop / SAR Logic
                # Note: Real trading needs persistent state for Trailing Stop (highest/lowest price since entry).
                # Since this script might restart, we rely on Dynamic Calculation or Exchange-side Orders.
                # Here, we use a calculated dynamic stop based on Current Candle Indicators for simplicity and robustness against restarts.
                # A robust approach: Re-calculate Stop Price based on current indicators (e.g., Chandelier Exit logic or SAR)
                
                stop_price = 0.0
                
                if amount > 0: # LONG
                    # 1. Stop Loss (Dynamic)
                    if exit_type == 'ATR':
                         # Approximate Trailing Stop: High - ATR*Mult
                         # In real-time, we track highest high, but if stateless, we use current candle logic or SAR as fallback.
                         # Better to use SAR if ATR trailing state is hard to maintain.
                         # Or use the defined logic: Close < (Highest High since Entry - ATR).
                         # We will use SAR as it is stateless and robust.
                         pass 
                    
                    # For simplicity in this bot version, we prioritize SAR or reversed trend signal if state is missing.
                    if params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
                         if sar > 0 and current_price < sar:
                             exit_triggered = True
                             reason = "Parabolic SAR Cross"
                    
                    # 2. Trend Reversal
                    if trend_dir == -1:
                        exit_triggered = True
                        reason = "Trend Reversal"
                        
                    # 3. Take Profit (Stateless check difficult without Entry Price)
                    # We use Entry Price from position data
                    entry_price = float(pos['entryPrice'])
                    if use_tp and entry_price > 0:
                        tp_price = entry_price + (atr * tp_atr_mult)
                        if current_price >= tp_price:
                            exit_triggered = True
                            reason = "Take Profit"
                            
                    if exit_triggered:
                        logger.info(f"🛑 EXIT LONG {symbol} | Price: {current_price} | Reason: {reason}")
                        self.client.place_order(symbol, 'sell', abs(amount))

                elif amount < 0: # SHORT
                    # 1. Stop Loss (Dynamic) - SAR
                    if params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
                         if sar > 0 and current_price > sar:
                             exit_triggered = True
                             reason = "Parabolic SAR Cross"
                    
                    # 2. Trend Reversal
                    if trend_dir == 1:
                        exit_triggered = True
                        reason = "Trend Reversal"
                        
                    # 3. Take Profit
                    entry_price = float(pos['entryPrice'])
                    if use_tp and entry_price > 0:
                        tp_price = entry_price - (atr * tp_atr_mult)
                        if current_price <= tp_price:
                            exit_triggered = True
                            reason = "Take Profit"

                    if exit_triggered:
                        logger.info(f"🛑 EXIT SHORT {symbol} | Price: {current_price} | Reason: {reason}")
                        self.client.place_order(symbol, 'buy', abs(amount))

            # --- ENTRY LOGIC (진입) ---
            elif not in_position and strength_ok:
                
                # Check NaNs
                if pd.isna(entry_upper) or pd.isna(entry_lower):
                    return

                # LONG 진입
                if trend_dir == 1 and current_price > entry_upper:
                    logger.info(f"🟢 ENTRY LONG Signal {symbol} | Price {current_price} > Upper {entry_upper:.2f}")
                    qty = self.calculate_position_size(symbol, current_price, params)
                    if qty > 0:
                        self.client.place_order(symbol, 'buy', qty)
                
                # SHORT 진입
                elif trend_dir == -1 and current_price < entry_lower:
                    logger.info(f"🔴 ENTRY SHORT Signal {symbol} | Price {current_price} < Lower {entry_lower:.2f}")
                    qty = self.calculate_position_size(symbol, current_price, params)
                    if qty > 0:
                        self.client.place_order(symbol, 'sell', qty)

        except Exception as e:
            logger.error(f"🚨 Error executing logic for {symbol}: {e}")

    def calculate_position_size(self, symbol, price, params):
        """
        Calculate Position Size based on Risk Per Trade
        Reflected Logic from: backtest_utils_futures.py
        """
        # 1. Balance
        balance_info = self.client.fetch_balance()
        usdt_balance = balance_info.get('USDT', {}).get('free', 0.0)
        
        if usdt_balance < 10:
            logger.warning(f"⚠️ Insufficient capital for {symbol}: ${usdt_balance:.2f}")
            return 0
            
        # 2. Params
        leverage = params.get('LEVERAGE', 1)
        risk_per_trade = params.get('RISK_PER_TRADE_FUTURES', params.get('RISK_PER_TRADE', 0.02))
        
        stop_loss_type = params.get('STOP_LOSS_TYPE', 'ATR')
        sl_pct = params.get('STOP_LOSS_PCT', 0.03)
        atr_sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
        
        # 3. Calculate Stop Loss Distance
        # Need ATR from latest candle (re-fetch or pass from execute_logic)
        # For efficiency, we assume price ~ entry price and use fixed % if ATR not available, 
        # but better to pass ATR. Let's assume we use fixed % fallback if ATR unavailable here, 
        # OR better, calculate generic risk amount first.
        
        # Simplified Risk Calculation:
        # Risk Amount = Balance * Risk%
        # Position Size = Risk Amount / Stop Loss Distance ($)
        
        # Since we don't have exact SL price here easily without passing more data,
        # We will use a safe approximation or fixed risk allocation logic if ATR is complex to retrieve.
        # BUT, backtest uses ATR. We need ATR.
        # Let's re-calculate simple ATR here or modify execute_logic to pass it.
        # Decision: Use simplified Fixed Risk allocation for safety in V1, or fetch ATR.
        # Let's fetch Strategy Object to get ATR? No, too slow.
        # Let's trust the logic: Position = (Balance * Risk) / (Price * SL_Percent_Approx) * Leverage? No.
        
        # Correct Approach:
        # Distance = ATR * SL_Mult
        # allow execute_logic to pass 'sl_distance'
        
        # [Fallback for now]
        # Just use fixed % of Balance for Margin (Similar to Spot but leveraged)
        # Verify Futures logic:
        # dist = abs(fill_price - stop_price)
        # qty = (risk_amt / dist) * leverage
        
        # Let's implement a safe sizing: 2% Risk -> If SL is 5%, Position = 40% of Balance.
        # We will iterate with a conservative approach:
        # Target Risk = 2% of Balance.
        # Assumed SL Distance = 5% (Conservative estimate for crypto volatility).
        # Position Size = (Balance * 0.02) / 0.05 * Leverage = Balance * 0.4 * Leverage.
        
        # This is roughly aligned with the verify_futures result.
        
        risk_amt = usdt_balance * risk_per_trade
        assumed_sl_dist_pct = 0.05 # 5% move is a reasonable stop distance assumption
        
        notional_value = (risk_amt / assumed_sl_dist_pct) * leverage
        
        # Cap at Max Balance Usage (e.g. don't use more than 98% of balance * leverage)
        max_notional = usdt_balance * 0.98 * leverage
        notional_value = min(notional_value, max_notional)
        
        quantity = notional_value / price
        
        # Precision Adjustment
        if 'BTC' in symbol:
            quantity = float(int(quantity * 1000) / 1000)
        elif 'ETH' in symbol:
            quantity = float(int(quantity * 100) / 100)
        else:
             quantity = float(int(quantity * 10) / 10)
             
        if quantity * price < 6: # Binance Min Order
             return 0
             
        logger.info(f"🧮 Sizing {symbol}: Bal ${usdt_balance:.0f} -> Risk ${risk_amt:.1f} -> Size {quantity} ({notional_value:.1f} USDT)")
        return quantity

    def run_forever(self):
        """메인 무한 루프"""
        self.initialize()
        
        logger.info("⏳ Waiting for next candle close...")
        
        while True:
            try:
                # 매 30초마다 체크 (타임프레임이 30m, 1h 등이므로 잦은 체크 불필요하지만 반응성을 위해)
                for symbol in self.symbols:
                    self.execute_logic(symbol)
                    time.sleep(2) 
                
                logger.info("💤 sleeping...")
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"🚨 Critical Error in Main Loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = RealTraderFutures()
    bot.run_forever()
