import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Project Root Setup
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from config.settings import BINANCE_API_KEY, BINANCE_SECRET
from src.data.binance_client import BinanceClient
from src.strategy.strategies import UltimateStrategy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_trader.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealTrader")

# --- 🏆 BTC UNIVERSAL STRATEGY PARAMETERS (VERIFIED) ---
# 2026-01-16 검증 완료: Sharpe 2.56, MDD -20.5%, p-value 0.01 (PASSED)
STRATEGY_PARAMS = {
    'TIMEFRAME': '5m',
    'LEVERAGE': 1.3,
    'RISK_PER_TRADE': 0.024, # 자본금의 2.4% 리스크 (검증된 최적값: MDD -26%)
    
    # Entry: Bollinger Bands
    'ENTRY_TYPE': 'BOLLINGER',
    'ENTRY_PERIOD': 31,
    'BB_STD': 1.6,
    
    # Trend: SuperTrend
    'TREND_FILTER_TYPE': 'SUPERTREND',
    'SUPERTREND_MULT': 4.9,
    'SUPERTREND_PERIOD': 44,
    'MA_PERIOD': 105, # (Not used for direction but kept for compatibility)
    
    # Strength Filters (MFI Only)
    'USE_ADX': False,
    'ADX_THRESHOLD': 30,
    'USE_VHF': False,
    'VHF_THRESHOLD': 0.52,
    'USE_MFI': True,
    'MFI_WINDOW': 20,
    'MFI_THRESHOLD': 32,
    'USE_RSI': False,
    'USE_STOCHASTIC': False,
    'USE_VOLUME_FILTER': False,
    
    # Exit: Parabolic SAR
    'EXIT_TYPE': 'PARABOLIC_SAR',
    'SAR_STEP': 0.045,
    'USE_TAKE_PROFIT': False
}

# --- 💰 CAPITAL ALLOCATION (USDT) ---
# BTC 단일 전략이므로 가용 자산의 대부분을 활용
# 로직 코드에서 동적으로 잔고를 조회하여 사용함
ALLOCATION = {
    'BTC/USDT': 'DYNAMIC' # 100% of Free Balance
}

class RealTrader:
    def __init__(self):
        self.client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        self.strategy = UltimateStrategy("Real_BTC_Bot", STRATEGY_PARAMS)
        self.symbols = ['BTC/USDT'] # [UPDATE] BTC Only
        
        # State tracking
        self.positions = {s: {'amount': 0.0} for s in self.symbols}
        
    def initialize(self):
        """초기 설정: 레버리지 설정 및 상태 점검"""
        logger.info("🤖 RealTrader BTC-Bot Initializing...")
        
        # 잔고 확인
        total, free = self.client.fetch_balance()
        # [SECURITY] 잔고 액수 로그 삭제 (Masking)
        logger.info(f"💰 Account Balance Check: OK (Has Funds)")
        
        if free < 20: # Minimum updated
            logger.warning("⚠️ Warning: Available balance is very low!")
            
        # 레버리지 설정
        target_lev = STRATEGY_PARAMS['LEVERAGE']
        for symbol in self.symbols:
            success = self.client.set_leverage(symbol, int(target_lev))
            if not success:
                logger.error(f"❌ Failed to set leverage for {symbol}")
                sys.exit(1)
                
            # 기존 포지션 확인
            pos = self.client.fetch_position(symbol)
            self.positions[symbol] = pos
            if pos['amount'] != 0:
                logger.info(f"⚠️ Existing Position Found for {symbol}: {pos['amount']} contracts")
            else:
                logger.info(f"✅ No existing position for {symbol}")
                
        logger.info("🚀 Initialization Complete. BTC Bot is Running...")

    def execute_logic(self, symbol):
        """핵심 매매 로직 (데이터 조회 -> 신호 -> 주문)"""
        try:
            # 1. 데이터 조회 (충분한 길이 확보 - 최근 48시간)
            # [FIX] fetch_ohlcv에 None을 주면 안됨. 명시적 날짜 계산.
            start_dt = datetime.now() - pd.Timedelta(days=2) # 2일치 데이터면 충분 (5분봉)
            start_str = start_dt.strftime("%Y-%m-%d")
            
            lookback = 300 # 지표 계산용 여유분
            df = self.client.fetch_ohlcv(symbol, STRATEGY_PARAMS['TIMEFRAME'], 
                                         start_date=start_str)
            
            if len(df) < lookback:
                logger.warning(f"Not enough data for {symbol}. Waiting...")
                return 

            # 2. 지표 계산 및 신호 생성
            # UltimateStrategy는 generate_signals에서 모든 지표를 계산함
            df = self.strategy.generate_signals(df)
            
            # 3. 최신 봉 정보 (확정된 봉 기준: -2번째 인덱스)
            # 실시간 봉(-1)은 변하므로, 직전 마감 봉(-2)을 보고 진입하는 것이 정석
            last_candle = df.iloc[-2] 
            current_price = self.client.get_market_price(symbol)
            
            # 신호 파싱
            # UltimateStrategy는 'entry_upper', 'entry_lower', 'strength_filter' 등을 계산해둠
            trend_dir = last_candle.get('trend_direction', 0)
            strength_ok = (last_candle.get('strength_filter', 1) == 1)
            
            # Entry Thresholds
            entry_upper = last_candle.get('entry_upper')
            entry_lower = last_candle.get('entry_lower')
            
            # Exit Thresholds
            sar = last_candle.get('psar') # Parabolic SAR value
            
            # 현재 포지션 상태
            current_pos = self.client.fetch_position(symbol)
            amount = current_pos['amount']
            in_position = abs(amount) > 0
            
            logger.info(f"🔍 {symbol} | Price: {current_price} | Trend: {trend_dir} | Strength: {strength_ok}")
            
            # --- EXIT LOGIC (청산) ---
            if in_position:
                # LONG 청산 조건: 가격이 SAR 아래로 내려감 (SAR는 상승 시 가격 밑에 있음)
                if amount > 0:
                    if current_price < sar:
                        logger.info(f"🛑 EXIT LONG Signal: Price {current_price} < SAR {sar}")
                        self.client.place_order(symbol, 'sell', abs(amount)) # Close Long
                
                # SHORT 청산 조건: 가격이 SAR 위로 올라감 (SAR는 하락 시 가격 위에 있음)
                elif amount < 0:
                    if current_price > sar:
                        logger.info(f"🛑 EXIT SHORT Signal: Price {current_price} > SAR {sar}")
                        self.client.place_order(symbol, 'buy', abs(amount)) # Close Short
                        
            # --- ENTRY LOGIC (진입) ---
            # 포지션이 없을 때만 진입 (피라미딩 없음)
            elif not in_position and strength_ok:
                
                # LONG 진입
                if trend_dir == 1 and current_price > entry_upper:
                    logger.info(f"🟢 ENTRY LONG Signal: Price {current_price} > Upper {entry_upper}")
                    qty = self.calculate_position_size(symbol, current_price)
                    if qty > 0:
                        self.client.place_order(symbol, 'buy', qty)
                
                # SHORT 진입
                elif trend_dir == -1 and current_price < entry_lower:
                    logger.info(f"🔴 ENTRY SHORT Signal: Price {current_price} < Lower {entry_lower}")
                    qty = self.calculate_position_size(symbol, current_price)
                    if qty > 0:
                        self.client.place_order(symbol, 'sell', qty)

        except Exception as e:
            logger.error(f"Error in execution for {symbol}: {e}")

    def calculate_position_size(self, symbol, price):
        """
        [Dynamic Sizing]
        Uses 98% of available USDT balance with Target Leverage.
        Order Cost = Balance * 0.98
        Notional Value = Order Cost * Leverage
        Quantity = Notional / Price
        """
        # Fetch latest Free Balance dynamically
        total, free_usdt = self.client.fetch_balance()
        
        # Use 98% of free balance to leave dust for fees
        trade_capital = free_usdt * 0.98
        
        if trade_capital < 10:
             logger.warning(f"⚠️ Insufficient capital: ${trade_capital:.2f}")
             return 0

        leverage = STRATEGY_PARAMS['LEVERAGE']
        
        # 투입 가능 명목 금액 (Notional Value)
        notional_value = trade_capital * leverage
        
        # 수량 계산 (코인 개수)
        quantity = notional_value / price
        
        # BTC Precision (0.001 usually)
        if symbol == 'BTC/USDT':
            quantity = float(int(quantity * 1000) / 1000)
        else:
            quantity = float(int(quantity * 100) / 100)
            
        # 바이낸스 최소 주문액 확인
        if quantity * price < 6:
            logger.warning(f"⚠️ Quantity too small: {quantity} {symbol}")
            return 0
            
        logger.info(f"🧮 Sizing: Balance [MASKED] -> Bet [MASKED] x {leverage}Lev = {quantity} BTC")
        return quantity

    def run_forever(self):
        """메인 무한 루프"""
        self.initialize()
        
        logger.info("⏳ Waiting for next candle close...")
        
        while True:
            try:
                # 현재 시간 확인
                now = datetime.now()
                
                # 매 1분마다 체크 (상태 기반이므로 중복 진입 안함)
                # 실제 API 부하는 execute_logic 안에서 데이터 조회 할 때 발생
                # 5분봉 전략이므로 1분 주기로 체크해도 충분함
                
                for symbol in self.symbols:
                    self.execute_logic(symbol)
                    time.sleep(2) # API Rate Limit 방지
                
                # 1분 대기
                logger.info("💤 Sleeping for 60s...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"🚨 Critical Error in Main Loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = RealTrader()
    bot.run_forever()
