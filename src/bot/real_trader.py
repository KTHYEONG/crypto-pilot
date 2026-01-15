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

# --- 🏆 UNIVERSAL STRATEGY PARAMETERS ---
# 우리가 찾은 최적의 파라미터 (2025 검증 완료)
STRATEGY_PARAMS = {
    'TIMEFRAME': '15m',
    'LEVERAGE': 1.5,  # 실전 안전 권장값 (1.5x)
    'RISK_PER_TRADE': 0.05, # 한 거래당 리스크 (자본금의 5%)
    
    # Entry
    'ENTRY_TYPE': 'KELTNER',
    'ENTRY_PERIOD': 80,
    'ATR_MULTIPLIER': 4.7, # Keltner uses ATR
    
    # Trend
    'TREND_FILTER_TYPE': 'SUPERTREND',
    'MA_PERIOD': 111, # Not used for SuperTrend but kept
    'SUPERTREND_MULT': 1.8,
    'SUPERTREND_PERIOD': 40,
    
    # Strength
    'USE_ADX': False,
    'ADX_THRESHOLD': 16,
    'USE_VHF': False,
    'VHF_THRESHOLD': 0.39,
    'USE_MFI': True,
    'MFI_WINDOW': 18,
    'MFI_THRESHOLD': 31,
    'USE_RSI': False,
    'USE_STOCHASTIC': False,
    
    # Exit
    'EXIT_TYPE': 'PARABOLIC_SAR',
    'SAR_STEP': 0.048
}

# --- 💰 CAPITAL ALLOCATION (USDT) ---
# 총 자본 100만원 약 750 USDT 가정
# 각 봇당 350 USDT 할당 (안전마진 포함)
ALLOCATION = {
    'BTC/USDT': 350,
    'ETH/USDT': 350
}

class RealTrader:
    def __init__(self):
        self.client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        self.strategy = UltimateStrategy("Real_Universal", STRATEGY_PARAMS)
        self.symbols = ['BTC/USDT', 'ETH/USDT']
        
        # State tracking
        self.positions = {s: {'amount': 0.0} for s in self.symbols}
        
    def initialize(self):
        """초기 설정: 레버리지 설정 및 상태 점검"""
        logger.info("🤖 RealTrader Initializing...")
        
        # 잔고 확인
        total, free = self.client.fetch_balance()
        logger.info(f"💰 Account Balance: Total ${total:.2f} | Available ${free:.2f}")
        
        if free < 50:
            logger.warning("⚠️ Warning: Available balance is very low for trading!")
            
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
                
        logger.info("🚀 Initialization Complete. Analysis Loop Starting...")

    def execute_logic(self, symbol):
        """핵심 매매 로직 (데이터 조회 -> 신호 -> 주문)"""
        try:
            # 1. 데이터 조회 (충분한 길이 확보)
            lookback = 300 # 지표 계산용 여유분
            df = self.client.fetch_ohlcv(symbol, STRATEGY_PARAMS['TIMEFRAME'], 
                                         start_date=None) # 최근 데이터 자동
            
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
        """할당된 자본(USDT)에 맞춰 주문 수량 계산"""
        budget_usdt = ALLOCATION.get(symbol, 100)
        leverage = STRATEGY_PARAMS['LEVERAGE']
        
        # 투입 가능 명목 금액 (Notional Value)
        notional_value = budget_usdt * leverage
        
        # 수량 계산 (코인 개수)
        quantity = notional_value / price
        
        # 최소 주문 수량 및 정밀도 보정 (간이 로직)
        # 실제로는 symbol info를 조회해서 precision을 맞춰야 함.
        # 여기서는 안전하게 소수점 3자리로 버림 (ETH 등 고려)
        if symbol == 'BTC/USDT':
            quantity = float(int(quantity * 1000) / 1000) # 0.001 단위
        else:
            quantity = float(int(quantity * 100) / 100)   # 0.01 단위
            
        # 바이낸스 최소 주문액 (약 5 USDT) 확인
        if quantity * price < 6:
            logger.warning(f"⚠️ Calculated quantity too small: {quantity} {symbol} (< $6)")
            return 0
            
        return quantity

    def run_forever(self):
        """메인 무한 루프"""
        self.initialize()
        
        logger.info("⏳ Waiting for next candle close...")
        
        while True:
            try:
                # 현재 시간 확인
                now = datetime.now()
                
                # 매 15분 단위 (00, 15, 30, 45분) 정각에 실행하되,
                # 데이터 집계 시간을 고려하여 10초 딜레이
                # 간단하게 1분마다 체크하여 로직 실행 (상태 기반이므로 중복 진입 안함)
                
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
