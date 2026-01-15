from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .indicators import add_common_indicators

class Strategy(ABC):
    def __init__(self, name, params):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, df):
        pass

    def prepare_data(self, df):
        """공통 데이터 전처리 및 지표 계산"""
        # params에 있는 지표 설정값들이 add_common_indicators로 전달됨
        return add_common_indicators(df, self.params)

class MasterStrategy(Strategy):
    """
    Adaptive Regime Trend Follower (Master Strategy)
    - Regime Filter: ADX (Trend Strength) + HMA/EMA (Trend Direction)
    - Entry: Donchian or Bollinger Breakout (Engine handles Donchian, here we prepare indicators)
    - Dynamic Risk: Volatility Targeting (Engine handles this using ATR)
    """
    def generate_signals(self, df):
        df = self.prepare_data(df)
        
        # 1. Regime Filter Line (Trend Direction)
        # EMA 또는 HMA 중 선택된 것을 'regime_line'으로 통일시켜 엔진이 읽게 함
        filter_type = self.params.get('REGIME_FILTER', 'EMA')
        
        if filter_type == 'HMA' and 'hma' in df.columns:
            df['regime_line'] = df['hma']
        elif 'ema_trend' in df.columns:
            df['regime_line'] = df['ema_trend']
        else:
            df['regime_line'] = df['ma50'] # Fallback
            
        # 2. ADX Filter (Trend Strength)
        # ADX가 특정 값보다 낮으면 횡보장으로 간주하여 진입 금지
        if self.params.get('USE_ADX', False) and 'adx' in df.columns:
            adx_threshold = self.params.get('ADX_THRESHOLD', 20)
            # ADX > Threshold 일 때만 1 (True)
            df['adx_filter'] = np.where(df['adx'] > adx_threshold, 1, 0)
        else:
            df['adx_filter'] = 1 # 기본 Pass
            
        return df

from .indicators_advanced import (
    calculate_sma, calculate_ema, calculate_hma, calculate_dema, calculate_tema,
    calculate_supertrend, calculate_atr, calculate_bollinger_bands,
    calculate_keltner_channel, calculate_adx, calculate_vhf, calculate_parabolic_sar,
    calculate_rsi, calculate_stochastic, calculate_macd, calculate_ichimoku, calculate_cci, calculate_mfi
)

class UltimateStrategy(Strategy):
    """
    The Ultimate Strategy: Dynamic combinations of all major indicators.
    """
    def generate_signals(self, df):
        # --- 1. Basic Indicators ---
        # (ATR is always essential for risk management)
        df['atr'] = calculate_atr(df, window=14)
        
        # --- 2. Entry Signal Setup ---
        entry_type = self.params.get('ENTRY_TYPE', 'DONCHIAN')
        entry_period = self.params.get('ENTRY_PERIOD', 20)
        
        if entry_type == 'DONCHIAN':
            # Donchian logic is traditionally handled in Engine using rolling max/min
            # We prepare the bands here for Engine to use
            df['entry_upper'] = df['high'].rolling(window=entry_period).max().shift(1)
            df['entry_lower'] = df['low'].rolling(window=entry_period).min().shift(1)
            
        elif entry_type == 'BOLLINGER':
            std_dev = self.params.get('BB_STD', 2.0)
            up, lo, _ = calculate_bollinger_bands(df, window=entry_period, std_dev=std_dev)
            df['entry_upper'] = up
            df['entry_lower'] = lo
            
        elif entry_type == 'KELTNER':
            # Keltner usually uses ATR as width
            up, lo = calculate_keltner_channel(df, window=entry_period, atr_mult=1.5)
            df['entry_upper'] = up
            df['entry_lower'] = lo
            
        elif entry_type == 'CCI':
            # CCI Breakout
            # Entry when CCI > 100 (Long), CCI < -100 (Short)
            # We map this to Entry Upper/Lower for engine compatibility
            # It's tricky because CCI is an oscillator, not a price level.
            # Only Engine handles price levels. To support CCI, we need to trick the engine.
            # Strategy: If CCI > 100, set entry_upper = close-epsilon (Trigger immediately)
            # Else set entry_upper = infinity
            
            # Since Engine logic is: if close > entry_upper -> LONG
            # We calculate CCI.
            df['cci'] = calculate_cci(df, window=entry_period)
            
            # Create price-based signals
            # If CCI > 100, we want to buy. So we put entry trigger slightly below close
            # If CCI <= 100, we put entry trigger at Infinity (impossible to hit)
            
            # This requires 'shift(1)' because we act on COMPLETED candle signal
            cci_prev = df['cci'].shift(1)
            
            # LONG Trigger
            df['entry_upper'] = np.where(cci_prev > 100, df['close'] * 0.99, np.inf)
            
            # SHORT Trigger
            df['entry_lower'] = np.where(cci_prev < -100, df['close'] * 1.01, -np.inf)
            
        # --- 3. Trend Direction Filter ---
        filter_type = self.params.get('TREND_FILTER_TYPE', 'EMA')
        ma_period = self.params.get('MA_PERIOD', 50)
        
        # Initialize Trend Direction (Default 0: No Trend)
        df['trend_direction'] = 0
            
        if filter_type == 'SMA':
            df['trend_line'] = calculate_sma(df['close'], ma_period)
            df['trend_direction'] = np.where(df['close'] > df['trend_line'], 1, -1)
            
        elif filter_type == 'EMA':
            df['trend_line'] = calculate_ema(df['close'], ma_period)
            df['trend_direction'] = np.where(df['close'] > df['trend_line'], 1, -1)
            
        elif filter_type == 'HMA':
            df['trend_line'] = calculate_hma(df['close'], ma_period)
            df['trend_direction'] = np.where(df['close'] > df['trend_line'], 1, -1)
            
        elif filter_type == 'DEMA':
            df['trend_line'] = calculate_dema(df['close'], ma_period)
            df['trend_direction'] = np.where(df['close'] > df['trend_line'], 1, -1)
            
        elif filter_type == 'TEMA':
            df['trend_line'] = calculate_tema(df['close'], ma_period)
            df['trend_direction'] = np.where(df['close'] > df['trend_line'], 1, -1)
            
        elif filter_type == 'SUPERTREND':
            mul = self.params.get('SUPERTREND_MULT', 3.0)
            per = self.params.get('SUPERTREND_PERIOD', 10)
            df['trend_direction'] = calculate_supertrend(df, period=per, multiplier=mul)
            
        elif filter_type == 'MACD':
            # MACD Line > Signal Line -> Upstream
            macd, signal, hist = calculate_macd(df)
            df['trend_direction'] = np.where(macd > signal, 1, -1)
            
        elif filter_type == 'ICHIMOKU':
            # Price > Cloud (Senkou A & B) -> Upstream
            t, k, sa, sb = calculate_ichimoku(df)
            # Cloud Top/Bottom
            cloud_top = np.maximum(sa, sb)
            cloud_bottom = np.minimum(sa, sb)
            
            df['trend_direction'] = 0
            df['trend_direction'] = np.where(df['close'] > cloud_top, 1, df['trend_direction'])
            df['trend_direction'] = np.where(df['close'] < cloud_bottom, -1, df['trend_direction'])
            
        # --- 4. Strength Filter ---
        # Initialize as Pass(1)
        df['strength_filter'] = 1
        
        if self.params.get('USE_ADX', False):
            adx_thresh = self.params.get('ADX_THRESHOLD', 20)
            df['adx'] = calculate_adx(df, window=14)
            # If ADX < threshold, set filter to 0 (Fail)
            df.loc[df['adx'] < adx_thresh, 'strength_filter'] = 0
            
        if self.params.get('USE_VHF', False):
            vhf_thresh = self.params.get('VHF_THRESHOLD', 0.4)
            df['vhf'] = calculate_vhf(df['close'], window=28)
            # If VHF < threshold (Choppy), set filter to 0
            df.loc[df['vhf'] < vhf_thresh, 'strength_filter'] = 0
        
        if self.params.get('USE_MFI', False):
            mfi_window = self.params.get('MFI_WINDOW', 14)
            mfi_thresh = self.params.get('MFI_THRESHOLD', 20) # Avoid low volume
            df['mfi'] = calculate_mfi(df, window=mfi_window)
            # MFI also works like RSI: Avoid extremes? Or avoid low volume?
            # Standard "Strength" filter: Avoid weak moves -> MFI > Threshold
            df.loc[df['mfi'] < mfi_thresh, 'strength_filter'] = 0

            
        # RSI Filter (Overbought/Oversold Avoidance)
        if self.params.get('USE_RSI', False):
            rsi_window = self.params.get('RSI_WINDOW', 14)
            rsi_overbought = self.params.get('RSI_OVERBOUGHT', 70)
            rsi_oversold = self.params.get('RSI_OVERSOLD', 30)
            df['rsi'] = calculate_rsi(df['close'], window=rsi_window)
            # Avoid entry when RSI is overbought (for LONG) or oversold (for SHORT)
            # For simplicity, we block both extremes
            df.loc[(df['rsi'] > rsi_overbought) | (df['rsi'] < rsi_oversold), 'strength_filter'] = 0
            
        # Stochastic Filter (Momentum Extremes)
        if self.params.get('USE_STOCHASTIC', False):
            stoch_window = self.params.get('STOCH_WINDOW', 14)
            stoch_overbought = self.params.get('STOCH_OVERBOUGHT', 80)
            stoch_oversold = self.params.get('STOCH_OVERSOLD', 20)
            stoch_k, stoch_d = calculate_stochastic(df, window=stoch_window)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            # Block extremes
            df.loc[(df['stoch_k'] > stoch_overbought) | (df['stoch_k'] < stoch_oversold), 'strength_filter'] = 0
            
        # --- 5. Exit Logic (Parabolic SAR) ---
        # ATR exit is handled by Engine risk management. 
        # But if Parabolic SAR is selected, we need to calculate it.
        if self.params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
            step = self.params.get('SAR_STEP', 0.02)
            sar_line, _ = calculate_parabolic_sar(df, step=step)
            df['parabolic_sar'] = sar_line
            
        return df

# --- Legacy Strategies ---
