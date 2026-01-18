
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .indicators_futures import add_common_indicators

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

from .indicators_advanced_futures import (
    calculate_sma, calculate_ema, calculate_hma, calculate_dema, calculate_tema,
    calculate_supertrend, calculate_atr, calculate_bollinger_bands,
    calculate_keltner_channel, calculate_adx, calculate_vhf, calculate_parabolic_sar,
    calculate_rsi, calculate_stochastic, calculate_macd, calculate_ichimoku, calculate_cci, calculate_mfi
)

class UltimateStrategy(Strategy):
    """
    The Ultimate Strategy: Dynamic combinations of all major indicators.
    (Aligned with Spot V2 Logic)
    """
    def generate_signals(self, df):
        # --- 1. Basic Indicators ---
        # (ATR is always essential for risk management)
        atr_period = self.params.get('ATR_PERIOD', 14)
        df['atr'] = calculate_atr(df, window=atr_period)
        
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
            # CCI Breakout (Realistic Implementation)
            # Entry when CCI crosses above 100 (Long) or below -100 (Short)
            # Use PREVIOUS candle's high/low as realistic entry trigger
            
            df['cci'] = calculate_cci(df, window=entry_period)
            
            # Shift CCI by 1 to avoid look-ahead bias
            cci_prev = df['cci'].shift(1)
            
            # LONG Trigger: If previous CCI > 100, enter at breakout above previous high
            prev_high = df['high'].shift(1)
            df['entry_upper'] = np.where(cci_prev > 100, prev_high, np.inf)
            
            # SHORT Trigger: If previous CCI < -100, enter at breakdown below previous low
            prev_low = df['low'].shift(1)
            df['entry_lower'] = np.where(cci_prev < -100, prev_low, -np.inf)
            
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
            fast = self.params.get('MACD_FAST', 12)
            slow = self.params.get('MACD_SLOW', 26)
            signal = self.params.get('MACD_SIGNAL', 9)
            macd_line, signal_line, hist = calculate_macd(df, fast=fast, slow=slow, signal=signal)
            df['trend_direction'] = np.where(macd_line > signal_line, 1, -1)
            
        elif filter_type == 'ICHIMOKU':
            # Price > Cloud (Senkou A & B) -> Upstream
            t_win = self.params.get('ICHIMOKU_TENKAN', 9)
            k_win = self.params.get('ICHIMOKU_KIJUN', 26)
            s_win = self.params.get('ICHIMOKU_SENKOU_B', 52)
            t, k, sa, sb = calculate_ichimoku(df, tenkan_window=t_win, kijun_window=k_win, senkou_span_b_window=s_win)
            # Cloud Top/Bottom
            cloud_top = np.maximum(sa, sb)
            cloud_bottom = np.minimum(sa, sb)
            
            df['trend_direction'] = 0
            df['trend_direction'] = np.where(df['close'] > cloud_top, 1, df['trend_direction'])
            df['trend_direction'] = np.where(df['close'] < cloud_bottom, -1, df['trend_direction'])
            
        # --- 4. Strength Filter ---
        # Initialize as Pass(1)
        df['strength_filter'] = 1
        
        strength_type = self.params.get('STRENGTH_FILTER_TYPE', 'NONE')
        
        if strength_type == 'ADX':
            adx_thresh = self.params.get('ADX_THRESHOLD', 20)
            strength_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
            df['adx'] = calculate_adx(df, window=strength_period)
            df.loc[df['adx'] < adx_thresh, 'strength_filter'] = 0
            
        elif strength_type == 'VHF':
            vhf_thresh = self.params.get('VHF_THRESHOLD', 0.4)
            strength_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
            df['vhf'] = calculate_vhf(df['close'], window=strength_period)
            df.loc[df['vhf'] < vhf_thresh, 'strength_filter'] = 0
            
        elif strength_type == 'MFI':
            mfi_thresh = self.params.get('MFI_THRESHOLD', 25)
            strength_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
            df['mfi'] = calculate_mfi(df, window=strength_period)
            df.loc[df['mfi'] < mfi_thresh, 'strength_filter'] = 0
            
        elif strength_type == 'RSI':
            # Use Futures specific parameter if available, else default to standard
            rsi_overbought = self.params.get('RSI_OVERBOUGHT_FUTURES', self.params.get('RSI_OVERBOUGHT', 75))
            rsi_oversold = self.params.get('RSI_OVERSOLD', 25)
            strength_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
            df['rsi'] = calculate_rsi(df['close'], window=strength_period)
            # Avoid extremes
            df.loc[(df['rsi'] > rsi_overbought) | (df['rsi'] < rsi_oversold), 'strength_filter'] = 0
            
        elif strength_type == 'STOCHASTIC':
            stoch_overbought = self.params.get('STOCH_OVERBOUGHT', 85)
            stoch_oversold = self.params.get('STOCH_OVERSOLD', 15)
            strength_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
            stoch_k, _ = calculate_stochastic(df, window=strength_period)
            df['stoch_k'] = stoch_k
            # Block extremes
            df.loc[(df['stoch_k'] > stoch_overbought) | (df['stoch_k'] < stoch_oversold), 'strength_filter'] = 0

        # --- 5. Exit Logic (Parabolic SAR) ---
        # Calculated here so it's available for the backtest engine
        if self.params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
            sar_step = self.params.get('SAR_STEP', 0.02)
            sar_line, _ = calculate_parabolic_sar(df, step=sar_step)
            df['parabolic_sar'] = sar_line
        else:
            df['parabolic_sar'] = 0.0 # Default
            
        # --- 6. Volume Filter (Ratio) ---
        if self.params.get('USE_VOLUME_FILTER', False):
            vol_ma_period = self.params.get('VOLUME_MA_PERIOD', 20)
            # Avoid division by zero
            vol_ma = df['volume'].rolling(window=vol_ma_period).mean()
            df['volume_ratio'] = df['volume'] / vol_ma.replace(0, 1)
        else:
            df['volume_ratio'] = 100.0 # Default Pass (High ratio)
            
        return df
