from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Strategy(ABC):
    def __init__(self, name, params):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, df):
        pass

class MasterStrategy(Strategy):
    """
    Adaptive Regime Trend Follower (Master Strategy)
    - Regime Filter: ADX (Trend Strength) + HMA/EMA (Trend Direction)
    - Entry: Donchian or Bollinger Breakout (Engine handles Donchian, here we prepare indicators)
    - Dynamic Risk: Volatility Targeting (Engine handles this using ATR)
    """
    def generate_signals(self, df):
        # Note: 이 전략은 더 이상 사용되지 않음 (UNIFIED 전략으로 통합됨)
        # Legacy compatibility 유지용
        
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
    calculate_rsi, calculate_stochastic, calculate_stoch_rsi, calculate_macd, calculate_ichimoku, calculate_cci, calculate_mfi,
    calculate_vwap, calculate_cmf, calculate_hurst_exponent, calculate_natr
)

class UltimateStrategy(Strategy):
    """
    The Ultimate Strategy: Dynamic combinations of all major indicators.
    """
    def generate_signals(self, df):
        # [ROBUSTNESS] Ensure clean column assignment
        # Data is already copied at loading stage (optimize/verify scripts)
        
        # --- 1. Basic Indicators (Lazy ATR) ---
        # ATR is needed only for certain exit/SL/sizing types
        use_tp = self.params.get('USE_TAKE_PROFIT', False)
        use_atr_sl = self.params.get('STOP_LOSS_TYPE') == 'ATR'
        use_trailing = self.params.get('EXIT_TYPE') == 'ATR' or self.params.get('TRAILING_ACTIVATION_ATR', 0) > 0
        
        if use_tp or use_atr_sl or use_trailing:
            atr_period = self.params.get('ATR_PERIOD', 14)
            df['atr'] = calculate_atr(df, window=atr_period).astype(np.float32)
        else:
            df['atr'] = np.float32(0.0) # Not used
        
        # [NEW] Always calculate Regime Indicators (Hurst, NATR, RSI)
        # 1. Hurst Exponent (Trend Strength/Regime)
        hurst_period = self.params.get('HURST_PERIOD', 200)
        df['hurst'] = calculate_hurst_exponent(df['close'], window=hurst_period)
        
        # 2. NATR (Volatility Regime)
        natr_period = self.params.get('STRENGTH_FILTER_PERIOD', 14) # Reuse existing period or default
        df['natr'] = calculate_natr(df, window=natr_period)
        
        # 3. RSI (Panic Exit)
        rsi_period = self.params.get('STRENGTH_FILTER_PERIOD', 14) 
        df['rsi'] = calculate_rsi(df['close'], window=rsi_period)

        # --- 2. Entry Signal Setup ---
        entry_type = self.params.get('ENTRY_TYPE', 'DONCHIAN')
        entry_period = self.params.get('ENTRY_PERIOD', 20)
        
        # Initialize defaults to avoid Engine .get() defaults
        df['entry_upper'] = np.nan
        df['entry_lower'] = np.nan
        
        if entry_type == 'DONCHIAN':
            df['entry_upper'] = df['high'].rolling(window=entry_period).max().shift(1)
            df['entry_lower'] = df['low'].rolling(window=entry_period).min().shift(1)
            
        elif entry_type == 'BOLLINGER':
            std_dev = self.params.get('BB_STD', 2.0)
            up, lo, _ = calculate_bollinger_bands(df, window=entry_period, std_dev=std_dev)
            df['entry_upper'] = up.shift(1)
            df['entry_lower'] = lo.shift(1)
            
        elif entry_type == 'KELTNER':
            k_mult = self.params.get('KELTNER_ATR_MULT', 1.5)
            up, lo = calculate_keltner_channel(df, window=entry_period, atr_mult=k_mult)
            df['entry_upper'] = up.shift(1)
            df['entry_lower'] = lo.shift(1)
            
        elif entry_type == 'CCI':
            df['cci'] = calculate_cci(df, window=entry_period)
            cci_prev = df['cci'].shift(1)
            cci_thresh = self.params.get('CCI_THRESHOLD', 100)
            prev_high = df['high'].shift(1)
            df['entry_upper'] = np.where(cci_prev > cci_thresh, prev_high, np.inf)
            prev_low = df['low'].shift(1)
            df['entry_lower'] = np.where(cci_prev < -cci_thresh, prev_low, -np.inf)
            
        # --- 3. Trend Direction Filter ---
        filter_type = self.params.get('TREND_FILTER_TYPE', 'EMA')
        ma_period = self.params.get('MA_PERIOD', 50)
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
            macd_line, signal_line, _ = calculate_macd(df, fast=self.params.get('MACD_FAST', 12), slow=self.params.get('MACD_SLOW', 26), signal=self.params.get('MACD_SIGNAL', 9))
            df['trend_direction'] = np.where(macd_line > signal_line, 1, -1)
        elif filter_type == 'ICHIMOKU':
            t, k, sa, sb = calculate_ichimoku(df, tenkan_window=self.params.get('ICHIMOKU_TENKAN', 9), kijun_window=self.params.get('ICHIMOKU_KIJUN', 26), senkou_span_b_window=self.params.get('ICHIMOKU_SENKOU_B', 52))
            cloud_top = np.maximum(sa, sb)
            cloud_bottom = np.minimum(sa, sb)
            df['trend_direction'] = 0
            # [FIX] Use .values to avoid index mismatch from shifted Ichimoku series
            df['trend_direction'] = np.where(df['close'].values > cloud_top.values, 1, df['trend_direction'].values)
            df['trend_direction'] = np.where(df['close'].values < cloud_bottom.values, -1, df['trend_direction'].values)
        elif filter_type == 'VWAP':
            vwap, _, _ = calculate_vwap(df, window=ma_period, std_mult=self.params.get('VWAP_STD_MULT', 1.5))
            df['trend_direction'] = np.where(df['close'] > vwap, 1, -1)
            
        # --- 4. Strength Filter ---
        df['strength_filter'] = 1
        strength_type = self.params.get('STRENGTH_FILTER_TYPE', 'NONE')
        strength_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
        
        if strength_type == 'ADX':
            df['adx'] = calculate_adx(df, window=strength_period)
            df.loc[df['adx'] < self.params.get('ADX_THRESHOLD', 20), 'strength_filter'] = 0
        elif strength_type == 'VHF':
            df['vhf'] = calculate_vhf(df['close'], window=strength_period)
            df.loc[df['vhf'] < self.params.get('VHF_THRESHOLD', 0.4), 'strength_filter'] = 0
        elif strength_type == 'MFI':
            df['mfi'] = calculate_mfi(df, window=strength_period)
            df.loc[df['mfi'] < self.params.get('MFI_THRESHOLD', 25), 'strength_filter'] = 0
        elif strength_type == 'RSI':
            # Use pre-calculated RSI
            rsi_upper = self.params.get('RSI_OVERBOUGHT', 75)
            rsi_lower = self.params.get('RSI_OVERSOLD', 25)
            df.loc[(df['rsi'] > rsi_upper) | (df['rsi'] < rsi_lower), 'strength_filter'] = 0
        elif strength_type == 'STOCHASTIC':
            stoch_k, _ = calculate_stochastic(df, window=strength_period)
            df.loc[(stoch_k > self.params.get('STOCH_OVERBOUGHT', 85)) | (stoch_k < self.params.get('STOCH_OVERSOLD', 15)), 'strength_filter'] = 0
        elif strength_type == 'STOCH_RSI':
            stoch_rsi_k, _ = calculate_stoch_rsi(df['close'], window=strength_period)
            df.loc[(stoch_rsi_k > self.params.get('STOCH_RSI_OVERBOUGHT', 80)) | (stoch_rsi_k < self.params.get('STOCH_RSI_OVERSOLD', 20)), 'strength_filter'] = 0
        elif strength_type == 'CMF':
            df['cmf'] = calculate_cmf(df, window=self.params.get('CMF_PERIOD', 20))
            df.loc[df['cmf'] < self.params.get('CMF_THRESHOLD', 0.05), 'strength_filter'] = 0
        elif strength_type == 'HURST':
            # Use pre-calculated Hurst
            df.loc[df['hurst'] < self.params.get('HURST_RANDOM_THRESHOLD', 0.50), 'strength_filter'] = 0
        elif strength_type == 'NATR':
             # Use pre-calculated NATR
             # NATR < Threshold: Low volatility -> Block Entry
             natr_thresh = self.params.get('NATR_THRESHOLD', 1.0)
             df.loc[df['natr'] < natr_thresh, 'strength_filter'] = 0

        # --- 5. Exit Logic (Parabolic SAR) ---
        if self.params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
            sar_line, _ = calculate_parabolic_sar(df, step=self.params.get('SAR_STEP', 0.02))
            df.loc[:, 'parabolic_sar'] = sar_line
        else:
            df.loc[:, 'parabolic_sar'] = 0.0
            
        # --- 6. Volume Filter ---
        if self.params.get('USE_VOLUME_FILTER', False):
            vol_ma = df['volume'].rolling(window=self.params.get('VOLUME_MA_PERIOD', 20)).mean()
            df.loc[:, 'volume_ratio'] = df['volume'] / vol_ma.replace(0, 1)
        else:
            df.loc[:, 'volume_ratio'] = 100.0  # High ratio to pass filter (aligned with Futures)
            
        return df

# --- Legacy Strategies ---
