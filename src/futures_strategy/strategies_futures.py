
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
    
    def get_required_warmup(self):
        """
        Calculate minimum warmup bars required based on strategy parameters.
        Analyzes all period-based parameters and returns max lookback with safety factor.
        
        Returns:
            int: Minimum number of bars to skip at start of backtest
        """
        # List of all period-related parameter keys
        period_keys = [
            'ENTRY_PERIOD', 'MA_PERIOD', 'ATR_PERIOD',
            'SUPERTREND_PERIOD', 'MACD_SLOW', 'ICHIMOKU_SENKOU_B',
            'STRENGTH_FILTER_PERIOD', 'VOLUME_MA_PERIOD',
            'CMF_PERIOD', 'HURST_PERIOD'
        ]
        
        max_period = 0
        for key in period_keys:
            if key in self.params:
                period = self.params[key]
                if period > max_period:
                    max_period = period
        
        # Safety Factor: 3x for EMA/ATR convergence
        # EMA needs ~3*period bars to reach 95% accuracy
        warmup = max_period * 3
        
        # Absolute minimum: 50 bars (even if no periods found)
        return max(warmup, 50)

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

from .indicators_advanced_futures import (
    calculate_sma, calculate_ema, calculate_hma, calculate_dema, calculate_tema,
    calculate_supertrend, calculate_atr, calculate_bollinger_bands,
    calculate_keltner_channel, calculate_adx, calculate_vhf, calculate_parabolic_sar,
    calculate_rsi, calculate_stochastic, calculate_stoch_rsi, calculate_macd, calculate_ichimoku, calculate_cci, calculate_mfi,
    calculate_vwap, calculate_cmf, calculate_hurst_exponent,
    calculate_efficiency_ratio, calculate_natr, calculate_garman_klass_vol
)

class UltimateStrategy(Strategy):
    """
    The Ultimate Strategy: Dynamic combinations of all major indicators.
    (Aligned with Spot V2 Logic)
    """
    def generate_signals(self, df):
        # [ROBUSTNESS] Clean column assignment (data copied at loader level)
        
        # --- 1. Basic Indicators (Always calculate ATR) ---
        # ATR is fundamental for volatility normalization, even if not used for Entry
        atr_period = self.params.get('ATR_PERIOD', 14)
        df.loc[:, 'atr'] = calculate_atr(df, window=atr_period)
        
        # [NEW] Always calculate RSI for Engine's Panic Exit checking
        # Even if strength_filter='NONE', engine might need RSI for exits
        rsi_period = self.params.get('STRENGTH_FILTER_PERIOD', 14) 
        df.loc[:, 'rsi'] = calculate_rsi(df['close'], window=rsi_period)
        
        # [NEW] Always calculate Hurst & NATR for Dynamic Risk Sizing (Regime Detection)
        hurst_period = self.params.get('HURST_PERIOD', 200)
        df.loc[:, 'hurst'] = calculate_hurst_exponent(df['close'], window=hurst_period)
        
        natr_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
        df.loc[:, 'natr'] = calculate_natr(df, window=natr_period)
        
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
            k_mult = self.params.get('KELTNER_ATR_MULT', 1.5)
            up, lo = calculate_keltner_channel(df, window=entry_period, atr_mult=k_mult)
            df['entry_upper'] = up
            df['entry_lower'] = lo
            
        elif entry_type == 'CCI':
            # CCI Breakout (Realistic Implementation)
            # Entry when CCI crosses above threshold (Long) or below -threshold (Short)
            
            df['cci'] = calculate_cci(df, window=entry_period)
            
            # Shift CCI by 1 to avoid look-ahead bias
            cci_prev = df['cci'].shift(1)
            cci_thresh = self.params.get('CCI_THRESHOLD', 100)
            
            # LONG Trigger: If previous CCI > threshold, enter at breakout above previous high
            prev_high = df['high'].shift(1)
            df['entry_upper'] = np.where(cci_prev > cci_thresh, prev_high, np.inf)
            
            # SHORT Trigger: If previous CCI < -threshold, enter at breakdown below previous low
            prev_low = df['low'].shift(1)
            df['entry_lower'] = np.where(cci_prev < -cci_thresh, prev_low, -np.inf)
            
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
            cloud_top = np.maximum(sa.to_numpy(), sb.to_numpy())
            cloud_bottom = np.minimum(sa.to_numpy(), sb.to_numpy())

            close_np = df['close'].to_numpy()
            trend = np.zeros(len(df), dtype=np.int8)
            trend[close_np > cloud_top] = 1
            trend[close_np < cloud_bottom] = -1
            df['trend_direction'] = trend
            
        elif filter_type == 'VWAP':
            # [NEW] VWAP Trend Filter
            # Price > VWAP: Long Only (1), Price < VWAP: Short Only (-1)
            # Institutional algorithm benchmark
            std_mult = self.params.get('VWAP_STD_MULT', 1.5)
            # Use rolling VWAP for consistency (use MA_PERIOD as window)
            vwap, vwap_upper, vwap_lower = calculate_vwap(df, window=ma_period, std_mult=std_mult)
            df['vwap'] = vwap
            df['vwap_upper'] = vwap_upper
            df['vwap_lower'] = vwap_lower
            df['trend_direction'] = np.where(df['close'] > df['vwap'], 1, -1)
            
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
 
        elif strength_type == 'STOCH_RSI':
            stoch_rsi_overbought = self.params.get('STOCH_RSI_OVERBOUGHT', 80)
            stoch_rsi_oversold = self.params.get('STOCH_RSI_OVERSOLD', 20)
            strength_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
            stoch_rsi_k, _ = calculate_stoch_rsi(df['close'], window=strength_period)
            df['stoch_rsi_k'] = stoch_rsi_k
            # Block extremes
            df.loc[(df['stoch_rsi_k'] > stoch_rsi_overbought) | (df['stoch_rsi_k'] < stoch_rsi_oversold), 'strength_filter'] = 0
 
        elif strength_type == 'CMF':
            # [NEW] Chaikin Money Flow - Institutional Accumulation/Distribution
            # 추세 추종: CMF > Threshold일 때만 진입 (양수 자금 유입 확인)
            # CMF < Threshold: 매도 압력 또는 낮은 매수 압력 -> Block Entry
            cmf_thresh = self.params.get('CMF_THRESHOLD', 0.05)
            cmf_period = self.params.get('CMF_PERIOD', 20)
            df['cmf'] = calculate_cmf(df, window=cmf_period)
            # Block when CMF is below threshold (insufficient buying pressure)
            df.loc[df['cmf'] < cmf_thresh, 'strength_filter'] = 0
 
            
        elif strength_type == 'HURST':
            # [NEW] Hurst Exponent - Market Regime Filter
            # H > Trend_Threshold: Trending -> Allow Trend Following
            # H < Random_Threshold: Random Walk -> Block Entry
            # 0.45 < H < 0.55: Random -> Block
            hurst_period = self.params.get('HURST_PERIOD', 100)
            hurst_trend_thresh = self.params.get('HURST_TREND_THRESHOLD', 0.60)
            hurst_random_thresh = self.params.get('HURST_RANDOM_THRESHOLD', 0.50)
            
            df.loc[:, 'hurst'] = calculate_hurst_exponent(df['close'], window=hurst_period)
            
            # Block if Random Walk (H near 0.5) or Mean-Reverting (H < 0.5)
            df.loc[df['hurst'] < hurst_random_thresh, 'strength_filter'] = 0
            # Only allow strong trending (H > trend_threshold)
            # Optional: You can also require H > trend_threshold explicitly
            # df.loc[df['hurst'] < hurst_trend_thresh, 'strength_filter'] = 0
            
        elif strength_type == 'ER':
            # [NEW] Kaufman Efficiency Ratio
            # ER > Threshold: Strong trend / low noise -> Allow Entry
            er_thresh = self.params.get('ER_THRESHOLD', 0.6)
            er_period = self.params.get('STRENGTH_FILTER_PERIOD', 10)
            df['er'] = calculate_efficiency_ratio(df['close'], window=er_period)
            df.loc[df['er'] < er_thresh, 'strength_filter'] = 0
            
        elif strength_type == 'NATR':
            # [NEW] Normalized ATR Filter
            # NATR < Threshold: Low volatility environment -> Block Entry (Optional logic)
            # NATR > Threshold: Sufficient volatility for profit
            natr_thresh = self.params.get('NATR_THRESHOLD', 1.0)
            natr_period = self.params.get('STRENGTH_FILTER_PERIOD', 14)
            df['natr'] = calculate_natr(df, window=natr_period)
            df.loc[df['natr'] < natr_thresh, 'strength_filter'] = 0

        # --- 4.1 Global Entry Filters (e.g. RSI_ENTRY_MAX) ---
        rsi_entry_max = self.params.get('RSI_ENTRY_MAX')
        if rsi_entry_max is not None:
            # RSI must be calculated (ensure it exists)
            if 'rsi' not in df.columns:
                rsi_period = self.params.get('STRENGTH_FILTER_PERIOD', 14) 
                df.loc[:, 'rsi'] = calculate_rsi(df['close'], window=rsi_period)
            
            # LONG Filter: Block if RSI > threshold
            df.loc[(df['trend_direction'] > 0) & (df['rsi'] > rsi_entry_max), 'strength_filter'] = 0
            # SHORT Filter: Block if RSI < (100 - threshold)
            df.loc[(df['trend_direction'] < 0) & (df['rsi'] < (100 - rsi_entry_max)), 'strength_filter'] = 0

        # --- 5. Exit Logic (Parabolic SAR) ---
        # Calculated here so it's available for the backtest engine
        if self.params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
            sar_step = self.params.get('SAR_STEP', 0.02)
            sar_line, _ = calculate_parabolic_sar(df, step=sar_step)
            df.loc[:, 'parabolic_sar'] = sar_line
        else:
            df.loc[:, 'parabolic_sar'] = 0.0 # Default
            
        # --- 6. Volume Filter (Ratio) ---
        if self.params.get('USE_VOLUME_FILTER', False):
            vol_ma_period = self.params.get('VOLUME_MA_PERIOD', 20)
            # [Statistical Upgrade] Log Z-Score Method
            # 1. Log-transform volume to normalize distribution (Log-Normal -> Normal)
            log_vol = np.log1p(df['volume'])
            
            # 2. Calculate Rolling Mean & Std of Log Volume
            log_vol_mean = log_vol.rolling(window=vol_ma_period).mean()
            log_vol_std = log_vol.rolling(window=vol_ma_period).std()
            
            # 3. Calculate Z-Score (Standardized Volume)
            # Z = (Current - Mean) / Std
            # This accounts for volume volatility, differentiating between noise and signal better than simple ratios.
            # Z-score is stored in 'volume_ratio' to maintain compatibility with Engine logic
            # [FIX] Safer division: handle NaN std and 0 std
            log_vol_std_safe = log_vol_std.replace(0, 1).fillna(1)
            z_score = (log_vol - log_vol_mean) / log_vol_std_safe
            df.loc[:, 'volume_ratio'] = z_score.fillna(-10.0) # Undefined -> Very low Z-score (Block trade)
        else:
            df.loc[:, 'volume_ratio'] = 100.0 # Default Pass (High ratio)
            
        # [OPTIMIZATION] Ensure all required columns exist for Engine
        # This removes the need for slow df.get() calls
        if 'entry_upper' not in df.columns:
            df['entry_upper'] = np.nan
        if 'entry_lower' not in df.columns:
            df['entry_lower'] = np.nan
        if 'trend_direction' not in df.columns:
            df['trend_direction'] = 0
            
        # [ROBUSTNESS] Final Cleanup & Sanitization
        # Fill NaNs in critical indicators to prevent Numba errors or Logical fails
        # Strategy usually has warmup phase, but clean data prevents edge-case crashes.
        
        # 1. Indicators: ffill first, then default defaults
        df['atr'] = df['atr'].ffill().bfill().fillna(df['close'] * 0.01)
        
        if 'natr' in df.columns:
            df['natr'] = df['natr'].ffill().fillna(1.0)
            
        if 'rsi' in df.columns:
            df['rsi'] = df['rsi'].ffill().fillna(50.0)
            
        if 'hurst' in df.columns:
            df['hurst'] = df['hurst'].ffill().fillna(0.5) # Neutral
            
        # 2. Signals
        df['strength_filter'] = df['strength_filter'].fillna(0).astype(int)
        df['trend_direction'] = df['trend_direction'].fillna(0).astype(int)
        
        # 3. Entry Bands (Keep NaN intact as they mean "No Signal", but ensure float type)
        df['entry_upper'] = df['entry_upper'].astype(float)
        df['entry_lower'] = df['entry_lower'].astype(float)

        return df
