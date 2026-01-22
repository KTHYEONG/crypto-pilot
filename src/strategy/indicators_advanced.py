import pandas as pd
import numpy as np
import talib
import functools
from numba import njit

# Global cache for indicators to prevent redundant calculations during optimization
_INDICATOR_CACHE = {}

def indicator_cache(func):
    """
    [DISABLED] Cache is disabled to prevent MemoryError and ID collisions during optimization.
    Since BacktestEngineFastSpot creates a new shallow copy of the DataFrame for every trial,
    id(df) changes, causing the cache to grow infinitely (Memory Leak) and potentially 
    causing collisions if IDs are reused (ValueError: shape mismatch).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@indicator_cache
def calculate_sma(series, window):
    return pd.Series(talib.SMA(series.values, timeperiod=window), index=series.index)

@indicator_cache
def calculate_ema(series, window):
    return pd.Series(talib.EMA(series.values, timeperiod=window), index=series.index)

@indicator_cache
def calculate_wma(series, window):
    return pd.Series(talib.WMA(series.values, timeperiod=window), index=series.index)

@indicator_cache
def calculate_hma(series, window):
    wma_half = calculate_wma(series, window // 2)
    wma_full = calculate_wma(series, window)
    return calculate_wma(2 * wma_half - wma_full, int(np.sqrt(window)))

@indicator_cache
def calculate_dema(series, window):
    return pd.Series(talib.DEMA(series.values, timeperiod=window), index=series.index)

@indicator_cache
def calculate_tema(series, window):
    return pd.Series(talib.TEMA(series.values, timeperiod=window), index=series.index)

@indicator_cache
def calculate_atr(df, window=14):
    return pd.Series(talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=window), index=df.index)

@indicator_cache
def calculate_bollinger_bands(df, window=20, std_dev=2.0):
    upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=window, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
    bandwidth = (upper - lower) / middle
    return pd.Series(upper, index=df.index), pd.Series(lower, index=df.index), pd.Series(bandwidth, index=df.index)

@indicator_cache
def calculate_keltner_channel(df, window=20, atr_mult=1.5):
    ema = calculate_ema(df['close'], window)
    atr = calculate_atr(df, window=10) 
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    return upper, lower

@indicator_cache
def calculate_supertrend(df, period=10, multiplier=3.0):
    atr = calculate_atr(df, window=period)
    hl2 = (df['high'] + df['low']) / 2
    
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    close = df['close'].values
    bu = basic_upper.values
    bl = basic_lower.values
    
    final_upper = np.zeros(len(df))
    final_lower = np.zeros(len(df))
    trend = np.zeros(len(df), dtype=int)
    
    upper = bu[0]
    lower = bl[0]
    curr_trend = 1
    
    final_upper[0] = upper
    final_lower[0] = lower
    trend[0] = curr_trend
    
    for i in range(1, len(df)):
        c = close[i]
        c_prev = close[i-1]
        
        if bu[i] < upper or c_prev > upper:
            upper = bu[i]
        
        if bl[i] > lower or c_prev < lower:
            lower = bl[i]
            
        if curr_trend == 1:
            if c < lower:
                curr_trend = -1
            else:
                curr_trend = 1
        else:
            if c > upper:
                curr_trend = 1
            else:
                curr_trend = -1
                
        final_upper[i] = upper
        final_lower[i] = lower
        trend[i] = curr_trend
        
    return pd.Series(trend, index=df.index)

@indicator_cache
def calculate_vhf(series, window=28):
    hcp = series.rolling(window).max()
    lcp = series.rolling(window).min()
    numerator = (hcp - lcp).abs()
    
    diff = series.diff().abs()
    denominator = diff.rolling(window).sum()
    
    return numerator / denominator

@indicator_cache
def calculate_adx(df, window=14):
    return pd.Series(talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=window), index=df.index)

@indicator_cache
def calculate_rsi(series, window=14):
    return pd.Series(talib.RSI(series.values, timeperiod=window), index=series.index)

@indicator_cache
def calculate_stochastic(df, window=14, smooth_k=3, smooth_d=3):
    stoch_k, stoch_d = talib.STOCH(
        df['high'].values, 
        df['low'].values, 
        df['close'].values, 
        fastk_period=window, 
        slowk_period=smooth_k, 
        slowk_matype=0, 
        slowd_period=smooth_d, 
        slowd_matype=0
    )
    return pd.Series(stoch_k, index=df.index), pd.Series(stoch_d, index=df.index)

@indicator_cache
def calculate_stoch_rsi(series, window=14, smooth_k=3, smooth_d=3):
    stoch_k, stoch_d = talib.STOCHRSI(
        series.values, 
        timeperiod=window, 
        fastk_period=smooth_k, 
        fastd_period=smooth_d, 
        fastd_matype=0
    )
    return pd.Series(stoch_k, index=series.index), pd.Series(stoch_d, index=series.index)

@indicator_cache
def calculate_macd(df, fast=12, slow=26, signal=9):
    macd, macdsignal, macdhist = talib.MACD(
        df['close'].values, 
        fastperiod=fast, 
        slowperiod=slow, 
        signalperiod=signal
    )
    return pd.Series(macd, index=df.index), pd.Series(macdsignal, index=df.index), pd.Series(macdhist, index=df.index)

@indicator_cache
def calculate_ichimoku(df, tenkan_window=9, kijun_window=26, senkou_span_b_window=52):
    high = df['high']
    low = df['low']
    
    tenkan_sen = (talib.MAX(high.values, timeperiod=tenkan_window) + talib.MIN(low.values, timeperiod=tenkan_window)) / 2
    kijun_sen = (talib.MAX(high.values, timeperiod=kijun_window) + talib.MIN(low.values, timeperiod=kijun_window)) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    sb_high = talib.MAX(high.values, timeperiod=senkou_span_b_window)
    sb_low = talib.MIN(low.values, timeperiod=senkou_span_b_window)
    senkou_span_b = (sb_high + sb_low) / 2
    
    ts = pd.Series(tenkan_sen, index=df.index)
    ks = pd.Series(kijun_sen, index=df.index)
    ssa = pd.Series(senkou_span_a, index=df.index).shift(kijun_window)
    ssb = pd.Series(senkou_span_b, index=df.index).shift(kijun_window)
    
    return ts, ks, ssa, ssb

@indicator_cache
def calculate_cci(df, window=20):
    return pd.Series(talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=window), index=df.index)

@indicator_cache
def calculate_mfi(df, window=14):
    return pd.Series(talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values.astype(float), timeperiod=window), index=df.index)

@indicator_cache
def calculate_parabolic_sar(df, step=0.02, max_step=0.2):
    return pd.Series(talib.SAR(df['high'].values, df['low'].values, acceleration=step, maximum=max_step), index=df.index), None 

@indicator_cache
def calculate_vwap(df, window=None, std_mult=1.5):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_volume = typical_price * df['volume']
    
    if window is None:
        vwap = tp_volume.cumsum() / df['volume'].cumsum()
        vwap_std = typical_price.expanding().std()
    else:
        vwap = tp_volume.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        vwap_std = typical_price.rolling(window=window).std()
    
    vwap_upper = vwap + (vwap_std * std_mult)
    vwap_lower = vwap - (vwap_std * std_mult)
    
    return vwap, vwap_upper, vwap_lower

@indicator_cache
def calculate_cmf(df, window=20):
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
    mf_volume = mf_multiplier * df['volume']
    cmf = mf_volume.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    return cmf

@njit
def _hurst_numba_logic(data, window):
    n = len(data)
    out = np.empty(n)
    out[:] = np.nan
    
    for i in range(window - 1, n):
        ts = data[i - window + 1 : i + 1]
        # De-mean
        m = np.mean(ts)
        ts_adj = ts - m
        # Rescaled Range (R/S)
        Y = np.cumsum(ts_adj)
        R = np.max(Y) - np.min(Y)
        S = np.std(ts)
        
        if S > 0 and R > 0:
            h = np.log(R / S) / np.log(window)
            if h < 0: h = 0.0
            if h > 1: h = 1.0
            out[i] = h
        else:
            out[i] = 0.5
            
    return out

@indicator_cache
def calculate_hurst_exponent(series, window=100):
    """
    Numba-accelerated Hurst Exponent calculation.
    Speed Gain: ~100x faster than rolling().apply()
    """
    vals = series.values.astype(np.float64)
    res = _hurst_numba_logic(vals, window)
    return pd.Series(res, index=series.index)

def clear_indicator_cache():
    global _INDICATOR_CACHE
    _INDICATOR_CACHE = {}
