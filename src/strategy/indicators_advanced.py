import pandas as pd
import numpy as np

def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, window):
    wma_half = calculate_wma(series, window // 2)
    wma_full = calculate_wma(series, window)
    return calculate_wma(2 * wma_half - wma_full, int(np.sqrt(window)))

def calculate_dema(series, window):
    ema1 = calculate_ema(series, window)
    ema2 = calculate_ema(ema1, window)
    return 2 * ema1 - ema2

def calculate_tema(series, window):
    ema1 = calculate_ema(series, window)
    ema2 = calculate_ema(ema1, window)
    ema3 = calculate_ema(ema2, window)
    return 3 * ema1 - 3 * ema2 + ema3

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_bollinger_bands(df, window=20, std_dev=2.0):
    sma = calculate_sma(df['close'], window)
    std = df['close'].rolling(window=window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    # Squeeze: Bandwidth
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth

def calculate_keltner_channel(df, window=20, atr_mult=1.5):
    ema = calculate_ema(df['close'], window)
    atr = calculate_atr(df, window=10) # ATR window usually fixed around 10-14
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    return upper, lower

def calculate_supertrend(df, period=10, multiplier=3.0):
    atr = calculate_atr(df, window=period)
    hl2 = (df['high'] + df['low']) / 2
    
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    # Init final values
    final_upper = pd.Series(index=df.index, dtype='float64')
    final_lower = pd.Series(index=df.index, dtype='float64')
    is_uptrend = pd.Series(index=df.index, dtype='int')
    
    # Iterate (Not vectorizable easily due to dependency on prev value)
    # Using numpy for iteration might be faster but simple loop is readable for logic check
    # We will initialize with basic first value
    up = basic_upper.iloc[0]
    lo = basic_lower.iloc[0]
    trend = 1
    
    upper_list = [up]
    lower_list = [lo]
    trend_list = [trend]
    
    close = df['close'].values
    bu = basic_upper.values
    bl = basic_lower.values
    
    for i in range(1, len(df)):
        c = close[i]
        c_prev = close[i-1]
        
        # Calculate Upper/Lower
        if bu[i] < upper_list[-1] or c_prev > upper_list[-1]:
            curr_upper = bu[i]
        else:
            curr_upper = upper_list[-1]
            
        if bl[i] > lower_list[-1] or c_prev < lower_list[-1]:
            curr_lower = bl[i]
        else:
            curr_lower = lower_list[-1]
            
        # Determine Trend
        prev_trend = trend_list[-1]
        
        if prev_trend == 1:
            if c < curr_lower:
                curr_trend = -1
            else:
                curr_trend = 1
        else:
            if c > curr_upper:
                curr_trend = 1
            else:
                curr_trend = -1
                
        upper_list.append(curr_upper)
        lower_list.append(curr_lower)
        trend_list.append(curr_trend)
        
    return pd.Series(trend_list, index=df.index)

def calculate_vhf(series, window=28):
    hcp = series.rolling(window).max()
    lcp = series.rolling(window).min()
    numerator = (hcp - lcp).abs()
    
    diff = series.diff().abs()
    denominator = diff.rolling(window).sum()
    
    return numerator / denominator

def calculate_adx(df, window=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = calculate_atr(df, window=window)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean().abs() / tr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/window).mean()
    return adx

def calculate_rsi(series, window=14):
    """
    Calculate Relative Strength Index (RSI)
    RSI > 70: Overbought (potential reversal down)
    RSI < 30: Oversold (potential reversal up)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(df, window=14, smooth_k=3, smooth_d=3):
    """
    Calculate Stochastic Oscillator (%K and %D)
    %K > 80: Overbought
    %K < 20: Oversold
    """
    low_min = df['low'].rolling(window=window).min()
    high_max = df['high'].rolling(window=window).max()
    
    # Fast %K
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    
    # Slow %K (smoothed)
    stoch_k = stoch_k.rolling(window=smooth_k).mean()
    
    # %D (signal line)
    stoch_d = stoch_k.rolling(window=smooth_d).mean()
    
    return stoch_k, stoch_d

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence)
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_ichimoku(df, tenkan_window=9, kijun_window=26, senkou_span_b_window=52):
    """
    Ichimoku Cloud
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    - Kijun-sen (Base Line): (26-period high + 26-period low)/2
    - Senkou Span A (Leading Span A): (Conversion + Base)/2
    - Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    """
    high = df['high']
    low = df['low']
    
    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(window=tenkan_window).max() + low.rolling(window=tenkan_window).min()) / 2

    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(window=kijun_window).max() + low.rolling(window=kijun_window).min()) / 2

    # Senkou Span A (Leading Span A) - Shifted forward by 26 periods (displacement)
    # Note: In backtesting, we look at value at t-26? 
    # Standard: Plot 26 periods ahead. So current cloud is based on past data.
    # Current Close vs Current Cloud Value (which was projected 26 periods ago)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)

    # Senkou Span B (Leading Span B)
    senkou_span_b = ((high.rolling(window=senkou_span_b_window).max() + low.rolling(window=senkou_span_b_window).min()) / 2).shift(kijun_window)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

def calculate_cci(df, window=20):
    """
    CCI (Commodity Channel Index)
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=window).mean()
    
    # Calculate Mean Absolute Deviation (MAD)
    # Pandas .mad() is deprecated, so we calculate manually using rolling mean of absolute difference
    mad = (tp - sma_tp).abs().rolling(window=window).mean()
    
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def calculate_mfi(df, window=14):
    """
    MFI (Money Flow Index) - Volume-weighted RSI
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=df.index)
    negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=df.index)
    
    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

def calculate_parabolic_sar(df, step=0.02, max_step=0.2):
    # This is a complex indicator to implement vectorially.
    # We'll use a simplified iterative version.
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_p = df['open'].values
    
    sar = np.zeros(len(df))
    trend = np.zeros(len(df)) # 1: up, -1: down
    ep = np.zeros(len(df)) # Extreme Point
    af = np.zeros(len(df)) # Acceleration Factor
    
    # Init
    trend[0] = 1 if close[0] > open_p[0] else -1
    sar[0] = low[0] if trend[0] == 1 else high[0]
    ep[0] = high[0] if trend[0] == 1 else low[0]
    af[0] = step
    
    for i in range(1, len(df)):
        prev_sar = sar[i-1]
        prev_trend = trend[i-1]
        prev_ep = ep[i-1]
        prev_af = af[i-1]
        
        # Calculate SAR
        curr_sar = prev_sar + prev_af * (prev_ep - prev_sar)
        
        # Trend Switch Logic
        curr_trend = prev_trend
        
        if prev_trend == 1:
            if low[i] < curr_sar:
                curr_trend = -1
                curr_sar = prev_ep # Reset SAR to EP
                curr_ep = low[i]
                curr_af = step
            else:
                curr_ep = max(prev_ep, high[i])
                if curr_ep > prev_ep and prev_af < max_step:
                    curr_af = prev_af + step
                else:
                    curr_af = prev_af
        else: # Down Trend
            if high[i] > curr_sar:
                curr_trend = 1
                curr_sar = prev_ep
                curr_ep = high[i]
                curr_af = step
            else:
                curr_ep = min(prev_ep, low[i])
                if curr_ep < prev_ep and prev_af < max_step:
                    curr_af = prev_af + step
                else:
                    curr_af = prev_af
                    
        sar[i] = curr_sar
        trend[i] = curr_trend
        ep[i] = curr_ep
        af[i] = curr_af
        
    return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)
