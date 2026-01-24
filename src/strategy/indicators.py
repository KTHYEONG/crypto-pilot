# LEGACY MODULE: 이 파일의 함수들은 더 이상 사용되지 않습니다.
# indicators_advanced.py의 버전을 사용하세요.

import pandas as pd
import numpy as np

def _legacy_calculate_sma(series, window):
    return series.rolling(window=window).mean()

def _legacy_calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def _legacy_calculate_atr(high, low, close, window=14):
    """Average True Range 계산"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_noise_ratio(open_price, high, low, close):
    """노이즈 비율 = 1 - abs(시가 - 종가) / (고가 - 저가)"""
    noise = 1 - abs(open_price - close) / (high - low)
    # 0으로 나누기 방지
    noise = noise.replace([np.inf, -np.inf], 0).fillna(0)
    return noise

def calculate_k(noise_ratio, window=20):
    """K = 최근 n일 노이즈 비율 평균"""
    return noise_ratio.rolling(window=window).mean()

def calculate_volatility(close, window):
    """변동성 = 전일 대비 등락률의 표준편차 * sqrt(252 or period)"""
    # 여기서는 단순 이동평균 표준편차 사용 또는 로그 수익률 표준편차
    # 사용자 로직 상 '최근 n일 변동성 평균'이라고 했으므로,
    # (고가 - 저가) / 시가 의 평균을 사용하는 것이 일반적인 변동성 돌파 전략의 맥락임.
    # 하지만 '변동성'의 정의가 모호하므로, 표준적인 Historical Volatility 사용
    return close.pct_change().rolling(window=window).std()

def calculate_range(high, low):
    return high - low

# --- New Indicators ---

def _legacy_calculate_rsi(close, window=14):
    """Relative Strength Index (RSI)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _legacy_calculate_bollinger_bands(close, window=20, num_std=2):
    """Bollinger Bands"""
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def _legacy_calculate_macd(close, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_donchian_channels(high, low, window=20):
    """Donchian Channels: 전일 기준 최근 N일 최고가/최저가"""
    # 당일 돌파 여부를 보기 위해, shift(1)된 데이터의 rolling max/min을 구함
    # 즉, 어제까지의 N일 최고가
    donchian_high = high.shift(1).rolling(window=window).max()
    donchian_low = low.shift(1).rolling(window=window).min()
    return donchian_high, donchian_low

def _legacy_calculate_wma(series, window):
    """Weighted Moving Average"""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def _legacy_calculate_hma(series, window):
    """Hull Moving Average"""
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    
    wma_half = _legacy_calculate_wma(series, half_length)
    wma_full = _legacy_calculate_wma(series, window)
    
    raw_hma = 2 * wma_half - wma_full
    return _legacy_calculate_wma(raw_hma, sqrt_length)

def _legacy_calculate_adx(high, low, close, window=14):
    """Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's Smoothing (ATR, +DI, -DI)
    # EWM alpha = 1/window is close to Wilder's methodology
    alpha = 1 / window
    
    atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean() / atr_smooth)
    
    # DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    return adx

def add_common_indicators(df, params):
    """⚠️ DEPRECATED: 이 함수는 더 이상 사용되지 않습니다.
    indicators_advanced.py의 함수들을 직접 사용하세요.
    """
    import warnings
    warnings.warn(
        "add_common_indicators는 deprecated되었습니다. "
        "indicators_advanced의 함수들을 직접 사용하세요.",
        DeprecationWarning,
        stacklevel=2
    )
    df = df.copy()
    
    # 기본 지표
    df['range'] = calculate_range(df['high'], df['low'])
    df['noise'] = calculate_noise_ratio(df['open'], df['high'], df['low'], df['close'])
    
    # 이동평균 (Legacy)
    df['ma50'] = _legacy_calculate_sma(df['close'], 50) 
    df['ema20'] = _legacy_calculate_ema(df['close'], 20)
    
    # ATR
    df['atr'] = _legacy_calculate_atr(df['high'], df['low'], df['close'], params.get('ATR_PERIOD', 14))
    
    # --- New Indicators (Conditional Calculation based on params) ---
    
    # 1. Donchian Channels (For Long/Short Trend Follower)
    if 'CHANNEL_PERIOD' in params:
        window = params['CHANNEL_PERIOD']
        df['donchian_high'], df['donchian_low'] = calculate_donchian_channels(df['high'], df['low'], window)

    # 2. Trend EMA (Regime Filter)
    if 'TREND_EMA_WINDOW' in params:
        window = params['TREND_EMA_WINDOW']
        df[f'ema_trend'] = calculate_ema(df['close'], window)
        
    # [NEW] Hull MA (Fast Trend)
    if params.get('REGIME_FILTER') == 'HMA' or 'HMA_WINDOW' in params:
        window = params.get('HMA_WINDOW', 50)
        df['hma'] = _legacy_calculate_hma(df['close'], window)

    # [NEW] ADX (Trend Strength)
    if params.get('USE_ADX', False) or 'ADX_THRESHOLD' in params:
        df['adx'] = _legacy_calculate_adx(df['high'], df['low'], df['close'], window=14)
    
    # 3. Volatility Indicator (Legacy Strategy A)
    if 'VOL_SHORT_WINDOW' in params:
        vol_short = params.get('VOL_SHORT_WINDOW', 5)
        vol_long = params.get('VOL_LONG_WINDOW', 20)
        daily_vol = (df['high'] - df['low']) / df['open']
        df['vol_short'] = daily_vol.rolling(window=vol_short).mean()
        df['vol_long'] = daily_vol.rolling(window=vol_long).mean()
    
    # 4. RSI
    if params.get('USE_RSI', False) or 'RSI_PERIOD' in params:
        rsi_window = params.get('RSI_PERIOD', 14)
        df['rsi'] = _legacy_calculate_rsi(df['close'], rsi_window)
        
    # 5. Bollinger Bands
    if params.get('USE_BOLLINGER', False) or 'BB_WINDOW' in params:
        bb_window = params.get('BB_WINDOW', 20)
        bb_std = params.get('BB_STD', 2.0)
        df['bb_upper'], df['bb_lower'] = _legacy_calculate_bollinger_bands(df['close'], bb_window, bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma50'] 

    # 6. MACD
    if params.get('USE_MACD', False) or 'MACD_FAST' in params:
        macd_fast = params.get('MACD_FAST', 12)
        macd_slow = params.get('MACD_SLOW', 26)
        macd_sig = params.get('MACD_SIGNAL', 9)
        df['macd'], df['macd_signal'], df['macd_hist'] = _legacy_calculate_macd(df['close'], macd_fast, macd_slow, macd_sig)

    # 7. MA Window for Strategy A
    if 'MA_WINDOW' in params:
        ma_window = params['MA_WINDOW']
        df[f'ma{ma_window}'] = _legacy_calculate_sma(df['close'], ma_window)

    return df
