"""
Optimization Configuration for Distinct Trading Modes
Defines specialized search spaces for SCALP, DAY, and SWING strategies.
Refactored to separate Period constraints and enhance market-specific logic.
"""

from copy import deepcopy

# =========================================================
# 1. PERIOD & THRESHOLD CONSTANTS (User Configurable)
# =========================================================

# Scalping: Fast reaction, tight ranges
SCALP_CONFIG = {
    'ENTRY_PERIOD': {'low': 10, 'high': 40, 'log': True},         # Log: 10→14→20→28→40
    'MA_PERIOD':    {'low': 5,  'high': 40, 'log': True},         # Log: 5→7→10→14→20→28→40
    'ATR_PERIOD':   {'low': 10, 'high': 20, 'log': True},         # Log: 10→14→20 (안정적 변동성 측정)
    'SL_PCT':       {'low': 0.005, 'high': 0.015,  'step': 0.001}, # Linear (비율) - 스캘핑: 타이트한 손절 (레버리지 10배 시 ROE -5% ~ -15%)
    'TP_ATR_MULT':  {'low': 0.8,   'high': 4.0,   'log': True},   # 최소값 복구(0.8), 최대값 확장 유지(4.0)
    'ADX_THRESH':   {'low': 25,    'high': 45,    'step': 1},     # Linear (임계값)
    'VOL_THRESHOLD': {'low': 1.5,  'high': 5.0,   'log': True},   # Log (배수)
    'MAX_HOLDING_BARS': {'low': 10, 'high': 50,   'log': True}    # Log (기간)
}

# Day Trading: Balanced approach (Standard)
DAY_CONFIG = {
    'ENTRY_PERIOD': {'low': 20, 'high': 100, 'log': True},        # Log: 20→28→40→56→80→100
    'MA_PERIOD':    {'low': 10, 'high': 100, 'log': True},        # Log: 10→14→20→28→40→56→80→100
    'ATR_PERIOD':   {'low': 10, 'high': 20,  'log': True},        # Log: 10→14→20
    'SL_PCT':       {'low': 0.015, 'high': 0.06, 'step': 0.005},  # Linear (비율)
    'TP_ATR_MULT':  {'low': 2.0,  'high': 12.0,  'log': True},    # 최소값 복구(2.0), 최대값 확장 유지(12.0)
    'ADX_THRESH':   {'low': 20,   'high': 35,   'step': 1},       # Linear (임계값)
    'VOL_THRESHOLD': {'low': 1.2,  'high': 3.0,  'log': True},    # Log (배수)
    'MAX_HOLDING_BARS': {'low': 20, 'high': 100, 'log': True}     # Log (기간)
}

# Swing Trading: Long-term trend following
SWING_CONFIG = {
    'ENTRY_PERIOD': {'low': 40, 'high': 200, 'log': True},         # Log: 40→56→80→113→160→200
    'MA_PERIOD':    {'low': 50, 'high': 200, 'log': True},         # Log: 50→71→100→141→200
    'ATR_PERIOD':   {'low': 14, 'high': 30,  'log': True},         # Log: 14→17→21→26→30
    'SL_PCT':       {'low': 0.06, 'high': 0.20, 'step': 0.01},     # Linear (비율)
    'TP_ATR_MULT':  {'low': 5.0,  'high': 25.0, 'log': True},      # 최소값 복구(5.0), 최대값 확장 유지(25.0)
    'ADX_THRESH':   {'low': 15,   'high': 30,   'step': 1},        # Linear (임계값)
    'VOL_THRESHOLD': {'low': 1.1,  'high': 2.5,  'log': True},     # Log (배수)
    'MAX_HOLDING_BARS': {'low': 50, 'high': 300, 'log': True},     # Log (기간)
    'TRAILING_ACTIVATION_ATR': {'low': 3.0, 'high': 8.0, 'log': True}  # Log (배수)
}

# =========================================================
# 2. BASE SEARCH SPACE
# =========================================================
BASE_SEARCH_SPACE = {
    'ENTRY_TYPE': {'type': 'categorical', 'choices': ['DONCHIAN', 'BOLLINGER', 'KELTNER', 'CCI']},
    'TREND_FILTER_TYPE': {'type': 'categorical', 'choices': ['SMA', 'EMA', 'HMA', 'DEMA', 'TEMA', 'SUPERTREND', 'MACD', 'ICHIMOKU', 'VWAP']},
    'STRENGTH_FILTER_TYPE': {'type': 'categorical', 'choices': ['NONE', 'ADX', 'VHF', 'MFI', 'RSI', 'STOCHASTIC', 'STOCH_RSI', 'CMF', 'HURST']},
    'EXIT_TYPE': {'type': 'categorical', 'choices': ['ATR', 'PARABOLIC_SAR']},
    'USE_TAKE_PROFIT': {'type': 'categorical', 'choices': [True, False]},
    'STOP_LOSS_TYPE': {'type': 'categorical', 'choices': ['FIXED', 'ATR']},
    'USE_VOLUME_FILTER': {'type': 'categorical', 'choices': [True, False]},
    
    # Common Indicator Parameters
    'BB_STD': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.1},  # Linear (표준편차)
    'KELTNER_ATR_MULT': {'type': 'float', 'low': 1.0, 'high': 2.5, 'step': 0.1}, # [NEW] Keltner 너비
    'CCI_THRESHOLD': {'type': 'int', 'low': 50, 'high': 150, 'step': 10},         # [NEW] CCI 돌파 기준
    'SUPERTREND_MULT': {'type': 'float', 'low': 1.0, 'high': 5.0, 'log': True},  # Log (배수)
    'SUPERTREND_PERIOD': {'type': 'int', 'low': 5, 'high': 50, 'log': True},     # Log (기간)
    'MACD_FAST': {'type': 'int', 'low': 5, 'high': 30, 'log': True},             # Log (기간)
    'MACD_SLOW': {'type': 'int', 'low': 20, 'high': 100, 'log': True},           # Log (기간)
    'MACD_SIGNAL': {'type': 'int', 'low': 5, 'high': 20, 'log': True},           # Log (기간)
    'ICHIMOKU_TENKAN': {'type': 'int', 'low': 7, 'high': 20, 'log': True},       # Log (기간)
    'ICHIMOKU_KIJUN': {'type': 'int', 'low': 20, 'high': 60, 'log': True},       # Log (기간)
    'ICHIMOKU_SENKOU_B': {'type': 'int', 'low': 40, 'high': 120, 'log': True},   # Log (기간)
    'STRENGTH_FILTER_PERIOD': {'type': 'int', 'low': 7, 'high': 50, 'log': True}, # Log (기간)
    'VHF_THRESHOLD': {'type': 'float', 'low': 0.2, 'high': 0.6, 'step': 0.01},   # Linear (임계값)
    'MFI_THRESHOLD': {'type': 'int', 'low': 10, 'high': 50, 'step': 5},          # Linear (임계값)
    'RSI_OVERBOUGHT': {'type': 'int', 'low': 65, 'high': 85, 'step': 1},         # Linear (임계값)
    'RSI_OVERSOLD': {'type': 'int', 'low': 15, 'high': 35, 'step': 1},           # Linear (임계값)
    'STOCH_OVERBOUGHT': {'type': 'int', 'low': 75, 'high': 95, 'step': 1},       # Linear (임계값)
    'STOCH_OVERSOLD': {'type': 'int', 'low': 5, 'high': 25, 'step': 1},          # Linear (임계값)
    'STOCH_RSI_OVERBOUGHT': {'type': 'int', 'low': 70, 'high': 90, 'step': 1},   # Linear (임계값)
    'STOCH_RSI_OVERSOLD': {'type': 'int', 'low': 10, 'high': 30, 'step': 1},     # Linear (임계값)
    'VOLUME_MA_PERIOD': {'type': 'int', 'low': 10, 'high': 50, 'log': True},     # Log (기간)
    'SAR_STEP': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.005},
    
    # VWAP Parameters
    'VWAP_STD_MULT': {'type': 'float', 'low': 0.5, 'high': 2.5, 'step': 0.1},    # VWAP 표준편차 밴드 (Mean Reversion 용)
    
    # CMF Parameters
    'CMF_PERIOD': {'type': 'int', 'low': 10, 'high': 40, 'log': True},           # Log (기간)
    'CMF_THRESHOLD': {'type': 'float', 'low': 0.0, 'high': 0.15, 'step': 0.01},  # Linear (임계값) - 추세 추종: 양수 자금 유입만 허용

    
    # Hurst Exponent Parameters
    'HURST_PERIOD': {'type': 'int', 'low': 100, 'high': 300, 'log': True},       # Log (기간) - 통계적 신뢰도를 위한 최소 100
    'HURST_TREND_THRESHOLD': {'type': 'float', 'low': 0.52, 'high': 0.65, 'step': 0.01},    # H > 임계값: Trending (금융 시계열 현실 반영)
    'HURST_RANDOM_THRESHOLD': {'type': 'float', 'low': 0.45, 'high': 0.50, 'step': 0.01},   # H < 임계값: Random (진입 금지)


}

def GET_SEARCH_SPACE(mode, market_type='futures'):
    """
    Returns the search space for a specific mode and market type.
    mode: 'SCALP', 'DAY', 'SWING'
    market_type: 'futures', 'spot'
    """
    space = deepcopy(BASE_SEARCH_SPACE)
    mode = mode.upper()
    market_type = market_type.lower()
    
    # Select Config based on Mode
    if mode == 'SCALP':
        cfg = SCALP_CONFIG
        # Timeframe: Spot vs Futures 분리
        if market_type == 'spot':
            # Upbit (CCXT): 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
            # 5m (ultra-fast), 15m (standard), 30m (low-freq scalping) - matches Futures
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['5m', '15m', '30m']}
        else:
            # Futures: All standard timeframes available
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['5m', '15m', '30m']}
        # Scalping specific: Tight ATR Multipliers
        space['ATR_STOP_LOSS_MULT'] = {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.2}
        space['ATR_MULTIPLIER'] = {'type': 'float', 'low': 1.5, 'high': 3.9, 'step': 0.2}

    elif mode == 'DAY':
        cfg = DAY_CONFIG
        # Timeframe: Spot vs Futures 분리
        if market_type == 'spot':
            # Upbit: 30m, 1h, 4h supported (2h not available in CCXT Upbit)
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['30m', '1h', '4h']}
        else:
            # Futures: 1h, 2h, 4h available
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['1h', '2h', '4h']}
        # Day Trading: Balanced
        space['ATR_STOP_LOSS_MULT'] = {'type': 'float', 'low': 1.5, 'high': 4.0, 'step': 0.5}
        space['ATR_MULTIPLIER'] = {'type': 'float', 'low': 2.0, 'high': 5.0, 'step': 0.5}

    elif mode == 'SWING':
        cfg = SWING_CONFIG
        # Timeframe: Spot vs Futures 분리
        if market_type == 'spot':
            # Upbit: 4h, 1d, 1w, 1M available (3d not supported)
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['4h', '1d', '1w']}
        else:
            # Futures: 4h, 1d, 3d available
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['4h', '1d', '3d']}
        # Swing: Loose stops
        space['ATR_STOP_LOSS_MULT'] = {'type': 'float', 'low': 3.0, 'high': 6.0, 'step': 0.5}
        space['ATR_MULTIPLIER'] = {'type': 'float', 'low': 3.0, 'high': 8.0, 'step': 0.5}

    else:
        # Fallback to DAY config
        cfg = DAY_CONFIG
        if market_type == 'spot':
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['30m', '1h', '4h']}
        else:
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['1h', '2h', '4h']}
        space['ATR_STOP_LOSS_MULT'] = {'type': 'float', 'low': 2.0, 'high': 4.0, 'step': 0.5}
        space['ATR_MULTIPLIER'] = {'type': 'float', 'low': 2.0, 'high': 5.0, 'step': 0.5}

    # Apply Mode-Specific Configs
    space['ENTRY_PERIOD'] = {'type': 'int', **cfg['ENTRY_PERIOD']}
    space['MA_PERIOD']    = {'type': 'int', **cfg['MA_PERIOD']}
    space['ATR_PERIOD']   = {'type': 'int', **cfg['ATR_PERIOD']}
    space['STOP_LOSS_PCT'] = {'type': 'float', **cfg['SL_PCT']}
    space['TAKE_PROFIT_ATR_MULT'] = {'type': 'float', **cfg['TP_ATR_MULT']}
    space['ADX_THRESHOLD'] = {'type': 'int', **cfg['ADX_THRESH']}
    
    # RVOL Filter (Mode-Specific Range)
    if 'VOL_THRESHOLD' in cfg:
        space['VOLUME_THRESHOLD_MULT'] = {'type': 'float', **cfg['VOL_THRESHOLD']}
    
    # Time-Based Exit (Opportunity Cost Management)
    if 'MAX_HOLDING_BARS' in cfg:
        space['MAX_HOLDING_BARS'] = {'type': 'int', **cfg['MAX_HOLDING_BARS']}
    
    # Trailing Stop Activation (Profit Protection for Swing)
    if 'TRAILING_ACTIVATION_ATR' in cfg:
        space['TRAILING_ACTIVATION_ATR'] = {'type': 'float', **cfg['TRAILING_ACTIVATION_ATR']}

    # === MARKET TYPE OVERRIDES ===
    if market_type == 'futures':
        # [Futures] Risk & Leverage - Aggressive Growth (안정성 60% : 수익률 40%)
        space['RISK_PER_TRADE'] = {'type': 'float', 'low': 0.005, 'high': 0.05, 'step': 0.005} # 0.5% ~ 5% (보수~공격 전범위)
        # Leverage 1x (현물 수준) ~ 10x (공격적 추세 추종)
        space['LEVERAGE'] = {'type': 'float', 'low': 1.0, 'high': 10.0, 'step': 0.5}
        
    else: 
        # [Spot] No Leverage, Higher Allocation per trade
        if 'LEVERAGE' in space: 
            del space['LEVERAGE']
        
        # Spot typically uses larger allocation per trade (e.g. 30% ~ 100% of cash)
        space['RISK_PER_TRADE_SPOT'] = {'type': 'float', 'low': 0.3, 'high': 1.0, 'step': 0.1}
        
        # Spot needs looser TP to catch big pumps
        space['TAKE_PROFIT_ATR_MULT']['high'] = max(space['TAKE_PROFIT_ATR_MULT']['high'], 8.0)

    return space
