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
    'TP_ATR_MULT':  {'low': 1.2,   'high': 4.5,   'log': True},   # [Hybrid] Min 1.2(Fee Safety), Max 4.5(Quick Ops)
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
    'TP_ATR_MULT':  {'low': 1.5,  'high': 12.0,  'log': True},    # [Hybrid] Min 1.5(Flexibility), Max 12.0(Trend)
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
    'TP_ATR_MULT':  {'low': 3.0,  'high': 30.0, 'log': True},     # [Hybrid] Min 3.0(Active Swing), Max 30.0(Moon Shot)
    'ADX_THRESH':   {'low': 15,   'high': 30,   'step': 1},        # Linear (임계값)
    'VOL_THRESHOLD': {'low': 1.1,  'high': 2.5,  'log': True},     # Log (배수)
    'MAX_HOLDING_BARS': {'low': 50, 'high': 300, 'log': True},     # Log (기간)
    'TRAILING_ACTIVATION_ATR': {'low': 3.0, 'high': 8.0, 'log': True}  # Log (배수)
}

# UNIFIED: All-in-one strategy (모든 범위 병합)
# 타임프레임: 1h, 4h, 1d 고정 (노이즈 제거)
# 파라미터: 각 지표의 최솟값 ~ 최댓값 사용
UNIFIED_CONFIG = {
    'ENTRY_PERIOD': {'low': 10, 'high': 200, 'log': True},         # 전체 범위: 10 (SCALP) ~ 200 (SWING)
    'MA_PERIOD':    {'low': 5,  'high': 200, 'log': True},         # 전체 범위: 5 (SCALP) ~ 200 (SWING)
    'ATR_PERIOD':   {'low': 5,  'high': 60,  'log': True},         # [Optimized] 10~30 -> 5~60 (민감~둔감 다양한 변동성 대응)
    'SL_PCT':       {'low': 0.005, 'high': 0.05, 'step': 0.005},   # [Optimized] Max 20% -> 5% (고배율 선물에서 5% 이상 손절은 의미 없음)
    'TP_ATR_MULT':  {'low': 1.5,  'high': 15.0, 'log': True},      # [Optimized] Max 30->15 (Realistic Big Win)
    'ADX_THRESH':   {'low': 15,   'high': 45,   'step': 1},        # 전체 범위: 15 (SWING) ~ 45 (SCALP)
    'VOL_THRESHOLD': {'low': 1.1,  'high': 3.0,  'log': True},     # [Optimized] Max 5.0 -> 3.0 (현실적인 거래량 돌파 기준)
    'MAX_HOLDING_BARS': {'low': 5, 'high': 500, 'log': True},      # [Max Profit] 200->500 (Catch Monster Trend)
    'TRAILING_ACTIVATION_ATR': {'low': 0.5, 'high': 8.0, 'log': False, 'step': 0.5}  # [Optimized] Min 0.0->0.5 (Avoid immediate whipsaw)
}

# =========================================================
# 2. BASE SEARCH SPACE
# =========================================================
BASE_SEARCH_SPACE = {
    'ENTRY_TYPE': {'type': 'categorical', 'choices': ['DONCHIAN', 'BOLLINGER', 'KELTNER', 'CCI']},
    'TREND_FILTER_TYPE': {'type': 'categorical', 'choices': ['SMA', 'EMA', 'HMA', 'DEMA', 'TEMA', 'SUPERTREND', 'MACD', 'ICHIMOKU', 'VWAP']},
    'STRENGTH_FILTER_TYPE': {'type': 'categorical', 'choices': ['NONE', 'ADX', 'VHF', 'MFI', 'RSI', 'STOCHASTIC', 'STOCH_RSI', 'CMF', 'HURST', 'ER', 'NATR']},
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
    
    # ER (Kaufman Efficiency Ratio) Parameters
    'ER_THRESHOLD': {'type': 'float', 'low': 0.3, 'high': 0.8, 'step': 0.05},    # H > 임계값: Trending
    
    # NATR (Normalized ATR) Parameters
    'NATR_THRESHOLD': {'type': 'float', 'low': 0.5, 'high': 2.0, 'step': 0.1},   # 최소 변동성 기준
    
    # Time-Based Exit Parameters
    'TIME_EXIT_PROFIT_THRESHOLD': {'type': 'float', 'low': 0.0, 'high': 2.0, 'step': 0.1},  # ATR 단위 최소 수익 (0 = 무조건 청산, 2 = 2 ATR 이상 수익일 때만 보유)
    
    # [NEW] Panic Exit Parameters
    'RSI_EXIT_THRESHOLD': {'type': 'int', 'low': 75, 'high': 95, 'step': 1}, # 75~95 (Long Exit), 25~5 (Short Exit)

    # [NEW] Safe Entry Filters
    'RSI_ENTRY_MAX': {'type': 'categorical', 'choices': [None, 70, 75, 80, 85]}, # Don't buy if RSI > X (Overbought Top)
    'NATR_ENTRY_MIN': {'type': 'float', 'low': 0.2, 'high': 1.5, 'step': 0.1}, # Don't buy if Volatility < X (Dead Market)

    # [NEW] Dynamic Risk Sizing Parameters (Relaxed for Stability)
    'USE_DYNAMIC_RISK': {'type': 'categorical', 'choices': [False, True]},
    'STRONG_REGIME_HURST': {'type': 'float', 'low': 0.52, 'high': 0.62, 'step': 0.01},    # 0.52만 넘어도 추세로 인정
    'STRONG_REGIME_NATR': {'type': 'float', 'low': 0.7, 'high': 1.6, 'step': 0.1},      # 낮은 변동성에서도 공격적 진입
    'STRONG_REGIME_MULTIPLIER': {'type': 'float', 'low': 1.2, 'high': 1.8, 'step': 0.1},
    'WEAK_REGIME_HURST': {'type': 'float', 'low': 0.40, 'high': 0.50, 'step': 0.01},     # 진짜 역추세일 때만 감액
    'WEAK_REGIME_MULTIPLIER': {'type': 'float', 'low': 0.5, 'high': 0.9, 'step': 0.1},   # 감액 폭을 완화 (최소 0.5배 유지)
    'PANIC_REGIME_NATR': {'type': 'float', 'low': 4.5, 'high': 7.5, 'step': 0.5},       # 진짜 패닉일 때만 방어
    'PANIC_REGIME_MULTIPLIER': {'type': 'float', 'low': 0.1, 'high': 0.4, 'step': 0.05},

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

    elif mode == 'UNIFIED' or mode == 'ALL':
        cfg = UNIFIED_CONFIG
        # [UNIFIED] Supports both Futures and Spot with stable timeframes
        # 15m, 30m added for Small Capital Rotation
        space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['15m', '30m', '1h', '4h', '1d']}
        # Wide ATR range to accommodate all strategies
        space['ATR_STOP_LOSS_MULT'] = {'type': 'float', 'low': 1.0, 'high': 6.0, 'step': 0.5}
        space['ATR_MULTIPLIER'] = {'type': 'float', 'low': 1.5, 'high': 8.0, 'step': 0.5}

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
        # [Futures] Baseline for small-cap realism:
        # keep enough upside, but avoid immediate account volatility explosion.
        space['RISK_PER_TRADE'] = {'type': 'float', 'low': 0.012, 'high': 0.05, 'step': 0.001}
        space['LEVERAGE'] = {'type': 'float', 'low': 1.0, 'high': 8.0, 'step': 1.0}
        
        # [UNIFIED-FUTURES] Small-cap production profile (about 0.8M KRW account)
        if mode == 'UNIFIED' or mode == 'ALL':
            # Structure choices: reduce combinatorial explosion and overfit risk.
            space['ENTRY_TYPE'] = {'type': 'categorical', 'choices': ['DONCHIAN', 'BOLLINGER', 'KELTNER']}
            space['TREND_FILTER_TYPE'] = {'type': 'categorical', 'choices': ['EMA', 'SUPERTREND', 'ICHIMOKU']}
            space['STRENGTH_FILTER_TYPE'] = {'type': 'categorical', 'choices': ['NONE', 'ADX', 'RSI', 'NATR', 'HURST', 'VHF']}
            space['USE_TAKE_PROFIT'] = {'type': 'categorical', 'choices': [True, False]}
            space['USE_VOLUME_FILTER'] = {'type': 'categorical', 'choices': [True, False]}

            # Timeframes: reduce fee/slippage tax while preserving responsiveness.
            space['TIMEFRAME'] = {'type': 'categorical', 'choices': ['30m', '1h', '4h']}

            # Risk / leverage: keep return potential but control drawdown speed.
            space['RISK_PER_TRADE'] = {'type': 'float', 'low': 0.012, 'high': 0.035, 'step': 0.001}
            space['LEVERAGE'] = {'type': 'float', 'low': 2.0, 'high': 6.0, 'step': 1.0}

            # Core execution bounds
            space['ENTRY_PERIOD'] = {'type': 'int', 'low': 12, 'high': 140, 'log': True}
            space['MA_PERIOD'] = {'type': 'int', 'low': 10, 'high': 170, 'log': True}
            space['ATR_PERIOD'] = {'type': 'int', 'low': 10, 'high': 40, 'log': True}
            space['STOP_LOSS_PCT'] = {'type': 'float', 'low': 0.008, 'high': 0.03, 'step': 0.002}
            space['ATR_STOP_LOSS_MULT'] = {'type': 'float', 'low': 1.5, 'high': 3.5, 'step': 0.25}
            space['TAKE_PROFIT_ATR_MULT'] = {'type': 'float', 'low': 1.8, 'high': 8.0, 'log': True}
            space['ATR_MULTIPLIER'] = {'type': 'float', 'low': 2.0, 'high': 5.0, 'step': 0.5}
            space['MAX_HOLDING_BARS'] = {'type': 'int', 'low': 40, 'high': 220, 'log': True}
            space['TRAILING_ACTIVATION_ATR'] = {'type': 'float', 'low': 1.0, 'high': 4.0, 'step': 0.5}
            space['TIME_EXIT_PROFIT_THRESHOLD'] = {'type': 'float', 'low': 0.2, 'high': 1.0, 'step': 0.1}
            space['RSI_EXIT_THRESHOLD'] = {'type': 'int', 'low': 82, 'high': 92, 'step': 1}

            # Indicator/detail bounds
            space['KELTNER_ATR_MULT'] = {'type': 'float', 'low': 1.2, 'high': 2.2, 'step': 0.1}
            space['BB_STD'] = {'type': 'float', 'low': 1.8, 'high': 2.6, 'step': 0.1}
            space['SUPERTREND_MULT'] = {'type': 'float', 'low': 1.2, 'high': 3.2, 'log': True}
            space['SUPERTREND_PERIOD'] = {'type': 'int', 'low': 7, 'high': 34, 'log': True}
            space['ICHIMOKU_TENKAN'] = {'type': 'int', 'low': 9, 'high': 18, 'log': True}
            space['ICHIMOKU_KIJUN'] = {'type': 'int', 'low': 24, 'high': 42, 'log': True}
            space['ICHIMOKU_SENKOU_B'] = {'type': 'int', 'low': 52, 'high': 90, 'log': True}
            space['STRENGTH_FILTER_PERIOD'] = {'type': 'int', 'low': 10, 'high': 35, 'log': True}
            space['ADX_THRESHOLD'] = {'type': 'int', 'low': 18, 'high': 35, 'step': 1}
            space['RSI_OVERBOUGHT'] = {'type': 'int', 'low': 68, 'high': 82, 'step': 1}
            space['RSI_OVERSOLD'] = {'type': 'int', 'low': 20, 'high': 32, 'step': 1}
            space['VOLUME_THRESHOLD_MULT'] = {'type': 'float', 'low': 1.2, 'high': 2.2, 'log': True}
            space['VOLUME_MA_PERIOD'] = {'type': 'int', 'low': 12, 'high': 30, 'log': True}
            space['SAR_STEP'] = {'type': 'float', 'low': 0.01, 'high': 0.03, 'step': 0.005}

            # Dynamic risk: keep enabled/usable with moderate multipliers.
            space['USE_DYNAMIC_RISK'] = {'type': 'categorical', 'choices': [False, True]}
            space['STRONG_REGIME_HURST'] = {'type': 'float', 'low': 0.53, 'high': 0.60, 'step': 0.01}
            space['STRONG_REGIME_NATR'] = {'type': 'float', 'low': 0.8, 'high': 1.5, 'step': 0.1}
            space['STRONG_REGIME_MULTIPLIER'] = {'type': 'float', 'low': 1.1, 'high': 1.4, 'step': 0.1}
            space['WEAK_REGIME_HURST'] = {'type': 'float', 'low': 0.43, 'high': 0.49, 'step': 0.01}
            space['WEAK_REGIME_MULTIPLIER'] = {'type': 'float', 'low': 0.6, 'high': 0.9, 'step': 0.1}
            space['PANIC_REGIME_NATR'] = {'type': 'float', 'low': 4.5, 'high': 7.0, 'step': 0.5}
            space['PANIC_REGIME_MULTIPLIER'] = {'type': 'float', 'low': 0.15, 'high': 0.35, 'step': 0.05}
        
        
        # [ISOLATION] Remove Spot-only safety filters to prevent ghost parameters in Futures
        if 'NATR_ENTRY_MIN' in space: del space['NATR_ENTRY_MIN']
        
    else: 
        # [Spot] No Leverage, Aggressive Allocation for Small Capital
        if 'LEVERAGE' in space: 
            del space['LEVERAGE']
        
        # [REVISED] Spot Allocation: Since there's no leverage, we use most of the capital.
        # AI will finding the best buffer (0.8 ~ 0.98).
        space['RISK_PER_TRADE_SPOT'] = {'type': 'float', 'low': 0.8, 'high': 0.98, 'step': 0.02}
        
        # [REVISED] Regime Multipliers for Spot
        # Strong: No room to grow beyond 1.0 if base risk is high.
        space['STRONG_REGIME_MULTIPLIER'] = {'type': 'float', 'low': 1.0, 'high': 1.0, 'step': 0.1}
        # Weak & Panic: Aggressive reduction to save seed.
        space['WEAK_REGIME_MULTIPLIER'] = {'type': 'float', 'low': 0.4, 'high': 0.6, 'step': 0.1}
        space['PANIC_REGIME_MULTIPLIER'] = {'type': 'float', 'low': 0.1, 'high': 0.3, 'step': 0.1}

        # [CRITICAL FIX] Spot Panic Exit Thresholds
        # Spot Altcoins have extreme volatility. Raise thresholds to avoid premature exit.
        space['RSI_EXIT_THRESHOLD'] = {'type': 'int', 'low': 88, 'high': 99, 'step': 1}
        
        # [NEW] Spot Entry Safety Filters
        space['RSI_ENTRY_MAX'] = {'type': 'int', 'low': 75, 'high': 95, 'step': 2} 
        space['NATR_ENTRY_MIN'] = {'type': 'float', 'low': 0.2, 'high': 2.0, 'step': 0.2} 
        
        space['PANIC_REGIME_NATR'] = {'type': 'float', 'low': 5.0, 'high': 12.0, 'step': 0.5}
        
        # Spot needs looser TP to catch big pumps
        if 'TAKE_PROFIT_ATR_MULT' in space:
            space['TAKE_PROFIT_ATR_MULT']['high'] = max(space['TAKE_PROFIT_ATR_MULT']['high'], 15.0)
            
        # [SPOT UPDATE] Mode-Specific Opportunity Cost Management
        if mode == 'SCALP':
            # Scalp (5m~30m): 500 bars @ 5m = ~41 hours (Max holding)
            space['MAX_HOLDING_BARS'] = {'type': 'int', 'low': 100, 'high': 500, 'log': True}
        elif mode == 'DAY':
             # Day (1h): 200 bars = ~8 days (Swing-like Day)
            space['MAX_HOLDING_BARS'] = {'type': 'int', 'low': 50, 'high': 200, 'log': True}
        else: # SWING
            # Swing (4h): 1000 bars = ~166 days (Long Trend)
            space['MAX_HOLDING_BARS'] = {'type': 'int', 'low': 100, 'high': 1000, 'log': True}

    return space
