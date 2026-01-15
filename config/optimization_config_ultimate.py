# Ultimate Strategy Optimization Configuration
# Defines the search space for "The Ultimate Strategy"

ULTIMATE_SEARCH_SPACE = {
    # === CORE SETTINGS ===
    # Using full timeframe range for comprehensive exploration: Scalping (3m, 5m) to Swing (4h)
    'TIMEFRAME': {'type': 'categorical', 'choices': ['3m', '5m', '15m', '30m', '1h', '2h', '4h']},
    'LEVERAGE': {'type': 'float', 'low': 1.0, 'high': 2.5, 'step': 0.05},  # 0.1 → 0.05
    
    # === ENTRY SIGNAL (Mutually Exclusive) ===
    # 1. DONCHIAN: Standard breakout (Classic)
    # 2. BOLLINGER: Volatility squeeze breakout
    # 3. KELTNER: ATR-based channel breakout
    # 4. CCI: Momentum Breakout
    'ENTRY_TYPE': {'type': 'categorical', 'choices': ['DONCHIAN', 'BOLLINGER', 'KELTNER', 'CCI']},
    'ENTRY_PERIOD': {'type': 'int', 'low': 10, 'high': 150, 'step': 1},  # 5 → 1
    
    # Bollinger-specific (Standard Deviation)
    'BB_STD': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.1},  # 0.5 → 0.1
    
    # === TREND DIRECTION FILTERS (Mutually Exclusive) ===
    # Selects ONE best trend filter from the list
    'TREND_FILTER_TYPE': {'type': 'categorical', 'choices': ['SMA', 'EMA', 'HMA', 'DEMA', 'TEMA', 'SUPERTREND', 'MACD', 'ICHIMOKU']},
    
    # Shared Period for MA-based filters (SMA, EMA, HMA, DEMA, TEMA)
    # Range extended down to 5 to allow short-term trend detection
    'MA_PERIOD': {'type': 'int', 'low': 5, 'high': 200, 'step': 1},  # 5 → 1
    
    # SuperTrend specific parameters
    'SUPERTREND_MULT': {'type': 'float', 'low': 1.0, 'high': 5.0, 'step': 0.1},  # 0.5 → 0.1
    'SUPERTREND_PERIOD': {'type': 'int', 'low': 5, 'high': 50, 'step': 1},  # 5 → 1
    
    # === STRENGTH FILTERS (Optional / Combinable) ===
    # These can be turned ON/OFF independently
    'USE_ADX': {'type': 'categorical', 'choices': [True, False]},
    'ADX_THRESHOLD': {'type': 'int', 'low': 10, 'high': 30, 'step': 1},  # 2 → 1
    
    'USE_VHF': {'type': 'categorical', 'choices': [True, False]},
    'VHF_THRESHOLD': {'type': 'float', 'low': 0.3, 'high': 0.6, 'step': 0.01},  # 0.05 → 0.01
    
    # MFI (Money Flow Index) Filter
    'USE_MFI': {'type': 'categorical', 'choices': [True, False]},
    'MFI_WINDOW': {'type': 'int', 'low': 10, 'high': 21, 'step': 1},
    'MFI_THRESHOLD': {'type': 'int', 'low': 15, 'high': 35, 'step': 1},
    
    # RSI (Relative Strength Index) Filter
    'USE_RSI': {'type': 'categorical', 'choices': [True, False]},
    'RSI_WINDOW': {'type': 'int', 'low': 10, 'high': 21, 'step': 1},  # Already 1
    'RSI_OVERBOUGHT': {'type': 'int', 'low': 65, 'high': 80, 'step': 1},  # 5 → 1
    'RSI_OVERSOLD': {'type': 'int', 'low': 20, 'high': 35, 'step': 1},  # 5 → 1
    
    # Stochastic Oscillator Filter
    'USE_STOCHASTIC': {'type': 'categorical', 'choices': [True, False]},
    'STOCH_WINDOW': {'type': 'int', 'low': 10, 'high': 21, 'step': 1},  # Already 1
    'STOCH_OVERBOUGHT': {'type': 'int', 'low': 75, 'high': 90, 'step': 1},  # 5 → 1
    'STOCH_OVERSOLD': {'type': 'int', 'low': 10, 'high': 25, 'step': 1},  # 5 → 1
    
    # === EXIT & RISK MANAGEMENT ===
    # ATR Exit vs Parabolic SAR Exit
    'EXIT_TYPE': {'type': 'categorical', 'choices': ['ATR', 'PARABOLIC_SAR']},
    
    # Dynamic Stop Loss (New Feature)
    # Switches between Fixed % (original) and ATR-based dynamic stop loss
    'STOP_LOSS_TYPE': {'type': 'categorical', 'choices': ['FIXED', 'ATR']},
    'STOP_LOSS_PCT': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.001}, # For FIXED
    'ATR_STOP_LOSS_MULT': {'type': 'float', 'low': 1.0, 'high': 4.0, 'step': 0.1}, # For ATR
    
    # ATR Multiplier (Used for Trailing Stop)
    'ATR_MULTIPLIER': {'type': 'float', 'low': 1.5, 'high': 6.0, 'step': 0.1},  # 0.5 → 0.1
    
    # Parabolic SAR Step (Acceleration Factor)
    'SAR_STEP': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.001},  # 0.01 → 0.001
    
    # Position Sizing Risk
    'RISK_PER_TRADE': {'type': 'float', 'low': 0.01, 'high': 0.04, 'step': 0.001},  # 0.01 → 0.001
}

COMMON_SEARCH_SPACE = {} # For compatibility with existing loader
