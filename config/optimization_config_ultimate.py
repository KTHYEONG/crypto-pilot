# Ultimate Strategy Optimization Configuration
# Defines the search space for "The Ultimate Strategy"

ULTIMATE_SEARCH_SPACE = {
    # === CORE SETTINGS ===
    # Using full timeframe range: Scalping (3m) to Swing (1d)
    'TIMEFRAME': {'type': 'categorical', 'choices': ['3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']},
    
    # LEVERAGE: Complete exploration range (0.1x ~ 3.0x)
    # 0.1~1.0x: Conservative (Pension fund level safety)
    # 1.0~2.0x: Standard (Most strategies fall here)
    # 2.0~3.0x: Aggressive (High risk tolerance)
    'LEVERAGE': {'type': 'float', 'low': 0.1, 'high': 3.0, 'step': 0.1},
    
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
    
    'SUPERTREND_MULT': {'type': 'float', 'low': 1.0, 'high': 5.0, 'step': 0.1},
    'SUPERTREND_PERIOD': {'type': 'int', 'low': 5, 'high': 50, 'step': 1},
    
    # MACD-specific [NEW]
    'MACD_FAST': {'type': 'int', 'low': 5, 'high': 30, 'step': 1},
    'MACD_SLOW': {'type': 'int', 'low': 20, 'high': 100, 'step': 1},
    'MACD_SIGNAL': {'type': 'int', 'low': 5, 'high': 20, 'step': 1},
    
    # Ichimoku-specific [NEW]
    'ICHIMOKU_TENKAN': {'type': 'int', 'low': 7, 'high': 20, 'step': 1},
    'ICHIMOKU_KIJUN': {'type': 'int', 'low': 20, 'high': 60, 'step': 1},
    'ICHIMOKU_SENKOU_B': {'type': 'int', 'low': 40, 'high': 120, 'step': 1},
    
    # === STRENGTH FILTERS (Mutually Exclusive) ===
    # Selects ONE filter to confirm trend strength
    'STRENGTH_FILTER_TYPE': {'type': 'categorical', 'choices': ['NONE', 'ADX', 'VHF', 'MFI', 'RSI', 'STOCHASTIC']},
    
    # Shared Period for Strength Filters [NEW]
    'STRENGTH_FILTER_PERIOD': {'type': 'int', 'low': 7, 'high': 50, 'step': 1},
    
    # Filter-Specific Parameters (Used only when corresponding type is selected)
    # Refined steps (1) and broader ranges for theoretical optimum
    'ADX_THRESHOLD': {'type': 'int', 'low': 15, 'high': 40, 'step': 1},
    'VHF_THRESHOLD': {'type': 'float', 'low': 0.2, 'high': 0.6, 'step': 0.01},
    'MFI_THRESHOLD': {'type': 'int', 'low': 10, 'high': 50, 'step': 1},
    'RSI_OVERBOUGHT': {'type': 'int', 'low': 65, 'high': 85, 'step': 1},
    'RSI_OVERSOLD': {'type': 'int', 'low': 15, 'high': 35, 'step': 1},
    'STOCH_OVERBOUGHT': {'type': 'int', 'low': 75, 'high': 95, 'step': 1},
    'STOCH_OVERSOLD': {'type': 'int', 'low': 5, 'high': 25, 'step': 1},
    
    # Volume Filter [NEW]
    'USE_VOLUME_FILTER': {'type': 'categorical', 'choices': [True, False]},
    'VOLUME_MA_PERIOD': {'type': 'int', 'low': 10, 'high': 50, 'step': 10},
    'VOLUME_THRESHOLD_MULT': {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.1},
    
    # ATR Period [NEW]
    'ATR_PERIOD': {'type': 'int', 'low': 7, 'high': 30, 'step': 1},
    
    # === EXIT & RISK MANAGEMENT ===
    # ATR Exit vs Parabolic SAR Exit (Main Trend Following Exit)
    'EXIT_TYPE': {'type': 'categorical', 'choices': ['ATR', 'PARABOLIC_SAR']},
    
    # Parabolic SAR Step (Acceleration Factor)
    'SAR_STEP': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.005},
    
    # Take Profit
    'USE_TAKE_PROFIT': {'type': 'categorical', 'choices': [True, False]},
    'TAKE_PROFIT_ATR_MULT': {'type': 'float', 'low': 2.0, 'high': 8.0, 'step': 1.0},
    
    # Dynamic Stop Loss
    'STOP_LOSS_TYPE': {'type': 'categorical', 'choices': ['FIXED', 'ATR']},
    'STOP_LOSS_PCT': {'type': 'float', 'low': 0.01, 'high': 0.1, 'step': 0.005}, # For FIXED
    'ATR_STOP_LOSS_MULT': {'type': 'float', 'low': 1.0, 'high': 6.0, 'step': 0.5}, # For ATR
    
    # Trailing Stop (ATR) - Used only if EXIT_TYPE == 'ATR'
    'ATR_MULTIPLIER': {'type': 'float', 'low': 2.0, 'high': 6.0, 'step': 0.5},
    
    # Position Sizing Risk (Split by Market Type)
    # FUTURES: 0.05% ~ 5% (Leverage assumed)
    'RISK_PER_TRADE_FUTURES': {'type': 'float', 'low': 0.0005, 'high': 0.05, 'step': 0.0005},
    
    # SPOT: 30% ~ 100% (No Leverage)
    'RISK_PER_TRADE_SPOT': {'type': 'float', 'low': 0.3, 'high': 1.0, 'step': 0.1},
}

COMMON_SEARCH_SPACE = {} # For compatibility with existing loader
