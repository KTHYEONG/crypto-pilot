from typing import Any, Dict

# ==============================================================================
# OPTIMIZATION V2 SEARCH SPACE & CONFIGURATION
# ==============================================================================
# This configuration file is explicitly designed for the optimize_futures_v2.py
# pipeline. It significantly reduces combinatorial explosion by strictly selecting
# orthogonal indicators rooted in quantitative finance principles.
# ==============================================================================

# Meta-parameters for Optuna TPE optimization
OPT_V2_CONFIG = {
    "total_trials": 2500,
    "n_startup_trials": 250,
    "seeds": [42, 137],  # Dual-seed approach for robust GP prior
    "n_jobs": 10,         # Default parallel workers
}

# The heavily refined and strictly typed search space
SEARCH_SPACE_V2: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # 1. STRUCTURAL COMBINATIONS (Categorical)
    # Total nominal combinations: 2 * 3 * 4 * 4 * 2 * 2 * 1 = 384
    # --------------------------------------------------------------------------
    "TIMEFRAME": {"type": "categorical", "choices": ["4h", "1d"]},
    "ENTRY_TYPE": {
        "type": "categorical",
        "choices": ["DONCHIAN", "BOLLINGER", "KELTNER"],
    },
    "TREND_FILTER_TYPE": {
        "type": "categorical",
        "choices": ["EMA", "SUPERTREND", "MACD", "VWAP"],
    },
    "STRENGTH_FILTER_TYPE": {
        "type": "categorical",
        "choices": ["NONE", "ADX", "HURST", "NATR"],
    },
    "STOP_LOSS_TYPE": {
        "type": "categorical",
        "choices": ["FIXED", "ATR"],
    },
    "USE_TAKE_PROFIT": {"type": "categorical", "choices": [True, False]},
    "EXIT_TYPE": {"type": "categorical", "choices": ["ATR", "PSAR"]},

    # --------------------------------------------------------------------------
    # 2. CORE SYSTEM PARAMETERS (Numeric Limits)
    # --------------------------------------------------------------------------
    "ENTRY_PERIOD": {"type": "int", "low": 14, "high": 120, "log": True},
    "MA_PERIOD": {"type": "int", "low": 20, "high": 150, "log": True},
    "ATR_PERIOD": {"type": "int", "low": 14, "high": 28, "step": 1},
    "MAX_HOLDING_BARS": {"type": "int", "low": 5, "high": 60, "log": True},
    "RISK_PER_TRADE": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.005},
    "LEVERAGE": {"type": "int", "low": 2, "high": 10},

    # --------------------------------------------------------------------------
    # 3. STOP/EXIT PARAMETERS
    # --------------------------------------------------------------------------
    "STOP_LOSS_PCT": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.005},
    "ATR_STOP_LOSS_MULT": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.25},
    "TAKE_PROFIT_ATR_MULT": {"type": "float", "low": 2.5, "high": 8.0, "log": True},
    "ATR_MULTIPLIER": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.25},
    "TRAILING_ACTIVATION_ATR": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},

    # --------------------------------------------------------------------------
    # 4. CONDITIONAL INDICATOR THRESHOLDS
    # --------------------------------------------------------------------------
    "ADX_THRESHOLD": {"type": "int", "low": 20, "high": 35, "step": 1},
    "HURST_PERIOD": {"type": "int", "low": 100, "high": 300, "log": True},
    "HURST_TREND_THRESHOLD": {"type": "float", "low": 0.53, "high": 0.60, "step": 0.01},
    "NATR_THRESHOLD": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.1},
    "BB_STD": {"type": "float", "low": 1.8, "high": 2.8, "step": 0.1},
    "KELTNER_ATR_MULT": {"type": "float", "low": 1.2, "high": 2.5, "step": 0.1},
    "SUPERTREND_MULT": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
    "SUPERTREND_PERIOD": {"type": "int", "low": 7, "high": 30, "step": 1},
    "MACD_FAST": {"type": "int", "low": 8, "high": 18, "step": 1},
    "MACD_SLOW": {"type": "int", "low": 21, "high": 45, "step": 1},
    "MACD_SIGNAL": {"type": "int", "low": 6, "high": 14, "step": 1},
    "VWAP_STD_MULT": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.1},
}
