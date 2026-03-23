from __future__ import annotations

import math
from typing import Any, Dict, List

# ==============================================================================
# OPTIMIZATION FUTURES SEARCH SPACE & CONFIGURATION
# ==============================================================================

OPT_FUTURES_CONFIG: Dict[str, Any] = {
    "total_trials": 5000,
    "n_startup_trials": 150,
    "seeds": [42],
    "n_jobs": 8,
    "task_workers": 1,
    "TARGET_TIMEFRAMES": ["4h"],
}

FUTURES_SYMBOLS: List[str] = [
    "ETH/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "NEAR/USDT",
    "LINK/USDT",
    "DOGE/USDT",
    "SUI/USDT",
    "1000PEPE/USDT",
    "FET/USDT",
    "APT/USDT",
]

# ==============================================================================
# OPTIMIZATION SPOT SEARCH SPACE & CONFIGURATION (shared-cash + CPCV)
# ==============================================================================

OPT_SPOT_CONFIG: Dict[str, Any] = {
    "total_trials": 1200,
    "n_startup_trials": 150,
    "tpe_n_startup_trials": 96,
    "tpe_pruner_n_startup_trials": 10,
    "tpe_pruner_n_warmup_steps": 2,
    "seeds": [42],
    "n_jobs": 2,
    "task_workers": 1,
    "TARGET_TIMEFRAMES": ["4h"],
    "SPOT_MAX_CONCURRENT_POSITIONS": 3,
    "SPOT_SHORTLIST_TOP_K": 50,
    "SPOT_OBJECTIVE_TRADE_FREQ_CAP": 120.0,
    "SPOT_PATH_CROSS_PATH_LOG_TW_STD_PENALTY_MULT": 2.0,
    "SPOT_MIN_PATH_CAGR_PENALTY_MULT": 1.5,
    "SPOT_MIN_TRADES_PER_CPCV_SEGMENT": 4,
    "SPOT_SEGMENT_TRADE_FAIL_PENALTY": 2.0,
    "SPOT_MDD_PENALTY_THRESHOLD_PCT": 35.0,
    "SPOT_OBJECTIVE_CVAR_PENALTY_THRESHOLD": 15.0,
    "SPOT_OBJECTIVE_CVAR25_LOG_TW_WEIGHT": 0.20,
    "SPOT_DD_DURATION_BARS_THRESHOLD": 100,
    "SPOT_DD_DURATION_PENALTY_PER_BAR": 0.001,
    "SPOT_HOLDOUT_MIN_PORTFOLIO_LONG_TRADES": 8,
    "SPOT_HOLDOUT_MAX_CVAR_PCT": 25.0,
    "SPOT_STRESS_SYMBOLS": [],
    "SPOT_SYMBOL_CLUSTER": {
        "KRW-BTC": "anchor",
        "KRW-ETH": "large_alt",
        "KRW-SOL": "large_alt",
        "KRW-LINK": "liquid_alt",
        "KRW-AVAX": "liquid_alt",
        "KRW-DOGE": "liquid_alt",
        "KRW-NEAR": "liquid_alt",
        "KRW-XRP": "liquid_alt",
        "KRW-ADA": "liquid_alt",
        "KRW-STX": "liquid_alt",
    },
    "SPOT_CLUSTER_WEIGHT": {
        "anchor": 1.0,
        "large_alt": 1.0,
        "liquid_alt": 0.95,
        "small_alt": 0.85,
    },
    # Fixed signal hyperparameters (not searched — dimensionality control)
    "BB_WINDOW": 20,
    "VOL_Z_WINDOW": 20,
    "VOL_EXPANSION_MULT": 1.05,
    "BTC_REGIME_SMA_PERIOD": None,
}

TARGET_TIMEFRAMES: List[str] = ["4h"]
WARMUP_PERIODS: Dict[str, int] = {"4h": 540}

SEARCH_SPACE_4H: Dict[str, Dict[str, Any]] = {
    "MACRO_EMA_PERIOD": {"type": "int", "low": 20, "high": 200, "step": 10},
    "KC_MULT": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
    "SQUEEZE_WINDOW": {"type": "int", "low": 3, "high": 12, "step": 1},
    "MOMENTUM_PERIOD": {"type": "int", "low": 10, "high": 50, "step": 5},
    "VOL_Z_THRESHOLD": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.25},
    "EXHAUSTION_MULT": {"type": "float", "low": 2.5, "high": 5.0, "step": 0.5},
    "ATR_PERIOD": {"type": "int", "low": 14, "high": 24, "step": 2},
    "LONG_ATR_MULT": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
    "LONG_TRAIL_MULT": {"type": "float", "low": 2.5, "high": 10.0, "step": 0.5},
    "LONG_SCALE_ATR_MULT": {"type": "float", "low": 2.0, "high": 8.0, "step": 0.5},
    "SHORT_ATR_MULT": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
    "SHORT_TP_MULT": {"type": "float", "low": 2.0, "high": 6.0, "step": 0.5},
    "SHORT_TRAIL_MULT": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
    "RISK_PER_TRADE": {"type": "float", "low": 0.02, "high": 0.08, "step": 0.01},
    "CVD_WINDOW": {"type": "int", "low": 3, "high": 9, "step": 2},
    "TAKER_RATIO_THRESHOLD": {"type": "float", "low": 1.05, "high": 1.50, "step": 0.05},
}

SEARCH_SPACE_SPOT_4H: Dict[str, Dict[str, Any]] = {
    "MACRO_EMA_PERIOD": {"type": "int", "low": 100, "high": 300, "step": 20},
    "FAST_EMA_PERIOD": {"type": "int", "low": 20, "high": 80, "step": 10},
    "ADX_PERIOD": {"type": "int", "low": 7, "high": 21, "step": 2},
    "ADX_THRESHOLD": {"type": "float", "low": 15.0, "high": 35.0, "step": 2.5},
    "KC_MULT": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
    "MOMENTUM_PERIOD": {"type": "int", "low": 10, "high": 40, "step": 5},
    "VOL_Z_THRESHOLD": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.25},
    "ATR_PERIOD": {"type": "int", "low": 10, "high": 24, "step": 2},
    "LONG_ATR_MULT": {"type": "float", "low": 1.5, "high": 3.5, "step": 0.5},
    "LONG_TRAIL_MULT": {"type": "float", "low": 2.5, "high": 8.0, "step": 0.5},
    "RISK_PER_TRADE": {"type": "float", "low": 0.0025, "high": 0.05, "step": 0.0025},
    "MAX_POSITION_PCT": {"type": "float", "low": 0.10, "high": 0.65, "step": 0.05},
    "TIME_STOP_BARS": {"type": "int", "low": 0, "high": 80, "step": 4},
}


def get_search_space_futures(tf: str) -> Dict[str, Dict[str, Any]]:
    return SEARCH_SPACE_4H.copy()


def get_search_space_spot(tf: str) -> Dict[str, Dict[str, Any]]:
    _ = tf
    return SEARCH_SPACE_SPOT_4H.copy()


def get_spot_effective_independent_trials(
    n_completed_trials: int,
    n_startup_trials: int,
) -> int:
    """
    Conservative effective independent trials under TPE autocorrelation (spot_enhance7).
    """
    n = max(0, int(n_completed_trials))
    n0 = max(1, int(n_startup_trials))
    # Penalize post-startup trials more: treat correlated trials as partially redundant.
    if n <= n0:
        return max(1, n)
    extra = n - n0
    effective = n0 + int(math.ceil(extra**0.65))
    return max(1, effective)


def get_quarterly_window(reference_date: Any = None) -> tuple[str, str, str, str]:
    import datetime

    from dateutil.relativedelta import relativedelta

    if reference_date is None:
        reference_date = datetime.date.today()
    elif isinstance(reference_date, str):
        reference_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d").date()
    elif isinstance(reference_date, datetime.datetime):
        reference_date = reference_date.date()
    current_quarter_start_month: int = ((reference_date.month - 1) // 3) * 3 + 1
    current_quarter_start: datetime.date = datetime.date(
        reference_date.year, current_quarter_start_month, 1
    )
    oos_end: datetime.date = current_quarter_start - datetime.timedelta(days=1)
    oos_start: datetime.date = current_quarter_start - relativedelta(months=6)
    is_start: datetime.date = oos_start - relativedelta(months=24)
    fetch_start: datetime.date = is_start - relativedelta(days=500)
    return (
        fetch_start.strftime("%Y-%m-%d"),
        is_start.strftime("%Y-%m-%d"),
        oos_start.strftime("%Y-%m-%d"),
        oos_end.strftime("%Y-%m-%d"),
    )
