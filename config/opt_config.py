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

SPOT_SYMBOLS: List[str] = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-ADA",
    "KRW-DOGE",
]

OPT_SPOT_CONFIG: Dict[str, Any] = {
    "total_trials": 540,
    "n_startup_trials": 96,
    "tpe_n_startup_trials": 96,
    "tpe_pruner_n_startup_trials": 10,
    "tpe_pruner_n_warmup_steps": 8,
    "tpe_pruner_patience": 2,
    "seeds": [42],
    "n_jobs": 4,
    # 0 = auto (logical CPUs) for multi-mode portfolio process-parallel TPE; >0 caps workers
    "task_workers": 0,
    "TARGET_TIMEFRAMES": ["4h"],
    "SPOT_MAX_CONCURRENT_POSITIONS": 5,
    "SPOT_SHORTLIST_TOP_K": 25,
    "SPOT_MIN_TRADES_PER_CPCV_SEGMENT": 4,
    "SPOT_SEGMENT_TRADE_FAIL_PENALTY": 2.0,
    "SPOT_HOLDOUT_MIN_PORTFOLIO_LONG_TRADES": 8,
    "SPOT_HOLDOUT_MIN_TAIL_RATIO": 2.0,
    "SPOT_HOLDOUT_MIN_CAGR_PCT": 30.0,
    "SPOT_HOLDOUT_MDD_LIMIT_PCT": 45.0,
    "SPOT_HOLDOUT_HWM_RECOVERY_MAX_DAYS": 300.0,
    "SPOT_HOLDOUT_ALPHA_DECAY_FLOOR_PCT": -50.0,
    "SPOT_GATE1_SQN_MIN": 1.5,
    # CPCV: mean(path Sortino) / (std(path Sortino) + eps); tune with composite objective
    "SPOT_GATE1_PATH_SORTINO_MIN": 0.5,
    "SPOT_GATE1_TAIL_RATIO_MIN": 1.5,
    "SPOT_DISCOVERY_DSR_MIN": -1.0,
    "SPOT_HOLDOUT_MAX_CVAR_PCT": 25.0,
    # Composite: min(path TW ratio) * mean(path Sortino) / (std(path Sortino) + eps)
    "SPOT_OBJECTIVE_SORTINO_EPS": 1e-6,
    "SPOT_OBJECTIVE_SORTINO_RATIO_CAP": 1.0e6,
    "SPOT_OBJECTIVE_PATH_SORTINO_CLIP": 500.0,
    "SPOT_STRESS_SYMBOLS": [],
    "SPOT_SYMBOL_CLUSTER": {
        "KRW-BTC": "anchor",
        "KRW-ETH": "large_alt",
        "KRW-XRP": "liquid_alt",
        "KRW-ADA": "liquid_alt",
        "KRW-DOGE": "liquid_alt",
    },
    "SPOT_CLUSTER_WEIGHT": {
        "anchor": 1.0,
        "large_alt": 1.0,
        "liquid_alt": 0.95,
        "small_alt": 0.85,
    },
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
    "ATR_PERIOD": {"type": "int", "low": 10, "high": 24, "step": 2},
    "HMM_TRAIN_WINDOW": {"type": "int", "low": 240, "high": 480, "step": 60},
    "HMM_RETRAIN_FREQ": {"type": "int", "low": 12, "high": 72, "step": 12},
    "FRAMA_PERIOD": {"type": "int", "low": 8, "high": 24, "step": 2},
    "FRAMA_MIN_SLOPE": {"type": "float", "low": 0.0001, "high": 0.005, "step": 0.0001},
    "EVR_WINDOW": {"type": "int", "low": 10, "high": 40, "step": 5},
    "EVR_THRESHOLD": {"type": "float", "low": 0.3, "high": 2.0, "step": 0.1},
    "GARCH_WINDOW": {"type": "int", "low": 180, "high": 360, "step": 30},
    "GARCH_RETRAIN_FREQ": {"type": "int", "low": 12, "high": 72, "step": 12},
    "GARCH_NU_FALLBACK": {"type": "float", "low": 4.0, "high": 10.0, "step": 0.5},
    "LONG_ATR_MULT": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.25},
    "LONG_TRAIL_MULT": {"type": "float", "low": 2.0, "high": 8.0, "step": 0.5},
    "LONG_TP_MULT": {"type": "float", "low": 2.0, "high": 8.0, "step": 0.5},
    "TP_LOCK_ATR_MULT": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.5},
    "LONG_TRAIL_LOCK_MULT": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.25},
    "RISK_PER_TRADE": {"type": "float", "low": 0.02, "high": 0.15, "step": 0.005},
    "MAX_POSITION_PCT": {"type": "float", "low": 0.15, "high": 0.80, "step": 0.05},
    "KILL_COOLDOWN_BARS": {"type": "int", "low": 3, "high": 12, "step": 3},
    "DELTA_GATE": {"type": "float", "low": 0.03, "high": 0.15, "step": 0.02},
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
    is_start: datetime.date = oos_start - relativedelta(months=36)
    fetch_start: datetime.date = is_start - relativedelta(days=500)
    return (
        fetch_start.strftime("%Y-%m-%d"),
        is_start.strftime("%Y-%m-%d"),
        oos_start.strftime("%Y-%m-%d"),
        oos_end.strftime("%Y-%m-%d"),
    )
