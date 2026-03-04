from typing import Any, Dict, List

# ==============================================================================
# OPTIMIZATION V2 SEARCH SPACE & CONFIGURATION
# ==============================================================================

OPT_V2_CONFIG: Dict[str, Any] = {
    "total_trials": 600,  # Total Optuna trials per timeframe (NSGA-II)
    "n_startup_trials": 60,  # Also used as NSGA-II population size
    "seeds": [42],
    "n_jobs": 2,
    "TARGET_TIMEFRAMES": ["1h"],
}

TARGET_TIMEFRAMES: List[str] = ["1h"]
WARMUP_PERIOD: int = 2160  # 3M lookback (2160 bars for 1h)

BASE_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "TSMOM_ENTRY_THRESHOLD": {"type": "float", "low": 0.5,  "high": 2.0, "step": 0.1},
    "TSMOM_WEIGHT_DECAY":    {"type": "float", "low": 0.0,  "high": 2.0, "step": 0.2},
    "ATR_WINDOW":            {"type": "int",   "low": 12,  "high": 48,  "step": 2},
    "ATR_MULTIPLIER":        {"type": "float", "low": 3.5, "high": 6.0, "step": 0.25},
    "ATR_PRC_WINDOW":        {"type": "int",   "low": 100, "high": 500, "step": 50},
    "VELOCITY_K":            {"type": "int",   "low": 6,   "high": 24,  "step": 2},
    "RISK_PER_TRADE":        {"type": "float", "low": 0.01, "high": 0.04, "step": 0.005},
}

SEARCH_SPACE_V2: Dict[str, Dict[str, Any]] = BASE_SEARCH_SPACE.copy()


def get_search_space_v2(tf: str) -> Dict[str, Dict[str, Any]]:
    """Return the search space for the given timeframe. Currently 1h only."""
    return SEARCH_SPACE_V2


def get_quarterly_window(reference_date=None) -> tuple[str, str, str, str]:
    import datetime
    from dateutil.relativedelta import relativedelta
    if reference_date is None: reference_date = datetime.date.today()
    elif isinstance(reference_date, str): reference_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d").date()
    elif isinstance(reference_date, datetime.datetime): reference_date = reference_date.date()
    current_quarter_start_month: int = ((reference_date.month - 1) // 3) * 3 + 1
    current_quarter_start: datetime.date = datetime.date(reference_date.year, current_quarter_start_month, 1)
    oos_end: datetime.date = current_quarter_start - datetime.timedelta(days=1)
    oos_start: datetime.date = current_quarter_start - relativedelta(months=6)
    is_start: datetime.date = oos_start - relativedelta(months=24)
    fetch_start: datetime.date = is_start - relativedelta(days=500)
    return (fetch_start.strftime("%Y-%m-%d"), is_start.strftime("%Y-%m-%d"), oos_start.strftime("%Y-%m-%d"), oos_end.strftime("%Y-%m-%d"))
