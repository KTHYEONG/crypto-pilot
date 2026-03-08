from typing import Any, Dict, List

# ==============================================================================
# OPTIMIZATION FUTURES SEARCH SPACE & CONFIGURATION
# ==============================================================================

OPT_FUTURES_CONFIG: Dict[str, Any] = {
    "total_trials": 5000,     # [CONSISTENCY] Massive increase to ensure global optimum is found
    "n_startup_trials": 150,  # [DIVERSITY] 150 population size gives 32 generations of deep evolution
    "seeds": [42],
    "n_jobs": 8,              # [BALANCE] Reduced slightly from 10 to 8 to minimize parallel race conditions
    "task_workers": 1,
    "TARGET_TIMEFRAMES": ["4h"],
}

TARGET_TIMEFRAMES: List[str] = ["4h"]
WARMUP_PERIODS: Dict[str, int] = {"1h": 2160, "4h": 540}  

# ==============================================================================
# TF-SPECIFIC SEARCH SPACES
# ==============================================================================

# [INSTITUTIONAL GROWTH] TTM Squeeze + Fat-Tail Momentum
SEARCH_SPACE_4H: Dict[str, Dict[str, Any]] = {
    # --- 1. Macro Trend Filter ---
    "MACRO_EMA_PERIOD":  {"type": "int",   "low": 50,   "high": 200, "step": 10}, 
    
    # --- 2. Squeeze Parameters (Relaxed for Opportunity) ---
    "KC_MULT":           {"type": "float", "low": 1.5,  "high": 3.0,  "step": 0.25}, 
    
    # --- 3. Momentum Breakout Trigger ---
    "MOMENTUM_PERIOD":   {"type": "int",   "low": 10,   "high": 40,  "step": 5},
    
    # --- 4. Exits (Asymmetric Hard Stop vs Fat-Tail Trail) ---
    "ATR_PERIOD":        {"type": "int",   "low": 14,   "high": 24,  "step": 2},
    "LONG_ATR_MULT":     {"type": "float", "low": 2.0,  "high": 5.0,  "step": 0.5}, # Initial Stop
    "LONG_TRAIL_MULT":   {"type": "float", "low": 2.5,  "high": 10.0, "step": 0.5}, # Fat tail trailing
    
    "SHORT_ATR_MULT":    {"type": "float", "low": 2.0,  "high": 4.0,  "step": 0.5},
    "SHORT_TP_MULT":     {"type": "float", "low": 2.0,  "high": 6.0,  "step": 0.5}, 
    
    # --- 5. Volatility Targeting (Compounding) ---
    "RISK_PER_TRADE":    {"type": "float", "low": 0.02, "high": 0.08, "step": 0.01}, 
}

def get_search_space_futures(tf: str) -> Dict[str, Dict[str, Any]]:
    return SEARCH_SPACE_4H.copy()


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
