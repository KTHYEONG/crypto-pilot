from typing import Any, Dict, List

# ==============================================================================
# OPTIMIZATION FUTURES SEARCH SPACE & CONFIGURATION
# ==============================================================================

OPT_FUTURES_CONFIG: Dict[str, Any] = {
    "total_trials": 5000,  # [CONSISTENCY] Massive increase to ensure global optimum is found
    "n_startup_trials": 150,  # [DIVERSITY] 150 population size gives 32 generations of deep evolution
    "seeds": [42],
    "n_jobs": 8,  # [BALANCE] Reduced slightly from 10 to 8 to minimize parallel race conditions
    "task_workers": 1,
    "TARGET_TIMEFRAMES": ["4h"],
}

# [STRATEGIC] Recommended symbols for 4H TSMOM Squeeze optimization
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
# OPTIMIZATION SPOT SEARCH SPACE & CONFIGURATION
# ==============================================================================

OPT_SPOT_CONFIG: Dict[str, Any] = {
    "total_trials": 5000,
    "n_startup_trials": 150,
    "seeds": [42],
    "n_jobs": 2,
    "task_workers": 1,
    "TARGET_TIMEFRAMES": ["4h"],
}

TARGET_TIMEFRAMES: List[str] = ["4h"]
WARMUP_PERIODS: Dict[str, int] = {"4h": 540}

# ==============================================================================
# TF-SPECIFIC SEARCH SPACES
# ==============================================================================

# [INSTITUTIONAL GROWTH] TTM Squeeze + Fat-Tail Momentum
SEARCH_SPACE_4H: Dict[str, Dict[str, Any]] = {
    # --- 1. Macro Trend Filter ---
    "MACRO_EMA_PERIOD": {"type": "int", "low": 20, "high": 200, "step": 10},
    # --- 2. Squeeze Parameters (Relaxed for Opportunity) ---
    "KC_MULT": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
    "SQUEEZE_WINDOW": {"type": "int", "low": 3, "high": 12, "step": 1},
    # --- 3. Momentum Breakout Trigger ---
    "MOMENTUM_PERIOD": {"type": "int", "low": 10, "high": 50, "step": 5},
    "VOL_Z_THRESHOLD": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.25},
    "EXHAUSTION_MULT": {"type": "float", "low": 2.5, "high": 5.0, "step": 0.5},
    # --- 4. Exits (Asymmetric Hard Stop vs Fat-Tail Trail + Scale-out) ---
    "ATR_PERIOD": {"type": "int", "low": 14, "high": 24, "step": 2},
    "LONG_ATR_MULT": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
    "LONG_TRAIL_MULT": {"type": "float", "low": 2.5, "high": 10.0, "step": 0.5},
    "LONG_SCALE_ATR_MULT": {"type": "float", "low": 2.0, "high": 8.0, "step": 0.5},
    "SHORT_ATR_MULT": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
    "SHORT_TP_MULT": {"type": "float", "low": 2.0, "high": 6.0, "step": 0.5},
    "SHORT_TRAIL_MULT": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
    # --- 5. Volatility Targeting (Compounding) ---
    "RISK_PER_TRADE": {"type": "float", "low": 0.02, "high": 0.08, "step": 0.01},
    # --- 6. Microstructure (CVD) Filters ---
    # CVD window: 4H 기준 3~9봉 (12~36시간)
    "CVD_WINDOW": {"type": "int", "low": 3, "high": 9, "step": 2},
    # Taker buy dominance threshold: 1.05 (5% 우위) ~ 1.50 (50% 우위)
    "TAKER_RATIO_THRESHOLD": {"type": "float", "low": 1.05, "high": 1.50, "step": 0.05},
}

SEARCH_SPACE_SPOT_4H: Dict[str, Dict[str, Any]] = {
    # --- 1. Macro Trend & Strength Filter (이중 정배열 및 가짜 반등 방어) ---
    "MACRO_EMA_PERIOD": {"type": "int", "low": 100, "high": 300, "step": 20},
    "FAST_EMA_PERIOD": {"type": "int", "low": 20, "high": 80, "step": 10},
    "ADX_PERIOD": {"type": "int", "low": 7, "high": 21, "step": 2},
    "ADX_THRESHOLD": {"type": "float", "low": 15.0, "high": 35.0, "step": 2.5},
    # --- 2. Squeeze Parameters ---
    "KC_MULT": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
    # --- 3. Momentum Breakout Trigger ---
    "MOMENTUM_PERIOD": {"type": "int", "low": 10, "high": 40, "step": 5},
    # --- 4. Exits (Long Only - 타이트한 손절, 넉넉한 익절) ---
    "ATR_PERIOD": {"type": "int", "low": 10, "high": 24, "step": 2},
    "LONG_ATR_MULT": {
        "type": "float",
        "low": 1.5,
        "high": 3.5,
        "step": 0.5,
    },  # 빠른 초기 손절
    "LONG_TRAIL_MULT": {
        "type": "float",
        "low": 2.5,
        "high": 8.0,
        "step": 0.5,
    },  # 여유로운 트레일링
    "LONG_TP_MULT": {
        "type": "float",
        "low": 2.0,
        "high": 12.0,
        "step": 1.0,
    },  # 급등 시 하드 익절
    # --- 5. Portfolio Risk Sizing (현물은 비중을 크게) ---
    "RISK_PER_TRADE": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.05},
}


def get_search_space_futures(tf: str) -> Dict[str, Dict[str, Any]]:
    return SEARCH_SPACE_4H.copy()


def get_search_space_spot(tf: str) -> Dict[str, Dict[str, Any]]:
    return SEARCH_SPACE_SPOT_4H.copy()


def get_quarterly_window(reference_date=None) -> tuple[str, str, str, str]:
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
