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
    "KRW-ETH",
    "KRW-POL",
    "KRW-XLM",
    "KRW-AWE",
    "KRW-AXS",
    "KRW-1INCH",
]

OPT_SPOT_CONFIG: Dict[str, Any] = {
    "total_trials": 1500,
    "n_startup_trials": 300,  # QMC(Sobol) 탐색량: 총 trials의 6% = 공간 커버리지와 TPE 학습 균형점
    "tpe_n_startup_trials": 256,  # Sweet spot: 26~35D combo-space에서 초기 커버리지/예산 균형
    "tpe_pruner_n_startup_trials": 10,
    "tpe_pruner_n_warmup_steps": 8,
    "tpe_pruner_patience": 2,
    "seeds": [42],
    "n_jobs": 6,
    "task_workers": 0,
    "TARGET_TIMEFRAMES": ["4h"],
    "SPOT_MAX_CONCURRENT_POSITIONS": 5,
    "CPCV_N_BLOCKS": 8,
    "CPCV_K_TEST": 3,
    "SPOT_SHORTLIST_TOP_K": 25,
    "SPOT_MIN_TRADES_PER_CPCV_SEGMENT": 8,
    "SPOT_SEGMENT_TRADE_FAIL_PENALTY": 2.0,
    "SPOT_HOLDOUT_MIN_PORTFOLIO_LONG_TRADES": 50,  # Dynamically calculated in opt_spot.py but used as a fallback
    "SPOT_HOLDOUT_MIN_TAIL_RATIO": 1.05,
    "SPOT_HOLDOUT_MIN_PROFIT_FACTOR": 1.2,
    "SPOT_HOLDOUT_MIN_CALMAR_RATIO": 1.2,
    "SPOT_HOLDOUT_MIN_CAGR_PCT": 18.0,
    "SPOT_HOLDOUT_MDD_LIMIT_PCT": 35.0,
    "SPOT_HOLDOUT_HWM_RECOVERY_MAX_DAYS": 270.0,
    "SPOT_HOLDOUT_MAX_CVAR_PCT": 10.0,
    "SPOT_HOLDOUT_ALPHA_DECAY_FLOOR_PCT": -75.0,
    "SPOT_GATE1_SQN_MIN": 2.0,
    "SPOT_GATE1_PATH_SORTINO_MIN": 0.5,
    "SPOT_GATE1_TAIL_RATIO_MIN": 1.1,
    "SPOT_DISCOVERY_DSR_MIN": 0.35,
    "SPOT_OBJECTIVE_DSR_TARGET": 0.35,
    "SPOT_OBJECTIVE_LAMBDA_UI": 0.02,
    "SPOT_OBJECTIVE_W_TRADE": 0.06,
    "SPOT_OBJECTIVE_W_SQN": 0.02,
    "SPOT_COMBO_TOP_K": 3,
    "SPOT_COMBO_QUICK_TRIALS": 55,
    "SPOT_COMBO_MIN_SIGNAL_RATE": 0.005,
    "SPOT_COMBO_QUICK_TRIALS_PHASE1": 18,
    "SPOT_COMBO_QUICK_TRIALS_PHASE1_MAX": 30,
    "SPOT_COMBO_QUICK_TRIALS_MAX": 70,
    "SPOT_COMBO_PRUNE_THRESHOLD": -0.5,
    "SPOT_COMBO_N_WORKERS": 6,
    "SPOT_COMBO_MIN_SCREEN_SCORE": 0.0,
    "SPOT_COMBO_AMBIGUITY_STD_RATIO": 0.15,
    "SPOT_COMBO_PHASE2_AMBIGUITY_BOOST": 4,
    "SPOT_STAGE1_TRIALS_PER_SIGNAL": 150,  # Stage1 ranking stability (tmp.md: 100→150)
    "SPOT_OBJECTIVE_MIN_SYMBOL_WIN_RATE": 0.45,
    "SPOT_OBJECTIVE_W_SYMBOL_EDGE": 0.08,
    "SPOT_OBJECTIVE_MIN_TRADES_PER_SYMBOL_EDGE": 5,
    "SPOT_STAGE1_TOP_K": 3,
    "SPOT_STAGE1_MIN_P10_GMGR": -0.5,
    "SPOT_BUCKET_TOP_EACH": 2,
    "SPOT_CONSTRAINT_PSR_FLOOR": 0.08,
    "SPOT_CONSTRAINT_DSR_FLOOR": 0.0,
    "SPOT_CONSTRAINT_MIN_MEAN_TRADES": 30.0,
    "SPOT_CONSTRAINT_MIN_PATH_TW_RATIO": 0.92,
    "SPOT_CONSTRAINT_MIN_MEAN_PF": 1.05,
    "SPOT_CONSTRAINT_MIN_MEAN_PATH_TAIL": 1.10,
    "SPOT_CONSTRAINT_MIN_WORST25_CALMAR": 0.2,
    "SPOT_OBJECTIVE_GAMMA": 1.5,
    "SPOT_OBJECTIVE_W_TAIL_RATIO": 0.60,
    "SPOT_OBJECTIVE_MAR": 0.0,
    "SPOT_OBJECTIVE_Z_CONF": 1.645,
    "SPOT_MIN_REGIME_ON_RATE": 0.22,
    "SPOT_REGIME_ON_RATE_PENALTY_WEIGHT": 0.25,
    "SPOT_TEMPORAL_DECAY_PEN_WEIGHT": 0.20,
    "SPOT_MIN_P10_GMGR_CAGR_PCT": 5.0,
    "SPOT_STAGE2_MAX_PER_SIGNAL_TYPE": 2,
    "SPOT_EXIT_FAMILY_PRIOR_SCALE": 1.0,
    "SPOT_OBJECTIVE_W_CALMAR": 0.10,
    "SPOT_PBO_MAX": 0.45,
    "SPOT_PBO_GATE_HARD": False,
    "SPOT_MULTI_WINDOW_OOS_ENABLED": True,
    "SPOT_MULTI_WINDOW_OOS_SUBS": 2,
    "SPOT_MULTI_WINDOW_BARS_PER_MONTH": 180,
    "SPOT_MULTI_WINDOW_MIN_POSITIVE": 3,
    "SPOT_MULTI_WINDOW_MIN_MEDIAN_CAGR_PCT": 15.0,
    "SPOT_MULTI_WINDOW_MAX_WORST_MDD_PCT": 35.0,
    "SPOT_REGIME_DIAGNOSTIC_ENABLED": True,
    "SPOT_REGIME_STRESS_MAX_MDD_PCT": 40.0,
    "SPOT_SYMBOL_CLUSTER": {
            "KRW-ETH": "cluster_0",
            "KRW-POL": "cluster_1",
            "KRW-XLM": "cluster_2",
            "KRW-AWE": "cluster_3",
            "KRW-AXS": "cluster_4",
            "KRW-1INCH": "cluster_5",
        },
}

TARGET_TIMEFRAMES: List[str] = ["4h"]
WARMUP_PERIODS: Dict[str, int] = {"4h": 540}

# Spot optimization: exclude institutional-only sizing for small KRW books (see tmp.md).
SPOT_EXCLUDED_SIZING_METHODS: frozenset[str] = frozenset({"liquidity_adjusted"})

# Shared-cash concurrency slippage (Reference ADV anchor; ~300B KRW / 4H notional proxy).
SLIPPAGE_GAMMA_BASE: float = 0.03
SLIPPAGE_REFERENCE_ADV_KRW: float = 3e10

ENGINE_PARAM_SPACE: Dict[str, Dict[str, Any]] = {
    "EXIT_FAMILY": {
        "type": "categorical",
        "choices": ("TREND_HOLD", "BALANCED", "FAST_REALIZE"),
    },
    "LONG_ATR_MULT": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.25},
    "TRAIL_ATR_MULT": {"type": "float", "low": 2.5, "high": 6.0, "step": 0.5},
    "USE_TRAILING_STOP": {"type": "categorical", "choices": (True, False)},
    "LONG_SCALE_ATR_MULT": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.5},
    "SCALE_OUT_PCT": {"type": "float", "low": 0.25, "high": 0.60, "step": 0.05},
    "TIME_STOP_BARS": {"type": "int", "low": 0, "high": 168, "step": 12},
    "RISK_PER_TRADE": {"type": "float", "low": 0.005, "high": 0.08},
    "MAX_EXPOSURE": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
    "RSI_EXIT_THRESHOLD": {"type": "float", "low": 75.0, "high": 92.0, "step": 1.0},
    "RSI_EXIT_PERIOD": {"type": "int", "low": 10, "high": 21, "step": 1},
    "BB_EXIT_PERIOD": {"type": "int", "low": 14, "high": 50, "step": 2},
    "BB_EXIT_STD": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
}

SPOT_SHARED_PARAM_SPACE: Dict[str, Dict[str, Any]] = {
    "ATR_PERIOD": {"type": "int", "low": 10, "high": 20, "step": 2},
    "KELLY_FRACTION": {"type": "float", "low": 0.2, "high": 0.8, "step": 0.1},
    "MAX_CAP_PER_COIN": {"type": "float", "low": 0.10, "high": 0.35},
    "MAX_PARTICIPATION_RATE": {"type": "float", "low": 0.005, "high": 0.05, "step": 0.005},
}


def get_search_space_futures(tf: str) -> Dict[str, Dict[str, Any]]:
    # Fallback to a default if not defined, but here we assume it might be needed by other files.
    # We will use a dummy or existing one if we can find it.
    return {}


def get_search_space_spot(tf: str) -> Dict[str, Dict[str, Any]]:
    _ = tf
    from src.spot_strategy.opt_spot_utils.opt_params import build_full_discovery_space

    return build_full_discovery_space()


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
    oos_start: datetime.date = current_quarter_start - relativedelta(months=12)
    is_start: datetime.date = oos_start - relativedelta(months=24)
    fetch_start: datetime.date = is_start - relativedelta(days=500)
    return (
        fetch_start.strftime("%Y-%m-%d"),
        is_start.strftime("%Y-%m-%d"),
        oos_start.strftime("%Y-%m-%d"),
        oos_end.strftime("%Y-%m-%d"),
    )
