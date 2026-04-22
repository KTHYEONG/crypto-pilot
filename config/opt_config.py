from __future__ import annotations

from typing import Any, Dict, List

# ==============================================================================
# OPTIMIZATION FUTURES SEARCH SPACE & CONFIGURATION (Cross-Sectional Ranking Edition)
# ==============================================================================

OPT_FUTURES_CONFIG: Dict[str, Any] = {
    "total_trials": 2000,
    "tpe_n_startup_trials": 384,
    "seeds": [42],
    "TARGET_TIMEFRAMES": ["1h"],
    # Risk & Portfolio (Phase D)
    "FUTURES_MAX_CONCURRENT_POSITIONS": 3,
    "FUTURES_MIN_PF": 1.35,
    "FUTURES_MAX_MDD": 30.0,
    "FUTURES_MIN_CAGR_PCT": 30.0,
    "FUTURES_DISCOVERY_LEVERAGE": 5,
    "FUTURES_PBO_MAX": 0.45,
    # Universal Cross-Sectional GP Miner Settings
    "FUTURES_ML_GP_POPULATION": 1000,
    "FUTURES_ML_GP_GENERATIONS": 20,
    "FUTURES_ML_GP_TARGET_HORIZON": 6,
    "FUTURES_ML_GP_HORIZONS": (6, 12, 24, 48),
    "FUTURES_ML_GP_PARSIMONY": 0.001,
    "FUTURES_ML_GP_USE_TBM_WEIGHT": True,
    "FUTURES_ML_PRE_GP_REGIME": False,
    "FUTURES_ML_PRE_GP_REGIME_STATES": 3,
    "FUTURES_ML_IC_FILTER_USE_HAC": True,
    "FUTURES_ML_IC_FILTER_USE_EWMA": False,
    "FUTURES_ML_IC_EWMA_HALF_LIFE": 540.0,
    "FUTURES_ML_IC_SYMBOL_BALANCE_MAX": 3.0,
    "FUTURES_ML_IC_REGIME_GATE": True,
    "FUTURES_ML_IC_FDR_Q": 0.15,
    "FUTURES_ML_GP_NSGA2_ENABLED": False,
    # HMM stable regime (fixed hyperparameters; not in Optuna search space)
    "FUTURES_HMM_K_STATES": 4,
    "FUTURES_HMM_KELLY_SHRINKAGE": 0.4,
    "FUTURES_HMM_CRISIS_THRESHOLD": 0.70,
    "FUTURES_HMM_TRANSITION_PRIOR_ALPHA": 0.2,
    "FUTURES_ML_PHASE_D_TRIALS": 500,
    "FUTURES_CPCV_N_BLOCKS": 8,
    "FUTURES_CPCV_K_TEST": 3,
    "FUTURES_WF_OOS_LEGS": 3,
    # R-6: per WF OOS leg, retrain systemic HMM on data strictly before leg start (GP frozen).
    "FUTURES_WF_HMM_LEG_REFIT": True,
    "FUTURES_WF_LEG_TW_MIN_ALL": 1.0,
    "FUTURES_WF_LEG_TW_MEAN_MIN": 1.05,
    # Phase 2: entry gate (rolling quantile), TBM horizon (1m bars), meta purge alignment
    "ENTRY_QUANTILE_WINDOW": 240,
    "FUTURES_ENTRY_NUMBA_THRESHOLD": 0.5,
    "FUTURES_TBM_TIME_STOP_BARS": 1440,
    "FUTURES_TBM_VOL_SCALE_WINDOW": 24,
    "FUTURES_META_VERTICAL_BARRIER_BARS": 24,
    "FUTURES_META_MIN_POS_ISOTONIC": 200,
    "FUTURES_USE_META_LABELER": False,
    "FUTURES_CRISIS_GATE_PROB_DEFAULT": 0.7,
    "FUTURES_MIN_TRADES_TARGET": 10,
    # Phase 3: WF refit HMM-only when Meta disabled; optional PBO/DSR hard gate after Optuna
    "FUTURES_ML_WF_REFIT_ENABLED": True,
    "FUTURES_ML_WF_REFIT_LEGS": 3,
    "FUTURES_PHASE3_HARD_GATE": True,
    # gate1_dsr ∈ [0,1] from CPCV paths (Bailey & López de Prado style)
    "FUTURES_ML_GATE1_DSR_MIN": 0.20,
}

# Cross-Sectional Strategy Parameter Space
SIGNAL_PARAM_SPACE_FUTURES: Dict[str, Dict[str, Any]] = {
    "ATR_PERIOD": {"type": "int", "low": 10, "high": 20, "step": 2},
    "LONG_ATR_MULT": {"type": "float", "low": 1.5, "high": 4.5, "step": 0.25},
    "LONG_TRAIL_MULT": {"type": "float", "low": 2.5, "high": 6.0, "step": 0.5},
    "SHORT_ATR_MULT": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.25},
    "SHORT_TP_MULT": {"type": "float", "low": 1.0, "high": 3.5, "step": 0.5},
    "SHORT_TRAIL_MULT": {"type": "float", "low": 1.5, "high": 4.5, "step": 0.5},
}

PORTFOLIO_PARAM_SPACE_FUTURES: Dict[str, Dict[str, Any]] = {
    "RISK_PER_TRADE": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.005},
    "MAX_EXPOSURE_PER_COIN": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.1},
    "DD_SCALING_THRESHOLD": {"type": "float", "low": 0.10, "high": 0.25, "step": 0.05},
}

ENGINE_PARAM_SPACE_FUTURES: Dict[str, Dict[str, Any]] = {
    **SIGNAL_PARAM_SPACE_FUTURES,
    **PORTFOLIO_PARAM_SPACE_FUTURES,
    "K_LONG": {"type": "int", "low": 1, "high": 4, "step": 1},
    "K_SHORT": {"type": "int", "low": 1, "high": 4, "step": 1},
    "REBALANCE_BARS": {"type": "categorical", "choices": (1, 3, 6, 12)},
    "MIN_SCORE_PERCENTILE": {"type": "float", "low": 0.50, "high": 0.85, "step": 0.05},
    "CRISIS_GATE_PROB": {"type": "float", "low": 0.50, "high": 0.85, "step": 0.05},
}

# Dynamic Universe Anchor Symbols
FUTURES_ANCHOR_SYMBOLS: List[str] = [
    "BTC/USDT",
    "ETH/USDT",
]

# This list will be overwritten by the dynamic screener
FUTURES_SYMBOLS: List[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "ADA/USDT",
    "BCH/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "AAVE/USDT",
    "FIL/USDT",
    "DOT/USDT",
]

FUTURES_SCREENER_CONFIG: Dict[str, Any] = {
    "BROAD_POOL_K": 80,
    "FINAL_POOL_K": 40,
    "MIN_ADV_USDT": 25_000_000,
    "MIN_CORR_BTC": 0.50,
    "MAX_BETA_BTC": 1.40,
    "MIN_BETA_BTC": 0.60,
    "FUNDING_RATE_MAX_ABS": 0.0008,
    "AMIHUD_PRUNE_RATIO": 0.75,
    "MIN_VOL_CV": 0.3,
}

# ==============================================================================
# SPOT CONFIGURATION (Unchanged)
# ==============================================================================
SPOT_ANCHOR_SYMBOLS: List[str] = ["KRW-ETH", "KRW-SOL", "KRW-XRP"]
SPOT_SYMBOLS: List[str] = ["KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-HBAR"]

OPT_SPOT_CONFIG: Dict[str, Any] = {
    "total_trials": 1500,
    "tpe_n_startup_trials": 256,
    "seeds": [42],
    "n_jobs": 3,
    "TARGET_TIMEFRAMES": ["4h"],
    "CPCV_N_BLOCKS": 8,
    "CPCV_K_TEST": 3,
}

def get_search_space_futures(tf: str, stage: int = 0) -> Dict[str, Dict[str, Any]]:
    _ = tf
    from src.domain.futures.opt_futures_utils.opt_params import build_full_discovery_space_futures
    return build_full_discovery_space_futures()

def get_search_space_spot(tf: str) -> Dict[str, Dict[str, Any]]:
    _ = tf
    from src.domain.spot.opt_spot_utils.opt_params import build_full_discovery_space
    return build_full_discovery_space()

def get_quarterly_window(reference_date: Any = None) -> tuple[str, str, str, str]:
    import datetime

    from dateutil.relativedelta import relativedelta
    if reference_date is None:
        reference_date = datetime.date.today()
    elif isinstance(reference_date, str):
        reference_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d").date()
    current_quarter_start_month: int = ((reference_date.month - 1) // 3) * 3 + 1
    current_quarter_start: datetime.date = datetime.date(
        reference_date.year, current_quarter_start_month, 1
    )
    oos_end: datetime.date = current_quarter_start - datetime.timedelta(days=1)
    oos_start: datetime.date = current_quarter_start - relativedelta(months=3)
    is_start: datetime.date = oos_start - relativedelta(months=15)
    fetch_start: datetime.date = is_start - relativedelta(days=365)
    return (
        fetch_start.strftime("%Y-%m-%d"),
        is_start.strftime("%Y-%m-%d"),
        oos_start.strftime("%Y-%m-%d"),
        oos_end.strftime("%Y-%m-%d"),
    )
