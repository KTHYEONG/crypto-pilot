from __future__ import annotations

from typing import Any

# ==============================================================================
# OPTIMIZATION FUTURES SEARCH SPACE & CONFIGURATION (Cross-Sectional Ranking Edition)
# ==============================================================================

OPT_FUTURES_CONFIG: dict[str, Any] = {
    # Reduced from 2000: CAWF-R has K=5 independent legs; more trials = more selection bias.
    # Bonferroni SR_bench = sqrt(2*ln(400))≈3.46 vs sqrt(2*ln(2000))≈5.30 — honest deflation.
    "total_trials": 1000,
    # Phase-D TPESampler: random trials before TPE; must be < n_ml_trials (see opt_main_futures).
    "tpe_n_startup_trials": 200,
    # Caps startup at this fraction of Phase-D n_trials (prevents 384/500 ~77% random when
    # total_trials-oriented tpe_n_startup is reused for FUTURES_ML_PHASE_D_TRIALS=500).
    "FUTURES_ML_PHASE_D_TPE_STARTUP_FRAC": 0.20,
    # Phase-D: trial 0 = deploy JSON in search space (TPE anchor; counts toward n_trials).
    "FUTURES_ML_PHASE_D_ENQUEUE_DEPLOY_JSON": True,
    "FUTURES_ML_PHASE_D_DEPLOY_JSON_REL": "results/best_futures_1h.json",
    "seeds": [42],
    "TARGET_TIMEFRAMES": ["4h"],
    # Risk & Portfolio (Phase D)
    "FUTURES_MAX_CONCURRENT_POSITIONS": 3,
    "FUTURES_MIN_PF": 1.35,
    "FUTURES_MAX_MDD": 30.0,
    "FUTURES_MIN_CAGR_PCT": 5.0,
    "FUTURES_DISCOVERY_LEVERAGE": 5,
    "FUTURES_PBO_MAX": 0.45,
    # Hardening: candidate PBO must be ≤ this to beat champion.json (skill P4 strict guard).
    "FUTURES_CHAMPION_PBO_STRICT_MAX": 0.50,
    # futures-opt Phase 3: trial-count PBO ceiling (-step per bucket); clamp avoids overkill.
    "FUTURES_MC_GATE_TRIAL_ADJUST_ENABLED": True,
    "FUTURES_MC_GATE_BUCKET_TRIALS": 100,
    "FUTURES_MC_PBO_STEP_PER_BUCKET": 0.01,
    "FUTURES_MC_PBO_CEILING_CLAMP_MIN": 0.38,
    # DSR base: +0.02 per 100-trial bucket (MC-DSR); at 500t→0.55, 2000t→0.85 (cap 0.95).
    "FUTURES_MC_DSR_TRIAL_ADJUST_ENABLED": True,
    "FUTURES_MC_DSR_STEP_PER_BUCKET": 0.02,
    "FUTURES_MC_DSR_FLOOR_CAP": 0.95,
    # Path A: True + PBO_MAX 0.50 for exploration; default off (session 42: 300t HOLD).
    "FUTURES_TIER1_SHIELD_MODE": False,
    # Universal Cross-Sectional Alpha Miner Settings
    "FUTURES_ML_ALPHA_POPULATION": 1000,
    "FUTURES_ALPHA_LONG_BIAS": 2.0,
    "FUTURES_ALPHA_ASYMMETRIC_FITNESS_WEIGHT": 0.7,
    "FUTURES_ALPHA_REGIME_SPECIFIC_LEARNING": True,
    # Tier 2: +2 generations for marginal symbolic-regression diversity (cheap vs full schema lift).
    "FUTURES_ML_ALPHA_GENERATIONS": 22,
    "FUTURES_ML_ALPHA_TARGET_HORIZON": 6,
    "FUTURES_ML_ALPHA_HORIZONS": (6, 12, 24, 48),
    # Cache-control refit: when True, bypass Alpha raw cache and force alpha retraining.
    "FUTURES_ML_FORCE_RETRAIN_ALPHA": True,
    "FUTURES_ML_ALPHA_PARSIMONY": 0.02,
    "FUTURES_ML_ALPHA_USE_TBM_WEIGHT": True,
    "FUTURES_ML_PRE_ALPHA_REGIME": False,
    "FUTURES_ML_PRE_ALPHA_REGIME_STATES": 3,
    "FUTURES_ML_IC_FILTER_USE_HAC": True,
    "FUTURES_ML_IC_FILTER_USE_EWMA": False,
    "FUTURES_ML_IC_EWMA_HALF_LIFE": 540.0,
    "FUTURES_ML_IC_HALF_LIFE": 2.3,
    # Tier 2 discovery: significantly relax cross-section balance cap for growth.
    "FUTURES_ML_IC_SYMBOL_BALANCE_MAX": 3.0,
    "FUTURES_ML_IC_REGIME_GATE": False,
    # Tier 2 discovery: FDR relaxation to expand candidate pool.
    "FUTURES_ML_IC_FDR_Q": 0.40,
    "FUTURES_ML_ALPHA_NSGA2_ENABLED": False,
    # NSGA-II population size. Generations = trials / population_size.
    # Target ≥ 10 generations → min trials = population_size * 10.
    # Sessions 27/35/38: NSGA-II 3-strike empirical failure. DSR collapses to ~0.45.
    # Root cause: TOPSIS equal-weight dilutes DSR primacy; Pareto front drifts outside valid region.
    "FUTURES_NSGA2_POPULATION_SIZE": 30,
    # HMM stable regime (fixed hyperparameters; not in Optuna search space)
    "FUTURES_HMM_K_STATES": 5,
    # Deployment floor: ≥0.45 (align HMM systemic prior with Phase D Kelly band).
    "FUTURES_HMM_KELLY_SHRINKAGE": 0.45,
    # Session 41 sweep: 0.65-0.68 heuristically best; 0.66 compromise vs 0.70 default.
    "FUTURES_HMM_CRISIS_THRESHOLD": 0.66,
    # Slightly stronger sticky transitions → stabler systemic HMM under leg refit (Path C).
    "FUTURES_HMM_TRANSITION_PRIOR_ALPHA": 0.50,
    # HMM Posterior Smoothing (EMA, DEMA, TEMA, HMA, KAMA, ALMA, JMA)
    "FUTURES_HMM_SMOOTHING_METHOD": "KAMA",
    "FUTURES_HMM_SMOOTHING_SPAN": 12,
    # Session 39: recovery override abandoned (P10 -0.027, DSR collapse). Keep disabled.
    "CRISIS_RECOVERY_TREND_THR": 1e9,
    "CRISIS_RECOVERY_FLOOR": 0.30,
    # CAWF-R: K=5 chronological AWF legs replace reshuffled CPCV paths.
    "FUTURES_AWF_K_LEGS": 5,
    "FUTURES_AWF_MIN_TRAIN_FRAC": 0.40,  # first leg trains on 40% of IS bars
    # PLGD objective weights (see objective_ml.py).
    "FUTURES_PLGD_LAMBDA_DEF": 0.5,   # Bonferroni trial-deflation strength
    "FUTURES_PLGD_LAMBDA_TAIL": 0.7,  # worst-leg tail penalty multiplier (k=5: 1 bad leg allowed)
    "FUTURES_AWF_NET_EDGE_MIN": 1.5,   # min EV/cost ratio (avg PnL / round-trip cost)
    # SPA bootstrap for post-run diagnostics (not used per-trial).
    "FUTURES_SPA_N_BOOTSTRAP": 2000,
    "FUTURES_SPA_P_VALUE_MAX": 0.10,   # SPA p-value ≤ 0.10 → reject H0: zero alpha
    # CPCV legacy params (kept for cv_utils compat; not used in AWF objective).
    "FUTURES_CPCV_N_BLOCKS": 8,
    "FUTURES_CPCV_K_TEST": 3,
    # Increased from 3: 5 WF OOS legs provide regime-diverse robustness verification.
    "FUTURES_WF_OOS_LEGS": 5,
    # R-6: per WF OOS leg, retrain systemic HMM on data strictly before leg start (GP frozen).
    "FUTURES_WF_HMM_LEG_REFIT": True,
    "FUTURES_WF_LEG_TW_MIN_ALL": 0.90,
    "FUTURES_WF_LEG_TW_MEAN_MIN": 1.00,
    # futures-opt P4: log reference + optional soft warn (not a hard gate).
    "FUTURES_ERGODICITY_GUIDELINE_PCT": 15.0,
    "FUTURES_ERGODICITY_HARD_GATE_ENABLED": True,
    # Phase 2: entry gate (rolling quantile), TBM horizon (1m bars), meta purge alignment
    "ENTRY_QUANTILE_WINDOW": 240,
    "FUTURES_ENTRY_NUMBA_THRESHOLD": 0.5,
    "FUTURES_TBM_TIME_STOP_BARS": 1440,
    "FUTURES_TBM_VOL_SCALE_WINDOW": 24,
    "FUTURES_META_VERTICAL_BARRIER_BARS": 24,
    "FUTURES_META_MIN_POS_ISOTONIC": 200,
    "FUTURES_USE_META_LABELER": True,
    "FUTURES_CRISIS_GATE_PROB_DEFAULT": 0.7,
    "FUTURES_MIN_TRADES_TARGET": 30,
    # Phase 3: WF refit HMM-only when Meta disabled; optional PBO/DSR hard gate after Optuna
    "FUTURES_ML_WF_REFIT_ENABLED": True,
    "FUTURES_ML_WF_REFIT_LEGS": 3,
    "FUTURES_PHASE3_HARD_GATE": True,
    # gate1_dsr ∈ [0,1] from CPCV paths (Bailey & López de Prado style)
    # AWF gate1_dsr = awf_pos_frac (fraction of legs with positive log-TW).
    # 0.40 → at least 2/5 legs positive (floor). MC adjustment raises to 0.48 at 400 trials.
    "FUTURES_ML_GATE1_DSR_MIN": 0.40,
    # Distributional hardening: 10th percentile CPCV path must still survive.
    # log(TW) > 0.0 ↔ TW > 1.0, so this is a direct worst-decile survival test.
    # AWF worst-leg floor: ≥-0.05 means at most 5% log loss on worst single leg.
    # Changed from -0.03 to -0.10 to accommodate v11 alpha volatility.
    "FUTURES_CPCV_P10_LOG_TW_MIN": -0.10,
    # Improvement 1: Friction-Aware EV Hurdle
    "FUTURES_ML_EV_HURDLE_RATIO": 0.0,
    # Improvement 2: Regime-Aware Dynamic Kelly Scaling
    "FUTURES_HMM_DYNAMIC_KELLY_ENABLED": True,
    # Improvement 3: PLGD Leg Stability Weight
    "FUTURES_PLGD_AWF_LEG_STABILITY_WEIGHT": 0.8,
    # Improvement 4: Friction-Aware Virtual Cost (bps per trade)
    "FUTURES_VIRTUAL_FRICTION_BPS": 3.5,
}

# Cross-Sectional Strategy Parameter Space
SIGNAL_PARAM_SPACE_FUTURES: dict[str, dict[str, Any]] = {
    # Multi-session stability: lower bound ≥26 (matches ML Phase D discovery band).
    "ATR_PERIOD": {"type": "int", "low": 26, "high": 40, "step": 2},
    "ATR_MULT": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.25},
    "TRAIL_MULT": {"type": "float", "low": 1.5, "high": 6.5, "step": 0.5},
    "TP_MULT": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.5},
}

PORTFOLIO_PARAM_SPACE_FUTURES: dict[str, dict[str, Any]] = {
    "RISK_PER_TRADE": {"type": "float", "low": 0.02, "high": 0.10, "step": 0.01},
    "MAX_EXPOSURE_PER_COIN": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.25},
    "DD_SCALING_THRESHOLD": {"type": "float", "low": 0.10, "high": 0.30, "step": 0.10},
}

ENGINE_PARAM_SPACE_FUTURES: dict[str, dict[str, Any]] = {
    **SIGNAL_PARAM_SPACE_FUTURES,
    **PORTFOLIO_PARAM_SPACE_FUTURES,
    "K_RANK": {"type": "int", "low": 1, "high": 4, "step": 1},
    "REBALANCE_BARS": {"type": "categorical", "choices": (1, 3, 6, 12)},
    "MIN_SCORE_PERCENTILE": {"type": "float", "low": 0.50, "high": 0.90, "step": 0.10},
    "CS_Z_SCORE_THRESHOLD": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
    "CRISIS_GAMMA": {"type": "float", "low": 1.0, "high": 5.0, "step": 1.0},
    "CRISIS_GATE_PROB": {"type": "float", "low": 0.50, "high": 0.90, "step": 0.10},
}

# Dynamic Universe Anchor Symbols
FUTURES_ANCHOR_SYMBOLS: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
]

# This list will be overwritten by the dynamic screener
FUTURES_SYMBOLS: list[str] = [
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
    "UNI/USDT",
    "DOT/USDT",
    "1000SHIB/USDT",
    "XLM/USDT",
    "APT/USDT",
]

FUTURES_SCREENER_CONFIG: dict[str, Any] = {
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
SPOT_ANCHOR_SYMBOLS: list[str] = ["KRW-ETH", "KRW-SOL", "KRW-XRP"]
SPOT_SYMBOLS: list[str] = ["KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-HBAR"]

OPT_SPOT_CONFIG: dict[str, Any] = {
    "total_trials": 1500,
    "tpe_n_startup_trials": 256,
    "seeds": [42],
    "n_jobs": 3,
    "TARGET_TIMEFRAMES": ["4h"],
    "CPCV_N_BLOCKS": 8,
    "CPCV_K_TEST": 3,
}

def get_search_space_futures(tf: str, stage: int = 0) -> dict[str, dict[str, Any]]:
    _ = tf
    from src.domain.futures.opt_futures_utils.opt_params import build_full_discovery_space_futures
    return build_full_discovery_space_futures()

def get_search_space_spot(tf: str) -> dict[str, dict[str, Any]]:
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
