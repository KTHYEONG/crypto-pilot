from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass as _dataclass
from typing import Any, Literal

from dateutil.relativedelta import relativedelta as _relativedelta

# ==============================================================================
# OPTIMIZATION FUTURES SEARCH SPACE & CONFIGURATION
# ==============================================================================

OPT_FUTURES_CONFIG: dict[str, Any] = {
    # 3-phase coordinate ascent total ≈ 260 (Bonferroni / reporting baseline).
    "total_trials": 1500,
    # High-speed JIT execution makes early pruning slower than full evaluation.
    # Set to False to completely eliminate SQLite DB WAL locking contention.
    "FUTURES_PRUNING_ENABLED": False,
    # Phase-D TPESampler startup; per-phase samplers in opt_main_futures override as needed.
    "tpe_n_startup_trials": 256,
    # Caps startup at this fraction of Phase-D n_trials (prevents 384/500 ~77% random when
    # total_trials-oriented tpe_n_startup is reused for FUTURES_ML_PHASE_D_TRIALS=500).
    "FUTURES_ML_PHASE_D_TPE_STARTUP_FRAC": 0.20,
    # Optimizer tail-stall guardrails (does not change JAX kernels).
    # Per-trial timeout in worker process; timed-out trials are marked FAIL and loop proceeds.
    "FUTURES_OPT_TRIAL_TIMEOUT_SEC": 180,
    # Per-chunk future timeout; prevents indefinite wait on a wedged worker batch.
    "FUTURES_OPT_CHUNK_TIMEOUT_SEC": 1200,
    # Optimized for 8-core CPU / 16GB RAM environment.
    # We use 4 workers to ensure each worker has enough RAM for JAX/Optimized kernels (approx 3-4GB per worker).
    "FUTURES_OPT_MAX_WORKERS": 4,
    "FUTURES_OPT_CHUNK_SIZE_CAP": 4,
    "FUTURES_OPT_ENABLE_PROGRESS_POLLER": False,
    # Phase-D: trial 0 = deploy JSON in search space (TPE anchor; counts toward n_trials).
    "FUTURES_ML_PHASE_D_ENQUEUE_DEPLOY_JSON": True,
    "FUTURES_ML_PHASE_D_DEPLOY_JSON_REL": "results/best_futures_1h.json",
    "seeds": [42],
    "TARGET_TIMEFRAMES": ["4h"],
    # --- TF Probe Integration (Phase-2) ---
    "ENABLE_TF_PROBE": False,
    "TF_PROBE_GRID": ["2h", "4h", "6h", "8h", "12h", "1d"],
    "TF_PROBE_MIN_TSTAT": 2.0,
    "TF_PROBE_REQUIRE_FDR": True,
    "TF_PROBE_MIN_FOLD_CONSISTENCY": 0.75,
    "TF_PROBE_MAX_WORKERS": 8,
    # Risk & Portfolio (Phase D)
    "FUTURES_EXECUTION_MODE": "intrabar_1m",
    "FUTURES_MAX_CONCURRENT_POSITIONS": 3,
    "FUTURES_MIN_PF": 1.50,
    "FUTURES_MAX_MDD": 20.0,
    "FUTURES_MIN_CAGR_PCT": 30.0,
    "FUTURES_DISCOVERY_LEVERAGE": 5,
    "FUTURES_PBO_MAX": 0.10,
    # Hardening: candidate PBO must be ≤ this to beat champion.json (skill P4 strict guard).
    "FUTURES_CHAMPION_PBO_STRICT_MAX": 0.15,
    # futures-opt Phase 3: trial-count PBO ceiling (-step per bucket); clamp avoids overkill.
    "FUTURES_MC_GATE_TRIAL_ADJUST_ENABLED": True,
    "FUTURES_MC_GATE_BUCKET_TRIALS": 100,
    "FUTURES_MC_PBO_STEP_PER_BUCKET": 0.01,
    "FUTURES_MC_PBO_CEILING_CLAMP_MIN": 0.12,
    # DSR base: +0.02 per 100-trial bucket (MC-DSR); at 500t→0.55, 2000t→0.85 (cap 0.95).
    "FUTURES_MC_DSR_TRIAL_ADJUST_ENABLED": True,
    "FUTURES_MC_DSR_STEP_PER_BUCKET": 0.02,
    "FUTURES_MC_DSR_FLOOR_CAP": 0.95,
    # Path A: True + PBO_MAX 0.50 for exploration; default off (session 42: 300t HOLD).
    "FUTURES_TIER1_SHIELD_MODE": False,
    # 3-Layer Tiered Hybrid Architecture (CS Rank + Diagonal Kelly).
    # True = tiered pipeline 실행 (Phase D allocation 스킵); False = Phase D 유지.
    "USE_CS_RANK_ENGINE": True,
    # Tiered L2 AWF Optuna 탐색 trial 수 (Phase D와 별도)
    "L2_OPTUNA_TRIALS": 200,
    # L2 Optuna 병렬 최적화 배치 사이즈 (1 이하는 순차 실행)
    "L2_OPTUNA_BATCH_SIZE": 6,
    # Universal Cross-Sectional Alpha Miner Settings
    "FUTURES_ML_ALPHA_POPULATION": 1500,
    "FUTURES_ALPHA_LONG_BIAS": 2.0,
    "FUTURES_ALPHA_ASYMMETRIC_FITNESS_WEIGHT": 0.7,
    "FUTURES_ALPHA_REGIME_SPECIFIC_LEARNING": True,
    # Tier 2: +2 generations for marginal symbolic-regression diversity (cheap vs full schema lift).
    "FUTURES_ML_ALPHA_GENERATIONS": 30,
    "FUTURES_ML_ALPHA_TARGET_HORIZON": 24,
    "FUTURES_ML_ALPHA_HORIZONS": (12, 24, 48, 72, 96),
    # Cache-control refit: when True, bypass Alpha raw cache and force alpha retraining.
    "FUTURES_ML_FORCE_RETRAIN_ALPHA": True,
    # Thematic breadth: LambdaRank buckets with multiple slots per theme. Total = 3 x value.
    # Raising adds interaction-style capacity without injecting systemic HMM into Trend slots.
    "FUTURES_ML_ALPHA_SLOTS_PER_THEME": 3,
    "FUTURES_ML_ALPHA_PARSIMONY": 0.02,
    "FUTURES_ML_ALPHA_USE_TBM_WEIGHT": True,
    "FUTURES_ML_PRE_ALPHA_REGIME": False,
    "FUTURES_ML_PRE_ALPHA_REGIME_STATES": 3,
    "FUTURES_ML_IC_FILTER_USE_HAC": True,
    "FUTURES_ML_IC_FILTER_USE_EWMA": False,
    "FUTURES_ML_IC_EWMA_HALF_LIFE": 540.0,
    "FUTURES_ML_IC_HALF_LIFE": 40,
    # Tier 2 discovery: significantly relax cross-section balance cap for growth.
    "FUTURES_ML_IC_SYMBOL_BALANCE_MAX": 3.0,
    "FUTURES_ML_IC_REGIME_GATE": True,
    # FDR: 0.10 기준으로 강화 (HORIZONS=(12,24,48,72,96) multi-horizon 유효성 확보)
    "FUTURES_ML_IC_FDR_Q": 0.10,
    # Step3: regime-conditional alpha utility pressure (disabled by default).
    "FUTURES_STEP3_REGIME_ALPHA_ENABLED": False,
    # CHOP-fragile rejection gate: active when chop support is sufficiently high.
    "FUTURES_STEP3_CHOP_SUPPORT_MIN": 0.25,
    "FUTURES_STEP3_CHOP_IC_MIN": -0.01,
    # Soft downweight for CHOP-fragile survivors (used in ensemble weighting).
    "FUTURES_STEP3_CHOP_WEIGHT_MULT": 0.50,
    # floor for soft downweighted component weights.
    "FUTURES_STEP3_WEIGHT_MULT_FLOOR": 0.20,
    "FUTURES_ML_ALPHA_NSGA2_ENABLED": False,  # Sessions 27/35/38: NSGA-II 3-strike failure → TPE
    # NSGA-II population size. Generations = trials / population_size.
    # Target ≥ 10 generations → min trials = population_size * 10.
    # Sessions 27/35/38: NSGA-II 3-strike empirical failure. DSR collapses to ~0.45.
    # Root cause: TOPSIS equal-weight dilutes DSR primacy; Pareto front drifts outside valid region.
    "FUTURES_NSGA2_POPULATION_SIZE": 30,
    # J-Score single-objective TPE params (Phase redesign: maximize compound growth)
    "FUTURES_J_LAMBDA_DOWNSIDE": 0.6,
    "FUTURES_J_PSI_DD": 0.3,
    "FUTURES_J_GAMMA_OVERFIT": 0.5,
    "FUTURES_J_GAMMA_REGIME": 0.4,
    "FUTURES_J_CONSISTENCY_FLOOR": 0.20,
    "FUTURES_J_MIN_TRADES_PER_LEG": 8,
    "FUTURES_J_HARD_FAIL_VALUE": -10.0,
    "FUTURES_DEPLOY_J_FLOOR": 0.0,
    "FUTURES_PRUNER_STARTUP_TRIALS": 40,
    "FUTURES_PRUNER_WARMUP_STEPS": 2,
    "FUTURES_PRUNER_TYPE": "wilcoxon",  # "wilcoxon", "successive_halving", "median"
    "FUTURES_PRUNER_WILCOXON_P": 0.10,
    "FUTURES_OPTUNA_DB_PATH": "",
    # S2: Auxiliary volatility-based CRISIS gate.
    # DISABLED: vol gate fires during profitable high-vol IS periods (CRISIS G=+0.193% in IS),
    # collapsing IS CAGR from ~30% to 2.6%. IS-OOS mismatch requires a smarter approach
    # (e.g., cross-sectional synchronized signal, not per-symbol rolling vol).
    "FUTURES_VOL_CRISIS_GATE_ENABLED": False,
    "FUTURES_VOL_CRISIS_WINDOW": 20,
    "FUTURES_VOL_CRISIS_MULT": 3.0,
    # Step2 policy redesign: tail8 is primarily a damping input, not hard-flat trigger.
    "FUTURES_POLICY_FLAT_TAIL8_THR": 0.96,
    "FUTURES_POLICY_FLAT_MIX_TAIL8_W": 0.15,
    "FUTURES_POLICY_FLAT_REALIZED_EXTREME_THR": 0.92,
    "FUTURES_POLICY_HAZARD_DAMP_THR": 0.52,
    "FUTURES_POLICY_HAZARD_DAMP_MAX": 0.62,
    "FUTURES_POLICY_TAIL8_DAMP_THR": 0.78,
    "FUTURES_POLICY_TAIL8_DAMP_MAX": 0.58,
    "FUTURES_POLICY_DAMP_CURVE_POWER": 1.25,
    "FUTURES_POLICY_DAMP_MIX_REALIZED_W": 0.65,
    "FUTURES_POLICY_DAMP_MIX_PRE_W": 0.20,
    "FUTURES_POLICY_DAMP_MIX_TAIL8_W": 0.15,
    "FUTURES_POLICY_DEFENSE_SOFT_THR": 0.38,
    "FUTURES_POLICY_DEFENSE_HARD_THR": 0.66,
    "FUTURES_POLICY_DEFENSE_NEAR_FLAT_THR": 0.84,
    "FUTURES_POLICY_DEFENSE_SOFT_MULT": 0.82,
    "FUTURES_POLICY_DEFENSE_HARD_MULT": 0.52,
    "FUTURES_POLICY_DEFENSE_NEAR_FLAT_MULT": 0.24,
    "FUTURES_POLICY_DEFENSE_TAIL_HARD_THR": 0.92,
    "FUTURES_POLICY_DEFENSE_HARD_REALIZED_THR": 0.55,
    "FUTURES_POLICY_DEFENSE_NEAR_FLAT_REALIZED_THR": 0.72,
    "FUTURES_POLICY_DEFENSE_HARD_TAIL_RANK_THR": 0.90,
    "FUTURES_POLICY_DEFENSE_NEAR_FLAT_TAIL_RANK_THR": 0.95,
    "FUTURES_POLICY_DEFENSE_SOFT_REALIZED_FLOOR": 0.30,
    "FUTURES_POLICY_DEFENSE_SOFT_TAIL_THR": 0.70,
    "FUTURES_POLICY_DEFENSE_SOFT_SUP_THR": 0.68,
    "FUTURES_POLICY_DEFENSE_HARD_SUP_THR": 0.86,
    "FUTURES_POLICY_DEFENSE_NEAR_FLAT_SUP_THR": 0.92,
    # Execution-oriented evaluation gate for Step2 contribution measurement.
    "FUTURES_POLICY_EXEC_STRONG_TAIL_Q": 0.88,
    "FUTURES_POLICY_EXEC_STRONG_TAIL_ABS_THR": 0.72,
    "FUTURES_POLICY_EXEC_DAMP_ACTIVE_THR": 0.72,
    "FUTURES_EXEC_TIER_SOFT_MAX_EXP": 0.80,
    "FUTURES_EXEC_TIER_HARD_MAX_EXP": 0.50,
    "FUTURES_EXEC_TIER_NEAR_FLAT_MAX_EXP": 0.25,
    # Session 39: recovery override abandoned (P10 -0.027, DSR collapse). Keep disabled.
    "CRISIS_RECOVERY_TREND_THR": 1e9,
    "CRISIS_RECOVERY_FLOOR": 0.30,
    # CAWF-R: K anchored legs in the OOS-pool slice (train [0, anchor_i), embargo, test window).
    "FUTURES_AWF_K_LEGS": 5,
    # IS-pool fraction (leading bars); trailing (1 - frac) builds tiled OOS test legs only.
    "FUTURES_AWF_IS_POOL_FRAC": 0.65,
    # Platt: prefer OOS-only windows in optimizer ``_fit_oos_platt_calibrators_from_maps``;
    # legacy tail fraction kept for deprecated tail-window path only.
    "FUTURES_CALIB_PLATT_TAIL_FRAC": 0.30,
    "FUTURES_CALIB_PLATT_MIN_OOS_BARS": 80,
    # If True and the primary OOS window has too few samples, widen to AWF OOS-pool start (still no IS-pool).
    "FUTURES_CALIB_PLATT_OOS_WIDEN_TO_POOL": True,
    # C1 inference panel minimum history: first_dt must be ≤ oos_start - N months.
    # Ensures common time axis spans ≥ N months → folds=2 with train_months=24.
    # (train_months + valid_months + test_months * 2 = 24+3+3*2 = 33 months required)
    "FUTURES_INFERENCE_MIN_HISTORY_MONTHS": 33,
    # Covariance lookback for portfolio_constructor (~30 calendar days in bars per TF).
    "FUTURES_PORTFOLIO_COV_LOOKBACK": 180,
    "FUTURES_PORTFOLIO_COV_LOOKBACK_BY_TF": {"1h": 720, "4h": 180, "1d": 30},
    "FUTURES_PORTFOLIO_KAPPA": 0.35,
    "FUTURES_PORTFOLIO_F_KELLY_MAX": 2.0,
    # L1-4h zero-event fix: minimum universe size for stable evidence window [LIMIT-01].
    "MIN_UNIVERSE_SIZE_FOR_EVIDENCE": 50,
    # L1-4h zero-event fix: canonical membership warm-up in calendar days (replaces W1/W2) [LIMIT-09].
    "MEMBERSHIP_WARMUP_DAYS": 42,
    # ATR stop multiplier (ATR_PERIOD lives in engine / Optuna params).
    "FUTURES_ATR_STOP_MULT": 2.5,
    "FUTURES_SIMPLE_ATR_STOP": True,
    # ATR period fixed (not an Optuna dimension).
    "FUTURES_ATR_PERIOD_FIXED": 30,
    # Rolling σ lookback ≈ 8 calendar days (bars per TF optional override).  # noqa: RUF003
    "FUTURES_COMPOSER_SIGMA_CALENDAR_DAYS": 8.0,
    "FUTURES_COMPOSER_SIGMA_LOOKBACK_BY_TF": {"1h": 192, "4h": 48, "1d": 8},
    # 1.0 = disabled (no crisis long-only magnitude scaling in execution stack).
    "FUTURES_CRISIS_LONG_MAG_SUPPRESS": 1.0,
    "FUTURES_EVENT_TURNOVER_REBALANCE": False,
    # Short borrow ~0.06%/day (fraction of notional per day).
    "FUTURES_SHORT_BORROW_DAILY": 0.0006,
    "SLIPPAGE_BPS_BUFFER_MULT": 1.0,
    "FUTURES_DEFAULT_BETA_ALPHA": 1.0,
    # Phase2 forecast-layer flags (default off for behavior safety).
    "COST_FORECAST_DYNAMIC": False,
    "COST_GATE_AMORTIZE": True,
    "KELLY_USE_RESIDUAL_VAR": False,
    # Dynamic cost forecast hyper-parameters.
    "FUTURES_COST_TAKER_FEE_BPS": 4.0,
    "FUTURES_COST_LATENCY_BUFFER_BPS": 0.5,
    "FUTURES_COST_IMPACT_COEF": 0.5,
    "FUTURES_COST_VOL_BUFFER_COEF": 0.0,
    "FUTURES_COST_FUNDING_EVENT_BUFFER_BPS": 0.0,
    "FUTURES_COST_ADV_LOOKBACK": 30,
    "FUTURES_COST_VOL_LOOKBACK": 20,
    "FUTURES_COST_ORDER_NOTIONAL_USDT": 0.0,
    "FUTURES_COST_UNCERTAINTY_RATIO": 0.1,
    # Stability fail → try runner-up phase-C trial once.
    "FUTURES_STABILITY_RUNNER_UP_RETRY": True,
    "FUTURES_DEFAULT_EV_HURDLE_BPS": 10.0,
    "FUTURES_DEFAULT_TIME_BARRIER_H": 24.0,
    "FUTURES_TMP_MD_CHAMPION_GATES_ENABLED": True,
    "FUTURES_TMP_LAYER1_MEDIAN_LOG_TW_MIN": 0.0,
    "FUTURES_TMP_LAYER1_POS_LEG_RATIO_MIN": 4.0 / 6.0,
    "FUTURES_TMP_LAYER1_MAX_DD_PCT": 12.0,
    "FUTURES_TMP_LAYER2_PSR_MIN": 0.95,
    "FUTURES_TMP_LAYER2_MIN_TRADES_PER_LEG": 20,
    # Layer 2 optional: median(leg_log_tw - stress·round_trip) > 0 (see tmp_md_champion).
    "FUTURES_TMP_LAYER2_FRICTION_STRESS_ENABLED": True,
    "FUTURES_TMP_LAYER2_FRICTION_STRESS_MULT": 1.2,
    # Layer 3: every stability-seed AWF replay must pass Layer-1 checks when hard gate on.
    "FUTURES_TMP_LAYER3_ALL_SEEDS_LAYER1": True,
    "FUTURES_TMP_LAYER3_HARD_GATE": True,
    "FUTURES_LEARNING_SEEDS": [42, 7],
    "FUTURES_STABILITY_SEEDS": [42, 7, 13],
    "FUTURES_COORD_PHASE_TRIALS": {"A": 120, "B": 240, "C": 120},
    "FUTURES_CHAMP_STABILITY_CV_MAX": 0.40,
    # When True, unified gates fail if multi-seed AWF replay CV exceeds max (see key above).
    "FUTURES_CHAMP_STABILITY_HARD_GATE": True,
    "EMBARGO_BARS_BY_TF": {"1h": 168, "4h": 42, "1d": 7},
    # Robust AWF scalar objective weights (see objective_ml.py).
    # median(log_TW) - lambda*MAD - psi*DD_max (fixed lambda, psi).
    "FUTURES_AWF_OBJ_LAMBDA_MAD": 1.5,
    "FUTURES_AWF_OBJ_PSI_DD": 0.3,
    # Phase-D pruning / auxiliary weights (robust AWF objective uses keys above).
    "FUTURES_PHASE_D_W_PF_LONG": 1.5,
    "FUTURES_PHASE_D_W_PF_SHORT": 1.5,
    "FUTURES_PHASE_D_PRUNE_MIN_POS_RATIO": 0.25,
    "FUTURES_AWF_NET_EDGE_MIN": 1.5,  # min EV/cost ratio (avg PnL / round-trip cost)
    # SPA bootstrap for post-run diagnostics (not used per-trial).
    "FUTURES_SPA_N_BOOTSTRAP": 2000,
    "FUTURES_SPA_P_VALUE_MAX": 0.10,  # SPA p-value ≤ 0.10 → reject H0: zero alpha
    # Worst AWF leg log-TW floor (10th percentile / min-leg semantics in objective).
    "FUTURES_AWF_P10_LOG_TW_MIN": -0.10,
    # Increased from 3: 5 WF OOS legs provide regime-diverse robustness verification.
    "FUTURES_WF_OOS_LEGS": 5,
    "FUTURES_ENSEMBLE_MAX_SIZE": 5,
    "FUTURES_META_ALLOC_WINDOW": 12,
    "FUTURES_META_ALLOC_ETA": 0.15,
    "FUTURES_WF_PHASE2_DRIFT_LOG": True,
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
    "FUTURES_IS_SURVIVAL_MIN_CAGR_PCT": 15.0,
    "FUTURES_IS_SURVIVAL_MIN_SHARPE": 1.0,
    # IS Alpha gate: cap BTC buy-and-hold benchmark to avoid unfair hurdle during bull markets.
    # 2023-2025 IS period = crypto bull run → BTC CAGR ≈ 150%+ makes IS_ALPHA_GATE impossible.
    # Cap at 35% (≈long-run BTC avg CAGR); strategy needs positive alpha vs this realistic floor.
    "FUTURES_IS_ALPHA_BTC_CAP_PCT": 35.0,
    # Phase 3: WF refit HMM-only when Meta disabled; optional PBO/DSR hard gate after Optuna
    "FUTURES_ML_WF_REFIT_ENABLED": False,
    "FUTURES_ML_WF_REFIT_LEGS": 3,
    "FUTURES_PHASE3_HARD_GATE": True,
    # gate1_dsr ∈ [0,1]: AWF positive-leg fraction proxy under MC-adjusted floor.
    # 0.40 → at least 2/5 legs positive (floor). MC adjustment raises to 0.48 at 400 trials.
    "FUTURES_ML_GATE1_DSR_MIN": 0.40,
    "FUTURES_AWF_POS_FRAC_MIN": 0.60,  # AWF 게이트: 최소 60% leg 수익 (≥3/5)
    "FUTURES_AWF_MU_LOG_MIN": 0.0,  # AWF 게이트: 평균 leg log-TW > 0
    "FUTURES_STEP2_CHOP_LOSS_SHARE_MAX": 0.60,
    "FUTURES_STEP2_CHOP_TRADE_SHARE_MAX": 0.70,
    "FUTURES_STEP2_FLIP_RATE_PROXY_MAX": 0.75,
    "FUTURES_STEP2_OBJ_CHOP_LOSS_W": 0.25,
    "FUTURES_STEP2_OBJ_CHOP_TRADE_W": 0.15,
    "FUTURES_STEP2_OBJ_FLIP_W": 0.10,
    # S4: Simplified deploy_score.
    # DISABLED: simplified 3-term formula (robust + worst_leg + chop) uses raw negative values
    # without bounded normalization → preferentially selects near-zero conservative trials
    # over genuine performers. Original 7-term bounded_center_score remains active.
    "FUTURES_DEPLOY_SCORE_SIMPLIFIED": False,
    # Step4: Optuna regime-aware deployability hardening.
    "FUTURES_STEP4_DEPLOYABILITY_ENABLED": True,
    "FUTURES_STEP4_OBJ_CHOP_TRADE_W": 0.10,
    "FUTURES_STEP4_OBJ_TURNOVER_W": 0.10,
    "FUTURES_STEP4_CHOP_TRADE_SHARE_MAX": 0.70,
    "FUTURES_STEP4_TURNOVER_COST_RATIO_MAX": 0.25,
    "FUTURES_STEP4_CHOP_PF_FLOOR": 0.95,
    # Ergodicity deviation penalty params (optimizer.py _evaluate_awf_phase_d_aggregate).
    "FUTURES_AWF_ERG_DEV_FLOOR": 1.5,
    "FUTURES_AWF_ERG_DEV_W": 0.001,
    # Worst AWF leg log-TW floor already enforced via FUTURES_AWF_P10_LOG_TW_MIN.
    # Improvement 1: Friction-Aware EV Hurdle
    "FUTURES_ML_EV_HURDLE_RATIO": 1.0,
    # Improvement 3: PLGD Leg Stability Weight
    "FUTURES_PLGD_AWF_LEG_STABILITY_WEIGHT": 0.8,
    # Improvement 4: Friction-Aware Virtual Cost (bps per trade)
    "FUTURES_VIRTUAL_FRICTION_BPS": 3.5,
    # Phase-1 architecture refactor blocks (non-breaking defaults).
    "FUTURES_VALIDATION_CONFIG": {
        "wf_n_legs": 6,
        "wf_purge_bars": 24,
        "wf_min_positive_leg_ratio": 0.70,
        "wf_worst_leg_tw_floor": 0.95,
        "wf_mean_leg_tw_floor": 1.00,
        "wf_ergodicity_guideline_pct": 15.0,
        "wf_ergodicity_hard_gate_enabled": True,
        "use_anchored_awf_geometry": True,
        "wf_anchored_is_pool_frac": 0.70,
    },
    # Phase-A: temporary gross cap (raise after PnL/cost tuning).
    "FUTURES_PHASE_A_MAX_GROSS_EXPOSURE": 1.5,
    "FUTURES_PORTFOLIO_POLICY": {
        "target_ann_vol": 0.35,
        "gross_exposure_cap": 1.80,
        "per_symbol_cap": 0.35,
        "top_k_long": 4,
        "top_k_short": 4,
        "entry_edge_threshold": 0.15,
        "rebalance_bars": 3,
        "min_long_pf": 1.05,
        "min_short_pf": 1.05,
        "min_is_net_alpha_pct": 0.0,
    },
    "FUTURES_CANDIDATE_SELECTOR": {
        "elite_top_n": 30,
        "basin_iqr_mult": 1.0,
    },
    # Direction A: regime-conditional score slope calibration (activated)
    "FUTURES_CANDIDATE_ENSEMBLE_SCORE_CALIBRATION_ENABLED": True,
    "FUTURES_CANDIDATE_ENSEMBLE_SCORE_Z_CLIP": 3.0,
    "FUTURES_CANDIDATE_ENSEMBLE_SCORE_CALIBRATION_MIN_OBS": 60,
    "FUTURES_CANDIDATE_ENSEMBLE_SCORE_SLOPE_K": 100.0,
    "FUTURES_CANDIDATE_FAMILIES": (),
    "FUTURES_CANDIDATE_ENABLED_VARIANTS": (),
    "FUTURES_CANDIDATE_SIDE_FLIP_VARIANTS": (),
    "FUTURES_CANDIDATE_DIAGNOSTIC_TOP_K": 20,
    "FUTURES_CANDIDATE_MIN_VARIANT_OOS_OBS": 100,
    "FUTURES_CANDIDATE_MIN_VARIANT_OOS_EDGE_BPS": 2.4,
    "FUTURES_CANDIDATE_MIN_VARIANT_OOS_HIT_RATE": 0.48,
    "FUTURES_CANDIDATE_REGIME_CELL_ADMISSION_ENABLED": True,
    "FUTURES_CANDIDATE_MIN_REGIME_CELL_OOS_OBS": 60,
    "FUTURES_CANDIDATE_MIN_REGIME_CELL_EDGE_BPS": 8.0,
    "FUTURES_CANDIDATE_MIN_REGIME_CELL_TSTAT": 1.0,
    "FUTURES_CANDIDATE_MAX_ADMITTED_CELLS_PER_VARIANT": 2,
    # Per-symbol beta scaling + idiosyncratic overlay (per_symbol_overlay.py)
    "FUTURES_BETA_WINDOW": 240,
    "FUTURES_BETA_MIN": 0.3,
    "FUTURES_BETA_MAX": 4.0,
    "FUTURES_BETA_PRIOR": 1.0,
    "FUTURES_IDIO_VOL_WINDOW": 72,
    "FUTURES_IDIO_ZHIST_WINDOW": 480,
    "FUTURES_IDIO_DD_WINDOW": 168,
    "FUTURES_IDIO_MAX_CUT": 0.5,
    "FUTURES_PER_SYMBOL_OVERLAY_ENABLED": True,
}

# Single-pass execution: ATR period & stop mult are config/defaults (not this search grid).
SIGNAL_PARAM_SPACE_FUTURES: dict[str, dict[str, Any]] = {
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
    "REBALANCE_BARS": {"type": "categorical", "choices": (1, 3, 6)},
    "label_horizon_bars": {"type": "categorical", "choices": (6, 12, 18)},
    "alpha_emit_select_q": {"type": "float", "low": 0.25, "high": 0.50, "step": 0.05},
    "alpha_emit_weight_k": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
    "MIN_SCORE_PERCENTILE": {"type": "float", "low": 0.40, "high": 0.90, "step": 0.10},
    "CS_Z_SCORE_THRESHOLD": {"type": "float", "low": 0.2, "high": 2.0, "step": 0.1},
    "CRISIS_GAMMA": {"type": "float", "low": 1.0, "high": 5.0, "step": 1.0},
    "CRISIS_GATE_PROB": {"type": "float", "low": 0.50, "high": 0.90, "step": 0.10},
    "DYNAMIC_RA_CRISIS_COEF": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.5},
    "DYNAMIC_RA_BEAR_COEF": {"type": "float", "low": 0.0, "high": 3.0, "step": 0.5},
    "NORM_VAR_CONSTANT": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.1},
}

# Dynamic Universe Anchor Symbols
FUTURES_ANCHOR_SYMBOLS: list[str] = [
    "BTCUSDT",
    "ETHUSDT",
]

# Tier 2: Institutional Macro Index (Stable Universe for HMM)
FUTURES_MACRO_INDEX_SYMBOLS: list[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "DOTUSDT",
    "BCHUSDT",
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
# FUTURES OPT CONFIGURATION & SEARCH SPACE
# ==============================================================================


def default_ev_hurdle_bps(cfg: dict[str, Any] | None = None) -> float:
    """Return canonical EV hurdle in bps from a config mapping."""
    source = OPT_FUTURES_CONFIG if cfg is None else cfg
    return float(source.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 10.0))


def get_search_space_futures(tf: str, stage: int = 0) -> dict[str, dict[str, Any]]:
    _ = tf
    return build_full_discovery_space_futures()


def build_full_discovery_space_futures() -> dict[str, Any]:
    return dict(ENGINE_PARAM_SPACE_FUTURES)


def get_quarterly_window(reference_date: Any = None, *, tf: str = "4h") -> tuple[str, str, str, str]:
    import datetime

    from dateutil.relativedelta import relativedelta

    if reference_date is None:
        reference_date = datetime.date.today()
    elif isinstance(reference_date, str):
        reference_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d").date()
    current_quarter_start_month: int = ((reference_date.month - 1) // 3) * 3 + 1
    current_quarter_start: datetime.date = datetime.date(reference_date.year, current_quarter_start_month, 1)
    oos_end: datetime.date = current_quarter_start - datetime.timedelta(days=1)
    oos_start: datetime.date = current_quarter_start - relativedelta(months=6)
    is_start: datetime.date = oos_start - relativedelta(months=24)
    from src.domain.futures.optimization.opt_data_utils import resolve_warmup_days_for_tf

    warmup_days = resolve_warmup_days_for_tf(tf)
    fetch_start: datetime.date = is_start - relativedelta(days=warmup_days)
    return (
        fetch_start.strftime("%Y-%m-%d"),
        is_start.strftime("%Y-%m-%d"),
        oos_start.strftime("%Y-%m-%d"),
        oos_end.strftime("%Y-%m-%d"),
    )


# ============================================================
# Layer-specific Optuna parameter spaces (신규 아키텍처용)
# ENGINE_PARAM_SPACE_FUTURES 하위 호환 유지 — 삭제 금지
# ============================================================

# Layer 1 Optuna study: signal quality(IC) 최적화 목적
# Tune: lookback, noise filter threshold, model HP — Sharpe/CAGR 미사용
L1_ALPHA_SPACE: dict[str, dict[str, Any]] = {
    "label_horizon_bars": {"type": "categorical", "choices": (6, 12, 18)},
    "alpha_emit_select_q": {"type": "float", "low": 0.25, "high": 0.50, "step": 0.05},
    "alpha_emit_weight_k": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
    "MIN_SCORE_PERCENTILE": {"type": "float", "low": 0.40, "high": 0.90, "step": 0.10},
}

# L2 allocation search space moved to src.domain.futures.allocation.search_space
# Active versionless constant: L2_SEARCH_SPACE


def resolve_config_by_tf(*, anchor_4h: int, tf: str) -> int:
    """Scale a 4h-anchored bar-count to *tf* via ``scale_bar_count``.

    This is the canonical way to derive a TF-specific value from a 4h reference,
    replacing the incomplete hand-maintained ``_BY_TF`` dict tables (W3/W4).
    The function does **not** require the dict to already contain *tf*.
    """
    from src.domain.futures.strategy.timeframe_contracts import scale_bar_count

    return scale_bar_count(anchor_4h, tf, base_tf="4h")


# ==============================================================================
# 3-WAY LAYERED WINDOW  (post-FTX regime-floor aware)
# ==============================================================================

# post-FTX 정착 + 연 경계: 이전 데이터는 비정상 regime 포함
REGIME_FLOOR: _dt.date = _dt.date(2023, 1, 1)


@_dataclass(frozen=True)
class LayeredWindow:
    """3-way sliding window 결과.

    Attributes:
        fetch_start: fetch 시작일 (warmup buffer 포함).
        l1_start: Layer1 CPCV 시작일 (≥ REGIME_FLOOR).
        l2_start: Layer2 AWF 시작일 (= L1 끝).
        holdout_start: Hold-out 시작일 (= L2 끝).
        holdout_end: Hold-out 끝일.
        regime_floor: 실제 적용된 floor 값 (감사용).
    """

    fetch_start: _dt.date
    l1_start: _dt.date
    l2_start: _dt.date
    holdout_start: _dt.date
    holdout_end: _dt.date
    regime_floor: _dt.date


def get_layered_window(
    reference_date: _dt.date | None = None,
    *,
    l1_months: int = 18,
    l2_months: int = 12,
    holdout_months: int = 6,
    regime_floor: _dt.date = REGIME_FLOOR,
    warmup_days: int | None = None,
    tf: str = "4h",
) -> LayeredWindow:
    """3-way sliding window. warmup_days=None이면 resolve_warmup_days_for_tf(tf)로 계산.
    [ADR_20260706_DATA_WINDOW_FLOOR_CONSISTENCY]
    """
    if reference_date is None:
        reference_date = _dt.date.today()

    if warmup_days is None:
        from src.domain.futures.optimization.opt_data_utils import resolve_warmup_days_for_tf

        warmup_days = resolve_warmup_days_for_tf(tf)

    # holdout_end = 현재 분기 시작 - 1일
    current_q_month: int = ((reference_date.month - 1) // 3) * 3 + 1
    current_q_start: _dt.date = _dt.date(reference_date.year, current_q_month, 1)
    holdout_end: _dt.date = current_q_start - _dt.timedelta(days=1)

    # 역산: holdout → L2 → L1
    holdout_start: _dt.date = holdout_end - _relativedelta(months=holdout_months) + _dt.timedelta(days=1)
    l2_start: _dt.date = holdout_start - _relativedelta(months=l2_months)
    l1_start_raw: _dt.date = l2_start - _relativedelta(months=l1_months)

    # REGIME_FLOOR 클램프
    l1_start: _dt.date = max(l1_start_raw, regime_floor)
    if l1_start > l2_start:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "REGIME_FLOOR clamp exceeded l2_start: l1_start=%s > l2_start=%s. "
            "L1 window is zero-length — check reference_date or regime_floor.",
            l1_start,
            l2_start,
        )

    # fetch warmup buffer
    fetch_start: _dt.date = l1_start - _dt.timedelta(days=warmup_days)

    return LayeredWindow(
        fetch_start=fetch_start,
        l1_start=l1_start,
        l2_start=l2_start,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        regime_floor=regime_floor,
    )


@_dataclass(slots=True, frozen=True)
class ValidationEpisode:
    """[ADR_20260705_L3_ROLLING_HOLDOUT_PANEL]"""

    episode_id: str
    reference_date: _dt.date
    role: Literal["promotion", "stress_only"]
    window: LayeredWindow


def build_validation_episode_panel(
    *,
    promotion_reference_dates: Sequence[str],
    stress_reference_dates: Sequence[str] = (),
    l1_months: int = 9,
    l2_months: int = 6,
    holdout_months: int = 3,
    warmup_days: int | None = None,
    tf: str = "4h",
) -> tuple[ValidationEpisode, ...]:
    episodes: list[ValidationEpisode] = []

    for d_str in promotion_reference_dates:
        ref = _dt.date.fromisoformat(d_str)
        window = get_layered_window(
            reference_date=ref,
            l1_months=l1_months,
            l2_months=l2_months,
            holdout_months=holdout_months,
            regime_floor=REGIME_FLOOR,
            warmup_days=warmup_days,
            tf=tf,
        )
        episodes.append(
            ValidationEpisode(
                episode_id=f"promotion_{d_str}",
                reference_date=ref,
                role="promotion",
                window=window,
            )
        )

    for d_str in stress_reference_dates:
        ref = _dt.date.fromisoformat(d_str)
        window = get_layered_window(
            reference_date=ref,
            l1_months=l1_months,
            l2_months=l2_months,
            holdout_months=holdout_months,
            regime_floor=_dt.date.min,
            warmup_days=warmup_days,
            tf=tf,
        )
        episodes.append(
            ValidationEpisode(
                episode_id=f"stress_only_{d_str}",
                reference_date=ref,
                role="stress_only",
                window=window,
            )
        )

    return tuple(episodes)
