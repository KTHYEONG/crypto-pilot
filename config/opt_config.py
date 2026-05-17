from __future__ import annotations

from typing import Any

# ==============================================================================
# OPTIMIZATION FUTURES SEARCH SPACE & CONFIGURATION
# ==============================================================================

OPT_FUTURES_CONFIG: dict[str, Any] = {
    # 3-phase coordinate ascent total ≈ 260 (Bonferroni / reporting baseline).
    "total_trials": 1500,
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
    # Risk & Portfolio (Phase D)
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
    # Thematic breadth: LambdaRank buckets (Trend / Vol+MR / Interaction) × slots each. Total = 3 × value.
    # Raising adds interaction-style capacity without injecting systemic HMM into Trend slots (helps long IC vs AWF pos_frac).
    "FUTURES_ML_ALPHA_SLOTS_PER_THEME": 8,
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
    # HMM stable regime (fixed hyperparameters; not in Optuna search space)
    # v10.0.0: 4 internal states (BULL_TREND, BEAR_TREND, CHOP_HIGH, CHOP_LOW).
    "FUTURES_HMM_K_STATES": 4,
    # HMM backend selector:
    # - "jax_gaussian" (default): existing JAX Gaussian backend
    # - "student_t": Student-t backend
    # - "skewed_t": Skewed-t backend with Parallel Associative Scan
    "FUTURES_HMM_BACKEND": "skewed_t",
    # Stability guard: JAX HMM backend intermittently segfaulted (exit 139) on long-trial runs.
    # v10.0.0: RESTORED for multivariate EM.
    "FUTURES_HMM_JAX_BACKEND_ENABLED": True,
    # Deployment floor: ≥0.45 (align HMM systemic prior with Phase D Kelly band).
    "FUTURES_HMM_KELLY_SHRINKAGE": 0.45,
    # Immovable at 0.66: ANY reduction (even 0.62) collapses IS AWF because IS period has
    # only 2.9% CRISIS — borderline bars (0.62-0.66) are profitable IS trades.
    # OOS CRISIS mismatch (2.9%→15.9%) handled by auxiliary volatility gate (not threshold).
    "FUTURES_HMM_CRISIS_THRESHOLD": 0.66,
    # CRISIS regime: hard zero-leverage kill-switch during crisis (0.0 = no position).
    # OOS data shows CRISIS PF=0.34 — any nonzero position destroys OOS compounding.
    "FUTURES_HMM_CRISIS_FLAT_LEV": 0.0,
    # Step6: split kill-switch policy. If new columns are absent, code falls back to
    # legacy hmm_prob_crisis behavior automatically.
    "FUTURES_HMM_SPLIT_KILLSWITCH_ENABLED": True,
    # pre_crisis: soft leverage damp (gross/leverage reduction only).
    "FUTURES_HMM_PRE_CRISIS_DAMP_THRESHOLD": 0.55,
    "FUTURES_HMM_PRE_CRISIS_DAMP_MIN_MULT": 0.50,
    # realized_crisis: hard flat (or configured flat leverage).
    "FUTURES_HMM_REALIZED_CRISIS_FLAT_THRESHOLD": 0.66,
    # tail risk overlay: high risk bars -> strong leverage cut (or optional hard flat).
    "FUTURES_HMM_TAIL_RISK_HIGH_THRESHOLD": 0.75,
    "FUTURES_HMM_TAIL_RISK_HIGH_LEV_MULT": 0.25,
    "FUTURES_HMM_TAIL_RISK_FORCE_FLAT": False,
    # Threshold calibration mode for Step6 split kill-switch.
    # fixed: use static thresholds below (backward compatible default)
    # is_quantile: use quantiles estimated on leading IS slice
    # rolling_quantile: use per-bar rolling quantile thresholds
    "FUTURES_HMM_THRESHOLD_MODE": "is_quantile",
    "FUTURES_HMM_THRESHOLD_IS_FRAC": 0.70,
    "FUTURES_HMM_THRESHOLD_ROLLING_WINDOW": 336,
    "FUTURES_HMM_THRESHOLD_ROLLING_MIN_PERIODS": 96,
    "FUTURES_HMM_PRE_CRISIS_Q": 0.75,
    "FUTURES_HMM_REALIZED_CRISIS_Q": 0.90,
    "FUTURES_HMM_TAIL_RISK_Q": 0.85,
    # S2: Auxiliary volatility-based CRISIS gate.
    # DISABLED: vol gate fires during profitable high-vol IS periods (CRISIS G=+0.193% in IS),
    # collapsing IS CAGR from ~30% to 2.6%. IS-OOS mismatch requires a smarter approach
    # (e.g., cross-sectional synchronized signal, not per-symbol rolling vol).
    "FUTURES_VOL_CRISIS_GATE_ENABLED": False,
    "FUTURES_VOL_CRISIS_WINDOW": 20,
    "FUTURES_VOL_CRISIS_MULT": 3.0,
    # Asymmetric Friction Reduction (Phase 6): stronger sticky penalty for training stability.
    # Raised with smoother posteriors (SPAN 8): compensates wigglier regimes vs span≈12 legacy.
    "FUTURES_HMM_STICKY_PENALTY_WEIGHT": 1100.0,
    # Min-duration (bars @ base TF) per state: [BULL_CALM, BULL_VOL_UP, BEAR, CHOP, CRISIS]
    # P0: was [1000,500,...] (~6w calm lock); shortened so regime errors recover in ~1w.
    # Note: sticky labels applied on 5-state output (RECOVERY merged into BULL_VOL_UP).
    "FUTURES_HMM_OUTPUT_STICKY_MIN_DURATION": [48, 32, 28, 16, 20],
    # Slightly stronger sticky transitions → stabler systemic HMM under leg refit (Path C).
    "FUTURES_HMM_TRANSITION_PRIOR_ALPHA": 0.50,
    # True per-bar TVTP controls (f2/vol_z driven).
    "FUTURES_HMM_TVTP_ENABLED": True,
    "FUTURES_HMM_TVTP_VOL_CENTER": 0.0,
    "FUTURES_HMM_TVTP_VOL_SCALE": 1.0,
    # diag_slope < 0: high vol lowers self-transition stickiness, increasing regime mobility.
    "FUTURES_HMM_TVTP_DIAG_SLOPE": -0.12,
    "FUTURES_HMM_TVTP_DIAG_BIAS": 0.0,
    "FUTURES_HMM_TVTP_DIAG_CLIP": 0.22,
    # Sticky prior multiplier = clip(1 + slope * avg_vol, min_mult, max_mult).
    "FUTURES_HMM_TVTP_STICKY_PRIOR_VOL_SLOPE": -0.30,
    "FUTURES_HMM_TVTP_STICKY_PRIOR_MIN_MULT": 1.01,
    "FUTURES_HMM_TVTP_STICKY_PRIOR_MAX_MULT": 1.35,
    # HMM Posterior Smoothing (EMA, DEMA, TEMA, HMA, KAMA, ALMA, JMA)
    "FUTURES_HMM_SMOOTHING_METHOD": "EMA",
    # Posterior smoothing: span=3 harmed HMM stability; 6–9 + higher STICKY_PENALTY (1100).
    "FUTURES_HMM_SMOOTHING_SPAN": 12,
    # Optional asymmetric EMA for crisis posterior (faster attack, slower decay).
    "FUTURES_HMM_CRISIS_ATTACK_SPAN": 2,
    "FUTURES_HMM_CRISIS_DECAY_SPAN": 9,
    # Step5: Tail-event supervised overlay (1~8 bar forward tail risk).
    "FUTURES_HMM_TAIL_OVERLAY_ENABLED": True,
    "FUTURES_HMM_TAIL_OVERLAY_HORIZON": 8,
    "FUTURES_HMM_TAIL_OVERLAY_LABEL_Q": 0.10,
    "FUTURES_HMM_TAIL_OVERLAY_MIN_TRAIN": 240,
    "FUTURES_HMM_TAIL_OVERLAY_MIN_POS": 20,
    "FUTURES_HMM_TAIL_OVERLAY_USE_ISOTONIC": True,
    "FUTURES_HMM_TAIL_OVERLAY_LR_C": 0.8,
    "FUTURES_HMM_SUP_HORIZONS": [4, 8, 16],
    "FUTURES_HMM_SUP_LABEL_Q10": 0.10,
    "FUTURES_HMM_SUP_LABEL_Q05": 0.05,
    "FUTURES_HMM_SUP_LABEL_Q03": 0.03,
    "FUTURES_HMM_SUP_MIN_POS": 12,
    "FUTURES_HMM_SUP_RANK_BLEND_W": 0.30,
    "FUTURES_HMM_SUP_RANK_BLEND_POW": 1.20,
    # Step2: hazard boost rebalance (posterior is untouched; boost hazard/policy path only).
    "FUTURES_HMM_STEP2_SUPERVISED_TOP_Q": 0.85,
    "FUTURES_HMM_STEP2_PRE_HAZARD_BOOST": 0.30,
    "FUTURES_HMM_STEP2_REALIZED_HAZARD_BOOST": 0.45,
    "FUTURES_HMM_STEP2_TAIL8_HAZARD_BOOST": 0.35,
    "FUTURES_HMM_STEP2_TAIL8_SUP_RANK_POW": 1.15,
    "FUTURES_HMM_STEP2_TAIL8_RANK_BLEND_W": 0.55,
    "FUTURES_HMM_STEP2_TAIL8_SUP_BLEND_W": 0.08,
    "FUTURES_HMM_STEP2_TAIL8_REALIZED_BLEND_W": 0.22,
    "FUTURES_HMM_STEP2_TAIL8_PRE_BLEND_W": 0.15,
    "FUTURES_HMM_STEP2_TAIL8_STRUCT_BLEND_W": 0.15,
    "FUTURES_HMM_STEP2_SUP_CRASH_Q05": 0.95,
    "FUTURES_HMM_STEP2_SUP_CRASH_Q03": 0.97,
    "FUTURES_HMM_STEP2_TAIL8_CRASH05_BLEND_W": 0.10,
    "FUTURES_HMM_STEP2_TAIL8_CRASH03_BLEND_W": 0.10,
    "FUTURES_HMM_STEP2_TAIL8_SOFT_BLEND_W": 0.10,
    "FUTURES_HMM_STEP2_TAIL8_HARD_BLEND_W": 0.10,
    "FUTURES_HMM_STEP2_TAIL8_NEAR_FLAT_BLEND_W": 0.10,
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
    # P2: asymmetric modulator ceiling (no bull amplification in OOS bear regime)
    "FUTURES_HMM_MOD_LONG_CEIL": 1.0,  # was 2.0
    # P2: backward-looking 168h BTC return suppression (no look-ahead bias)
    "FUTURES_HMM_BEAR_TREND_SUPPRESS_THR": -0.08,  # P2.1 revert: -0.05 over-suppressed
    "FUTURES_HMM_BEAR_TREND_DAMP": 0.40,           # P2.1 revert: 0.25 over-suppressed
    # Session 39: recovery override abandoned (P10 -0.027, DSR collapse). Keep disabled.
    "CRISIS_RECOVERY_TREND_THR": 1e9,
    "CRISIS_RECOVERY_FLOOR": 0.30,
    # CAWF-R: K anchored legs in the OOS-pool slice (train [0, anchor_i), embargo, test window).
    "FUTURES_AWF_K_LEGS": 5,
    # IS-pool fraction (leading bars); trailing (1 − frac) builds tiled OOS test legs only.
    "FUTURES_AWF_IS_POOL_FRAC": 0.65,
    # Platt: prefer OOS-only windows in optimizer ``_fit_oos_platt_calibrators_from_maps``;
    # legacy tail fraction kept for deprecated tail-window path only.
    "FUTURES_CALIB_PLATT_TAIL_FRAC": 0.30,
    "FUTURES_CALIB_PLATT_MIN_OOS_BARS": 80,
    # If True and the primary OOS window has too few samples, widen to AWF OOS-pool start (still no IS-pool).
    "FUTURES_CALIB_PLATT_OOS_WIDEN_TO_POOL": True,
    # Covariance lookback for portfolio_constructor (~30 calendar days in bars per TF).
    "FUTURES_PORTFOLIO_COV_LOOKBACK": 180,
    "FUTURES_PORTFOLIO_COV_LOOKBACK_BY_TF": {"1h": 720, "4h": 180, "1d": 30},
    "FUTURES_PORTFOLIO_KAPPA": 0.35,
    "FUTURES_PORTFOLIO_F_KELLY_MAX": 2.0,
    # ATR stop multiplier (ATR_PERIOD lives in engine / Optuna params).
    "FUTURES_ATR_STOP_MULT": 2.5,
    "FUTURES_SIMPLE_ATR_STOP": True,
    # ATR period fixed (not an Optuna dimension).
    "FUTURES_ATR_PERIOD_FIXED": 30,
    # Rolling σ lookback ≈ 8 calendar days (bars per TF optional override).
    "FUTURES_COMPOSER_SIGMA_CALENDAR_DAYS": 8.0,
    "FUTURES_COMPOSER_SIGMA_LOOKBACK_BY_TF": {"1h": 192, "4h": 48, "1d": 8},
    # 1.0 = disabled (no crisis long-only magnitude scaling in execution stack).
    "FUTURES_CRISIS_LONG_MAG_SUPPRESS": 1.0,
    "FUTURES_EVENT_TURNOVER_REBALANCE": False,
    # Short borrow ~0.06%/day (fraction of notional per day).
    "FUTURES_SHORT_BORROW_DAILY": 0.0006,
    "SLIPPAGE_BPS_BUFFER_MULT": 1.0,
    "FUTURES_DEFAULT_BETA_ALPHA": 1.0,
    "FUTURES_DEFAULT_BETA_REGIME_BULL": 1.0,
    "FUTURES_DEFAULT_BETA_REGIME_BEAR": 0.15,
    # Stability fail → try runner-up phase-C trial once.
    "FUTURES_STABILITY_RUNNER_UP_RETRY": True,
    "FUTURES_DEFAULT_BETA_REGIME_CRISIS": 0.3,
    "FUTURES_DEFAULT_BETA_REGIME_RECOVERY": 0.0,
    "FUTURES_DEFAULT_BETA_REGIME_CHOP": 0.25,
    "FUTURES_DEFAULT_EV_HURDLE_BPS": 40.0,
    # Step1: posterior-aware regime policy (disabled by default for backward compatibility).
    "FUTURES_REGIME_POLICY_ENABLED": False,
    "FUTURES_REGIME_CONFIDENCE_ENTROPY_MULT": 0.50,
    "FUTURES_REGIME_MULT_MIN": 0.10,
    "FUTURES_REGIME_MULT_MAX": 1.50,
    "FUTURES_REGIME_LONG_BULL_W": 0.35,
    "FUTURES_REGIME_LONG_BEAR_PENALTY": 0.35,
    "FUTURES_REGIME_LONG_CHOP_PENALTY": 0.55,
    "FUTURES_REGIME_LONG_CRISIS_PENALTY": 0.90,
    "FUTURES_REGIME_SHORT_BEAR_W": 0.45,
    "FUTURES_REGIME_SHORT_BULL_PENALTY": 0.25,
    "FUTURES_REGIME_SHORT_CHOP_PENALTY": 0.45,
    "FUTURES_REGIME_SHORT_CRISIS_W": 0.15,
    "FUTURES_REGIME_EV_CHOP_ADD_BPS": 8.0,
    "FUTURES_REGIME_EV_CRISIS_ADD_BPS": 12.0,
    "FUTURES_REGIME_EV_ENTROPY_ADD_BPS": 6.0,
    "FUTURES_PORTFOLIO_REGIME_DAMP_ENABLED": False,
    "FUTURES_PORTFOLIO_CHOP_GROSS_DAMP": 0.50,
    "FUTURES_PORTFOLIO_CRISIS_GROSS_DAMP": 0.80,
    "FUTURES_PORTFOLIO_ENTROPY_GROSS_DAMP": 0.35,
    "FUTURES_PORTFOLIO_BEAR_GROSS_DAMP": 0.10,
    "FUTURES_PORTFOLIO_GROSS_FLOOR_MULT": 0.15,
    "FUTURES_PORTFOLIO_CRISIS_LONG_SUPPRESS_THR": 0.60,
    "FUTURES_PORTFOLIO_CRISIS_LONG_SUPPRESS_MULT": 0.10,
    "FUTURES_DEFAULT_TIME_BARRIER_H": 24.0,
    "FUTURES_TMP_MD_CHAMPION_GATES_ENABLED": True,
    "FUTURES_TMP_LAYER1_MEDIAN_LOG_TW_MIN": 0.0,
    "FUTURES_TMP_LAYER1_POS_LEG_RATIO_MIN": 4.0 / 6.0,
    "FUTURES_TMP_LAYER1_MAX_DD_PCT": 12.0,
    "FUTURES_TMP_LAYER2_PSR_MIN": 0.95,
    "FUTURES_TMP_LAYER2_MIN_TRADES_PER_LEG": 20,
    # Layer 2 optional: median(leg_log_tw − stress·round_trip) > 0 (see tmp_md_champion).
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
    "FUTURES_AWF_NET_EDGE_MIN": 1.5,   # min EV/cost ratio (avg PnL / round-trip cost)
    # SPA bootstrap for post-run diagnostics (not used per-trial).
    "FUTURES_SPA_N_BOOTSTRAP": 2000,
    "FUTURES_SPA_P_VALUE_MAX": 0.10,   # SPA p-value ≤ 0.10 → reject H0: zero alpha
    # Worst AWF leg log-TW floor (10th percentile / min-leg semantics in objective).
    "FUTURES_AWF_P10_LOG_TW_MIN": -0.10,
    # Increased from 3: 5 WF OOS legs provide regime-diverse robustness verification.
    "FUTURES_WF_OOS_LEGS": 5,
    "FUTURES_ENSEMBLE_MAX_SIZE": 5,
    "FUTURES_META_ALLOC_WINDOW": 12,
    "FUTURES_META_ALLOC_ETA": 0.15,
    # R-6: per WF OOS leg, retrain systemic HMM on data strictly before leg start (GP frozen).
    # Per AWF leg, rerun full universe ML (alpha + systemic HMM + fusion) at the leg train
    # cutoff before slicing that leg's test window. Disable for faster iteration (single merge).
    "FUTURES_WF_HMM_LEG_REFIT": True,
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
    "FUTURES_AWF_POS_FRAC_MIN": 0.60,   # AWF 게이트: 최소 60% leg 수익 (≥3/5)
    "FUTURES_AWF_MU_LOG_MIN": 0.0,       # AWF 게이트: 평균 leg log-TW > 0
    # Step2: regime-aware deployability pressure.
    # Note: CHOP trade share is structurally ~0.68; threshold must exceed this to avoid
    # penalizing all trials. Set 0.70 (tuned_v1 best-validated setting).
    "FUTURES_STEP2_REGIME_DEPLOY_ENABLED": True,
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
    # Improvement 2: Regime-Aware Dynamic Kelly Scaling
    "FUTURES_HMM_DYNAMIC_KELLY_ENABLED": True,
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
    "BTC/USDT",
    "ETH/USDT",
]

# Tier 2: Institutional Macro Index (Stable Universe for HMM)
FUTURES_MACRO_INDEX_SYMBOLS: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "DOT/USDT",
    "BCH/USDT",
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
# SPOT CONFIGURATION (Standardized)
# ==============================================================================
SPOT_ANCHOR_SYMBOLS: list[str] = ["KRW-ETH", "KRW-SOL", "KRW-XRP"]
SPOT_SYMBOLS: list[str] = ["KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-HBAR"]

SPOT_SHARED_PARAM_SPACE: dict[str, dict[str, Any]] = {
    "TIMEFRAME": {"type": "categorical", "choices": ["4h"]},
    "RISK_PER_TRADE": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.01},
}

SPOT_EXCLUDED_SIZING_METHODS: list[str] = ["inv_vol_parity", "liquidity_adjusted"]

SLIPPAGE_GAMMA_BASE: float = 0.6
SLIPPAGE_REFERENCE_ADV_KRW: float = 1_000_000_000.0

ENGINE_PARAM_SPACE: dict[str, dict[str, Any]] = {
    "MAX_CONCURRENT_POSITIONS": {"type": "int", "low": 3, "high": 10},
    "MIN_PF": {"type": "float", "low": 1.2, "high": 2.0},
    "MAX_MDD": {"type": "float", "low": 10.0, "high": 25.0},
}

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
    return build_full_discovery_space_futures()

def build_full_discovery_space_futures() -> dict[str, Any]:
    return dict(ENGINE_PARAM_SPACE_FUTURES)

def get_spot_effective_independent_trials(n_done: int, n_startup: int) -> int:
    """Heuristic for independent trials (deflating multiple testing bias)."""
    return max(1, n_done - n_startup)

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
