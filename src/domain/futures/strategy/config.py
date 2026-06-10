from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_DEFAULT_COST_MODEL = ExecutionCostModel()
_DEFAULT_RT_BPS: float = _DEFAULT_COST_MODEL.round_trip_bps()  # ≈ 7.5
_DEFAULT_MAX_EXPECTED_HOLDING_BARS = 36


@dataclass(slots=True, frozen=True)
class BlendConfig:
    """Enhanced strategy blending parameters."""

    clip_z: float = 3.0
    min_symbols: int = 5
    ic_window_bars: int = 180
    ic_shrinkage: float = 0.5
    min_mean_ic: float = 0.02
    min_t_stat: float = 2.0
    min_hit_ratio: float = 0.45
    sigma_lookback: int = 30

    def __post_init__(self) -> None:
        """Validate blending parameters."""
        if self.clip_z <= 0.0:
            raise ValueError("clip_z must be positive")
        if self.min_symbols < 1:
            raise ValueError("min_symbols must be >= 1")
        if self.ic_window_bars < self.sigma_lookback:
            raise ValueError("ic_window_bars must satisfy >= sigma_lookback")
        if not (0.0 < self.ic_shrinkage <= 1.0):
            raise ValueError("ic_shrinkage must satisfy 0 < ic_shrinkage <= 1")
        if not (0.0 <= self.min_hit_ratio <= 1.0):
            raise ValueError("min_hit_ratio must satisfy 0 <= min_hit_ratio <= 1")
        if self.sigma_lookback < 1:
            raise ValueError("sigma_lookback must be >= 1")


@dataclass(slots=True, frozen=True)
class RegimeConfig:
    """Continuous risk overlay and regime quality settings."""

    overlay_target_vol_ann: float = 0.40
    overlay_vol_ewma_span: int = 30
    overlay_vol_scale_clip: tuple[float, float] = (0.25, 1.5)
    overlay_trend_snr_span: int = 60
    crisis_target_arl_bars: int = 500
    crisis_gross_floor: float = 0.15
    regime_overlay_min_lift_tstat: float = 1.0
    regime_min_n_eff: int = 60
    regime_quality_gate_enabled: bool = True
    regime_transition_occupancy: float = 0.10

    # P0 — Regime Lift Proof Gate
    regime_lift_proof_enabled: bool = True
    regime_lift_nw_tstat_threshold: float = 1.5
    regime_lift_fold_pass_ratio: float = 0.60
    regime_lift_max_holding_bars: int = 6

    # P1 — Hysteresis + persistence-targeted band
    trend_hysteresis_enter: float = 0.35
    trend_hysteresis_exit: float = 0.15
    persistence_target_dwell: float = 6.0

    def __post_init__(self) -> None:
        """Validate regime parameters."""
        if self.overlay_target_vol_ann <= 0.0:
            raise ValueError("overlay_target_vol_ann must be positive")
        if self.overlay_vol_ewma_span < 1:
            raise ValueError("overlay_vol_ewma_span must be >= 1")
        if self.overlay_trend_snr_span < 2:
            raise ValueError("overlay_trend_snr_span must be >= 2")
        lo, hi = self.overlay_vol_scale_clip
        if not (0.0 < lo <= hi):
            raise ValueError("overlay_vol_scale_clip must satisfy 0 < lo <= hi")
        if self.crisis_target_arl_bars < 2:
            raise ValueError("crisis_target_arl_bars must be >= 2")
        if not (0.0 <= self.crisis_gross_floor <= 1.0):
            raise ValueError("crisis_gross_floor must satisfy 0 <= gross_floor <= 1.0")
        if self.regime_min_n_eff < 2:
            raise ValueError("regime_min_n_eff must be >= 2")
        if not (0.0 < self.regime_transition_occupancy < 0.5):
            raise ValueError("regime_transition_occupancy must satisfy 0 < value < 0.5")
        if self.regime_lift_nw_tstat_threshold < 0.0:
            raise ValueError("regime_lift_nw_tstat_threshold must be >= 0")
        if not (0.0 < self.regime_lift_fold_pass_ratio <= 1.0):
            raise ValueError("regime_lift_fold_pass_ratio must be in (0, 1]")
        if self.regime_lift_max_holding_bars < 1:
            raise ValueError("regime_lift_max_holding_bars must be >= 1")
        if self.trend_hysteresis_exit >= self.trend_hysteresis_enter:
            raise ValueError("trend_hysteresis_exit must be < trend_hysteresis_enter")
        if self.trend_hysteresis_enter <= 0.0:
            raise ValueError("trend_hysteresis_enter must be > 0")
        if self.persistence_target_dwell < 2.0:
            raise ValueError("persistence_target_dwell must be >= 2")


@dataclass(slots=True, frozen=True)
class StrategyConfig:
    """Top-level strategy switch."""

    name: str = "candidate_ml"
    blend: BlendConfig = field(default_factory=BlendConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    candidate: CandidateStrategyConfig = field(default_factory=lambda: CandidateStrategyConfig())

    def __post_init__(self) -> None:
        """Validate top-level strategy name."""
        if self.name not in {
            "candidate_ml",
            "rule_baseline",
        }:
            raise ValueError(f"unsupported strategy name: {self.name}")


@dataclass(slots=True, frozen=True)
class CandidateStrategyConfig:
    """Candidate strategy routing config."""

    name: Literal["candidate_ml", "rule_baseline"] = "candidate_ml"
    timeframe: str = "4h"
    seed: int = 42
    train_months: int = 24
    valid_months: int = 3
    test_months: int = 6
    max_holding_bars: int = _DEFAULT_MAX_EXPECTED_HOLDING_BARS
    purge_bars: int | None = None
    embargo_bars: int | None = None
    purge_safety_mult: float = 1.2
    _purge_bars_input: int | None = field(init=False, repr=False, compare=False)
    _embargo_bars_input: int | None = field(init=False, repr=False, compare=False)
    # Deprecated: use ExecutionCostModel fields instead; kept for explicit override only
    cost_floor_bps: float = _DEFAULT_RT_BPS
    gate_label_column: Literal[
        "profitable_after_hurdle_label",
        "barrier_first_label",
        "gross_direction_label",
    ] = "profitable_after_hurdle_label"
    gate_calibration_method: Literal["sigmoid", "isotonic", "none"] = "isotonic"
    gate_calibration_fallback_raw: bool = True
    min_gate_calibration_obs: int = 100
    min_gate_calibration_pos: int = 10
    min_gate_probability_std: float = 0.03
    percentile_gate_enabled: bool = False
    percentile_gate_threshold: float = 0.70  # top 30% signals pass
    gate_lgbm_max_depth: int = 3
    gate_lgbm_reg_lambda: float = 30.0
    edge_lgbm_max_depth: int = 3
    edge_lgbm_reg_lambda: float = 30.0
    ml_fit_fraction: float = 0.60
    ml_calibration_fraction: float = 0.20
    model_early_stop_fraction: float = 0.15
    calibration_fit_fraction: float = 0.50
    promotion_decision_split: Literal["fit", "calibration", "fit_calibration"] = "fit_calibration"
    min_promotion_calibration_edge_bps: float = 1.0
    min_promotion_calibration_obs: int = 100
    min_listing_age_days: int = 180
    min_candidate_obs: int = 200
    min_symbol_oos_blocks: int = 3
    min_rule_net_bps: float = 0.0
    min_rule_ir_t: float = 1.0
    min_rule_hit_rate: float = 0.50
    max_rule_turnover_per_bar: float = 0.50
    max_symbol_weight: float = 0.10
    gross_cap: float = 1.20
    net_cap: float = 0.30
    beta_cap: float = 0.50
    target_ann_vol: float = 0.35
    kelly_fraction: float = 0.25
    sizing_mode: Literal["stop_risk", "calibrated_event_kelly"] = "stop_risk"
    event_risk_budget: float = 0.0025
    kelly_min_bin_ess: int = 100
    min_expected_net_bps: float = 1.0
    max_expected_shortfall_bps: float = 300.0
    shortfall_threshold_basis: Literal["absolute_bps", "stop_relative"] = "absolute_bps"
    max_expected_shortfall_stop_mult: float = 1.25
    selection_use_expected_utility: bool = True
    selection_min_expected_utility_bps: float = 0.0
    selection_shortfall_mode: Literal["hard", "penalty_only", "catastrophic"] = "penalty_only"
    catastrophic_shortfall_bps: float = 300.0
    catastrophic_shortfall_stop_mult: float = 1.50
    selection_sensitivity_enabled: bool = True
    selection_gate_grid: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55)
    selection_edge_grid_bps: tuple[float, ...] = (0.0, 1.0, 5.0)
    selection_q10_grid_bps: tuple[float, ...] = (80.0, 150.0, 250.0, 400.0)
    selection_waterfall_diagnostics_enabled: bool = True
    selection_shadow_profiles_enabled: bool = True
    selection_shadow_utility_floors_bps: tuple[float, ...] = (-50.0, -25.0, 0.0)
    selection_shadow_breakeven_floor_fractions: tuple[float, ...] = (0.0, 0.25, 0.50)
    selection_shadow_top_quantile: float = 0.10
    selection_shadow_max_profiles: int = 20
    selection_utility_mode: Literal["additive_drag", "expected_edge_direct"] = "expected_edge_direct"
    selection_shadow_utility_modes: tuple[str, ...] = ("additive_drag", "expected_edge_direct")
    breakeven_floor_mode: Literal["static", "fold_adaptive"] = "static"
    breakeven_floor_cost_quantile: float = 0.50
    enabled_candidate_variants: tuple[str, ...] = ()
    disabled_candidate_variants: tuple[str, ...] = ()
    side_flip_candidate_variants: tuple[str, ...] = ()
    diagnostic_top_k: int = 10
    min_variant_oos_obs: int = 100
    min_variant_oos_edge_bps: float = 1.0
    # Phase 2 signal pruning: additional profit floor (cost-based minimum).
    # Effective floor = max(min_variant_oos_edge_bps, min_variant_oos_profit_bps).
    # Set > 0 to prune high-frequency noise signals near breakeven.
    # Example: 15.0 eliminates fzs/rsi/bollinger/vrr/rr (6~12bps OOS profit).
    min_variant_oos_profit_bps: float = 0.0
    # Any ATR-stop strategy has median<0 + mean>0 as a structural property.
    # Use mean_edge + hit_or_payoff as economic gates; median is a soft diagnostic.
    min_variant_oos_median_edge_bps: float = -100.0
    median_gate_skew_exempt_archetypes: tuple[str, ...] = (
        "trend_continuation",
        "time_series_momentum",
    )
    # p10 for crypto futures with 1.5-2.5x ATR stops is structurally -300~-500bps.
    # Primary tail guard is q10_fail_rate; p10 is a hard outlier filter only.
    min_variant_oos_p10_edge_bps: float = -600.0
    p10_edge_relative_to_stop: bool = False
    p10_min_fraction_of_stop: float = 1.5
    min_variant_oos_hit_rate: float = 0.50
    min_variant_oos_payoff_ratio: float = 1.20
    max_variant_oos_q10_fail_rate: float = 0.65
    max_variant_event_fraction_per_bar: float = 0.25
    regime_diagnostic_enabled: bool = True
    min_regime_variant_oos_obs: int = 40
    min_regime_variant_oos_edge_bps: float = 2.0
    # Regime-cell conditional admission: promotes a variant if it passes the
    # Bayesian posterior probability gate (P(μ > δ | data) ≥ min_admission_posterior_prob)
    # even when the global-pooled gates fail.
    # Targets carry/reversion specialists diluted by out-of-regime OOS samples.
    regime_cell_admission_enabled: bool = True
    min_regime_cell_oos_obs: int = 10  # NW variance stability floor only; not a domain gate
    min_regime_cell_edge_bps: float = 8.0  # δ: minimum profitable edge (breakeven proxy)
    max_admitted_cells_per_variant: int = 2
    # Bayesian posterior probability admission (replaces flat obs/tstat thresholds)
    min_admission_posterior_prob: float = 0.70   # P(μ > δ | data) gate; Bounds: [0.5, 1.0)
    admission_use_newey_west: bool = True         # True=NW autocorr-corrected; False=IID legacy
    admission_tau_prior_bps: float = 15.0        # fallback cross-cell std when < 2 cells; Bounds: (0, ∞)
    allocation_backend: Literal["ensemble_b0", "ml_edge"] = "ensemble_b0"
    ensemble_shrinkage_k: float = 50.0
    # EB adaptive shrinkage: k_eff = within_var / between_var (James-Stein principle).
    # When True, replaces fixed k=50 with data-derived k_eff capped at k_max.
    # This preserves high-edge rare signals from being swamped by high-frequency noise.
    ensemble_adaptive_shrinkage: bool = True
    ensemble_shrinkage_k_max: float = 50.0  # upper bound for k_eff; Bounds: (0, ∞)
    # Frequency cap: clip n to n_cap before computing w=n/(n+k) to prevent
    # high-frequency noise cells from dominating the global pull.
    ensemble_freq_n_cap: int = 200  # max effective n per cell; 0=disabled
    # Floor: cell mu_net < this → contribution zeroed (not-predicted rather than negative).
    ensemble_min_cell_edge_floor_bps: float = 0.0
    # Variant-edge hierarchical prior (3-level James-Stein extension).
    # Restores within-cell discrimination by blending variant fit-window mean
    # toward its archetype-regime cell anchor (mode cell).
    # vmean_v = w_v*raw_v + (1-w_v)*anchor; w_v = n_eff/(n_eff+k_v)
    ensemble_variant_prior_enabled: bool = True
    ensemble_variant_shrinkage_k: float = 30.0   # anchor pull for variant James-Stein; Bounds: (0, ∞)
    ensemble_variant_min_obs: int = 40            # below → w_v≈0, fallback to cell anchor
    # Conditioning axis: "auto" (default) picks archetype_regime vs archetype_only via
    # in-fold purged validation Rank IC gain — data-driven, fold-adaptive.
    # "archetype_regime" forces regime conditioning regardless of evidence.
    # "archetype_only" strips regime_code from alpha (most conservative).
    ensemble_conditioning: Literal["archetype_regime", "archetype_only", "auto"] = "auto"
    ensemble_internal_val_fraction: float = 0.25
    ensemble_min_conditioning_ic_gain: float = 0.01
    # mu-quality shrinkage: lam = clip(val_rank_ic / mu_quality_ic_full_scale, 0, 1)
    # final_mu = lam * mu_pred + (1-lam) * cross_sectional_mean(mu_pred)
    # Disabled by default: shrinkage can collapse predictions and worsen OOS rank IC.
    mu_quality_shrinkage_enabled: bool = False
    mu_quality_ic_full_scale: float = 0.05
    # Set to False to remove hard regime-based signal masking; regime moves to
    # sizing multiplier layer (see regime_as_size_multiplier).
    regime_signal_gating_enabled: bool = False
    mean_reversion_regime_entry_gating_enabled: bool = True
    standalone_breakeven_hard_gate_enabled: bool = True
    # When True, apply per-regime continuous weight multipliers at sizing stage.
    regime_as_size_multiplier: bool = False
    regime_size_multipliers: tuple[tuple[str, float], ...] = (
        ("crash", 0.3),
        ("bear_volatile", 0.5),
        ("transition", 0.7),
        ("bull_volatile", 0.85),
        ("bull_quiet", 1.0),
        ("bear_quiet", 0.8),
    )
    max_exit_policy_variants_per_signal: int = 2
    diagnose_signal_cells: bool = True
    # "variant": promote at family:variant granularity (regime pooled).
    # "signal_cell": legacy per-cell AND-conjunction (fail-closed).
    promotion_level: Literal["signal_cell", "variant"] = "variant"
    fdr_alpha: float = 0.10
    fdr_gate_enabled: bool = False
    spa_gate_enabled: bool = False
    spa_p_value_max: float = 0.10
    spa_n_bootstrap: int = 2000
    # Vol-normalised percentile edge (atr_bps column required; skeleton if absent)
    edge_percentile_vol_normalized: bool = False
    min_signal_cell_oos_obs: int = 80
    max_signal_cell_event_fraction_per_bar: float = 0.12
    candidate_identity_features_enabled: bool = True
    market_state_features_enabled: bool = True
    static_universe_features_enabled: bool = False
    signal_context_features_enabled: bool = True
    score_pct_variant_hist_window_bars: int = 2160
    exclude_immediate_return_features: bool = True
    promotion_filter_enabled: bool = True
    selection_policy: Literal["hard", "validation_quantile", "utility_topk"] = "utility_topk"
    selection_scope: Literal["per_timestamp"] = "per_timestamp"
    selection_max_events_per_bar: int | None = None
    selection_top_quantile: float = 0.10
    min_net_floor_cost_fraction: float = 0.50
    min_oos_rank_ic: float = 0.01
    min_ic_tstat: float = 0.8
    min_oos_log_growth_uplift: float = 0.0
    max_oos_edge_decay_bps: float = 50.0
    min_gate_brier_skill: float = 0.0
    min_gate_decile_lift: float = 0.02
    min_edge_rank_ic: float = 0.02
    signal_prequalify_min_obs: int = 30
    signal_prequalify_method: Literal["block_bootstrap", "concurrency_t", "mean"] = "block_bootstrap"
    signal_prequalify_min_tstat: float = 1.5
    signal_prequalify_bootstrap_n: int = 1000
    overlay_sizing_enabled: bool = True
    edge_gate_mode: Literal["overlay_lift", "rank_ic"] = "rank_ic"
    edge_gate_min_lift_tstat: float = 1.0
    edge_gate_min_n_eff: int = 60
    edge_uplift_bootstrap_samples: int = 500
    edge_uplift_confidence: float = 0.90
    min_risk_unit_bps: float = 25.0
    candidate_rebalance_bars: Literal[1] = 1
    exit_policy_mode: Literal["label_only", "engine_aligned"] = "engine_aligned"
    candidate_families: tuple[str, ...] = (
        "trend_ma",
        "trend_donchian",
        "vol_breakout",
        "bollinger_reversion",
        "rsi_reversion",
        "funding_carry",
        "oi_volume_impulse",
        "btc_regime_pullback",
        "funding_zscore_carry",
        "vol_regime_reversion",
        "oi_volume_confirmed_breakout",
        "trend_pullback_continuation",
        "dual_momentum",
        "residual_reversion",
        "mtf_trend_pullback",
        "mtf_breakout_retest",
        "oi_price_divergence",
        "oi_breakout_confirm",
        "basis_zscore_reversion",
        "basis_momentum",
        "taker_imbalance_momentum",
        "taker_exhaustion_reversal",
        "funding_extreme_reversal",
        "vol_term_structure_gate",
    )
    # Execution cost model (SSOT; replaces flat 24bps)
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    maker_ratio: float = 0.75
    slippage_bps: float = 1.0
    impact_coeff_bps: float = 0.5
    impact_adv_proxy_field: str = "turnover_proxy"
    cost_stress_multiplier: float = 1.5
    cost_amortize_by_holding: bool = True
    # Signal-only validation mode (--mode signal; skips ML training)
    signal_only: bool = False
    # Walk-forward
    wf_enabled: bool = True
    wf_scheme: Literal["anchored", "rolling", "single"] = "anchored"
    wf_n_folds: int = 4
    min_fit_obs: int = 200
    min_wf_fold_pass_ratio: float = 0.60
    fold_survival_metric: Literal[
        "predicted_mu_tstat", "realized_selected_edge", "realized_log_growth"
    ] = "realized_selected_edge"
    min_fold_selected_events: int = 20
    min_fold_realized_edge_bps: float = 8.0  # >= 1.07x RT cost (7.5bps); 0.0은 +0.001bps도 통과
    min_fold_log_growth: float = 0.0
    # Edge model utility parameters
    downside_penalty: float = 0.3
    turnover_penalty: float = 0.5
    concentration_penalty: float = 0.0
    # Deprecated: use ExecutionCostModel fields instead; kept for explicit override only
    expected_cost_bps: float = _DEFAULT_RT_BPS
    edge_prediction_min_std_bps: float = 3.0
    edge_prediction_min_positive_rate: float = 0.01
    edge_prior_enabled: bool = True
    use_empirical_bayes_shrinkage: bool = True
    edge_prior_min_obs: int = 100
    edge_prior_shrinkage_obs: int = 500
    edge_prior_max_deviation_bps: float = 30.0
    max_variant_selection_fraction: float = 0.4
    edge_residual_model_enabled: bool = False
    max_drawdown_cap: float = 0.25
    # Deployment integrity: prevent near-zero-trading variants from "passing"
    min_deployment_trade_count: int = 20
    min_deployment_capital_fraction: float = 0.05
    # Edge attribution diagnostics
    edge_attribution_enabled: bool = True
    # Ablation evaluation alignment (RC1 fix)
    eval_apply_candidate_barriers: bool = True
    # Promotion gate — RC2 fix: prevent degenerate near-zero-deployment passes
    mar_min_drawdown_floor: float = 0.01      # MAR = 0 when max_dd < this (ratio of two noise values)
    min_cagr_for_promotion: float = 0.15      # crypto 위험 대비 최소 15% (0.02는 예금 이하)
    enforce_deployment_in_compound_gate: bool = True
    # Downside target — RC3 fix: clip paper-MAE to realizable stop loss
    q10_bound_to_stop: bool = True
    # Blend survival gate (signal→ML handoff) — mean-based, no double-count
    blend_survival_use_mean: bool = True
    blend_survival_min_net_stress_bps: float = 0.0
    blend_survival_require_promoted: bool = True
    kelly_use_probability_adjusted_mu: bool = True
    kelly_downside_variance_floor_enabled: bool = True
    candidate_metadata_forward_fill: bool = True

    def __post_init__(self) -> None:
        """Validate candidate strategy parameters."""
        if self.name not in {"candidate_ml", "rule_baseline"}:
            raise ValueError("candidate strategy name must be 'candidate_ml' or 'rule_baseline'")
        if self.train_months <= 0 or self.valid_months <= 0 or self.test_months <= 0:
            raise ValueError("all month windows must be positive")
        if self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1")
        if self.purge_bars is not None and self.purge_bars < 0:
            raise ValueError("purge bars must be non-negative")
        if self.embargo_bars is not None and self.embargo_bars < 0:
            raise ValueError("purge and embargo bars must be non-negative")
        if self.purge_safety_mult < 1.0:
            raise ValueError("purge_safety_mult must be >= 1.0")
        object.__setattr__(self, "_purge_bars_input", self.purge_bars)
        object.__setattr__(self, "_embargo_bars_input", self.embargo_bars)
        derived_purge_bars = (
            int(self.purge_bars)
            if self.purge_bars is not None
            else max(0, math.ceil(int(self.max_holding_bars) * self.purge_safety_mult))
        )
        derived_embargo_bars = derived_purge_bars if self.embargo_bars is None else int(self.embargo_bars)
        object.__setattr__(self, "purge_bars", derived_purge_bars)
        object.__setattr__(self, "embargo_bars", derived_embargo_bars)
        if self.gate_label_column not in {
            "profitable_after_hurdle_label",
            "barrier_first_label",
            "gross_direction_label",
        }:
            raise ValueError("unsupported gate_label_column")
        if self.gate_calibration_method not in {"sigmoid", "isotonic", "none"}:
            raise ValueError("gate_calibration_method must be sigmoid, isotonic, or none")
        if self.min_gate_calibration_obs < 1:
            raise ValueError("min_gate_calibration_obs must be >= 1")
        if self.min_gate_calibration_pos < 1:
            raise ValueError("min_gate_calibration_pos must be >= 1")
        if self.min_gate_probability_std < 0.0:
            raise ValueError("min_gate_probability_std must be non-negative")
        if not (0.1 <= self.ml_fit_fraction < 1.0):
            raise ValueError("ml_fit_fraction must be in [0.1, 1.0)")
        if not (0.0 <= self.ml_calibration_fraction < 1.0):
            raise ValueError("ml_calibration_fraction must be in [0.0, 1.0)")
        if not (0.0 <= self.model_early_stop_fraction < 1.0):
            raise ValueError("model_early_stop_fraction must be in [0.0, 1.0)")
        if not (0.0 < self.calibration_fit_fraction < 1.0):
            raise ValueError("calibration_fit_fraction must be in (0.0, 1.0)")
        if self.ml_fit_fraction + self.ml_calibration_fraction >= 1.0:
            raise ValueError("ml_fit_fraction + ml_calibration_fraction must be < 1.0")
        if self.promotion_decision_split not in {"fit", "calibration", "fit_calibration"}:
            raise ValueError("unsupported promotion_decision_split")
        if self.min_promotion_calibration_edge_bps < 0.0:
            raise ValueError("min_promotion_calibration_edge_bps must be non-negative")
        if self.min_promotion_calibration_obs < 1:
            raise ValueError("min_promotion_calibration_obs must be >= 1")
        if not (0.0 < self.kelly_fraction <= 0.25):
            raise ValueError("kelly_fraction must be in range (0.0, 0.25]")
        if self.sizing_mode not in {"stop_risk", "calibrated_event_kelly"}:
            raise ValueError("unsupported sizing_mode")
        if self.event_risk_budget <= 0.0:
            raise ValueError("event_risk_budget must be positive")
        if self.kelly_min_bin_ess < 1:
            raise ValueError("kelly_min_bin_ess must be >= 1")
        if self.cost_floor_bps < 0.0:
            raise ValueError("cost_floor_bps must be non-negative")
        if self.shortfall_threshold_basis not in {"absolute_bps", "stop_relative"}:
            raise ValueError("unsupported shortfall_threshold_basis")
        if self.max_expected_shortfall_stop_mult < 0.0:
            raise ValueError("max_expected_shortfall_stop_mult must be non-negative")
        if self.catastrophic_shortfall_stop_mult < 0.0:
            raise ValueError("catastrophic_shortfall_stop_mult must be non-negative")
        if (
            self.max_symbol_weight < 0.0
            or self.gross_cap < 0.0
            or self.net_cap < 0.0
            or self.beta_cap < 0.0
            or self.target_ann_vol < 0.0
            or self.max_expected_shortfall_bps < 0.0
            or self.catastrophic_shortfall_bps < 0.0
        ):
            raise ValueError("cap and penalty parameters must be non-negative")
        if self.gross_cap < self.max_symbol_weight:
            raise ValueError("gross cap must be at least max symbol weight")
        if self.min_candidate_obs <= 0 or self.min_symbol_oos_blocks <= 0:
            raise ValueError("minimum observations and blocks must be positive")
        if self.min_fold_selected_events < 1:
            raise ValueError("min_fold_selected_events must be >= 1")
        if self.diagnostic_top_k < 1:
            raise ValueError("diagnostic_top_k must be >= 1")
        if self.min_variant_oos_obs < 1:
            raise ValueError("min_variant_oos_obs must be >= 1")
        if self.allocation_backend not in {"ensemble_b0", "ml_edge"}:
            raise ValueError("allocation_backend must be ensemble_b0 or ml_edge")
        if self.ensemble_shrinkage_k <= 0.0:
            raise ValueError("ensemble_shrinkage_k must be positive")
        if self.ensemble_conditioning not in {"archetype_regime", "archetype_only", "auto"}:
            raise ValueError("ensemble_conditioning must be archetype_regime, archetype_only, or auto")
        if not (0.0 < self.ensemble_internal_val_fraction < 0.5):
            raise ValueError("ensemble_internal_val_fraction must be in (0.0, 0.5)")
        if self.mu_quality_ic_full_scale <= 0.0:
            raise ValueError("mu_quality_ic_full_scale must be positive")
        if not (0.0 <= self.min_variant_oos_hit_rate <= 1.0):
            raise ValueError("min_variant_oos_hit_rate must satisfy 0 <= value <= 1")
        if self.min_variant_oos_payoff_ratio < 0.0:
            raise ValueError("min_variant_oos_payoff_ratio must be non-negative")
        if not (0.0 <= self.max_variant_oos_q10_fail_rate <= 1.0):
            raise ValueError("max_variant_oos_q10_fail_rate must satisfy 0 <= value <= 1")
        if not (0.0 < self.selection_top_quantile <= 1.0):
            raise ValueError("selection_top_quantile must satisfy 0 < value <= 1")
        if not (0.0 <= self.min_net_floor_cost_fraction <= 2.0):
            raise ValueError("min_net_floor_cost_fraction must be in [0.0, 2.0]")
        if not (-1.0 <= self.min_oos_rank_ic <= 1.0):
            raise ValueError("min_oos_rank_ic must satisfy -1 <= value <= 1")
        if self.min_ic_tstat < 0.0:
            raise ValueError("min_ic_tstat must be non-negative")
        if self.max_oos_edge_decay_bps < 0.0:
            raise ValueError("max_oos_edge_decay_bps must be non-negative")
        if self.selection_utility_mode not in {"additive_drag", "expected_edge_direct"}:
            raise ValueError("unsupported selection_utility_mode")
        if self.breakeven_floor_mode not in {"static", "fold_adaptive"}:
            raise ValueError("unsupported breakeven_floor_mode")
        if not (0.0 <= self.breakeven_floor_cost_quantile <= 1.0):
            raise ValueError("breakeven_floor_cost_quantile must be in [0.0, 1.0]")
        if self.selection_shortfall_mode not in {"hard", "penalty_only", "catastrophic"}:
            raise ValueError("selection_shortfall_mode must be hard, penalty_only, or catastrophic")
        if self.selection_policy not in {"hard", "validation_quantile", "utility_topk"}:
            raise ValueError("selection_policy must be hard, validation_quantile, or utility_topk")
        if self.selection_scope != "per_timestamp":
            raise ValueError("selection_scope must be per_timestamp")
        if self.selection_max_events_per_bar is not None and self.selection_max_events_per_bar < 1:
            raise ValueError("selection_max_events_per_bar must be >= 1 when provided")
        if self.exit_policy_mode not in {"label_only", "engine_aligned"}:
            raise ValueError("exit_policy_mode must be label_only or engine_aligned")
        if self.catastrophic_shortfall_bps < self.max_expected_shortfall_bps:
            raise ValueError("catastrophic_shortfall_bps must be >= max_expected_shortfall_bps")
        if any((value < 0.0) or (value > 1.0) for value in self.selection_gate_grid):
            raise ValueError("selection_gate_grid values must be within [0.0, 1.0]")
        if any(value < 0.0 for value in self.selection_edge_grid_bps):
            raise ValueError("selection_edge_grid_bps values must be non-negative")
        if any(value < 0.0 for value in self.selection_q10_grid_bps):
            raise ValueError("selection_q10_grid_bps values must be non-negative")
        if any(not math.isfinite(floor) for floor in self.selection_shadow_utility_floors_bps):
            raise ValueError("selection_shadow_utility_floors_bps must be finite")
        if any(
            (not math.isfinite(frac) or frac < 0.0)
            for frac in self.selection_shadow_breakeven_floor_fractions
        ):
            raise ValueError("selection_shadow_breakeven_floor_fractions must be finite and non-negative")
        if not (0.0 < self.selection_shadow_top_quantile <= 1.0):
            raise ValueError("selection_shadow_top_quantile must be in (0.0, 1.0]")
        if self.selection_shadow_max_profiles < 1:
            raise ValueError("selection_shadow_max_profiles must be >= 1")
        if not (0.0 <= self.maker_ratio <= 1.0):
            raise ValueError("maker_ratio must be in [0.0, 1.0]")
        if any(v < 0.0 for v in (self.maker_fee_bps, self.taker_fee_bps, self.slippage_bps, self.impact_coeff_bps)):
            raise ValueError("fee, slippage, and impact parameters must be non-negative")
        if self.cost_stress_multiplier < 1.0:
            raise ValueError("cost_stress_multiplier must be >= 1.0")
        if self.wf_scheme not in {"anchored", "rolling", "single"}:
            raise ValueError("wf_scheme must be anchored, rolling, or single")
        if self.fold_survival_metric not in {
            "predicted_mu_tstat",
            "realized_selected_edge",
            "realized_log_growth",
        }:
            raise ValueError("unsupported fold_survival_metric")
        if self.wf_n_folds < 1:
            raise ValueError("wf_n_folds must be >= 1")
        if self.min_fit_obs < 1:
            raise ValueError("min_fit_obs must be >= 1")
        if not (0.0 <= self.min_wf_fold_pass_ratio <= 1.0):
            raise ValueError("min_wf_fold_pass_ratio must be in [0.0, 1.0]")
        if self.min_gate_brier_skill < -1.0:
            raise ValueError("min_gate_brier_skill must be >= -1.0")
        if self.min_gate_decile_lift < 0.0:
            raise ValueError("min_gate_decile_lift must be non-negative")
        if not (-1.0 <= self.min_edge_rank_ic <= 1.0):
            raise ValueError("min_edge_rank_ic must satisfy -1 <= value <= 1")
        if self.signal_prequalify_method not in {"block_bootstrap", "concurrency_t", "mean"}:
            raise ValueError("unsupported signal_prequalify_method")
        if self.signal_prequalify_bootstrap_n < 1:
            raise ValueError("signal_prequalify_bootstrap_n must be >= 1")
        if self.edge_gate_mode not in {"overlay_lift", "rank_ic"}:
            raise ValueError("unsupported edge_gate_mode")
        if self.edge_gate_min_n_eff < 1:
            raise ValueError("edge_gate_min_n_eff must be >= 1")
        if self.edge_uplift_bootstrap_samples < 1:
            raise ValueError("edge_uplift_bootstrap_samples must be >= 1")
        if not (0.0 < self.edge_uplift_confidence < 1.0):
            raise ValueError("edge_uplift_confidence must be in (0.0, 1.0)")
        if self.min_risk_unit_bps <= 0.0:
            raise ValueError("min_risk_unit_bps must be positive")
        if self.candidate_rebalance_bars != 1:
            raise ValueError("candidate_rebalance_bars must be fixed to 1")
        if self.downside_penalty < 0.0 or self.turnover_penalty < 0.0 or self.concentration_penalty < 0.0:
            raise ValueError("penalty parameters must be non-negative")
        if self.edge_prediction_min_std_bps < 0.0:
            raise ValueError("edge_prediction_min_std_bps must be non-negative")
        if not (0.0 <= self.edge_prediction_min_positive_rate <= 1.0):
            raise ValueError("edge_prediction_min_positive_rate must be within [0.0, 1.0]")
        if self.edge_prior_min_obs < 1:
            raise ValueError("edge_prior_min_obs must be >= 1")
        if self.edge_prior_shrinkage_obs < 1:
            raise ValueError("edge_prior_shrinkage_obs must be >= 1")
        if self.edge_prior_max_deviation_bps <= 0.0:
            raise ValueError("edge_prior_max_deviation_bps must be > 0")
        if not (0.0 < self.max_variant_selection_fraction <= 1.0):
            raise ValueError("max_variant_selection_fraction must be in (0.0, 1.0]")
        if self.min_variant_oos_obs < 1:
            raise ValueError("min_variant_oos_obs must be >= 1")
        if not (0.0 < self.max_variant_event_fraction_per_bar <= 1.0):
            raise ValueError("max_variant_event_fraction_per_bar must be in (0.0, 1.0]")
        if self.min_regime_variant_oos_obs < 1:
            raise ValueError("min_regime_variant_oos_obs must be >= 1")
        if self.min_regime_cell_oos_obs < 1:
            raise ValueError("min_regime_cell_oos_obs must be >= 1")
        if not math.isfinite(self.min_regime_cell_edge_bps):
            raise ValueError("min_regime_cell_edge_bps must be finite")
        if self.max_admitted_cells_per_variant < 1:
            raise ValueError("max_admitted_cells_per_variant must be >= 1")
        if not (0.5 <= self.min_admission_posterior_prob < 1.0):
            raise ValueError("min_admission_posterior_prob must be in [0.5, 1.0)")
        if not (math.isfinite(self.admission_tau_prior_bps) and self.admission_tau_prior_bps > 0.0):
            raise ValueError("admission_tau_prior_bps must be finite and > 0")
        if self.max_exit_policy_variants_per_signal < 1:
            raise ValueError("max_exit_policy_variants_per_signal must be >= 1")
        if self.min_signal_cell_oos_obs < 1:
            raise ValueError("min_signal_cell_oos_obs must be >= 1")
        if self.max_signal_cell_event_fraction_per_bar <= 0.0:
            raise ValueError("max_signal_cell_event_fraction_per_bar must be positive")
        if not (0.0 < self.max_drawdown_cap <= 1.0):
            raise ValueError("max_drawdown_cap must be in (0.0, 1.0]")
        for variant in self.enabled_candidate_variants:
            if variant.count(":") != 1:
                raise ValueError("enabled_candidate_variants entries must be formatted as family:variant")
        for variant in self.disabled_candidate_variants:
            if variant.count(":") != 1:
                raise ValueError("disabled_candidate_variants entries must be formatted as family:variant")
        for variant in self.side_flip_candidate_variants:
            if variant.count(":") != 1:
                raise ValueError("side_flip_candidate_variants entries must be formatted as family:variant")
        if self.min_deployment_trade_count < 0:
            raise ValueError("min_deployment_trade_count must be >= 0")
        if not (0.0 <= self.min_deployment_capital_fraction <= 1.0):
            raise ValueError("min_deployment_capital_fraction must be in [0.0, 1.0]")
        if self.blend_survival_min_net_stress_bps < -50.0:
            raise ValueError("blend_survival_min_net_stress_bps too permissive (< -50)")


def with_max_holding_bars(
    cfg: CandidateStrategyConfig,
    *,
    max_holding_bars: int | None = None,
) -> CandidateStrategyConfig:
    """Return a config instance with purge/embargo derived from the provided horizon."""
    resolved_holding_bars = max(
        1,
        int(max_holding_bars) if max_holding_bars is not None else int(cfg.max_holding_bars),
    )
    if (
        resolved_holding_bars == int(cfg.max_holding_bars)
        and cfg.purge_bars == (
            int(cfg._purge_bars_input)
            if cfg._purge_bars_input is not None
            else max(0, math.ceil(resolved_holding_bars * cfg.purge_safety_mult))
        )
        and cfg.embargo_bars
        == (
            int(cfg._embargo_bars_input)
            if cfg._embargo_bars_input is not None
            else (
                int(cfg._purge_bars_input)
                if cfg._purge_bars_input is not None
                else max(0, math.ceil(resolved_holding_bars * cfg.purge_safety_mult))
            )
        )
    ):
        return cfg
    return replace(
        cfg,
        max_holding_bars=resolved_holding_bars,
        purge_bars=cfg._purge_bars_input,
        embargo_bars=cfg._embargo_bars_input,
    )


def resolve_purge_and_embargo_bars(
    cfg: CandidateStrategyConfig,
    *,
    max_holding_bars: int | None = None,
) -> tuple[int, int]:
    """Return purge and embargo bars, auto-derived from holding horizon when unset."""
    resolved_cfg = with_max_holding_bars(cfg, max_holding_bars=max_holding_bars)
    purge_bars = resolved_cfg.purge_bars
    embargo_bars = resolved_cfg.embargo_bars
    if purge_bars is None or embargo_bars is None:
        raise ValueError("purge/embargo bars must be materialized before use")
    return int(purge_bars), int(embargo_bars)
