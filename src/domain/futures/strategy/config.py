from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_DEFAULT_COST_MODEL = ExecutionCostModel()
_DEFAULT_RT_BPS: float = _DEFAULT_COST_MODEL.round_trip_bps()  # ≈ 7.5


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
    """Rule-based 5-state soft posterior regime settings."""

    enabled: bool = False  # regime provider module removed; keep False until re-implemented
    vol_window: int = 30
    vol_crisis_pct: float = 0.95
    vol_high_pct: float = 0.70
    trend_ma_fast: int = 12
    trend_ma_slow: int = 48
    trend_thr: float = 0.0
    dd_crisis_thr: float = -0.20
    corr_crisis_thr: float = 0.80
    smooth_ewma_bars: int = 6
    gross_floor: float = 0.15

    def __post_init__(self) -> None:
        """Validate regime parameters."""
        if self.vol_window < 1:
            raise ValueError("vol_window must be >= 1")
        if not (0.0 < self.vol_high_pct < self.vol_crisis_pct < 1.0):
            raise ValueError(
                "volatility percentiles must satisfy "
                "0 < vol_high_pct < vol_crisis_pct < 1.0"
            )
        if self.trend_ma_fast >= self.trend_ma_slow:
            raise ValueError("trend_ma_fast must be less than trend_ma_slow")
        if self.trend_ma_fast < 1:
            raise ValueError("trend_ma_fast must be >= 1")
        if self.dd_crisis_thr >= 0.0:
            raise ValueError("dd_crisis_thr must be negative")
        if not (0.0 <= self.corr_crisis_thr <= 1.0):
            raise ValueError("corr_crisis_thr must satisfy 0 <= corr_crisis_thr <= 1.0")
        if self.smooth_ewma_bars < 1:
            raise ValueError("smooth_ewma_bars must be >= 1")
        if not (0.0 <= self.gross_floor <= 1.0):
            raise ValueError("gross_floor must satisfy 0 <= gross_floor <= 1.0")


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
    purge_bars: int = 18
    embargo_bars: int = 18
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
    min_gate_probability: float = 0.55
    min_expected_net_bps: float = 1.0
    max_expected_shortfall_bps: float = 300.0
    shortfall_threshold_basis: Literal["absolute_bps", "stop_relative"] = "absolute_bps"
    max_expected_shortfall_stop_mult: float = 1.25
    selection_gate_mode: Literal["off", "soft_floor", "hard_floor"] = "soft_floor"
    selection_min_gate_probability_floor: float = 0.35
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
    selection_shadow_gate_modes: tuple[str, ...] = ("off", "soft_floor", "hard_floor")
    selection_shadow_gate_floors: tuple[float, ...] = (0.0, 0.30, 0.35, 0.40)
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
    max_variant_oos_q10_fail_rate: float = 0.90
    max_variant_event_fraction_per_bar: float = 0.25
    regime_diagnostic_enabled: bool = True
    min_regime_variant_oos_obs: int = 40
    min_regime_variant_oos_edge_bps: float = 2.0
    # Set to False to remove hard regime-based signal masking; regime moves to
    # sizing multiplier layer (see regime_as_size_multiplier).
    regime_signal_gating_enabled: bool = False
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
    # Vol-normalised percentile edge (atr_bps column required; skeleton if absent)
    edge_percentile_vol_normalized: bool = False
    min_signal_cell_oos_obs: int = 80
    max_signal_cell_event_fraction_per_bar: float = 0.12
    candidate_identity_features_enabled: bool = True
    market_state_features_enabled: bool = True
    static_universe_features_enabled: bool = False
    exclude_immediate_return_features: bool = True
    promotion_filter_enabled: bool = True
    selection_policy: Literal["hard", "validation_quantile", "utility_topk"] = "utility_topk"
    selection_scope: Literal["per_timestamp"] = "per_timestamp"
    selection_max_events_per_bar: int | None = None
    selection_top_quantile: float = 0.10
    min_net_floor_cost_fraction: float = 0.50
    min_oos_rank_ic: float = 0.01
    min_oos_log_growth_uplift: float = 0.0
    max_oos_edge_decay_bps: float = 50.0
    min_gate_brier_skill: float = 0.0
    min_gate_decile_lift: float = 0.02
    min_edge_rank_ic: float = 0.02
    signal_prequalify_min_obs: int = 30
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
        "cross_sectional_momentum",
        "funding_zscore_carry",
        "vol_regime_reversion",
        "btc_corr_regime",
        "funding_acceleration_carry",
        "btc_residual_momentum",
        "oi_volume_confirmed_breakout",
        "trend_pullback_continuation",
        "dual_momentum",
        "liquidation_wick_reversal",
        "squeeze_unwind",
        "residual_reversion",
    )
    # Execution cost model (SSOT; replaces flat 24bps)
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    maker_ratio: float = 0.75
    slippage_bps: float = 1.0
    impact_coeff_bps: float = 0.0
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
    min_fold_realized_edge_bps: float = 15.0  # >= 2x RT cost (7.5bps); 0.0은 +0.001bps도 통과
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
    edge_prior_min_obs: int = 100
    edge_prior_shrinkage_obs: int = 500
    edge_prior_max_deviation_bps: float = 30.0
    max_variant_selection_fraction: float = 0.4
    edge_residual_model_enabled: bool = True
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
        if self.purge_bars < 0 or self.embargo_bars < 0:
            raise ValueError("purge and embargo bars must be non-negative")
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
        if self.selection_gate_mode not in {"off", "soft_floor", "hard_floor"}:
            raise ValueError("unsupported selection_gate_mode")
        if not (0.0 <= self.selection_min_gate_probability_floor <= 1.0):
            raise ValueError("selection_min_gate_probability_floor must be in [0.0, 1.0]")
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
        valid_gate_modes = {"off", "soft_floor", "hard_floor"}
        if any(mode not in valid_gate_modes for mode in self.selection_shadow_gate_modes):
            raise ValueError("selection_shadow_gate_modes contains unsupported mode")
        if any((floor < 0.0 or floor > 1.0) for floor in self.selection_shadow_gate_floors):
            raise ValueError("selection_shadow_gate_floors must be in [0.0, 1.0]")
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
