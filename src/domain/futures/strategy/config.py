from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
    cost_floor_bps: float = 24.0
    gate_label_column: Literal[
        "profitable_after_hurdle_label",
        "barrier_first_label",
        "gross_direction_label",
    ] = "profitable_after_hurdle_label"
    gate_calibration_method: Literal["sigmoid", "isotonic", "none"] = "sigmoid"
    min_gate_calibration_obs: int = 100
    min_gate_calibration_pos: int = 10
    min_gate_probability_std: float = 0.03
    ml_fit_fraction: float = 0.60
    ml_calibration_fraction: float = 0.20
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
    min_gate_probability: float = 0.55
    min_expected_net_bps: float = 1.0
    max_expected_shortfall_bps: float = 80.0
    selection_shortfall_mode: Literal["hard", "penalty_only", "catastrophic"] = "hard"
    catastrophic_shortfall_bps: float = 300.0
    selection_sensitivity_enabled: bool = True
    selection_gate_grid: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55)
    selection_edge_grid_bps: tuple[float, ...] = (0.0, 1.0, 5.0)
    selection_q10_grid_bps: tuple[float, ...] = (80.0, 150.0, 250.0, 400.0)
    enabled_candidate_variants: tuple[str, ...] = ()
    side_flip_candidate_variants: tuple[str, ...] = ()
    diagnostic_top_k: int = 10
    min_variant_oos_obs: int = 100
    min_variant_oos_edge_bps: float = 1.0
    min_variant_oos_hit_rate: float = 0.50
    min_variant_oos_payoff_ratio: float = 1.20
    max_variant_oos_q10_fail_rate: float = 0.90
    candidate_identity_features_enabled: bool = True
    market_state_features_enabled: bool = True
    promotion_filter_enabled: bool = True
    selection_policy: Literal["hard", "validation_quantile", "utility_topk"] = "utility_topk"
    selection_top_quantile: float = 0.10
    min_net_floor_cost_fraction: float = 0.50
    min_oos_rank_ic: float = 0.01
    min_oos_log_growth_uplift: float = 0.0
    max_oos_edge_decay_bps: float = 50.0
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
    )
    # Edge model utility parameters
    downside_penalty: float = 1.0
    turnover_penalty: float = 0.5
    concentration_penalty: float = 0.0
    expected_cost_bps: float = 24.0
    max_drawdown_cap: float = 0.25

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
        if self.ml_fit_fraction + self.ml_calibration_fraction >= 1.0:
            raise ValueError("ml_fit_fraction + ml_calibration_fraction must be < 1.0")
        if not (0.0 < self.kelly_fraction <= 0.25):
            raise ValueError("kelly_fraction must be in range (0.0, 0.25]")
        if self.cost_floor_bps < 0.0:
            raise ValueError("cost_floor_bps must be non-negative")
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
        if self.selection_shortfall_mode not in {"hard", "penalty_only", "catastrophic"}:
            raise ValueError("selection_shortfall_mode must be hard, penalty_only, or catastrophic")
        if self.selection_policy not in {"hard", "validation_quantile", "utility_topk"}:
            raise ValueError("selection_policy must be hard, validation_quantile, or utility_topk")
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
        if self.downside_penalty < 0.0 or self.turnover_penalty < 0.0 or self.concentration_penalty < 0.0:
            raise ValueError("penalty parameters must be non-negative")
        if not (0.0 < self.max_drawdown_cap <= 1.0):
            raise ValueError("max_drawdown_cap must be in (0.0, 1.0]")
        for variant in self.enabled_candidate_variants:
            if variant.count(":") != 1:
                raise ValueError("enabled_candidate_variants entries must be formatted as family:variant")
        for variant in self.side_flip_candidate_variants:
            if variant.count(":") != 1:
                raise ValueError("side_flip_candidate_variants entries must be formatted as family:variant")
