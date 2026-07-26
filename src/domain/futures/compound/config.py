from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True, frozen=True)
class ClusterConfig:
    k_clusters: int = 4
    min_cluster_size: int = 5
    feature_lookback_hours: int = 480
    winsorize_pct: float = 0.05
    live_refit_days: int = 30

    def __post_init__(self) -> None:
        assert self.k_clusters >= 2, "k_clusters must be >= 2"
        assert self.min_cluster_size >= 1
        assert self.feature_lookback_hours > 0
        assert 0 <= self.winsorize_pct <= 0.5
        assert self.live_refit_days > 0


@dataclass(slots=True, frozen=True)
class L2BenchmarkConfig:
    mode: Literal["risk_matched_crypto", "cash_collateral"] = "risk_matched_crypto"
    crypto_symbols: tuple[str, str] = ("BTCUSDT", "ETHUSDT")
    crypto_weights: tuple[float, float] = (0.5, 0.5)
    volatility_lookback_days: int = 60
    target_ann_vol: float = 0.15

    def __post_init__(self) -> None:
        assert self.volatility_lookback_days > 0
        assert self.target_ann_vol > 0
        assert len(self.crypto_symbols) == len(self.crypto_weights) >= 2
        assert abs(sum(self.crypto_weights) - 1.0) < 1e-10, "crypto_weights must sum to 1"


@dataclass(slots=True, frozen=True)
class L2GateConfig:
    min_oos_days: int = 365
    min_active_days_ratio: float = 0.10
    min_rebalances: int = 30
    min_excess_growth_probability: float = 0.90
    min_bootstrap_sharpe_probability: float = 0.90
    min_deflated_sharpe_probability: float = 0.90
    max_drawdown: float = 0.20
    min_daily_cvar95: float = -0.025
    max_annual_volatility: float = 0.25
    max_cost_drag_ratio: float = 0.50
    max_capacity_utilisation_p95: float = 0.10
    min_positive_outer_folds: int = 3
    stressed_cost_multiplier: float = 2.0

    def __post_init__(self) -> None:
        assert self.min_oos_days > 0
        assert 0 < self.min_active_days_ratio <= 1
        assert self.min_rebalances > 0
        assert 0 < self.min_excess_growth_probability <= 1
        assert 0 < self.min_bootstrap_sharpe_probability <= 1
        assert 0 < self.min_deflated_sharpe_probability <= 1
        assert self.max_drawdown > 0
        assert self.min_daily_cvar95 < 0
        assert self.max_annual_volatility > 0
        assert self.max_cost_drag_ratio > 0
        assert self.max_capacity_utilisation_p95 > 0
        assert self.min_positive_outer_folds >= 1
        assert self.stressed_cost_multiplier >= 1.0


@dataclass(slots=True, frozen=True)
class DataPlaneConfig:
    base_timeframe: str = "1h"
    max_symbols: int = 120
    min_listing_days: int = 90
    liquidity_lookback_days: int = 30
    min_median_daily_quote_volume: float = 20_000_000.0
    min_core_coverage: float = 0.98
    alpha_block_bars: int = 2048

    def __post_init__(self) -> None:
        assert self.max_symbols > 0
        assert self.min_listing_days > 0
        assert self.liquidity_lookback_days > 0
        assert self.min_median_daily_quote_volume > 0
        assert 0 < self.min_core_coverage <= 1
        assert self.alpha_block_bars > 0


@dataclass(slots=True, frozen=True)
class L1EstimatorConfig:
    n_folds: int = 4
    prior_effective_n: float = 30.0
    active_effective_n: float = 20.0
    retire_effective_n: float = 60.0
    retire_probability_max: float = 0.20
    retire_consecutive_versions: int = 3
    purge_bars: int = 25
    embargo_bars: int = 1

    def __post_init__(self) -> None:
        assert self.n_folds > 0
        assert self.prior_effective_n > 0
        assert self.active_effective_n > 0
        assert self.retire_effective_n > self.active_effective_n
        assert 0 <= self.retire_probability_max <= 1
        assert self.retire_consecutive_versions > 0
        assert self.purge_bars > 0
        assert self.embargo_bars >= 0


@dataclass(slots=True, frozen=True)
class FactorRiskConfig:
    ewm_half_life_days: int = 60
    max_cluster_factors: int = 8
    variance_floor: float = 1e-8

    def __post_init__(self) -> None:
        assert self.ewm_half_life_days > 0
        assert self.max_cluster_factors > 0
        assert self.variance_floor > 0


@dataclass(slots=True, frozen=True)
class AllocatorConfig:
    rebalance_bars: int = 4
    fractional_kelly: float = 0.20
    uncertainty_z: float = 0.50
    target_ann_vol: float = 0.15
    gross_cap: float = 1.00
    net_cap: float = 0.30
    per_symbol_cap: float = 0.10
    beta_cap: float = 0.25
    turnover_l2: float = 1e-3
    max_iterations: int = 64
    objective_tolerance: float = 1e-10
    portfolio_nav_usdt: float = 100_000.0
    risk_scale: float = 1.0

    def __post_init__(self) -> None:
        assert self.rebalance_bars > 0
        assert 0 < self.fractional_kelly <= 1
        assert self.uncertainty_z >= 0
        assert self.target_ann_vol > 0
        assert self.net_cap <= self.gross_cap
        assert self.per_symbol_cap <= self.gross_cap
        assert self.beta_cap >= 0
        assert self.turnover_l2 >= 0
        assert 1 <= self.max_iterations <= 64
        assert self.objective_tolerance > 0
        assert self.portfolio_nav_usdt > 0
        assert self.risk_scale > 0


@dataclass(slots=True, frozen=True)
class RiskOverlayConfig:
    soft_drawdown_start: float = 0.08
    drawdown_second_knot: float = 0.15
    hard_drawdown: float = 0.20
    hard_drawdown_cooldown_bars: int = 168

    def __post_init__(self) -> None:
        assert 0 <= self.soft_drawdown_start < self.drawdown_second_knot < self.hard_drawdown
        assert self.hard_drawdown_cooldown_bars > 0


@dataclass(slots=True, frozen=True)
class L1Config:
    max_residual_correlation: float = 0.60
    min_positive_folds: int = 4
    total_outer_folds: int = 5
    min_effective_days: float = 180.0
    min_effective_events_short: int = 1000
    bootstrap_iterations: int = 1000
    min_posterior_probability: float = 0.65
    fdr_q_threshold: float = 0.10
    sign_consistency_min: float = 0.80
    cost_stress_multiplier: float = 2.0

    def __post_init__(self) -> None:
        assert 0 < self.max_residual_correlation <= 1
        assert self.min_positive_folds > 0
        assert self.total_outer_folds >= self.min_positive_folds
        assert self.min_effective_days > 0
        assert self.min_effective_events_short > 0
        assert self.bootstrap_iterations > 0
        assert 0 < self.min_posterior_probability <= 1
        assert 0 < self.fdr_q_threshold <= 1
        assert 0 < self.sign_consistency_min <= 1
        assert self.cost_stress_multiplier >= 1


@dataclass(slots=True, frozen=True)
class L3ValidationConfig:
    holdout_days: int = 90
    min_holdout_days: int = 30
    l2_prior_effective_days_cap: int = 60
    promote_probability: float = 0.65
    reject_probability: float = 0.20
    max_drawdown: float = 0.20
    max_daily_cvar95: float = 0.04

    def __post_init__(self) -> None:
        assert self.holdout_days >= self.min_holdout_days > 0
        assert self.l2_prior_effective_days_cap > 0
        assert 0 < self.reject_probability < self.promote_probability <= 1
        assert 0 < self.max_drawdown <= 1
        assert 0 < self.max_daily_cvar95 <= 1


@dataclass(slots=True, frozen=True)
class L1MultiscaleConfig:
    n_folds: int = 5
    purge_bars: int = 25
    embargo_bars: int = 1
    min_positive_folds: int = 4
    bootstrap_iterations: int = 1000


@dataclass(slots=True, frozen=True)
class CalibrationConfig:
    ridge_lambda_scale: float = 0.01
    family_shrink: float = 0.5
    min_fold_obs: int = 1000
    n_folds: int = 5
    purge_bars: int = 2
    embargo_bars: int = 42

    def __post_init__(self) -> None:
        assert self.ridge_lambda_scale >= 0
        assert 0 <= self.family_shrink <= 1
        assert self.min_fold_obs > 0
        assert self.n_folds >= 3
        assert self.purge_bars >= 0
        assert self.embargo_bars >= 0


@dataclass(slots=True, frozen=True)
class AdmissionConfig:
    n_bootstrap: int = 500
    block_size: int = 42
    fdr_q_threshold: float = 0.10
    default_cost_bps: float = 8.0
    sign_consistency_min: float = 0.6
    composite_sign_consistency_min: float = 0.5
    composite_p_value_max: float = 0.5

    def __post_init__(self) -> None:
        assert self.n_bootstrap > 0
        assert self.block_size > 0
        assert 0 < self.fdr_q_threshold <= 1
        assert self.default_cost_bps >= 0
        assert 0 < self.sign_consistency_min <= 1
        assert 0 < self.composite_sign_consistency_min <= 1
        assert 0 <= self.composite_p_value_max <= 1


@dataclass(slots=True, frozen=True)
class RiskModelConfig:
    ewm_half_life_bars: int = 60
    shrink_delta: float = 0.3
    variance_floor: float = 1e-10
    min_history_bars: int = 60

    def __post_init__(self) -> None:
        assert self.ewm_half_life_bars > 0
        assert 0 <= self.shrink_delta <= 1
        assert self.variance_floor > 0
        assert self.min_history_bars > 0


@dataclass(slots=True, frozen=True)
class BaselineAllocConfig:
    target_ann_vol: float = 0.20
    per_symbol_cap: float = 0.05
    gross_cap: float = 2.0

    def __post_init__(self) -> None:
        assert self.target_ann_vol > 0
        assert 0 < self.per_symbol_cap <= 1
        assert self.gross_cap > 0


@dataclass(slots=True, frozen=True)
class DenseSimConfig:
    bars_per_year: float = 2190.0

    def __post_init__(self) -> None:
        assert self.bars_per_year > 0


@dataclass(slots=True, frozen=True)
class LadderConfig:
    cost_bps: float = 8.0
    n_bootstrap: int = 1000

    def __post_init__(self) -> None:
        assert self.cost_bps >= 0
        assert self.n_bootstrap > 0


@dataclass(slots=True, frozen=True)
class DynamicCompoundingConfig:
    kelly_fraction: float = 0.20
    target_ann_vol: float = 0.15
    vol_lookback_bars: int = 180
    vol_scale_max: float = 1.5
    max_gross_leverage: float = 1.00
    max_long_leverage: float = 0.70
    max_short_leverage: float = 0.30
    sigma_floor: float = 1e-4
    funding_carry_enabled: bool = True
    soft_drawdown_limit: float = 0.10
    hard_drawdown_limit: float = 0.18
    alpha_smooth: float = 0.15
    band_frac: float = 0.30
    dd_scale_floor: float = 0.25
    dd_cooldown_bars: int = 60
    min_vol_samples: int = 60
    use_rank_conviction: bool = True
    mdd_risk_budget: float = 0.20
    max_ann_vol_budget: float = 0.25
    risk_safety_factor: float = 0.75
    min_mdd_vol_ratio: float = 0.50

    def __post_init__(self) -> None:
        assert 0 < self.kelly_fraction <= 1
        assert self.target_ann_vol > 0
        assert self.vol_lookback_bars > 0
        assert self.vol_scale_max >= 1.0
        assert self.max_long_leverage > 0
        assert self.max_short_leverage > 0
        assert self.max_gross_leverage >= self.max_long_leverage + self.max_short_leverage
        assert self.sigma_floor > 0
        assert 0 < self.soft_drawdown_limit < self.hard_drawdown_limit <= 1
        assert 0 < self.alpha_smooth <= 1
        assert self.band_frac >= 0
        assert self.dd_scale_floor > 0, "[LIMIT-01] dd_scale_floor must be > 0"
        assert self.dd_cooldown_bars > 0
        assert self.min_vol_samples > 0
        assert self.mdd_risk_budget > 0
        assert self.max_ann_vol_budget > 0
        assert self.risk_safety_factor > 0
        assert self.min_mdd_vol_ratio > 0


@dataclass(slots=True, frozen=True)
class HandoffConfig:
    max_pairwise_correlation: float = 0.80
    min_positive_outer_folds: int = 4
    target_ann_vol: float = 0.15
    max_ann_vol: float = 0.20
    max_drawdown: float = 0.20
    cost_stress_multiplier: float = 2.0
    n_bootstrap: int = 1_000
    dedup_rho_threshold: float = 0.90
    min_dedup_observations: int = 1_000

    def __post_init__(self) -> None:
        assert 0 < self.max_pairwise_correlation <= 1
        assert self.min_positive_outer_folds > 0
        assert self.target_ann_vol > 0
        assert self.max_ann_vol > 0
        assert self.max_drawdown > 0
        assert self.cost_stress_multiplier >= 1
        assert self.n_bootstrap > 0
        assert 0 < self.dedup_rho_threshold <= 1
        assert self.min_dedup_observations >= 1


@dataclass(slots=True, frozen=True)
class CompoundEngineConfig:
    data: DataPlaneConfig = field(default_factory=DataPlaneConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    l1: L1EstimatorConfig = field(default_factory=L1EstimatorConfig)
    l1_multiscale: L1MultiscaleConfig = field(default_factory=L1MultiscaleConfig)
    factor_risk: FactorRiskConfig = field(default_factory=FactorRiskConfig)
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)
    risk: RiskOverlayConfig = field(default_factory=RiskOverlayConfig)
    l2_gate: L2GateConfig = field(default_factory=L2GateConfig)
    l2_benchmark: L2BenchmarkConfig = field(default_factory=L2BenchmarkConfig)
    l3: L3ValidationConfig = field(default_factory=L3ValidationConfig)
    risk_model: RiskModelConfig = field(default_factory=RiskModelConfig)
    baseline_alloc: BaselineAllocConfig = field(default_factory=BaselineAllocConfig)
    dense_sim: DenseSimConfig = field(default_factory=DenseSimConfig)
    ladder: LadderConfig = field(default_factory=LadderConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    handoff: HandoffConfig = field(default_factory=HandoffConfig)
    dynamic_compounding: DynamicCompoundingConfig = field(default_factory=DynamicCompoundingConfig)

    def __post_init__(self) -> None:
        assert isinstance(self.data, DataPlaneConfig)
        assert isinstance(self.cluster, ClusterConfig)
        assert isinstance(self.l1, L1EstimatorConfig)
        assert isinstance(self.l1_multiscale, L1MultiscaleConfig)
        assert isinstance(self.factor_risk, FactorRiskConfig)
        assert isinstance(self.allocator, AllocatorConfig)
        assert isinstance(self.risk, RiskOverlayConfig)
        assert isinstance(self.l2_gate, L2GateConfig)
        assert isinstance(self.l2_benchmark, L2BenchmarkConfig)
        assert isinstance(self.l3, L3ValidationConfig)
        assert isinstance(self.risk_model, RiskModelConfig)
        assert isinstance(self.baseline_alloc, BaselineAllocConfig)
        assert isinstance(self.dense_sim, DenseSimConfig)
        assert isinstance(self.ladder, LadderConfig)
        assert isinstance(self.calibration, CalibrationConfig)
        assert isinstance(self.admission, AdmissionConfig)
        assert isinstance(self.handoff, HandoffConfig)
        assert isinstance(self.dynamic_compounding, DynamicCompoundingConfig)
