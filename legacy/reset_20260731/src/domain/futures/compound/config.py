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
    min_oos_days: int = 340
    min_active_days_ratio: float = 0.10
    min_rebalances: int = 30
    min_excess_growth_probability: float = 0.90
    min_deflated_sharpe_probability: float = 0.90
    max_drawdown: float = 0.20
    min_daily_cvar95: float = -0.025
    max_annual_volatility: float = 0.25
    max_cost_drag_ratio: float = 0.50
    max_name_weight_p95: float = 0.10
    min_positive_outer_folds: int = 3
    stressed_cost_multiplier: float = 2.0
    max_spa_pvalue: float = 0.10
    l1_prior_effective_days_cap: int = 90
    l1_prior_min_active_days: int = 30

    def __post_init__(self) -> None:
        assert self.min_oos_days > 0
        assert 0 < self.min_active_days_ratio <= 1
        assert self.min_rebalances > 0
        assert 0 < self.min_excess_growth_probability <= 1
        assert 0 < self.min_deflated_sharpe_probability <= 1
        assert self.max_drawdown > 0
        assert self.min_daily_cvar95 < 0
        assert self.max_annual_volatility > 0
        assert self.max_cost_drag_ratio > 0
        assert self.max_name_weight_p95 > 0
        assert self.min_positive_outer_folds >= 1
        assert self.stressed_cost_multiplier >= 1.0
        assert 0 < self.max_spa_pvalue <= 1
        assert self.l1_prior_effective_days_cap > 0
        assert self.l1_prior_min_active_days >= 0


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
    min_holdout_growth_probability: float = 0.50

    def __post_init__(self) -> None:
        assert self.holdout_days >= self.min_holdout_days > 0
        assert self.l2_prior_effective_days_cap > 0
        assert 0 < self.reject_probability < self.promote_probability <= 1
        assert 0 < self.max_drawdown <= 1
        assert 0 < self.max_daily_cvar95 <= 1
        assert 0 < self.min_holdout_growth_probability <= 1


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
    spread_bps: float = 2.0
    impact_coeff: float = 0.10
    nav_usdt: float = 100_000.0
    min_quote_volume_usdt: float = 1_000.0

    def __post_init__(self) -> None:
        assert self.bars_per_year > 0
        assert self.spread_bps >= 0
        assert self.impact_coeff >= 0
        assert self.nav_usdt > 0
        assert self.min_quote_volume_usdt > 0


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
    target_ann_vol: float = 0.12
    vol_lookback_bars: int = 180
    vol_scale_max: float = 1.5
    max_gross_leverage: float = 1.00
    max_long_leverage: float = 0.70
    max_short_leverage: float = 0.30
    sigma_floor: float = 1e-4
    funding_carry_enabled: bool = True
    soft_drawdown_limit: float = 0.10
    hard_drawdown_limit: float = 0.18
    alpha_smooth: float = 0.08
    band_frac: float = 0.60
    dd_scale_floor: float = 0.25
    dd_cooldown_bars: int = 60
    min_vol_samples: int = 60
    use_rank_conviction: bool = True
    mdd_risk_budget: float = 0.20
    max_ann_vol_budget: float = 0.25
    risk_safety_factor: float = 0.75
    min_mdd_vol_ratio: float = 0.50
    mdd_budget: float = 0.107
    mdd_parity_lookback_days: int = 180
    mdd_parity_max_scale: float = 3.0
    max_net_exposure: float = 0.10

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
        assert self.mdd_budget > 0
        assert self.mdd_parity_lookback_days > 0
        assert self.mdd_parity_max_scale >= 1.0
        assert 0.0 <= self.max_net_exposure <= 1.0


@dataclass(slots=True, frozen=True)
class L1RoutingConfig:
    enabled: bool = True
    family_top_k: int = 2
    symbol_top_n: int = 10
    normalization_warmup_bars: int = 500
    signal_clip: float = 3.0
    min_rank_folds: int = 1

    def __post_init__(self) -> None:
        assert self.family_top_k >= 1, f"family_top_k must be >= 1, got {self.family_top_k}"
        assert self.symbol_top_n >= 1, f"symbol_top_n must be >= 1, got {self.symbol_top_n}"
        assert self.normalization_warmup_bars > 0, f"normalization_warmup_bars must be > 0, got {self.normalization_warmup_bars}"
        assert self.signal_clip > 0, f"signal_clip must be > 0, got {self.signal_clip}"
        assert self.min_rank_folds >= 1, f"min_rank_folds must be >= 1, got {self.min_rank_folds}"


@dataclass(slots=True, frozen=True)
class L1LegConfig:
    horizon_band_bars: tuple[int, ...] = (6, 12, 24)
    modes: tuple[str, ...] = ("xs", "ts")
    min_turnover_per_bar: float = 0.005
    cost_safety_margin: float = 1.5
    min_positive_fold_ratio: float = 0.50
    max_leg_weight: float = 0.25
    max_name_weight: float = 0.10
    warmup_folds: int = 4
    familywise_error_rate: float = 0.10
    min_cross_section: int = 10
    bars_per_year: float = 2190.0
    n_bootstrap: int = 2000
    min_growth_posterior_probability: float = 0.90
    stress_cost_multiplier: float = 2.0
    shrinkage_prior_obs: float = 2000.0
    leg_prior_lookback_bars: int = 1080
    handoff_posterior_floor: float = 0.50
    prior_only_folds: int = 1

    def __post_init__(self) -> None:
        for h in self.horizon_band_bars:
            assert h > 0, f"horizon_band_bars entries must be > 0, got {h}"
        for m in self.modes:
            assert m in ("xs", "ts"), f"mode must be 'xs' or 'ts', got {m}"
        assert self.min_turnover_per_bar > 0
        assert self.cost_safety_margin >= 1.0
        assert 0 < self.min_positive_fold_ratio <= 1
        assert 0 < self.max_leg_weight <= 1
        assert 0 < self.max_name_weight <= 1
        assert self.warmup_folds >= 4, f"warmup_folds must be >= 4, got {self.warmup_folds}"
        assert self.min_cross_section > 0
        assert self.bars_per_year > 0
        assert self.n_bootstrap > 0
        assert 0 < self.min_growth_posterior_probability <= 1
        assert self.stress_cost_multiplier >= 1.0
        assert 0 < self.familywise_error_rate < 1, f"familywise_error_rate must be in (0,1), got {self.familywise_error_rate}"
        assert self.shrinkage_prior_obs > 0, f"shrinkage_prior_obs must be > 0, got {self.shrinkage_prior_obs}"
        assert self.leg_prior_lookback_bars > 0, f"leg_prior_lookback_bars must be > 0, got {self.leg_prior_lookback_bars}"
        assert 0 <= self.handoff_posterior_floor < self.min_growth_posterior_probability, (
            f"handoff_posterior_floor={self.handoff_posterior_floor} must be in "
            f"[0, {self.min_growth_posterior_probability})"
        )
        assert self.prior_only_folds >= 1, f"prior_only_folds must be >= 1, got {self.prior_only_folds}"


@dataclass(slots=True, frozen=True)
class HandoffConfig:
    max_pairwise_correlation: float = 0.80
    min_positive_outer_folds: int = 4
    target_ann_vol: float = 0.12
    max_ann_vol: float = 0.12
    max_drawdown: float = 0.15
    cost_stress_multiplier: float = 2.0
    n_bootstrap: int = 1_000
    dedup_rho_threshold: float = 0.90
    min_dedup_observations: int = 1_000
    min_sleeve_posterior_probability: float = 0.95
    min_oos_posterior_probability: float = 0.55
    min_oos_effective_blocks: int = 5
    hac_lag_cap: int = 120
    family_screen_alpha: float = 0.05
    min_family_ic_samples: int = 30
    min_growth_posterior_probability: float = 0.90
    screen_cost_bps: float = 8.0

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
        assert 0.5 < self.min_sleeve_posterior_probability < 1.0
        assert self.hac_lag_cap >= 1
        assert 0 < self.family_screen_alpha <= 1
        assert self.min_family_ic_samples >= 1
        assert 0.5 < self.min_growth_posterior_probability < 1.0
        assert self.screen_cost_bps >= 0


@dataclass(slots=True, frozen=True)
class RegimeRouterConfig:
    trend_lookback_bars: int = 42
    regime_history_bars: int = 126
    min_dwell_bars: int = 12
    stress_enter_quantile: float = 0.80
    stress_exit_quantile: float = 0.70
    trend_enter_tstat: float = 1.25
    trend_exit_tstat: float = 0.75
    min_effective_blocks: int = 20
    min_evidence_bars: int = 900
    min_posterior_probability: float = 0.90
    max_expert_weight: float = 0.50
    n_bootstrap: int = 1_000
    regime_overlay_floor: float = 0.5

    def __post_init__(self) -> None:
        assert self.trend_lookback_bars > 0
        assert self.regime_history_bars > self.trend_lookback_bars
        assert self.min_dwell_bars > 0
        assert 0 < self.stress_exit_quantile < self.stress_enter_quantile < 1
        assert self.trend_enter_tstat > self.trend_exit_tstat > 0
        assert self.min_effective_blocks > 0
        assert self.min_evidence_bars > 0
        assert 0.5 < self.min_posterior_probability < 1
        assert 0 < self.max_expert_weight <= 1
        assert self.n_bootstrap > 0
        assert 0 < self.regime_overlay_floor <= 1.0


@dataclass(slots=True, frozen=True)
class CompoundEngineConfig:
    strategy_code_version: str = "compound-2026-07-26"
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
    l1_leg: L1LegConfig = field(default_factory=L1LegConfig)
    dynamic_compounding: DynamicCompoundingConfig = field(default_factory=DynamicCompoundingConfig)
    regime_router: RegimeRouterConfig = field(default_factory=RegimeRouterConfig)
    l1_routing: L1RoutingConfig = field(default_factory=L1RoutingConfig)

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
        assert isinstance(self.l1_leg, L1LegConfig)
        assert isinstance(self.dynamic_compounding, DynamicCompoundingConfig)
        assert isinstance(self.regime_router, RegimeRouterConfig)
        assert isinstance(self.l1_routing, L1RoutingConfig)
