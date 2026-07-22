from __future__ import annotations

from dataclasses import dataclass, field


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
class CompoundEngineConfig:
    data: DataPlaneConfig = field(default_factory=DataPlaneConfig)
    l1: L1EstimatorConfig = field(default_factory=L1EstimatorConfig)
    factor_risk: FactorRiskConfig = field(default_factory=FactorRiskConfig)
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)
    risk: RiskOverlayConfig = field(default_factory=RiskOverlayConfig)
    l3: L3ValidationConfig = field(default_factory=L3ValidationConfig)

    def __post_init__(self) -> None:
        assert isinstance(self.data, DataPlaneConfig)
        assert isinstance(self.l1, L1EstimatorConfig)
        assert isinstance(self.factor_risk, FactorRiskConfig)
        assert isinstance(self.allocator, AllocatorConfig)
        assert isinstance(self.risk, RiskOverlayConfig)
        assert isinstance(self.l3, L3ValidationConfig)
