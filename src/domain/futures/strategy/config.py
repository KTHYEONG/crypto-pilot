from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class MomentumConfig:
    """XS momentum sleeve parameters."""

    lookback_bars: int = 6
    top_ratio: float = 0.30
    bottom_ratio: float = 0.30
    min_symbols_for_xs: int = 5
    edge_scale_per_bar: float = 1e-3

    def __post_init__(self) -> None:
        """Validate momentum parameter bounds."""
        if self.lookback_bars < 1:
            raise ValueError("lookback_bars must be >= 1")
        if not (0.0 < self.top_ratio <= 0.5):
            raise ValueError("top_ratio must satisfy 0 < top_ratio <= 0.5")
        if not (0.0 < self.bottom_ratio <= 0.5):
            raise ValueError("bottom_ratio must satisfy 0 < bottom_ratio <= 0.5")


@dataclass(slots=True, frozen=True)
class SleeveConfig:
    """Enhanced strategy multi-sleeve switch and parameter settings."""

    # TS momentum disabled: negative IC at all tested horizons (4h t=-6.8, 1d t=-2.3)
    ts_momentum_enabled: bool = False
    ts_momentum_lookback: int = 36
    ts_momentum_skip: int = 1
    reversal_enabled: bool = True
    reversal_lookback: int = 6
    carry_enabled: bool = True
    carry_smooth: int = 6

    def __post_init__(self) -> None:
        """Validate sleeve parameters."""
        if self.ts_momentum_lookback < 1:
            raise ValueError("ts_momentum_lookback must be >= 1")
        if self.ts_momentum_skip < 0:
            raise ValueError("ts_momentum_skip must be >= 0")
        if self.reversal_lookback < 1:
            raise ValueError("reversal_lookback must be >= 1")
        if self.carry_smooth < 1:
            raise ValueError("carry_smooth must be >= 1")


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

    enabled: bool = True
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
            raise ValueError("volatility percentiles must satisfy 0 < vol_high_pct < vol_crisis_pct < 1.0")
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

    name: str = "momentum_v0"
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    sleeves: SleeveConfig = field(default_factory=SleeveConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

