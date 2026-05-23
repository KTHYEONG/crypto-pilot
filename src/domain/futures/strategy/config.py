from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.core.settings import SLIPPAGE_BPS, TAKER_FEE_BPS


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

    name: str = "momentum_v0"
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    sleeves: SleeveConfig = field(default_factory=SleeveConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    ml: StrategyMLConfig = field(default_factory=lambda: StrategyMLConfig())

    def __post_init__(self) -> None:
        """Validate top-level strategy name."""
        if self.name not in {"momentum_v0", "eh_st_v1", "ml_lambdamart_v1", "xs_reversal"}:
            raise ValueError(f"unsupported strategy name: {self.name}")


@dataclass(slots=True, frozen=True)
class StrategyMLConfig:
    """ML strategy configuration for LambdaMART + quantile calibrator."""

    name: Literal["ml_lambdamart_v1"] = "ml_lambdamart_v1"
    timeframe: str = "4h"
    seed: int = 42
    n_jobs: int = 4
    min_group_size: int = 8
    label_horizon_bars: int = 6
    train_months: int = 24
    valid_months: int = 3
    test_months: int = 3
    purge_bars: int = 6
    embargo_bars: int = 1
    max_features: int = 64
    alpha_clip_bps: float = 75.0
    lambda_tail: float = 0.25
    ranker_n_estimators: int = 800
    calibrator_n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.80
    bagging_fraction: float = 0.80
    lambda_l2: float = 5.0
    early_stopping_rounds: int = 75
    # per-side 비용: 레이블 생성 시 round-trip(x2)으로 환산됨 (labels.py 참조)
    fee_bps: float = TAKER_FEE_BPS       # Taker 수수료 per side (canonical: core/settings.py)
    slippage_bps: float = SLIPPAGE_BPS   # 슬리피지 per side (canonical: core/settings.py)

    def __post_init__(self) -> None:
        """Validate ML strategy parameters."""
        if self.purge_bars < self.label_horizon_bars:
            raise ValueError("purge_bars must be >= label_horizon_bars")
        if self.embargo_bars < 1:
            raise ValueError("embargo_bars must be >= 1")
        if self.max_features > 64:
            raise ValueError("max_features must be <= 64")
        if self.alpha_clip_bps <= 0.0:
            raise ValueError("alpha_clip_bps must be > 0")
        if self.min_group_size < 2:
            raise ValueError("min_group_size must be >= 2")
        if not (0.0 < self.learning_rate <= 0.2):
            raise ValueError("learning_rate must satisfy 0 < lr <= 0.2")
        if self.num_leaves > 31:
            raise ValueError("num_leaves must be <= 31")
        if self.max_depth > 6:
            raise ValueError("max_depth must be <= 6")
        if self.min_data_in_leaf < 10:
            raise ValueError("min_data_in_leaf must be >= 10")
