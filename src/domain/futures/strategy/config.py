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
    min_listing_age_days: int = 90
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
    candidate_families: tuple[str, ...] = (
        "trend_ma",
        "trend_donchian",
        "vol_breakout",
        "bollinger_reversion",
        "rsi_reversion",
        "funding_carry",
        "oi_volume_impulse",
        "btc_regime_pullback",
    )

    def __post_init__(self) -> None:
        """Validate candidate strategy parameters."""
        if self.name not in {"candidate_ml", "rule_baseline"}:
            raise ValueError("candidate strategy name must be 'candidate_ml' or 'rule_baseline'")
        if self.train_months <= 0 or self.valid_months <= 0 or self.test_months <= 0:
            raise ValueError("all month windows must be positive")
        if self.purge_bars < 0 or self.embargo_bars < 0:
            raise ValueError("purge and embargo bars must be non-negative")
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
        ):
            raise ValueError("cap and penalty parameters must be non-negative")
        if self.gross_cap < self.max_symbol_weight:
            raise ValueError("gross cap must be at least max symbol weight")
        if self.min_candidate_obs <= 0 or self.min_symbol_oos_blocks <= 0:
            raise ValueError("minimum observations and blocks must be positive")
