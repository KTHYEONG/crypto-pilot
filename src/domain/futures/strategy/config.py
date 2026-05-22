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
class StrategyConfig:
    """Top-level strategy switch."""

    name: str = "momentum_v0"
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
