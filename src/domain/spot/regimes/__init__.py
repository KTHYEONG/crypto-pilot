from __future__ import annotations

from src.domain.spot.regimes import (
    ema_atr,
    market_breadth,
    trend_quality,
)
from src.domain.spot.regimes.base import IRegime
from src.domain.spot.regimes.registry import REGIME_REGISTRY, register_regime

__all__ = ["REGIME_REGISTRY", "IRegime", "register_regime"]
