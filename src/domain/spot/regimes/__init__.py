from __future__ import annotations

from src.domain.spot.regimes.registry import REGIME_REGISTRY, register_regime

from src.domain.spot.regimes import ema_atr  # noqa: F401
from src.domain.spot.regimes import market_breadth  # noqa: F401
from src.domain.spot.regimes import trend_quality  # noqa: F401
from src.domain.spot.regimes.base import IRegime

__all__ = ["IRegime", "REGIME_REGISTRY", "register_regime"]
