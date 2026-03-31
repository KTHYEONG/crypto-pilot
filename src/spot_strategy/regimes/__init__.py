from __future__ import annotations

from src.spot_strategy.regimes.registry import REGIME_REGISTRY, register_regime

from src.spot_strategy.regimes import ema_atr  # noqa: F401
from src.spot_strategy.regimes import market_breadth  # noqa: F401
from src.spot_strategy.regimes import trend_quality  # noqa: F401
from src.spot_strategy.regimes.base import IRegime

__all__ = ["IRegime", "REGIME_REGISTRY", "register_regime"]
