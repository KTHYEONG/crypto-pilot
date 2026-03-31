from __future__ import annotations

from src.spot_strategy.sizing.registry import SIZING_REGISTRY, register_sizing

from src.spot_strategy.sizing import inv_vol_parity  # noqa: F401
from src.spot_strategy.sizing import liquidity_adjusted  # noqa: F401
from src.spot_strategy.sizing import profit_factor_kelly  # noqa: F401
from src.spot_strategy.sizing import rolling_kelly  # noqa: F401
from src.spot_strategy.sizing import vol_target  # noqa: F401
from src.spot_strategy.sizing.base import ISizing

__all__ = ["ISizing", "SIZING_REGISTRY", "register_sizing"]
