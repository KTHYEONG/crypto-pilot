from __future__ import annotations

from src.futures_strategy.sizing.registry import FUTURES_SIZING_REGISTRY, register_futures_sizing

from src.futures_strategy.sizing import inv_vol_parity_futures  # noqa: F401
from src.futures_strategy.sizing import profit_factor_kelly_futures  # noqa: F401
from src.futures_strategy.sizing import vol_target_futures  # noqa: F401

__all__ = [
    "FUTURES_SIZING_REGISTRY",
    "register_futures_sizing",
]
