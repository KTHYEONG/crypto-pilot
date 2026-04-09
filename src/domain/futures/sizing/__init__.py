from __future__ import annotations

from src.domain.futures.sizing.registry import FUTURES_SIZING_REGISTRY, register_futures_sizing

from src.domain.futures.sizing import inv_vol_parity_futures  # noqa: F401
from src.domain.futures.sizing import profit_factor_kelly_futures  # noqa: F401
from src.domain.futures.sizing import vol_target_futures  # noqa: F401

__all__ = [
    "FUTURES_SIZING_REGISTRY",
    "register_futures_sizing",
]
