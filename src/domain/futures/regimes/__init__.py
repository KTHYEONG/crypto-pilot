"""Futures regime registry (EMA-ATR, funding, trend quality, breadth, none)."""

from __future__ import annotations

from src.domain.futures.regimes.registry import FUTURES_REGIME_REGISTRY, register_regime

# Side-effect: populate registry
from src.domain.futures.regimes import ema_atr_futures  # noqa: F401
from src.domain.futures.regimes import funding_rate_regime  # noqa: F401
from src.domain.futures.regimes import market_breadth_futures  # noqa: F401
from src.domain.futures.regimes import none_futures  # noqa: F401
from src.domain.futures.regimes import trend_quality_futures  # noqa: F401

__all__ = [
    "FUTURES_REGIME_REGISTRY",
    "register_regime",
]
