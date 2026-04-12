"""Futures regime registry (EMA-ATR, funding, trend quality, breadth, none)."""

from __future__ import annotations

# Side-effect: populate registry
from src.domain.futures.regimes import (
    adaptive_vol_futures,
    ema_atr_futures,
    # funding_rate_regime, # Removed per redesign plan
    # market_breadth_futures, # Removed per redesign plan
    none_futures,
    trend_quality_futures,
)
from src.domain.futures.regimes.registry import FUTURES_REGIME_REGISTRY, register_regime

__all__ = [
    "FUTURES_REGIME_REGISTRY",
    "register_regime",
]
