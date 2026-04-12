"""Futures signal plugins registry."""

from __future__ import annotations

# Side-effect imports populate registry
from src.domain.futures.signals import (
    adx_breakout_futures,
    # bb_squeeze_futures, # Removed per redesign plan
    cs_momentum_futures,
    macd_hist_div_futures,
    rsm_vt_futures,
    supertrend_futures,
    vwap_mr_futures,
)
from src.domain.futures.signals.registry import FUTURES_SIGNAL_REGISTRY, register_futures_signal

__all__ = [
    "FUTURES_SIGNAL_REGISTRY",
    "register_futures_signal",
]
