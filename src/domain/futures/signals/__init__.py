"""Futures signal plugins registry."""
from __future__ import annotations

from src.domain.futures.signals.registry import FUTURES_SIGNAL_REGISTRY, register_futures_signal

# Side-effect imports populate registry
from src.domain.futures.signals import adx_breakout_futures  # noqa: F401
from src.domain.futures.signals import bb_squeeze_futures  # noqa: F401
from src.domain.futures.signals import macd_hist_div_futures  # noqa: F401
from src.domain.futures.signals import rsm_vt_futures  # noqa: F401
from src.domain.futures.signals import supertrend_futures  # noqa: F401

__all__ = [
    "FUTURES_SIGNAL_REGISTRY",
    "register_futures_signal",
]
