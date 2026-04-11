from __future__ import annotations

from src.domain.spot.signals import (
    adx_breakout,
    bb_squeeze,
    frama_evr_signal,
    kc_pullback,
    macd_hist_div,
    obv_ma_breakout,
    rs_momentum,
    rsi2_pullback,
    stochrsi_cross,
    supertrend_signal,
    vix_fix,
)
from src.domain.spot.signals.base import ISignal, SignalOutput
from src.domain.spot.signals.registry import SIGNAL_REGISTRY, register_signal

__all__ = [
    "SIGNAL_REGISTRY",
    "ISignal",
    "SignalOutput",
    "register_signal",
]
