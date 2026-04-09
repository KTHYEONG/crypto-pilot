from __future__ import annotations

from src.domain.spot.signals.registry import SIGNAL_REGISTRY, register_signal

from src.domain.spot.signals import adx_breakout  # noqa: F401
from src.domain.spot.signals import bb_squeeze  # noqa: F401
from src.domain.spot.signals import frama_evr_signal  # noqa: F401
from src.domain.spot.signals import kc_pullback  # noqa: F401
from src.domain.spot.signals import macd_hist_div  # noqa: F401
from src.domain.spot.signals import obv_ma_breakout  # noqa: F401
from src.domain.spot.signals import rsi2_pullback  # noqa: F401
from src.domain.spot.signals import rs_momentum  # noqa: F401
from src.domain.spot.signals import stochrsi_cross  # noqa: F401
from src.domain.spot.signals import supertrend_signal  # noqa: F401
from src.domain.spot.signals import vix_fix  # noqa: F401
from src.domain.spot.signals.base import ISignal, SignalOutput

__all__ = [
    "ISignal",
    "SIGNAL_REGISTRY",
    "SignalOutput",
    "register_signal",
]
