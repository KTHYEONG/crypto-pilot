from __future__ import annotations

from src.spot_strategy.signals.registry import SIGNAL_REGISTRY, register_signal

from src.spot_strategy.signals import adx_breakout  # noqa: F401
from src.spot_strategy.signals import bb_squeeze  # noqa: F401
from src.spot_strategy.signals import frama_evr_signal  # noqa: F401
from src.spot_strategy.signals import kc_pullback  # noqa: F401
from src.spot_strategy.signals import macd_hist_div  # noqa: F401
from src.spot_strategy.signals import obv_ma_breakout  # noqa: F401
from src.spot_strategy.signals import rsi2_pullback  # noqa: F401
from src.spot_strategy.signals import rs_momentum  # noqa: F401
from src.spot_strategy.signals import stochrsi_cross  # noqa: F401
from src.spot_strategy.signals import supertrend_signal  # noqa: F401
from src.spot_strategy.signals import vix_fix  # noqa: F401
from src.spot_strategy.signals.base import ISignal, SignalOutput

__all__ = [
    "ISignal",
    "SIGNAL_REGISTRY",
    "SignalOutput",
    "register_signal",
]
