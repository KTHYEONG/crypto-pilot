from __future__ import annotations

from typing import Any, Dict, Type, TypeVar

from src.spot_strategy.signals.base import ISignal

SIGNAL_REGISTRY: Dict[str, ISignal] = {}

T = TypeVar("T")


def register_signal(cls: Type[T]) -> Type[T]:
    inst = cls()
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define non-empty str name")
    SIGNAL_REGISTRY[name] = inst  # type: ignore[assignment]
    return cls


def _load_signal_plugin_modules() -> None:
    """
    Import all signal modules so @register_signal runs.

    Callers that only do ``from ...registry import SIGNAL_REGISTRY`` must still
    populate the registry; importing the package __init__ would recurse here, so
    we load leaf modules directly.
    """
    from importlib import import_module

    _plugin_modules: tuple[str, ...] = (
        "src.spot_strategy.signals.adx_breakout",
        "src.spot_strategy.signals.bb_squeeze",
        "src.spot_strategy.signals.frama_evr_signal",
        "src.spot_strategy.signals.kc_pullback",
        "src.spot_strategy.signals.macd_hist_div",
        "src.spot_strategy.signals.obv_ma_breakout",
        "src.spot_strategy.signals.rsi2_pullback",
        "src.spot_strategy.signals.rs_momentum",
        "src.spot_strategy.signals.stochrsi_cross",
        "src.spot_strategy.signals.supertrend_signal",
        "src.spot_strategy.signals.vix_fix",
    )
    for _mod in _plugin_modules:
        import_module(_mod)


_load_signal_plugin_modules()
