from __future__ import annotations

from typing import TypeVar

from src.domain.spot.signals.base import ISignal

SIGNAL_REGISTRY: dict[str, ISignal] = {}

T = TypeVar("T")


def register_signal(cls: type[T]) -> type[T]:
    inst = cls()
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define non-empty str name")
    SIGNAL_REGISTRY[name] = inst  # type: ignore[assignment]
    return cls


def _load_signal_plugin_modules() -> None:
    """Import all signal modules so @register_signal runs.

    Callers that only do ``from ...registry import SIGNAL_REGISTRY`` must still
    populate the registry; importing the package __init__ would recurse here, so
    we load leaf modules directly.
    """
    from importlib import import_module

    _plugin_modules: tuple[str, ...] = (
        "src.domain.spot.signals.adx_breakout",
        "src.domain.spot.signals.bb_squeeze",
        "src.domain.spot.signals.frama_evr_signal",
        "src.domain.spot.signals.kc_pullback",
        "src.domain.spot.signals.macd_hist_div",
        "src.domain.spot.signals.obv_ma_breakout",
        "src.domain.spot.signals.rsi2_pullback",
        "src.domain.spot.signals.rs_momentum",
        "src.domain.spot.signals.stochrsi_cross",
        "src.domain.spot.signals.supertrend_signal",
        "src.domain.spot.signals.vix_fix",
    )
    for _mod in _plugin_modules:
        import_module(_mod)


_load_signal_plugin_modules()
