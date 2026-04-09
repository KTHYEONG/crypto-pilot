from __future__ import annotations

from typing import Dict, Type, TypeVar

from src.domain.futures.signals.base import IFuturesSignal

FUTURES_SIGNAL_REGISTRY: Dict[str, IFuturesSignal] = {}

T = TypeVar("T")


def register_futures_signal(cls: Type[T]) -> Type[T]:
    inst = cls()
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define non-empty str name")
    FUTURES_SIGNAL_REGISTRY[name] = inst  # type: ignore[assignment]
    return cls
