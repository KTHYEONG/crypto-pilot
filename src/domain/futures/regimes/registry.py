from __future__ import annotations

from typing import Dict, Type, TypeVar

from src.domain.futures.regimes.base import IFuturesRegime

FUTURES_REGIME_REGISTRY: Dict[str, IFuturesRegime] = {}

T = TypeVar("T")


def register_regime(cls: Type[T]) -> Type[T]:
    inst = cls()
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define non-empty str name")
    FUTURES_REGIME_REGISTRY[name] = inst  # type: ignore[assignment]
    return cls
