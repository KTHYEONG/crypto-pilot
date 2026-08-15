from __future__ import annotations

from typing import TypeVar

from src.domain.spot.regimes.base import IRegime

REGIME_REGISTRY: dict[str, IRegime] = {}

T = TypeVar("T")


def register_regime(cls: type[T]) -> type[T]:
    inst = cls()
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define non-empty str name")
    REGIME_REGISTRY[name] = inst  # type: ignore[assignment]
    return cls
