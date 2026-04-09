from __future__ import annotations

from typing import Dict, Type, TypeVar

from src.domain.spot.sizing.base import ISizing

SIZING_REGISTRY: Dict[str, ISizing] = {}

T = TypeVar("T")


def register_sizing(cls: Type[T]) -> Type[T]:
    inst = cls()
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define non-empty str name")
    SIZING_REGISTRY[name] = inst  # type: ignore[assignment]
    return cls
