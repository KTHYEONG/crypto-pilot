from __future__ import annotations

from typing import Dict, Type, TypeVar

from src.domain.futures.sizing.base import IFuturesSizing

FUTURES_SIZING_REGISTRY: Dict[str, IFuturesSizing] = {}

T = TypeVar("T")


def register_futures_sizing(cls: Type[T]) -> Type[T]:
    inst = cls()
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define non-empty str name")
    FUTURES_SIZING_REGISTRY[name] = inst  # type: ignore[assignment]
    return cls
