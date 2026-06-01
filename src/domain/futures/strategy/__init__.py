"""Public exports for the futures strategy package."""

from __future__ import annotations

from typing import Any

from src.domain.futures.strategy.config import (
    CandidateStrategyConfig,
    MomentumConfig,
    StrategyConfig,
    StrategyMLConfig,
)

__all__ = [
    "CandidateStrategyConfig",
    "MomentumConfig",
    "StrategyConfig",
    "StrategyMLConfig",
    "build_strategy_alpha",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve heavy exports to avoid eager import side effects."""
    if name == "build_strategy_alpha":
        from src.domain.futures.strategy.builder import build_strategy_alpha

        return build_strategy_alpha
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
