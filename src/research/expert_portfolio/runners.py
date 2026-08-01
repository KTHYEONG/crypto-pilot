"""Central runner-key dispatch table for causal expert component runs.

``resolve_component_runner`` is the only place a runner key maps to a callable.
The application layer never branches on a runner key: it resolves the runner,
loads the causal data declared for that runner, and invokes it.  Unknown runner
keys fail closed before any data execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from src.research.baseline.backtest import BacktestResult, run_backtest, run_directional_backtest
from src.research.contracts import CostModel, StrategySpec
from src.research.expert_portfolio.contracts import ExpertDefinition

ComponentRunner = Callable[[ExpertDefinition, Mapping[str, pd.DataFrame], "ComponentRunRequest"], BacktestResult]


@dataclass(frozen=True, slots=True)
class ComponentRunRequest:
    """Immutable causal inputs shared by every component runner."""

    costs: CostModel
    signal_delay_bars: int = 0
    start: str | None = None
    end: str | pd.Timestamp | None = None


def _run_single_symbol_backtest(
    definition: ExpertDefinition,
    data: Mapping[str, pd.DataFrame],
    request: ComponentRunRequest,
) -> BacktestResult:
    symbol = definition.symbols[0]
    return run_backtest(
        data["ohlcv"],
        StrategySpec(symbol=symbol),
        request.costs,
        signal_delay_bars=request.signal_delay_bars,
    )


def _run_single_symbol_directional(
    definition: ExpertDefinition,
    data: Mapping[str, pd.DataFrame],
    request: ComponentRunRequest,
) -> BacktestResult:
    symbol = definition.symbols[0]
    return run_directional_backtest(
        data["ohlcv"],
        StrategySpec(symbol=symbol),
        request.costs,
        data["funding"],
        signal_delay_bars=request.signal_delay_bars,
    )


_RUNNERS: dict[str, ComponentRunner] = {
    "run_backtest": _run_single_symbol_backtest,
    "run_directional_backtest": _run_single_symbol_directional,
}

_RUNNER_DATA_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "run_backtest": ("ohlcv",),
    "run_directional_backtest": ("ohlcv", "funding"),
}


def resolve_component_runner(runner_key: str) -> ComponentRunner:
    """Resolve a registered runner key or fail closed before data execution."""
    try:
        return _RUNNERS[runner_key]
    except KeyError as exc:
        raise ValueError(f"runner '{runner_key}' is not registered") from exc


def component_data_requirements(runner_key: str) -> tuple[str, ...]:
    """Return the causal data slots a runner requires from the component context."""
    try:
        return _RUNNER_DATA_REQUIREMENTS[runner_key]
    except KeyError as exc:
        raise ValueError(f"runner '{runner_key}' is not registered") from exc
