"""OHLCV strategy-aware execution replay (delegates to the causal window engine).

``strategy_aware_execution_replay`` is a thin single-panel adapter: it keeps
the historical input validation, builds one ``ExecutionReplayWindow``, and
delegates to ``replay_execution_windows`` so exactly one accounting
implementation exists (migration oracle retirement). Independent verification
of the economics lives in ``simulate_inventory_ledger`` recomputation plus
the hand-calculated regression tests, not in a second replay formula.
"""

from __future__ import annotations

import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.execution.batch import replay_execution_windows
from src.mhs.types import ExecutionSpec

from . import _ExecutionBound
from .contracts import (
    ExecutionReplayWindow,
    StrategyExecutionReplayResult,
)


def strategy_aware_execution_replay(
    target_weights: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    minute_highs: pd.DataFrame,
    minute_lows: pd.DataFrame,
    minute_closes: pd.DataFrame,
    minute_marks: pd.DataFrame | None,
    bar_funding: pd.DataFrame,
    initial_equity: float,
    execution_bound: _ExecutionBound,
    spec: ExecutionSpec,
) -> StrategyExecutionReplayResult:
    """Replay the target into timestamp-sorted proxy fills and an inventory ledger.

    The replay is the single timestamp-sorted proxy-event loop: before each
    decision it marks simulated units and applies funding accrued since the
    prior event, then converts target notional at the decision mark into
    desired units using current ledger equity, subtracts simulated units, and
    nets opposite fast/slow intents before any market intent is created. It
    must not create a bar-wise target-weight path that implicitly rebalances
    without a proxy event.

    Units and last-price state are aligned NumPy vectors; mark-to-market and
    interval funding use masked vector operations and intents are created only
    for the active columns (finite non-zero targets plus non-zero held units),
    so the work scales with the active roster instead of the full union width.

    Live forward collection (Phase 4B) records one ``ForwardExecutionObservation``
    per signal intent; this OHLCV replay cannot observe queue position, partial
    fills, or rejections, so those assumptions are reported as unsupported.
    """
    if initial_equity <= 0:
        raise DataIntegrityError("initial_equity must be > 0")
    if execution_bound not in (
        "OHLCV_STRICT_PROXY",
        "OHLCV_TOUCH_PROXY",
        "OHLCV_IMMEDIATE_TAKER",
        "OHLCV_LADDERED_PROXY",
        "OHLCV_PEG_CHASE_PROXY",
    ):
        raise ValueError(f"unknown execution_bound '{execution_bound}'")
    if len(target_weights) != len(signal_available_at):
        raise DataIntegrityError("signal_available_at must align with target_weights")
    if not (
        minute_highs.index.equals(minute_lows.index)
        and minute_highs.index.equals(minute_closes.index)
    ):
        raise DataIntegrityError("minute frames must share an identical index")
    if (
        list(minute_highs.columns) != list(minute_lows.columns)
        or list(minute_highs.columns) != list(minute_closes.columns)
    ):
        raise DataIntegrityError("minute frames must share an identical column order")
    for sym in target_weights.columns:
        if sym not in minute_highs.columns:
            raise DataIntegrityError(f"target references unavailable symbol {sym}")
    if minute_marks is not None:
        if (
            not minute_marks.index.equals(minute_closes.index)
            or list(minute_marks.columns) != list(minute_closes.columns)
        ):
            raise DataIntegrityError("minute_marks must exactly align to minute_closes")
        marks: pd.DataFrame = minute_marks
    else:
        marks = minute_closes

    minute_grid = minute_closes.index
    if not bar_funding.index.equals(minute_grid):
        raise DataIntegrityError("bar_funding must align exactly to the minute grid")
    symbols = list(target_weights.columns)
    window = ExecutionReplayWindow(
        window_start=minute_grid[0],
        window_end=minute_grid[-1],
        columns=tuple(symbols),
        symbols=tuple(symbols),
        minute_grid=minute_grid,
        highs=minute_highs[symbols],
        lows=minute_lows[symbols],
        closes=minute_closes[symbols],
        marks=marks[symbols] if minute_marks is not None else None,
        bar_funding=bar_funding[symbols],
        target_weights=target_weights,
        signal_available_at=signal_available_at,
    )
    return _delegate_single_panel_window(
        window, initial_equity=initial_equity, execution_bound=execution_bound, spec=spec,
        min_equity_fraction=None,
    )


def _delegate_single_panel_window(
    window: ExecutionReplayWindow,
    *,
    initial_equity: float,
    execution_bound: _ExecutionBound,
    spec: ExecutionSpec,
    min_equity_fraction: float | None,
) -> StrategyExecutionReplayResult:
    """Delegate one single-panel window to the causal window engine.

    The legacy single-panel oracle retires as an independent accounting
    implementation: the window batch API owns the economics, so oracle and
    windowed replays agree by construction instead of by duplicated formula.
    """
    return replay_execution_windows(
        (window,), initial_equity, execution_bound, spec,
        retain_event_snapshots=True, min_equity_fraction=min_equity_fraction,
    )