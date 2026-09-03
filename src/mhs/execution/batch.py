"""Batch replay API over execution windows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace as dataclass_replace

import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.types import ExecutionSpec

from . import _ExecutionBound
from . import accumulator as _accumulator
from .accumulator import _BoundExecutionReplayAccumulator
from .contracts import (
    BatchReplayOutcome,
    ExecutionReplayWindow,
    IsolatedBoundFailure,
    StrategyExecutionReplayResult,
)


def replay_execution_windows(
    windows: Iterable[ExecutionReplayWindow],
    initial_equity: float,
    execution_bound: _ExecutionBound,
    spec: ExecutionSpec,
    retain_event_snapshots: bool = False,
    min_equity_fraction: float | None = None,
) -> StrategyExecutionReplayResult:
    """Stateful windowed replay equivalent to ``strategy_aware_execution_replay``.

    Windows are consumed one at a time through a private bound-specific
    accumulator: cash, units, last prices, the last finite-close mark
    provenance, and the streamed ledger carry into the next window, and a
    completed window's frames are released before the next is read. Each
    window's grid covers the strict timeout overlap of its final order plus
    the boundary bars needed for decision-time funding/MTM, so an order never
    crosses a window boundary unresolved. The six ledger series are computed
    per window in chronological order and concatenated once, matching the
    single-panel oracle at ``rtol=atol=1e-12`` where the inputs are equal.

    ``retain_event_snapshots`` defaults to ``False`` for bounded memory: the
    dense per-fill ``simulated_units``/``simulated_notional_weights`` event
    tables are then empty (correctly columned) and ``event_snapshots_retained``
    is ``False``, so empty tables cannot be mistaken for no fills. Diagnostic
    callers that compare event snapshots (the single-panel oracle and
    equivalence tests) must explicitly opt in with ``True``; the ledger, fills,
    gaps, termination data, and numerical results are identical either way.
    """
    it = iter(windows)
    first = next(it, None)
    if first is None:
        raise DataIntegrityError("at least one execution window is required")
    accumulator = _accumulator._BoundExecutionReplayAccumulator(
        first, initial_equity, execution_bound, spec, retain_event_snapshots, min_equity_fraction,
    )
    accumulator.consume(first)
    del first
    for w in it:
        accumulator.consume(w)
        del w
    return accumulator.finalize()


def replay_execution_window_batch_isolated(
    windows: Iterable[ExecutionReplayWindow],
    initial_equity: float,
    bounds: Iterable[tuple[_ExecutionBound, ExecutionSpec]],
    retain_event_snapshots: bool = False,
    min_equity_fraction: float | None = None,
    isolated_bound_indices: frozenset[int] = frozenset(),
) -> BatchReplayOutcome:
    bound_list = list(bounds)
    if not bound_list:
        raise ValueError("bounds must be non-empty")
    for idx in isolated_bound_indices:
        if idx < 0 or idx >= len(bound_list):
            raise ValueError(f"isolated index {idx} out of range for {len(bound_list)} bounds")
    it = iter(windows)
    first = next(it, None)
    if first is None:
        raise DataIntegrityError("at least one execution window is required")
    accumulators: list[_BoundExecutionReplayAccumulator | None] = [
        _accumulator._BoundExecutionReplayAccumulator(
            first, initial_equity, bound, spec, retain_event_snapshots, min_equity_fraction,
        )
        for (bound, spec) in bound_list
    ]
    active: list[bool] = [True] * len(bound_list)
    windows_consumed: list[int] = [0] * len(bound_list)
    failures: list[IsolatedBoundFailure] = []

    def _try_consume(idx: int, w: ExecutionReplayWindow) -> None:
        if not active[idx]:
            return
        try:
            assert accumulators[idx] is not None
            accumulators[idx].consume(w)  # type: ignore[union-attr]
            windows_consumed[idx] += 1
        except DataIntegrityError as exc:
            if idx in isolated_bound_indices:
                failures.append(
                    IsolatedBoundFailure(
                        bound_index=idx,
                        execution_bound=str(bound_list[idx][0]),
                        error_class=type(exc).__name__,
                        message=str(exc),
                        windows_consumed=windows_consumed[idx],
                    )
                )
                active[idx] = False
                accumulators[idx] = None
            else:
                raise

    for idx in range(len(bound_list)):
        _try_consume(idx, first)
    del first
    for w in it:
        for idx in range(len(bound_list)):
            _try_consume(idx, w)
        del w
    results: list[StrategyExecutionReplayResult | None] = []
    for idx in range(len(bound_list)):
        if not active[idx]:
            results.append(None)
            continue
        try:
            assert accumulators[idx] is not None
            results.append(accumulators[idx].finalize())  # type: ignore[union-attr]
        except DataIntegrityError as exc:
            if idx in isolated_bound_indices:
                failures.append(
                    IsolatedBoundFailure(
                        bound_index=idx,
                        execution_bound=str(bound_list[idx][0]),
                        error_class=type(exc).__name__,
                        message=str(exc),
                        windows_consumed=windows_consumed[idx],
                    )
                )
                results.append(None)
                active[idx] = False
                accumulators[idx] = None
            else:
                raise
    return BatchReplayOutcome(results=tuple(results), isolated_failures=tuple(failures))


def _rescale_window_weights(
    w: ExecutionReplayWindow,
    scale: pd.Series,
) -> ExecutionReplayWindow:
    """Apply the two-pass rescaling formula to one loaded window (verbatim)."""
    scaled = w.target_weights.mul(
        scale.reindex(w.target_weights.index, method="ffill").fillna(1.0),
        axis=0,
    )
    original_active = (
        w.target_weights.notna() & w.target_weights.ne(0.0)
    ).any(axis=0)
    scaled_active = (scaled.notna() & scaled.ne(0.0)).any(axis=0)
    if (
        list(scaled.columns) != list(w.target_weights.columns)
        or not bool((original_active == scaled_active).all())
    ):
        raise DataIntegrityError(
            "pnl-vol-target scaling changed a window's active roster; "
            "the scale must preserve the zero pattern across replay passes"
        )
    return dataclass_replace(w, target_weights=scaled)


def replay_execution_windows_coupled(
    windows: Iterable[ExecutionReplayWindow],
    initial_equity: float,
    reference_bound: tuple[_ExecutionBound, ExecutionSpec],
    scaled_bounds: Sequence[tuple[_ExecutionBound, ExecutionSpec]],
    scale_fn: Callable[[pd.Series], pd.Series],
    retain_event_snapshots: bool = False,
    min_equity_fraction: float | None = None,
    isolated_bound_indices: frozenset[int] = frozenset(),
) -> tuple[StrategyExecutionReplayResult, BatchReplayOutcome]:
    """One-pass coupled reference/rescaled replay (D1).

    One market-data stream: the reference accumulator consumes window W, the
    daily reference-return prefix is extended, ``scale_fn`` recomputes the
    full-prefix scale, W's target weights are rescaled with the exact two-pass
    formula, and every scaled bound consumes the SAME already-loaded W. This
    eliminates the second generation pass (measured at 72.7% of the two-pass
    cost) while peak residency stays one loaded window.

    HAZARD handled explicitly (fail-closed): a non-final window's grid_end is
    the last order's timeout bar, not a day boundary, so a decision day ``d``
    may need the return of day ``d-1`` while ``d-1``'s last bars sit beyond
    everything consumed so far. The gate checks whether W's own grid START is
    already covered by everything consumed strictly BEFORE W: a window's own
    span is internally contiguous, so only the boundary with the prior window
    can hide a gap. Checking decision days against coverage already widened
    by W's own consumption would make the guard vacuous (a window's own
    decisions always lie inside its own span, and W's own span provably
    completes any day it starts inside). If W starts strictly after the last
    consumed bar, ``DataIntegrityError`` is raised before W is consumed and
    the caller falls back to the exact two-pass path. The very first window
    is exempt (no prior window can have left a gap, and the reference's own
    first-day return is legitimately NaN). Only call this for
    streaming-capable modes (``is_streaming_scale_mode``); other modes keep
    the two-pass path.
    """
    bound_list = list(scaled_bounds)
    if not bound_list:
        raise ValueError("scaled_bounds must be non-empty")
    for idx in isolated_bound_indices:
        if idx < 0 or idx >= len(bound_list):
            raise ValueError(f"isolated index {idx} out of range for {len(bound_list)} bounds")
    it = iter(windows)
    first = next(it, None)
    if first is None:
        raise DataIntegrityError("at least one execution window is required")

    reference = _accumulator._BoundExecutionReplayAccumulator(
        first, initial_equity, reference_bound[0], reference_bound[1],
        retain_event_snapshots, min_equity_fraction,
    )
    scaled_accumulators: list[_BoundExecutionReplayAccumulator | None] = [
        _accumulator._BoundExecutionReplayAccumulator(
            first, initial_equity, bound, spec, retain_event_snapshots, min_equity_fraction,
        )
        for bound, spec in bound_list
    ]
    active: list[bool] = [True] * len(bound_list)
    windows_consumed: list[int] = [0] * len(bound_list)

    def _try_consume_scaled(idx: int, w: ExecutionReplayWindow) -> None:
        if not active[idx]:
            return
        try:
            assert scaled_accumulators[idx] is not None
            scaled_accumulators[idx].consume(w)  # type: ignore[union-attr]
            windows_consumed[idx] += 1
        except DataIntegrityError as exc:
            if idx in isolated_bound_indices:
                active[idx] = False
                scaled_accumulators[idx] = None
                _isolated_failures.append(
                    IsolatedBoundFailure(
                        bound_index=idx,
                        execution_bound=str(bound_list[idx][0]),
                        error_class=type(exc).__name__,
                        message=str(exc),
                        windows_consumed=windows_consumed[idx],
                    )
                )
            else:
                raise

    _isolated_failures: list[IsolatedBoundFailure] = []
    coverage_ns = -1

    def _couple_window(w: ExecutionReplayWindow) -> None:
        nonlocal coverage_ns
        # 1) Fail-closed completeness gate: a window is internally contiguous
        #    (a single date_range), so its own span always finishes any day it
        #    starts inside -- the only place a gap can hide is the BOUNDARY
        #    with the prior window. If W starts strictly after everything
        #    consumed so far, the reference prefix (and therefore any day's
        #    resampled return spanning that boundary) has a hole; checking
        #    decision days against coverage already widened by W's own
        #    consumption would make the guard vacuous, since a window's own
        #    decisions always lie inside its own span.
        prior_coverage_ns = coverage_ns
        window_start_ns = int(w.minute_grid[0].value)
        if prior_coverage_ns >= 0 and window_start_ns > prior_coverage_ns:
            raise DataIntegrityError(
                "coupled replay: window starting "
                f"{w.window_start.isoformat()} begins after the last consumed "
                f"bar (coverage={prior_coverage_ns}); the reference prefix has "
                "a gap -- fall back to the exact two-pass path"
            )
        # 2) The unscaled reference consumes W now that the gate has cleared.
        reference.consume(w)
        window_end_ns = int(w.minute_grid[-1].value)
        coverage_ns = max(coverage_ns, window_end_ns)
        # 3) Recompute the full-prefix scale from the reference equity chunks.
        equity_prefix = (
            pd.concat(
                [
                    pd.Series(chunk, index=times)
                    for chunk, times in zip(
                        reference.equity_chunks, reference.equity_times, strict=True,
                    )
                ],
            )
            if reference.equity_chunks
            else pd.Series(dtype="float64")
        )
        daily_returns = equity_prefix.resample("1D").last().pct_change()
        scale = scale_fn(daily_returns)
        # 4) Rescale W in place (exact two-pass formula) and fan it to the
        #    already-constructed scaled bounds.
        rescaled = _rescale_window_weights(w, scale)
        for idx in range(len(bound_list)):
            _try_consume_scaled(idx, rescaled)

    coverage_ns = -1
    _couple_window(first)
    del first
    for w in it:
        _couple_window(w)
        del w

    reference_result = reference.finalize()
    results: list[StrategyExecutionReplayResult | None] = []
    for idx in range(len(bound_list)):
        if not active[idx]:
            results.append(None)
            continue
        try:
            assert scaled_accumulators[idx] is not None
            results.append(scaled_accumulators[idx].finalize())  # type: ignore[union-attr]
        except DataIntegrityError as exc:
            if idx in isolated_bound_indices:
                _isolated_failures.append(
                    IsolatedBoundFailure(
                        bound_index=idx,
                        execution_bound=str(bound_list[idx][0]),
                        error_class=type(exc).__name__,
                        message=str(exc),
                        windows_consumed=windows_consumed[idx],
                    )
                )
                results.append(None)
                active[idx] = False
                scaled_accumulators[idx] = None
            else:
                raise
    return reference_result, BatchReplayOutcome(
        results=tuple(results), isolated_failures=tuple(_isolated_failures),
    )


def replay_execution_window_batch(
    windows: Iterable[ExecutionReplayWindow],
    initial_equity: float,
    bounds: Iterable[tuple[_ExecutionBound, ExecutionSpec]],
    retain_event_snapshots: bool = False,
    min_equity_fraction: float | None = None,
) -> tuple[StrategyExecutionReplayResult, ...]:
    """Replay one shared window stream into N independent bounds.

    The N ``(execution_bound, spec)`` pairs consume identical immutable market
    windows; only their state, fill rule, and cost spec differ. Each yielded
    window is consumed by every bound's accumulator, then released before the
    next is requested, so a single loaded ``ExecutionReplayWindow`` stays the
    memory boundary and the window iterator is exhausted exactly once (never
    materialized or recreated). Every bound's accumulator is byte-identical to
    the single-bound ``replay_execution_windows`` path. A fatal
    ``DataIntegrityError`` raised by an earlier bound propagates unchanged; no
    later bound result is fabricated.
    """
    outcome = replay_execution_window_batch_isolated(
        windows, initial_equity, bounds, retain_event_snapshots, min_equity_fraction, isolated_bound_indices=frozenset(),
    )
    # isolated set empty guarantees no None results
    return tuple(result for result in outcome.results if result is not None)


def replay_execution_window_pair(
    windows: Iterable[ExecutionReplayWindow],
    initial_equity: float,
    spec: ExecutionSpec,
    retain_event_snapshots: bool = False,
) -> tuple[StrategyExecutionReplayResult, StrategyExecutionReplayResult]:
    """Replay one shared window stream into an independent strict/stress pair.

    The strict and immediate-taker bounds consume identical immutable market
    windows; only their state and fill rule differ. Each yielded window is
    consumed by the strict accumulator, then by the stress accumulator, and
    released before the next is requested, so a single loaded
    ``ExecutionReplayWindow`` remains the memory boundary and the window
    iterator is never materialized or recreated. A fatal ``DataIntegrityError``
    raised by the strict bound propagates unchanged; no stress result is
    fabricated.
    """
    strict, stress = replay_execution_window_batch(
        windows, initial_equity,
        [("OHLCV_STRICT_PROXY", spec), ("OHLCV_IMMEDIATE_TAKER", spec)],
        retain_event_snapshots=retain_event_snapshots,
    )
    return strict, stress
