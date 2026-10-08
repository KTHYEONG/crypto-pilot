"""Bound-invariant staging of one execution window, shared read-only across bounds.

A batch replays one window into several bound accumulators (base, stress,
scaled bounds) whose state, fill rule and cost spec differ but whose market
planes, availability overlays and decision positions are identical. Staging
them once per window object halves decode work and the resident overlay
working set for a two-bound replay without changing any value.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

from .contracts import ExecutionReplayWindow


@dataclasses.dataclass(frozen=True, slots=True)
class StagedWindowArrays:
    """Validated numeric planes of one window, identical for every bound.

    All arrays are read-only (``flags.writeable is False``); an accumulator that
    attempted an in-place write would raise instead of corrupting another bound.

    Attributes:
        window: The exact window object these arrays were staged from.
        local_cols: Window roster in canonical order.
        grid: Window bar labels (UTC).
        grid_ns: ``grid`` as int64 nanoseconds.
        bar_ns: Bar width in nanoseconds (``grid_ns[1] - grid_ns[0]``).
        marks_values: Valuation marks (``marks`` or ``closes`` fallback), ``(n_grid, n_local)``.
        highs_values: Bar highs, same shape.
        lows_values: Bar lows, same shape.
        closes_values: Bar closes, same shape.
        mark_valid: Finite strictly positive mark mask, same shape.
        funding_matrix: Per-bar funding rates, same shape.
        last_close_idx: Last finite-close row at or before each row, ``-1`` when none.
        quote_volumes: Per-bar quote volume (ones when the window carries none).
        last_liquid_idx: Last positive-volume row at or before each row, ``-1`` when none.
        funding_known: Per-bar funding knowledge (all True when absent).
        avail_ns: Per-bar availability nanoseconds (grid copy when not explicit).
        avail_explicit: Whether ``bar_available_at`` supplied ``avail_ns``.
        mark_avail: Mark availability (one bar behind ``avail_ns``), ``(n_grid, n_local)``.
        decision_ns_all: Decision labels in nanoseconds.
        spos_all: First grid position strictly after each signal availability.
        dpos_all: Grid position of each decision label (searchsorted left).
        on_grid_all: Whether each decision label is an exact grid label.
        target_values: Target weights, ``(n_decisions, n_local)``.
    """

    window: ExecutionReplayWindow
    local_cols: tuple[str, ...]
    grid: pd.DatetimeIndex
    grid_ns: np.ndarray
    bar_ns: int
    marks_values: np.ndarray
    highs_values: np.ndarray
    lows_values: np.ndarray
    closes_values: np.ndarray
    mark_valid: np.ndarray
    funding_matrix: np.ndarray
    last_close_idx: np.ndarray
    quote_volumes: np.ndarray
    last_liquid_idx: np.ndarray
    funding_known: np.ndarray
    avail_ns: np.ndarray
    avail_explicit: bool
    mark_avail: np.ndarray
    decision_ns_all: np.ndarray
    spos_all: np.ndarray
    dpos_all: np.ndarray
    on_grid_all: np.ndarray
    target_values: np.ndarray


def _stage_overlays(
    w: ExecutionReplayWindow, local_cols: list[str], n_grid: int, n_local: int, grid_ns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    """Decode quote-volume, funding-knowledge and availability overlays."""
    n_grid_int = int(n_grid)
    if w.quote_volumes is not None:
        qv = np.full((n_grid_int, n_local), 1.0, dtype="float64")
        for j, sym in enumerate(local_cols):
            if sym in w.quote_volumes.columns:
                qv[:, j] = w.quote_volumes[sym].to_numpy(dtype="float64")
    else:
        qv = np.ones((n_grid_int, n_local), dtype="float64")
    last_liquid_idx = np.maximum.accumulate(np.where(qv > 0.0, np.arange(n_grid_int)[:, None], -1), axis=0)
    if w.funding_known is not None:
        fknown = np.zeros((n_grid_int, n_local), dtype=bool)
        for j, sym in enumerate(local_cols):
            if sym in w.funding_known.columns:
                fknown[:, j] = w.funding_known[sym].to_numpy(dtype=bool)
    else:
        fknown = np.ones((n_grid_int, n_local), dtype=bool)
    if w.bar_available_at is not None and len(w.bar_available_at) == n_grid_int:
        avail_ns = np.asarray(w.bar_available_at, dtype="datetime64[ns]").astype("int64")
        avail_explicit = True
    else:
        avail_ns = grid_ns.copy()
        avail_explicit = False
    bar_step = int(avail_ns[1] - avail_ns[0]) if len(avail_ns) > 1 else 0
    mark_avail = np.repeat((avail_ns - bar_step)[:, None], n_local, axis=1)
    return qv, last_liquid_idx, fknown, avail_ns, avail_explicit, mark_avail


def stage_window_arrays(w: ExecutionReplayWindow) -> StagedWindowArrays:
    """Validate and decode the bound-invariant planes of one execution window.

    Args:
        w: Window to stage; its frames are only read.
    Returns:
        Read-only staged arrays bit-identical to the per-bound arrays built at
        HEAD by ``_BoundExecutionReplayAccumulator._consume_validate_window``.
    Raises:
        DataIntegrityError: In this order: ``"an execution window must span at
            least two grid bars"``, ``"bar_funding must align exactly to the
            window minute grid"``, ``"bar_funding must be finite"``, ``"finite
            marks must be strictly positive"``.
    """
    local_cols = list(w.symbols)
    n_local = len(local_cols)
    grid = w.minute_grid
    grid_ns = np.asarray(grid, dtype="datetime64[ns]").astype("int64")
    n_grid = len(grid_ns)
    if n_grid < 2:
        raise DataIntegrityError("an execution window must span at least two grid bars")
    if not w.bar_funding.index.equals(grid):
        raise DataIntegrityError("bar_funding must align exactly to the window minute grid")
    bar_ns = int(grid_ns[1] - grid_ns[0])
    marks = w.marks if w.marks is not None else w.closes
    marks_values = marks[local_cols].to_numpy(dtype="float64")
    highs_values = w.highs[local_cols].to_numpy(dtype="float64")
    lows_values = w.lows[local_cols].to_numpy(dtype="float64")
    closes_values = w.closes[local_cols].to_numpy(dtype="float64")
    close_finite = np.isfinite(closes_values)
    sym_finite = np.isfinite(marks_values)
    mark_valid = sym_finite & (marks_values > 0.0)
    if n_local:
        funding_matrix = np.stack(
            [w.bar_funding[s].to_numpy(dtype="float64") for s in local_cols], axis=1,
        )
    else:
        funding_matrix = np.zeros((n_grid, 0), dtype="float64")
    if not np.isfinite(funding_matrix).all():
        raise DataIntegrityError("bar_funding must be finite")
    finite_marks = marks_values[sym_finite]
    if (finite_marks <= 0).any():
        raise DataIntegrityError("finite marks must be strictly positive")
    qv, last_liquid_idx, fknown, avail_ns, avail_explicit, mark_avail = _stage_overlays(
        w, local_cols, n_grid, n_local, grid_ns
    )
    close_row = np.where(close_finite, np.arange(n_grid)[:, None], -1)
    last_close_idx = np.maximum.accumulate(close_row, axis=0)
    decision_ns_all = np.asarray(w.target_weights.index, dtype="datetime64[ns]").astype("int64")
    signal_ns_all = np.asarray(w.signal_available_at, dtype="datetime64[ns]").astype("int64")
    spos_all = np.searchsorted(grid_ns, signal_ns_all, side="right")
    dpos_all = np.searchsorted(grid_ns, decision_ns_all, side="left")
    dpos_clipped = np.minimum(dpos_all, n_grid - 1)
    on_grid_all = np.where(dpos_all < n_grid, grid_ns[dpos_clipped] == decision_ns_all, False)
    target_values = w.target_weights[local_cols].to_numpy(dtype="float64")
    staged = StagedWindowArrays(
        window=w, local_cols=tuple(local_cols), grid=grid, grid_ns=grid_ns, bar_ns=bar_ns,
        marks_values=marks_values, highs_values=highs_values, lows_values=lows_values,
        closes_values=closes_values, mark_valid=mark_valid, funding_matrix=funding_matrix,
        last_close_idx=last_close_idx, quote_volumes=qv, last_liquid_idx=last_liquid_idx,
        funding_known=fknown, avail_ns=avail_ns, avail_explicit=avail_explicit,
        mark_avail=mark_avail, decision_ns_all=decision_ns_all, spos_all=spos_all,
        dpos_all=dpos_all, on_grid_all=on_grid_all, target_values=target_values,
    )
    for field in (
        staged.grid_ns, staged.marks_values, staged.highs_values, staged.lows_values,
        staged.closes_values, staged.mark_valid, staged.funding_matrix, staged.last_close_idx,
        staged.quote_volumes, staged.last_liquid_idx, staged.funding_known, staged.avail_ns,
        staged.mark_avail, staged.decision_ns_all, staged.spos_all, staged.dpos_all,
        staged.on_grid_all, staged.target_values,
    ):
        field.flags.writeable = False
    return staged


class WindowStaging:
    """Lazy per-window staging shared by every accumulator consuming one window object.

    Successful staging is memoized; a failure is not, so each bound that asks
    re-stages and raises the identical error inside its own ``consume`` and the
    batch isolation of per-bound failures is unchanged.
    """

    def __init__(self, window: ExecutionReplayWindow) -> None:
        self._window = window
        self._staged: StagedWindowArrays | None = None

    @property
    def window(self) -> ExecutionReplayWindow:
        return self._window

    def arrays(self) -> StagedWindowArrays:
        """Return the staged arrays, staging on first successful request.

        Raises:
            DataIntegrityError: As ``stage_window_arrays``; never memoized.
        """
        if self._staged is None:
            self._staged = stage_window_arrays(self._window)
        return self._staged
