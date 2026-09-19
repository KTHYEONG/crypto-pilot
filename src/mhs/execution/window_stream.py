"""Completed three-minute execution window stream owned by the execution core."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator, Mapping
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.execution.contracts import ExecutionReplayWindow, align_funding_with_knowledge, funding_coverage_gaps
from src.mhs.marks import _build_window_frames, _load_window_minute_frames
from src.mhs.resources import (
    MhsExecutionAllocation,
    assert_mhs_allocation_budget,
    plan_mhs_execution_bars,
)
from src.mhs.types import ExecutionSpec

MhsExecutionWindow = ExecutionReplayWindow


def _resolve_ns_vectorized(
    spos_all: np.ndarray,
    full_grid_ns: np.ndarray,
    n_grid: int,
    timeout_ns_delta: int,
) -> np.ndarray:
    """Vectorized ``resolve_ns`` computation for the window generator.

    Bit-identical to the scalar per-decision loop: ``resolve_ns[i]`` is the
    exact timeout bar ``full_grid_ns[spos_all[i]] + timeout_ns_delta`` when it
    lies on the grid, else ``-1``.  ``searchsorted`` (``side="left"``) keeps the
    same semantics; the ``np.minimum`` guards keep out-of-range positions from
    raising instead of silently skipping (matching the scalar ``continue``).
    """
    resolve_ns = np.full(len(spos_all), -1, dtype="int64")
    s = np.minimum(spos_all, n_grid - 1)
    timeout_ns = full_grid_ns[s] + timeout_ns_delta
    tpos = np.searchsorted(full_grid_ns, timeout_ns, side="left")
    valid = (spos_all < n_grid) & (tpos < n_grid) & (full_grid_ns[np.minimum(tpos, n_grid - 1)] == timeout_ns)
    resolve_ns[valid] = timeout_ns[valid]
    return resolve_ns


def _estimate_mhs_execution_allocation(*, n_symbols: int, n_columns: int, bound_count: int) -> MhsExecutionAllocation:
    """Conservative working-set model from projected shapes and live bounds."""
    if isinstance(bound_count, bool) or not isinstance(bound_count, int) or bound_count <= 0:
        raise ValueError(f"bound_count must be a positive integer, got {bound_count!r}")
    n_sym = max(int(n_symbols), 0)
    n_col = max(int(n_columns), 0)
    fixed_bytes = 262144 + n_sym * 4096 * int(bound_count) + n_col * 256
    per_symbol = 112 + int(bound_count) * 64
    bytes_per_bar = max(n_sym, 1) * per_symbol
    decoder_bytes = n_sym * 65536 + 1048576
    return MhsExecutionAllocation(
        fixed_bytes=int(fixed_bytes), bytes_per_bar=int(bytes_per_bar), decoder_bytes=int(decoder_bytes)
    )


def _minimum_mhs_execution_bars(timeout_ns_delta: int, step_ns: int) -> int:
    """Smallest grid preserving two completed bars and the strict timeout span."""
    if timeout_ns_delta <= 0:
        return 2
    timeout_bars = (int(timeout_ns_delta) + int(step_ns) - 1) // int(step_ns)
    return max(2, int(timeout_bars) + 1)


def _materialize_execution_piece(
    *,
    piece_grid: pd.DatetimeIndex,
    piece_weights: pd.DataFrame,
    piece_signals: pd.DatetimeIndex,
    roster: list[str],
    columns: tuple[str, ...],
    root: str,
    timeframe: Literal["3m"],
    funding_by_symbol: dict[str, pd.Series],
    funding_failures: Mapping[str, str] | None,
    allocation: MhsExecutionAllocation,
    budget_bytes: int | None,
    reserve_bytes: int | None,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    logical_partition: tuple[int, int],
    initial_swap_bytes: int | None = None,
) -> ExecutionReplayWindow:
    """Materialize completed three-minute trade OHLCV, funding and publication evidence for one replay piece. Emit no external mark plane; the execution engine uses the same trade close series for inventory valuation.

    Args:
        initial_swap_bytes: Observed run-entry process-tree swap baseline; existing swapped pages are not classified as growth.
    """
    bars = len(piece_grid)
    estimated = int(allocation.fixed_bytes) + bars * int(allocation.bytes_per_bar) + int(allocation.decoder_bytes)
    assert_mhs_allocation_budget(
        estimated_bytes=estimated,
        budget_bytes=budget_bytes,
        reserve_bytes=reserve_bytes,
        stage="process_execution_piece",
        initial_swap_bytes=initial_swap_bytes,
    )
    symbol_frames = _load_window_minute_frames(root, roster, piece_grid[0], piece_grid[-1], timeframe)
    aligned = _build_window_frames(symbol_frames, roster, piece_grid[0], piece_grid[-1], piece_grid, timeframe)
    if aligned is None:
        highs = pd.DataFrame(index=piece_grid)
        lows = pd.DataFrame(index=piece_grid)
        closes = pd.DataFrame(index=piece_grid)
    else:
        highs, lows, closes = aligned
    for s in roster:
        if s not in highs.columns:
            highs[s] = np.nan
            lows[s] = np.nan
            closes[s] = np.nan
    highs = highs.reindex(columns=roster)
    lows = lows.reindex(columns=roster)
    closes = closes.reindex(columns=roster)
    minute_period = piece_grid[1] - piece_grid[0] if len(piece_grid) > 1 else pd.Timedelta(minutes=1)
    funding_alignment = align_funding_with_knowledge(
        funding_by_symbol, piece_grid, symbols=roster, source_failures=funding_failures
    )
    minute_funding = funding_alignment.rates
    funding_known = funding_alignment.known
    coverage_gaps = funding_coverage_gaps(funding_alignment, piece_grid)
    quote_volumes = pd.DataFrame(
        {
            s: symbol_frames[s]["quote_vol"]
            for s in roster
            if s in symbol_frames and "quote_vol" in symbol_frames[s].columns
        },
        index=piece_grid,
    )
    for s in roster:
        if s not in quote_volumes.columns:
            quote_volumes[s] = np.nan
    quote_volumes = quote_volumes.reindex(columns=roster)
    bar_available_at = piece_grid + minute_period
    window = ExecutionReplayWindow(
        window_start=window_start,
        window_end=window_end,
        columns=columns,
        symbols=tuple(roster),
        minute_grid=piece_grid,
        highs=highs,
        lows=lows,
        closes=closes,
        marks=None,
        bar_funding=minute_funding,
        target_weights=piece_weights[roster] if len(piece_weights) else piece_weights.reindex(columns=roster),
        signal_available_at=piece_signals,
        quote_volumes=quote_volumes,
        funding_known=funding_known,
        bar_available_at=bar_available_at,
        logical_partition=logical_partition,
        funding_coverage_gaps=coverage_gaps,
        funding_knowledge_source=funding_alignment.knowledge_source,
    )
    del symbol_frames
    del aligned
    gc.collect()
    return window


def _iter_mhs_execution_windows(
    target_weights: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    root: str,
    timeframe: Literal["3m"],
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series],
    spec: ExecutionSpec,
    funding_failures: Mapping[str, str] | None = None,
    *,
    required_symbols: Callable[[], frozenset[str]] | None = None,
    budget_bytes: int | None = None,
    reserve_bytes: int | None = None,
    execution_bound_count: int = 2,
    initial_swap_bytes: int | None = None,
) -> Iterator[MhsExecutionWindow]:
    """Stream chronologically completed three-minute trade bars and funding knowledge for an exact target path. The OHLCV mode leaves `ExecutionReplayWindow.marks` absent so the shared accounting engine values positions from 3m closes; bar completion remains the earliest publication time."""
    if len(target_weights) != len(signal_available_at):
        raise DataIntegrityError("signal_available_at must align with target_weights")
    if start >= end:
        raise DataIntegrityError("start must precede end")
    columns = tuple(target_weights.columns)
    freq = "3min"
    step = pd.Timedelta(minutes=3)
    step_ns = 180_000_000_000
    timeout_ns_delta = int(spec.passive_timeout_minutes) * 60_000_000_000
    start_ns = int(start.value)
    fence_max_ns = int((end - step).value)
    signal_ns = np.asarray(signal_available_at, dtype="datetime64[ns]").astype("int64")
    spos_idx = np.where(
        signal_ns < start_ns,
        0,
        (signal_ns - start_ns) // step_ns + 1,
    )
    spos_label_ns = start_ns + spos_idx * step_ns
    timeout_label_ns = spos_label_ns + timeout_ns_delta
    on_grid = timeout_ns_delta % step_ns == 0
    valid = on_grid & (spos_label_ns <= fence_max_ns) & (timeout_label_ns <= fence_max_ns)
    resolve_ns = np.where(valid, timeout_label_ns, -1).astype("int64")

    if target_weights.empty:
        completed_grid = pd.date_range(start, end - step, freq=freq, tz="UTC")
        yield ExecutionReplayWindow(
            window_start=start,
            window_end=end,
            columns=columns,
            symbols=(),
            minute_grid=completed_grid,
            highs=pd.DataFrame(index=completed_grid),
            lows=pd.DataFrame(index=completed_grid),
            closes=pd.DataFrame(index=completed_grid),
            marks=None,
            bar_funding=pd.DataFrame(index=completed_grid),
            target_weights=target_weights,
            signal_available_at=signal_available_at,
        )
        return

    decision_times = pd.DatetimeIndex(target_weights.index)
    max_window = pd.Timedelta(days=31)
    bounds: list[tuple[int, int]] = []
    i0 = 0
    while i0 < len(decision_times):
        i1 = i0 + 1
        while i1 < len(decision_times) and decision_times[i1] - decision_times[i0] <= max_window:
            i1 += 1
        bounds.append((i0, i1))
        i0 = i1

    if (
        isinstance(execution_bound_count, bool)
        or not isinstance(execution_bound_count, int)
        or execution_bound_count <= 0
    ):
        raise ValueError(f"execution_bound_count must be a positive integer, got {execution_bound_count!r}")
    bound_count = int(execution_bound_count)
    minimum_bars = _minimum_mhs_execution_bars(timeout_ns_delta, step_ns)
    budgeted = budget_bytes is not None or reserve_bytes is not None
    prev_active: set[str] = set()
    for wi, (i0, i1) in enumerate(bounds):
        w_weights = target_weights.iloc[i0:i1]
        w_signals = signal_available_at[i0:i1]
        is_last = wi == len(bounds) - 1
        grid_start = start if wi == 0 else decision_times[i0 - 1]
        if is_last:
            grid_end = end
        else:
            max_resolve = int(resolve_ns[i0:i1].max())
            if max_resolve < 0:
                max_resolve = int(
                    np.asarray(decision_times[i1 - 1] + pd.Timedelta(hours=2), dtype="datetime64[ns]").astype("int64")
                )
            grid_end = pd.Timestamp(max_resolve, unit="ns", tz="UTC")
        if grid_end > end:
            grid_end = end
        fence_last = end - step
        effective_end = fence_last if is_last or grid_end >= end else min(grid_end, fence_last)
        minute_grid = pd.date_range(grid_start, effective_end, freq=freq, tz="UTC")
        if len(minute_grid) < 2:
            raise DataIntegrityError("evaluation range supplies no legal completed window")
        if not budgeted:
            non_zero = w_weights.notna() & w_weights.ne(0.0)
            active = set(w_weights.columns[non_zero.any(axis=0)])
            if required_symbols is not None:
                live_required = set(required_symbols())
                unknown = live_required - set(columns)
                if unknown:
                    raise DataIntegrityError(
                        f"required symbols {sorted(unknown)} are not in canonical columns; "
                        "missing held symbols must never disappear"
                    )
                roster_set = active | live_required
            else:
                roster_set = active | prev_active
            prev_active = active
            roster = [s for s in columns if s in roster_set]
            legacy_alloc = _estimate_mhs_execution_allocation(
                n_symbols=len(roster), n_columns=len(columns), bound_count=bound_count
            )
            window = _materialize_execution_piece(
                piece_grid=minute_grid,
                piece_weights=w_weights,
                piece_signals=w_signals,
                roster=roster,
                columns=columns,
                root=root,
                timeframe=timeframe,
                funding_by_symbol=funding_by_symbol,
                funding_failures=funding_failures,
                allocation=legacy_alloc,
                budget_bytes=budget_bytes,
                reserve_bytes=reserve_bytes,
                initial_swap_bytes=initial_swap_bytes,
                window_start=grid_start,
                window_end=grid_end,
                logical_partition=(i0, i1),
            )
            yield window
            del window
            gc.collect()
            continue
        full_grid = minute_grid
        full_ns = np.asarray(full_grid, dtype="datetime64[ns]").astype("int64")
        n_full = len(full_grid)
        active_full = set(w_weights.columns[(w_weights.notna() & w_weights.ne(0.0)).any(axis=0)])
        if required_symbols is not None:
            live_now = set(required_symbols())
            unknown = live_now - set(columns)
            if unknown:
                raise DataIntegrityError(
                    f"required symbols {sorted(unknown)} are not in canonical columns; "
                    "missing held symbols must never disappear"
                )
        else:
            live_now = set(prev_active)
        roster_full = [s for s in columns if s in (active_full | live_now)]
        allocation = _estimate_mhs_execution_allocation(
            n_symbols=len(roster_full), n_columns=len(columns), bound_count=bound_count
        )
        decision_pos = np.searchsorted(
            full_ns,
            np.asarray(decision_times[i0:i1], dtype="datetime64[ns]").astype("int64"),
            side="left",
        )
        timeout_pos = np.full((i1 - i0), -1, dtype=np.intp)
        for j in range(i1 - i0):
            r = int(resolve_ns[i0 + j])
            if r >= 0:
                timeout_pos[j] = int(np.searchsorted(full_ns, np.int64(r), side="left"))
            else:
                signal_pos = int(np.searchsorted(full_ns, signal_ns[i0 + j], side="right"))
                timeout_pos[j] = min(signal_pos + minimum_bars - 1, n_full - 1)
        minimum_piece_bars = max(
            minimum_bars,
            int(np.max(timeout_pos - decision_pos + 1)),
        )
        if n_full < minimum_piece_bars:
            raise DataIntegrityError(
                f"logical partition [{i0}, {i1}) needs {minimum_piece_bars} bars to preserve the strict timeout span "
                f"but only {n_full} completed bars remain before the fence"
            )
        planned = plan_mhs_execution_bars(
            requested_bars=n_full,
            minimum_bars=minimum_piece_bars,
            allocation=allocation,
            budget_bytes=budget_bytes,
            reserve_bytes=reserve_bytes,
        )
        if planned >= n_full:
            non_zero = w_weights.notna() & w_weights.ne(0.0)
            active = set(w_weights.columns[non_zero.any(axis=0)])
            if required_symbols is not None:
                live_required = set(required_symbols())
                unknown = live_required - set(columns)
                if unknown:
                    raise DataIntegrityError(
                        f"required symbols {sorted(unknown)} are not in canonical columns; "
                        "missing held symbols must never disappear"
                    )
                roster_set = active | live_required
            else:
                roster_set = active | prev_active
            prev_active = active
            roster = [s for s in columns if s in roster_set]
            piece_allocation = _estimate_mhs_execution_allocation(
                n_symbols=len(roster), n_columns=len(columns), bound_count=bound_count
            )
            window = _materialize_execution_piece(
                piece_grid=full_grid,
                piece_weights=w_weights,
                piece_signals=w_signals,
                roster=roster,
                columns=columns,
                root=root,
                timeframe=timeframe,
                funding_by_symbol=funding_by_symbol,
                funding_failures=funding_failures,
                allocation=piece_allocation,
                budget_bytes=budget_bytes,
                reserve_bytes=reserve_bytes,
                initial_swap_bytes=initial_swap_bytes,
                window_start=grid_start,
                window_end=grid_end,
                logical_partition=(i0, i1),
            )
            yield window
            del window
            gc.collect()
            continue
        g0 = 0
        d = 0
        n_dec = i1 - i0
        while d < n_dec:
            g1 = min(g0 + planned - 1, n_full - 1)
            d1 = d
            while d1 < n_dec and int(decision_pos[d1]) <= g1 and int(timeout_pos[d1]) <= g1:
                d1 += 1
            if d1 == d:
                next_decision = int(decision_pos[d])
                if next_decision <= g1:
                    g1 = next_decision - 1
                if g1 < g0 + 1:
                    raise DataIntegrityError(
                        f"physical piece at logical partition [{i0}, {i1}) leaves order {i0 + d} unresolved "
                        "merely to satisfy a narrower IO budget"
                    )
                if required_symbols is not None:
                    live_required = set(required_symbols())
                    unknown = live_required - set(columns)
                    if unknown:
                        raise DataIntegrityError(
                            f"required symbols {sorted(unknown)} are not in canonical columns; "
                            "missing held symbols must never disappear"
                        )
                    roster_set = live_required | prev_active
                else:
                    roster_set = set(prev_active)
                roster = [s for s in columns if s in roster_set]
                piece_grid = full_grid[g0 : g1 + 1]
                empty_weights = target_weights.iloc[0:0].reindex(columns=roster)
                empty_signals = signal_available_at[0:0]
                piece_allocation = _estimate_mhs_execution_allocation(
                    n_symbols=len(roster),
                    n_columns=len(columns),
                    bound_count=bound_count,
                )
                window = _materialize_execution_piece(
                    piece_grid=piece_grid,
                    piece_weights=empty_weights,
                    piece_signals=empty_signals,
                    roster=roster,
                    columns=columns,
                    root=root,
                    timeframe=timeframe,
                    funding_by_symbol=funding_by_symbol,
                    funding_failures=funding_failures,
                    allocation=piece_allocation,
                    budget_bytes=budget_bytes,
                    reserve_bytes=reserve_bytes,
                    initial_swap_bytes=initial_swap_bytes,
                    window_start=piece_grid[0],
                    window_end=piece_grid[-1] + step,
                    logical_partition=(i0, i1),
                )
                yield window
                del window
                gc.collect()
                g0 = g1
                continue
            piece_weights = target_weights.iloc[i0 + d : i0 + d1]
            piece_signals = signal_available_at[i0 + d : i0 + d1]
            piece_active = set(piece_weights.columns[(piece_weights.notna() & piece_weights.ne(0.0)).any(axis=0)])
            if required_symbols is not None:
                live_required = set(required_symbols())
                unknown = live_required - set(columns)
                if unknown:
                    raise DataIntegrityError(
                        f"required symbols {sorted(unknown)} are not in canonical columns; "
                        "missing held symbols must never disappear"
                    )
                roster_set = piece_active | live_required
            else:
                roster_set = piece_active | prev_active
            prev_active = set(piece_active)
            roster = [s for s in columns if s in roster_set]
            piece_allocation = _estimate_mhs_execution_allocation(
                n_symbols=len(roster), n_columns=len(columns), bound_count=bound_count
            )
            piece_grid = full_grid[g0 : g1 + 1]
            piece_end = grid_end if (g1 == n_full - 1 and d1 == n_dec) else piece_grid[-1] + step
            window = _materialize_execution_piece(
                piece_grid=piece_grid,
                piece_weights=piece_weights,
                piece_signals=piece_signals,
                roster=roster,
                columns=columns,
                root=root,
                timeframe=timeframe,
                funding_by_symbol=funding_by_symbol,
                funding_failures=funding_failures,
                allocation=piece_allocation,
                budget_bytes=budget_bytes,
                reserve_bytes=reserve_bytes,
                initial_swap_bytes=initial_swap_bytes,
                window_start=piece_grid[0],
                window_end=piece_end,
                logical_partition=(i0, i1),
            )
            yield window
            del window
            gc.collect()
            d = d1
            if g1 == n_full - 1:
                break
            g0 = g1
        if g0 < n_full - 1 and d >= n_dec:
            tail_start = g0
            while tail_start < n_full - 1:
                tail_end = min(tail_start + planned - 1, n_full - 1)
                tail_grid = full_grid[tail_start : tail_end + 1]
                if required_symbols is not None:
                    live_required = set(required_symbols())
                    unknown = live_required - set(columns)
                    if unknown:
                        raise DataIntegrityError(
                            f"required symbols {sorted(unknown)} are not in canonical columns; "
                            "missing held symbols must never disappear"
                        )
                    roster_set = set(live_required) | prev_active
                else:
                    roster_set = set(prev_active)
                roster = [s for s in columns if s in roster_set]
                empty_weights = target_weights.iloc[0:0].reindex(columns=roster)
                empty_signals = signal_available_at[0:0]
                piece_allocation = _estimate_mhs_execution_allocation(
                    n_symbols=len(roster), n_columns=len(columns), bound_count=bound_count
                )
                piece_end = grid_end if tail_end == n_full - 1 else tail_grid[-1] + step
                window = _materialize_execution_piece(
                    piece_grid=tail_grid,
                    piece_weights=empty_weights,
                    piece_signals=empty_signals,
                    roster=roster,
                    columns=columns,
                    root=root,
                    timeframe=timeframe,
                    funding_by_symbol=funding_by_symbol,
                    funding_failures=funding_failures,
                    allocation=piece_allocation,
                    budget_bytes=budget_bytes,
                    reserve_bytes=reserve_bytes,
                    initial_swap_bytes=initial_swap_bytes,
                    window_start=tail_grid[0],
                    window_end=piece_end,
                    logical_partition=(i0, i1),
                )
                yield window
                del window
                gc.collect()
                if tail_end == n_full - 1:
                    break
                tail_start = tail_end


__all__ = [
    "MhsExecutionWindow",
    "_estimate_mhs_execution_allocation",
    "_iter_mhs_execution_windows",
    "_materialize_execution_piece",
    "_minimum_mhs_execution_bars",
    "_resolve_ns_vectorized",
]
