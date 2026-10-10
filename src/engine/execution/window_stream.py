"""Completed three-minute execution window stream owned by the execution core."""

from __future__ import annotations

import functools
import gc
from collections.abc import Callable, Iterator, Mapping
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.instrument_settlements import InstrumentSettlementRegistry, settlement_registry_for_root
from src.core.marks import _build_window_frames, _load_window_minute_frames, _missing_execution_sources
from src.core.parallel import collect_window_garbage
from src.core.resources import (
    MhsExecutionAllocation,
    assert_mhs_allocation_budget,
    plan_mhs_execution_bars,
)
from src.core.types import ExecutionSpec
from src.core.venue_halts import VenueHaltRegistry, venue_halt_registry_for_root
from src.engine.execution.contracts import ExecutionReplayWindow, align_funding_with_knowledge, funding_coverage_gaps

from .settlement import settled_before_piece, settlement_events_for_piece

MhsExecutionWindow = ExecutionReplayWindow


def _single_panel_execution_window(
    target_weights: pd.DataFrame, signal_available_at: pd.DatetimeIndex,
    highs: pd.DataFrame, lows: pd.DataFrame, closes: pd.DataFrame,
    marks: pd.DataFrame | None, bar_funding: pd.DataFrame,
) -> ExecutionReplayWindow:
    """Wrap a validated caller-owned panel in the shared window contract."""
    symbols = list(target_weights.columns)
    grid = closes.index
    return ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=tuple(symbols), symbols=tuple(symbols),
        minute_grid=grid, highs=highs[symbols], lows=lows[symbols], closes=closes[symbols],
        marks=marks[symbols] if marks is not None else None, bar_funding=bar_funding[symbols],
        target_weights=target_weights, signal_available_at=signal_available_at,
    )


def _resolve_settlement_registry(
    settlement_registry: InstrumentSettlementRegistry | None, root: str,
) -> InstrumentSettlementRegistry:
    """Resolve the window-stream registry without branching at the call site."""
    if settlement_registry is not None:
        return settlement_registry
    return settlement_registry_for_root(root)


def _resolve_venue_halt_registry(
    venue_halts: VenueHaltRegistry | None, root: str,
) -> VenueHaltRegistry:
    """Resolve the window-stream halt registry without branching at the call site."""
    if venue_halts is not None:
        return venue_halts
    return venue_halt_registry_for_root(root)


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


def _first_active_ordinals(target_weights: pd.DataFrame) -> np.ndarray:
    """Return, per column, the ordinal of the first decision with a finite nonzero target.

    A position can only open from a finite nonzero target, so this table bounds every inventory
    any execution bound can carry from a target path alone, without consulting a consumer.
    Columns never targeted get ``len(target_weights)``.

    Args:
        target_weights: Decision-indexed targets in canonical column order.

    Returns:
        ``int64`` array of length ``len(target_weights.columns)``.
    """
    n_rows = len(target_weights)
    n_cols = len(target_weights.columns)
    first = np.full(n_cols, n_rows, dtype=np.int64)
    if n_rows == 0 or n_cols == 0:
        return first
    active = (target_weights.notna() & target_weights.ne(0.0)).to_numpy()
    has_any = active.any(axis=0)
    first[has_any] = np.argmax(active[:, has_any], axis=0)
    return first


def _roster_requirement(
    required_symbols: Callable[[], frozenset[str]] | None,
    columns: tuple[str, ...],
    first_active: np.ndarray,
    stop_ordinal: int,
) -> frozenset[str]:
    """Resolve the symbols a piece must load regardless of its own intents.

    With a live callback the requirement is the consumer's actual carried inventory and
    unresolved orders, validated against the canonical columns here and nowhere else. Without one,
    the requirement is every column whose first finite nonzero target precedes ``stop_ordinal``:
    held, blocked-exit and NaN-hold inventory can only exist on such columns, and the result is a
    pure function of the target path, so a generated stream stays valid for every bound and pass
    that replays it.

    Args:
        required_symbols: Live requirement callback, or None for the target-only carry.
        columns: Canonical column order.
        first_active: Output of ``_first_active_ordinals`` for the same targets.
        stop_ordinal: Exclusive global decision ordinal already scheduled at this piece.

    Returns:
        Requirement set (subset of ``columns``).

    Raises:
        DataIntegrityError: The callback names a symbol outside ``columns``
            ("required symbols [...] are not in canonical columns; missing held symbols must
            never disappear").
    """
    if required_symbols is not None:
        live = set(required_symbols())
        unknown = live - set(columns)
        if unknown:
            raise DataIntegrityError(
                f"required symbols {sorted(unknown)} are not in canonical columns; "
                "missing held symbols must never disappear"
            )
        return frozenset(live)
    return frozenset(
        col for col, ordinal in zip(columns, first_active, strict=True) if int(ordinal) < stop_ordinal
    )


def _piece_roster(
    columns: tuple[str, ...],
    *,
    piece_active: frozenset[str],
    previous_active: frozenset[str],
    requirement: frozenset[str],
) -> list[str]:
    """Union the piece's own intents, the previous piece's intents, and the requirement.

    One rule for every branch and both modes. It is a superset of each branch's former roster, so
    the refactor can add but never drop a symbol any branch loaded before.

    Returns:
        Roster in canonical ``columns`` order.
    """
    roster_set = set(piece_active) | set(previous_active) | set(requirement)
    return [s for s in columns if s in roster_set]


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
    settlement_registry: InstrumentSettlementRegistry | None = None,
    venue_halts: VenueHaltRegistry | None = None,
    replay_start: pd.Timestamp | None = None,
    replay_end: pd.Timestamp | None = None,
    initial_swap_bytes: int | None = None,
) -> ExecutionReplayWindow:
    """Materialize completed three-minute trade OHLCV, funding and publication evidence for one replay piece. Emit no external mark plane; the execution engine uses the same trade close series for inventory valuation. Every window carries the evidenced settlement events of its roster so all replay paths settle delisted inventory identically; symbols delivered before the piece with exact-zero targets leave the roster.

    Args:
        initial_swap_bytes: Observed run-entry process-tree swap baseline; existing swapped pages are not classified as growth.
    """
    from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY as _EMPTY_REG
    from src.core.venue_halts import EMPTY_VENUE_HALT_REGISTRY as _EMPTY_HALTS

    _reg = settlement_registry if settlement_registry is not None else _EMPTY_REG
    _halts = venue_halts if venue_halts is not None else _EMPTY_HALTS
    _rs = replay_start if replay_start is not None else piece_grid[0]
    _re = replay_end if replay_end is not None else piece_grid[-1] + (piece_grid[1] - piece_grid[0] if len(piece_grid) > 1 else pd.Timedelta(minutes=3))
    drop = settled_before_piece(_reg, roster, piece_grid, piece_weights)
    if drop:
        roster = [s for s in roster if s not in drop]
        if len(piece_weights.columns):
            keep = [c for c in piece_weights.columns if c not in drop]
            piece_weights = piece_weights.reindex(columns=keep)
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
    settlement_events = settlement_events_for_piece(
        _reg, roster, piece_grid, quote_volumes,
        replay_start=_rs, replay_end=_re,
    )
    halt_step = piece_grid[1] - piece_grid[0] if len(piece_grid) > 1 else pd.Timedelta(minutes=3)
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
        settlement_events=settlement_events,
        venue_halts=_halts.overlapping(piece_grid[0], piece_grid[-1] + halt_step),
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
    settlement_registry: InstrumentSettlementRegistry | None = None,
    venue_halts: VenueHaltRegistry | None = None,
) -> Iterator[MhsExecutionWindow]:
    """Stream chronologically completed three-minute trade bars and funding knowledge for an exact target path. The OHLCV mode leaves `ExecutionReplayWindow.marks` absent so the shared accounting engine values positions from 3m closes; bar completion remains the earliest publication time. Rosters always cover carried inventory — live requirements when supplied, otherwise every column targeted so far. Every window carries the evidenced settlement events of its roster so all replay paths settle delisted inventory identically; symbols delivered before the piece with exact-zero targets leave the roster.

    Raises:
        DataIntegrityError: A column with at least one finite non-zero target has no
            execution source file under ``root``: "execution source missing for <n>
            targeted symbol(s): <SYM> (first_target=<iso>), ... timeframe=<tf>
            root=<root> replay=[<start iso>, <end iso>)". Raised before the first window
            is materialized: replaying a target on a symbol that was never collected
            would drop its P&L, fees and funding from the ledger (phantom performance).
    """
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
    first_active = _first_active_ordinals(target_weights)
    targeted = [c for c, o in zip(columns, first_active, strict=True) if int(o) < len(target_weights)]
    missing = _missing_execution_sources(root, targeted, timeframe)
    if missing:
        ordinal_by_symbol = {c: int(o) for c, o in zip(columns, first_active, strict=True)}
        details = ", ".join(
            f"{sym} (first_target={decision_times[ordinal_by_symbol[sym]].isoformat()})"
            for sym in missing
        )
        raise DataIntegrityError(
            f"execution source missing for {len(missing)} targeted symbol(s): {details} "
            f"timeframe={timeframe} root={root} replay=[{start.isoformat()}, {end.isoformat()})"
        )
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
    materialize = functools.partial(
        _materialize_execution_piece,
        settlement_registry=_resolve_settlement_registry(settlement_registry, root),
        venue_halts=_resolve_venue_halt_registry(venue_halts, root),
        replay_start=start,
        replay_end=end,
    )
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
            requirement = _roster_requirement(required_symbols, columns, first_active, i1)
            roster = _piece_roster(
                columns, piece_active=frozenset(active), previous_active=frozenset(prev_active),
                requirement=requirement,
            )
            prev_active = active
            legacy_alloc = _estimate_mhs_execution_allocation(
                n_symbols=len(roster), n_columns=len(columns), bound_count=bound_count
            )
            window = materialize(
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
            collect_window_garbage()
            continue
        full_grid = minute_grid
        full_ns = np.asarray(full_grid, dtype="datetime64[ns]").astype("int64")
        n_full = len(full_grid)
        active_full = set(w_weights.columns[(w_weights.notna() & w_weights.ne(0.0)).any(axis=0)])
        planning_requirement = _roster_requirement(required_symbols, columns, first_active, i1)
        roster_full = _piece_roster(
            columns, piece_active=frozenset(active_full),
            previous_active=frozenset(prev_active), requirement=planning_requirement,
        )
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
                requirement = planning_requirement
            else:
                requirement = _roster_requirement(None, columns, first_active, i1)
            roster = _piece_roster(
                columns, piece_active=frozenset(active),
                previous_active=frozenset(prev_active), requirement=requirement,
            )
            prev_active = active
            piece_allocation = _estimate_mhs_execution_allocation(
                n_symbols=len(roster), n_columns=len(columns), bound_count=bound_count
            )
            window = materialize(
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
            collect_window_garbage()
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
                requirement = _roster_requirement(required_symbols, columns, first_active, i0 + d)
                roster = _piece_roster(
                    columns, piece_active=frozenset(),
                    previous_active=frozenset(prev_active), requirement=requirement,
                )
                piece_grid = full_grid[g0 : g1 + 1]
                empty_weights = target_weights.iloc[0:0].reindex(columns=roster)
                empty_signals = signal_available_at[0:0]
                piece_allocation = _estimate_mhs_execution_allocation(
                    n_symbols=len(roster),
                    n_columns=len(columns),
                    bound_count=bound_count,
                )
                window = materialize(
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
                collect_window_garbage()
                g0 = g1
                continue
            piece_weights = target_weights.iloc[i0 + d : i0 + d1]
            piece_signals = signal_available_at[i0 + d : i0 + d1]
            piece_active = set(piece_weights.columns[(piece_weights.notna() & piece_weights.ne(0.0)).any(axis=0)])
            requirement = _roster_requirement(required_symbols, columns, first_active, i0 + d1)
            roster = _piece_roster(
                columns, piece_active=frozenset(piece_active),
                previous_active=frozenset(prev_active), requirement=requirement,
            )
            prev_active = set(piece_active)
            piece_allocation = _estimate_mhs_execution_allocation(
                n_symbols=len(roster), n_columns=len(columns), bound_count=bound_count
            )
            piece_grid = full_grid[g0 : g1 + 1]
            piece_end = grid_end if (g1 == n_full - 1 and d1 == n_dec) else piece_grid[-1] + step
            window = materialize(
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
            collect_window_garbage()
            d = d1
            if g1 == n_full - 1:
                break
            g0 = g1
        if g0 < n_full - 1 and d >= n_dec:
            tail_start = g0
            while tail_start < n_full - 1:
                tail_end = min(tail_start + planned - 1, n_full - 1)
                tail_grid = full_grid[tail_start : tail_end + 1]
                requirement = _roster_requirement(required_symbols, columns, first_active, i1)
                roster = _piece_roster(
                    columns, piece_active=frozenset(),
                    previous_active=frozenset(prev_active), requirement=requirement,
                )
                empty_weights = target_weights.iloc[0:0].reindex(columns=roster)
                empty_signals = signal_available_at[0:0]
                piece_allocation = _estimate_mhs_execution_allocation(
                    n_symbols=len(roster), n_columns=len(columns), bound_count=bound_count
                )
                piece_end = grid_end if tail_end == n_full - 1 else tail_grid[-1] + step
                window = materialize(
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
                collect_window_garbage()
                if tail_end == n_full - 1:
                    break
                tail_start = tail_end


__all__ = [
    "MhsExecutionWindow",
    "_estimate_mhs_execution_allocation",
    "_first_active_ordinals",
    "_iter_mhs_execution_windows",
    "_materialize_execution_piece",
    "_minimum_mhs_execution_bars",
    "_piece_roster",
    "_resolve_ns_vectorized",
    "_roster_requirement",
]
