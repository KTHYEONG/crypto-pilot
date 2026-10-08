"""Independent single-panel inventory-ledger recomputation oracle (tests and diagnostics only)."""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

from . import _MarkSource
from .accounting import QTY_EPS
from .contracts import ExecutionDataGap, SimulatedInventoryLedgerResult


def _funding_knowledge_matrix(
    funding_known: pd.DataFrame | None, marks: pd.DataFrame
) -> np.ndarray | None:
    """Validate the funding-knowledge frame and read it once as a bool matrix.

    None keeps the legacy semantics in which every bar is known. A present frame must align
    exactly to ``marks`` and be genuinely boolean, so an object- or float-encoded frame can
    never launder an unknown bar into a silently charged rate.
    """
    if funding_known is None:
        return None
    if not funding_known.index.equals(marks.index):
        raise DataIntegrityError("funding_known must share an identical index with marks")
    if list(funding_known.columns) != list(marks.columns):
        raise DataIntegrityError("funding_known must share an identical column order")
    if not bool(funding_known.dtypes.eq("bool").all()):
        raise DataIntegrityError("funding_known must be a boolean frame")
    return np.asarray(funding_known.to_numpy(dtype=bool))


def _validated_availability(
    bar_available_at: pd.DatetimeIndex | None, grid: pd.DatetimeIndex
) -> pd.DatetimeIndex | None:
    """Validate the per-label bar availability stamps, or None for label-time fills.

    The stamps must be one-to-one with the grid, tz-aware UTC, strictly increasing, and never
    earlier than their own label: an inverted or duplicated stamp could not identify a single
    bar, so it fails closed instead of booking a fill on an arbitrary label.
    """
    if bar_available_at is None:
        return None
    if not isinstance(bar_available_at, pd.DatetimeIndex):
        raise DataIntegrityError("bar_available_at must be a DatetimeIndex")
    if len(bar_available_at) != len(grid):
        raise DataIntegrityError("bar_available_at must align one-to-one with the mark grid")
    if bar_available_at.tz is None or str(bar_available_at.tz) != "UTC":
        raise DataIntegrityError("bar_available_at must be tz-aware UTC")
    if not bar_available_at.is_monotonic_increasing or not bar_available_at.is_unique:
        raise DataIntegrityError("bar_available_at must be strictly increasing")
    if bool((bar_available_at < grid).any()):
        raise DataIntegrityError("bar_available_at must not precede its own bar label")
    return bar_available_at


def _fill_label_positions(
    fill_ts: pd.DatetimeIndex,
    grid: pd.DatetimeIndex,
    availability: pd.DatetimeIndex | None,
) -> np.ndarray:
    """Resolve every fill timestamp to the bar-label position it books on.

    With explicit availability stamps the lookup is an exact position match: a fill recorded
    anywhere but on that grid has no identifiable bar and fails closed rather than being snapped
    to a neighbouring label.
    """
    if availability is None:
        if not fill_ts.isin(grid).all():
            raise DataIntegrityError("fills must occur on the mark grid")
        return np.asarray(np.searchsorted(grid, fill_ts), dtype=np.intp)
    positions = np.asarray(availability.get_indexer(fill_ts), dtype=np.intp)
    if bool((positions < 0).any()):
        raise DataIntegrityError("fills must occur on the bar availability grid")
    return positions


class _SymbolPlane(NamedTuple):
    """One symbol's contribution to the six ledger series, plus its first data-gap bar."""

    mtm: np.ndarray
    funding: np.ndarray
    notional: np.ndarray
    notional_before: np.ndarray
    units_state: np.ndarray
    first_held_mark: int | None
    first_held_funding: int | None


def _symbol_plane(
    m: np.ndarray,
    f: np.ndarray,
    sym_finite: np.ndarray,
    known: np.ndarray,
    positions: np.ndarray,
    quantities: np.ndarray,
    grid_index: np.ndarray,
) -> _SymbolPlane:
    """Carry one symbol's inventory across the panel: mark, known funding, then fills.

    Units are the cumulative sum of the symbol's booked fill deltas, so a fill booked at bar
    ``p`` only changes ``units_before`` from ``p + 1`` onward. An unavailable mark is valued at
    exactly zero for a flat position so cash equity stays finite before the first tradable mark;
    a held position at an unavailable mark is carried at its last known mark and reported as
    primary-invalid, which keeps the arithmetic finite and positive instead of leaking
    ``0 * NaN`` or a negative cash shortfall.
    """
    n_grid = len(m)
    deltas = np.zeros(n_grid, dtype="float64")
    np.add.at(deltas, positions, quantities)
    units_state = np.cumsum(deltas)
    units_before = np.zeros(n_grid, dtype="float64")
    units_before[1:] = units_state[:-1]

    last_index = np.maximum.accumulate(np.where(sym_finite, grid_index, 0))
    valuation = np.where(
        sym_finite | (units_state != 0.0),
        np.where(sym_finite, m, m[last_index]),
        0.0,
    )

    held = np.abs(units_before) >= QTY_EPS
    joint = np.zeros(n_grid, dtype=bool)
    joint[1:] = sym_finite[1:] & sym_finite[:-1]
    held_mark = held & ~joint

    mtm = np.zeros(n_grid, dtype="float64")
    mtm[1:] = np.where(joint[1:], units_before[1:] * (m[1:] - m[:-1]), 0.0)

    funding = np.where(sym_finite & known, f * units_before * m, 0.0)
    held_funding = held & (~known | (~sym_finite & (f != 0.0)))

    return _SymbolPlane(
        mtm=mtm,
        funding=funding,
        notional=units_state * valuation,
        notional_before=units_before * valuation,
        units_state=units_state,
        first_held_mark=int(np.argmax(held_mark)) if bool(held_mark.any()) else None,
        first_held_funding=int(np.argmax(held_funding)) if bool(held_funding.any()) else None,
    )


def simulated_inventory_ledger(
    simulated_fills: pd.DataFrame,
    marks: pd.DataFrame,
    bar_funding: pd.DataFrame,
    initial_equity: float,
    fill_source: str,
    mark_source: _MarkSource,
    retain_simulated_units: bool = False,
    *,
    funding_known: pd.DataFrame | None = None,
    bar_available_at: pd.DatetimeIndex | None = None,
) -> SimulatedInventoryLedgerResult:
    """Recompute a cash-and-inventory ledger from a fill stream as an independent oracle.

    This is a verification oracle, not a production PnL source: Research GO, OOS, capital, and
    capacity evidence come from the streamed accumulator ledger of ``replay_execution_windows`` and
    the batch replays. The oracle re-derives the same six series over one full panel by a
    different arithmetic path (per-symbol cumulative sums of fill deltas instead of the
    accumulator's per-window step of booked post-fill levels), so agreement at
    ``rtol=atol=1e-12`` is evidence that window splitting, carry, and booking did not leak or
    invent cash. It must stay out of production imports so it can never become the thing it
    checks.

    Event order per bar: units held since the previous bar are marked first, then known funding
    is charged on the pre-fill quantity times the bar mark, then the bar's fills and fees are
    applied. A fill cannot earn or lose PnL before its bar.

    Held state uses ``abs(units) >= QTY_EPS`` (the accumulator's predicate) because cumulative
    sums of tranche quantities leave sub-epsilon dust after a full close; treating dust as held
    would report fictitious missing marks or funding.

    Args:
        simulated_fills: Fills with ``timestamp``, ``symbol``, ``quantity_delta``, ``fill_price``,
            ``fee_bps`` (extra columns ignored). ``timestamp`` is the bar label when
            ``bar_available_at`` is None, else the bar availability stamp the accumulator records.
        marks: Valuation marks on the full bar-label grid (tz-aware UTC), canonical columns.
        bar_funding: Per-bar funding rates, identical index and column order to ``marks``; finite.
        initial_equity: Starting cash, strictly positive.
        fill_source: Execution bound label carried into the result and gaps.
        mark_source: Valuation source label carried into the result.
        retain_simulated_units: Materialize the dense per-bar units frame (diagnostics only).
        funding_known: Optional boolean frame, identical index and column order to ``marks``;
            ``False`` marks a bar whose funding is unknown. None means every bar is known.
            Unknown funding is never charged; holding through it invalidates the ledger.
        bar_available_at: Optional availability stamp per label, same length as ``marks.index``,
            tz-aware UTC, strictly increasing, each stamp at or after its label. When given, each
            fill timestamp must equal exactly one stamp and is booked at that stamp's label.

    Returns:
        Ledger series on ``marks.index``; ``primary_valid`` is False with ``MISSING_DATA`` when a
        held position (``abs(units_before) >= QTY_EPS``) spans a bar without a finite mark pair,
        holds through unknown funding, or holds through a non-finite mark with a nonzero known
        rate. ``data_gaps`` carries the first ``MISSING_HELD_MARK`` and first
        ``MISSING_HELD_FUNDING``.

    Raises:
        DataIntegrityError: Misaligned or non-UTC inputs, non-finite funding or fills,
            non-positive finite marks, unknown fill symbols, fills off the label (or availability)
            grid, a non-boolean or misaligned ``funding_known``, a malformed
            ``bar_available_at``, non-positive or non-finite pre-trade equity, or non-positive
            equity.
    """
    if initial_equity <= 0:
        raise DataIntegrityError("initial_equity must be > 0")
    if not marks.index.equals(bar_funding.index):
        raise DataIntegrityError("marks and bar_funding must share an identical index")
    if list(marks.columns) != list(bar_funding.columns):
        raise DataIntegrityError("marks and bar_funding must share an identical column order")
    if marks.index.tz is None or bar_funding.index.tz is None:
        raise DataIntegrityError("marks and bar_funding must be tz-aware UTC")
    finite = marks.to_numpy(dtype="float64")
    if not np.isfinite(bar_funding.to_numpy()).all():
        raise DataIntegrityError("bar_funding must be finite")
    finite_positive = finite[np.isfinite(finite)]
    if (finite_positive <= 0).any():
        raise DataIntegrityError("finite marks must be strictly positive")

    marks_values = finite
    finite = np.isfinite(marks_values)
    funding_rates = bar_funding.to_numpy(dtype="float64")
    columns = list(marks.columns)
    grid = marks.index
    known_rates = _funding_knowledge_matrix(funding_known, marks)
    availability = _validated_availability(bar_available_at, grid)

    fills = simulated_fills.copy()
    if fills.empty:
        fills = pd.DataFrame(
            columns=["timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"],
        )
    fills = fills.sort_values("timestamp").reset_index(drop=True)
    fill_ts = pd.DatetimeIndex(pd.to_datetime(fills["timestamp"], utc=True))
    if not fill_ts.is_monotonic_increasing:
        raise DataIntegrityError("simulated fills must be timestamp-sorted")
    unknown_syms = set(fills["symbol"]) - set(columns)
    if unknown_syms:
        raise DataIntegrityError(f"fills reference unknown symbols: {sorted(unknown_syms)}")
    # Multiple intents for one symbol can legitimately resolve on the same
    # coarse execution bar (especially at 5m resolution). The ledger applies
    # them in stable input order while aggregating units, cash flow, and fees
    # at that grid position.

    n_grid = len(grid)

    fill_positions = _fill_label_positions(fill_ts, grid, availability)
    delta_positions: dict[str, list[int]] = {c: [] for c in columns}
    delta_quantities: dict[str, list[float]] = {c: [] for c in columns}
    fill_flow = np.zeros(n_grid, dtype="float64")
    fee_by_ts = np.zeros(n_grid, dtype="float64")
    turnover_terms: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for k, row in enumerate(fills.itertuples(index=False)):
        pos = int(fill_positions[k])
        sym = str(row.symbol)
        qty = float(row.quantity_delta)
        price = float(row.fill_price)
        fee_bps = float(row.fee_bps)
        if not (np.isfinite(qty) and np.isfinite(price) and np.isfinite(fee_bps)):
            raise DataIntegrityError("simulated fills, prices, and fees must be finite")
        fee = fee_bps / 1e4 * abs(qty) * price
        delta_positions[sym].append(pos)
        delta_quantities[sym].append(qty)
        fill_flow[pos] += -(qty * price + fee)
        fee_by_ts[pos] += fee
        turnover_terms[pos].append((qty, price))

    notional = np.zeros(n_grid, dtype="float64")
    notional_before = np.zeros(n_grid, dtype="float64")
    mtm = np.zeros(n_grid, dtype="float64")
    funding_charge = np.zeros(n_grid, dtype="float64")
    funding_by_symbol: dict[str, float] = {}
    funding_daily_cols: dict[str, np.ndarray] = {}
    funding_days, funding_day_positions = np.unique(grid.normalize(), return_inverse=True)
    units_state_by_symbol: list[np.ndarray] | None = [] if retain_simulated_units else None
    grid_index = np.arange(n_grid)
    first_held_mark: tuple[str, int] | None = None
    first_held_funding: tuple[str, int] | None = None
    all_known = np.ones(n_grid, dtype=bool)

    for j, sym in enumerate(columns):
        plane = _symbol_plane(
            marks_values[:, j],
            funding_rates[:, j],
            finite[:, j],
            all_known if known_rates is None else np.asarray(known_rates[:, j], dtype=bool),
            np.asarray(delta_positions[sym], dtype=np.intp),
            np.asarray(delta_quantities[sym], dtype="float64"),
            grid_index,
        )
        mtm += plane.mtm
        funding_charge += plane.funding
        notional += plane.notional
        notional_before += plane.notional_before
        funding_by_symbol[sym] = float(plane.funding.sum())
        funding_daily_cols[sym] = np.bincount(funding_day_positions, weights=plane.funding, minlength=len(funding_days))
        if units_state_by_symbol is not None:
            units_state_by_symbol.append(plane.units_state)
        if plane.first_held_mark is not None and first_held_mark is None:
            first_held_mark = (sym, plane.first_held_mark)
        if plane.first_held_funding is not None and first_held_funding is None:
            first_held_funding = (sym, plane.first_held_funding)

    cash_after = initial_equity + np.cumsum(fill_flow - funding_charge)
    cash_pre_fill = np.empty(n_grid, dtype="float64")
    cash_pre_fill[0] = initial_equity - funding_charge[0]
    cash_pre_fill[1:] = cash_after[:-1] - funding_charge[1:]

    equity_values_arr = cash_after + notional

    turnover_arr = np.zeros(n_grid, dtype="float64")
    for pos, terms in turnover_terms.items():
        pre_trade_equity = cash_pre_fill[pos] + notional_before[pos]
        if not np.isfinite(pre_trade_equity) or pre_trade_equity <= 0:
            raise DataIntegrityError(
                f"pre-trade equity must be positive and finite "
                f"(ts={grid[pos]!r} pre_trade_equity={pre_trade_equity!r})"
            )
        turnover_arr[pos] = sum(
            abs(qty * price) / pre_trade_equity for qty, price in terms
        )

    equity = pd.Series(equity_values_arr, index=grid, dtype="float64")
    if not np.isfinite(equity_values_arr).all() or (equity_values_arr <= 0).any():
        raise DataIntegrityError("simulated inventory equity must be finite and strictly positive")
    simulated_units_df = (
        pd.DataFrame(np.column_stack(units_state_by_symbol), index=grid, columns=columns)
        if units_state_by_symbol is not None
        else None
    )
    ledger_gaps: list[ExecutionDataGap] = []
    if first_held_mark is not None:
        sym, pos = first_held_mark
        ledger_gaps.append(
            ExecutionDataGap(
                code="MISSING_HELD_MARK", symbol=sym, timestamp=grid[pos],
                execution_bound=fill_source,
            )
        )
    if first_held_funding is not None:
        sym, pos = first_held_funding
        ledger_gaps.append(
            ExecutionDataGap(
                code="MISSING_HELD_FUNDING", symbol=sym, timestamp=grid[pos],
                execution_bound=fill_source,
            )
        )
    ledger_gaps.sort(key=lambda g: (g.timestamp, g.code))
    primary_valid = first_held_mark is None and first_held_funding is None
    funding_daily = pd.DataFrame(funding_daily_cols, index=pd.DatetimeIndex(funding_days), dtype="float64")
    funding_daily.index = pd.DatetimeIndex(funding_daily.index, tz="UTC")
    return SimulatedInventoryLedgerResult(
        equity=equity,
        net_returns=equity.pct_change().dropna(),
        simulated_units=simulated_units_df,
        mark_to_market_pnl=pd.Series(mtm, index=grid, dtype="float64"),
        funding_charge=pd.Series(funding_charge, index=grid, dtype="float64"),
        fee_charge=pd.Series(fee_by_ts, index=grid, dtype="float64"),
        fill_turnover=pd.Series(turnover_arr, index=grid, dtype="float64"),
        fill_source=fill_source,
        mark_source=mark_source,
        primary_valid=primary_valid,
        invalid_reasons=() if primary_valid else ("MISSING_DATA",),
        data_gaps=tuple(ledger_gaps),
        funding_by_symbol=funding_by_symbol,
        funding_by_symbol_daily=funding_daily,
    )
