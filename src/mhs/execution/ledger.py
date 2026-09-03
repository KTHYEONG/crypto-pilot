"""Simulated inventory ledger engine."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

from . import _MarkSource
from .contracts import ExecutionDataGap, SimulatedInventoryLedgerResult


def simulated_inventory_ledger(
    simulated_fills: pd.DataFrame,
    marks: pd.DataFrame,
    bar_funding: pd.DataFrame,
    initial_equity: float,
    fill_source: str,
    mark_source: _MarkSource,
    retain_simulated_units: bool = False,
) -> SimulatedInventoryLedgerResult:
    """Compound a timestamp-sorted proxy fill stream into a cash-and-inventory ledger.

    ``marks`` and ``bar_funding`` share an identical UTC index and ordered
    symbol columns. For every interval the units held since the preceding event
    are marked first, then funding published in the interval is charged against
    the pre-fill quantity times mark price, then timestamp-sorted fills and
    their fees are applied. A proxy fill cannot earn or lose PnL before its
    timestamp.

    Symbols are streamed one at a time into six aggregate one-dimensional
    ledger series so only one symbol-length work-buffer set exists at any
    moment; the dense ``simulated_units`` matrix is materialized only when
    ``retain_simulated_units`` is requested by a diagnostic caller.
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
    grid_set = set(grid)

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
    if not fill_ts.isin(grid_set).all():
        raise DataIntegrityError("fills must occur on the mark grid")
    # Multiple intents for one symbol can legitimately resolve on the same
    # coarse execution bar (especially at 5m resolution). The ledger applies
    # them in stable input order while aggregating units, cash flow, and fees
    # at that grid position.

    n_grid = len(grid)

    fill_positions = np.searchsorted(grid, fill_ts)
    delta_positions: dict[str, list[int]] = {c: [] for c in columns}
    delta_quantities: dict[str, list[float]] = {c: [] for c in columns}
    fill_flow = np.zeros(n_grid, dtype="float64")
    fee_by_ts = np.zeros(n_grid, dtype="float64")
    turnover_terms: dict[pd.Timestamp, list[tuple[float, float]]] = defaultdict(list)
    turnover_pos: dict[pd.Timestamp, int] = {}
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
        turnover_terms[row.timestamp].append((qty, price))
        turnover_pos[row.timestamp] = pos

    notional = np.zeros(n_grid, dtype="float64")
    notional_before = np.zeros(n_grid, dtype="float64")
    mtm = np.zeros(n_grid, dtype="float64")
    funding_charge = np.zeros(n_grid, dtype="float64")
    primary_valid = True
    invalid_reasons: set[str] = set()
    units_state_by_symbol: list[np.ndarray] | None = [] if retain_simulated_units else None
    grid_index = np.arange(n_grid)
    first_held_mark: tuple[str, int] | None = None
    first_held_funding: tuple[str, int] | None = None

    for j, sym in enumerate(columns):
        m = marks_values[:, j]
        f = funding_rates[:, j]
        sym_finite = finite[:, j]

        d = np.zeros(n_grid, dtype="float64")
        np.add.at(
            d,
            np.asarray(delta_positions[sym], dtype=np.intp),
            np.asarray(delta_quantities[sym], dtype="float64"),
        )
        units_state = np.cumsum(d)
        units_before = np.zeros(n_grid, dtype="float64")
        units_before[1:] = units_state[:-1]

        # An unavailable mark is valued at exactly zero for a flat position, so
        # cash equity stays finite before the first tradable mark. A held position
        # at an unavailable mark is reported below as primary-invalid and is carried
        # at its last known mark so the ledger arithmetic stays finite and positive
        # instead of leaking ``0 * NaN`` or a negative cash shortfall.
        last_index = np.maximum.accumulate(np.where(sym_finite, grid_index, 0))
        forward = m[last_index]
        valuation = np.where(
            sym_finite | (units_state != 0.0),
            np.where(sym_finite, m, forward),
            0.0,
        )

        held = units_before != 0.0
        joint = np.zeros(n_grid, dtype=bool)
        joint[1:] = sym_finite[1:] & sym_finite[:-1]
        held_mark_trigger = held & ~joint
        if np.any(held_mark_trigger):
            primary_valid = False
            invalid_reasons.add("MISSING_DATA")
            if first_held_mark is None:
                first_held_mark = (sym, int(np.argmax(held_mark_trigger)))

        delta_price = np.zeros(n_grid, dtype="float64")
        delta_price[1:] = m[1:] - m[:-1]
        mtm[1:] += np.where(joint[1:], units_before[1:] * delta_price[1:], 0.0)

        charged = f * units_before * m
        charged = np.where(sym_finite, charged, 0.0)
        held_funding_trigger = ~sym_finite & held & (f != 0.0)
        if np.any(held_funding_trigger):
            primary_valid = False
            invalid_reasons.add("MISSING_DATA")
            if first_held_funding is None:
                first_held_funding = (sym, int(np.argmax(held_funding_trigger)))
        funding_charge += charged

        notional += units_state * valuation
        notional_before += units_before * valuation
        if units_state_by_symbol is not None:
            units_state_by_symbol.append(units_state)

    cash_after = initial_equity + np.cumsum(fill_flow - funding_charge)
    cash_pre_fill = np.empty(n_grid, dtype="float64")
    cash_pre_fill[0] = initial_equity - funding_charge[0]
    cash_pre_fill[1:] = cash_after[:-1] - funding_charge[1:]

    equity_values_arr = cash_after + notional

    turnover_arr = np.zeros(n_grid, dtype="float64")
    for ts, terms in turnover_terms.items():
        pos = turnover_pos[ts]
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
        invalid_reasons=tuple(sorted(invalid_reasons)),
        data_gaps=tuple(ledger_gaps),
    )
