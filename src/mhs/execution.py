"""Execution layer: passive fills, funding panel, and the simulated inventory ledger.

The Research-GO PnL source is ``simulated_inventory_ledger`` fed by the
OHLCV strict-proxy ``strategy_aware_execution_replay``; ``mhs_ledger_pnl`` is
the pinned target-weight *pre-screen* proxy only and must never back Research
GO, OOS, capital, or capacity claims.
"""

import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.contracts import ExecutionSpec
from src.research.baseline.backtest import _align_funding_rates
from src.research.technical_experts.cross_sectional import (
    XsCompositeSpec,
    run_xs_composite_ledger,
    run_xs_composite_ledger_multi_tier,
)

# Conservative extra settlement/slippage penalty applied to a stress-ledger
# (OHLCV_IMMEDIATE_TAKER) UNKNOWN_TERMINATION forced exit (spec §2.17/§7.5).
TERMINATION_STRESS_PENALTY_BPS = 50.0

_ExecutionBound = Literal[
    "OHLCV_STRICT_PROXY", "OHLCV_TOUCH_PROXY", "OHLCV_IMMEDIATE_TAKER", "OHLCV_LADDERED_PROXY"
]
_MarkSource = Literal["MARK_PRICE", "OHLCV_CLOSE_FALLBACK"]

_ExecutionGapCode = Literal[
    "MISSING_DECISION_MARK",
    "MISSING_ACTIVE_ORDER_OHLCV",
    "MISSING_HELD_MARK",
    "MISSING_HELD_FUNDING",
    "MISSING_FORCED_EXIT_CLOSE",
]


def passive_fill_shortfall_bps(
    decision_price: float,
    adverse_path: np.ndarray,
    timeout_price: float,
    side: int,
    spec: ExecutionSpec,
) -> float:
    """Implementation shortfall of one passive order against its decision price.

    ``side=+1`` is a buy and ``adverse_path`` carries the window's lows;
    ``side=-1`` is a sell and ``adverse_path`` carries the window's highs. A
    fill costs exactly the maker fee; a no-fill crosses at the timeout price
    and pays the all-in taker cost, so fee and adverse selection are always
    accounted together.
    """
    if decision_price <= 0 or timeout_price <= 0:
        raise ValueError("decision_price and timeout_price must be > 0")
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    if adverse_path.size == 0:
        raise ValueError("adverse_path must not be empty")

    extreme = float(np.min(adverse_path)) if side == 1 else float(np.max(adverse_path))
    if side == 1:
        filled = (
            extreme < decision_price if spec.require_trade_through else extreme <= decision_price
        )
    else:
        filled = (
            extreme > decision_price if spec.require_trade_through else extreme >= decision_price
        )
    if filled:
        return float(spec.maker_fee_bps)
    move_bps = side * (timeout_price / decision_price - 1.0) * 1e4
    return float(move_bps + spec.taker_fee_bps + spec.taker_slippage_bps)


def notional_weighted_shortfall_bps(
    shortfalls: Iterable[float],
    notionals: Iterable[float],
) -> float:
    """Notional-weighted mean of per-fill implementation shortfalls in bps.

    The unweighted ``np.mean`` of per-fill shortfalls over-weights the many
    small fills that dominate the count but carry little capital. The
    economically correct aggregate weights each fill's shortfall by its
    absolute notional ``abs(qty) * fill_price``:
    ``sum(shortfall_i * notional_i) / sum(notional_i)``. Returns ``nan``
    (never ``0.0``, which would read as free execution, and never a
    ``ZeroDivisionError``) when no fills occurred or the total notional is
    zero.
    """
    shortfall_arr = np.asarray(list(shortfalls), dtype="float64")
    notional_arr = np.asarray(list(notionals), dtype="float64")
    if shortfall_arr.size == 0 or notional_arr.size == 0:
        return float("nan")
    total_notional = float(notional_arr.sum())
    if total_notional <= 0.0 or not np.isfinite(total_notional):
        return float("nan")
    return float(np.sum(shortfall_arr * notional_arr) / total_notional)


def laddered_fill_schedule(
    decision_price: float,
    side: int,
    adverse: np.ndarray,
    closes: np.ndarray,
    tranche_count: int,
    spec: ExecutionSpec,
    require_strict: bool,
) -> list[tuple[int, float, float, float]]:
    """Split one order into an escalating ladder of ``tranche_count`` limit sub-windows.

    The OHLCV execution window ``[0, len(adverse))`` is split into
    ``tranche_count`` equal-width sub-windows (the last absorbs any remainder
    bars); each sub-window reuses the existing binary trade-through predicate
    (``require_strict`` selects the strict ``<``/``>`` vs the touch ``<=``/``>=``
    comparison operators used at the inline STRICT/TOUCH branches) against its
    own limit price. Tranche 1 rests at ``decision_price``; tranche ``k > 1``
    reprices linearly toward the market by ``side * (k-1)/tranche_count`` of the
    gap to the previous sub-window's closing anchor ``closes[sub_end_{k-1}]``
    (the boundary bar's close, matching the codebase's timeout-close
    convention). A tranche whose sub-window trades through fills its accumulated
    carried share at its own limit price with the maker fee; only the final
    tranche, if it never trades through, converts its remaining share to an
    immediate market fill at the final sub-window's close with the all-in taker
    cost. Non-final tranches that fail carry their share forward without a
    market fallback.

    ``closes`` must be at least as long as ``adverse`` (the production callers
    pass one extra boundary close at index ``len(adverse)``); the market
    fallback uses that boundary close when present so ``tranche_count == 1``
    reproduces the pre-existing STRICT/TOUCH single-fill fallback exactly.
    Returns ``(relative_fill_position, fill_price, fee_bps, qty_fraction)``
    tuples in fill order with ``qty_fraction`` summing to 1.0.
    """
    if tranche_count < 1:
        raise ValueError(f"tranche_count must be >= 1, got {tranche_count}")
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    if decision_price <= 0:
        raise ValueError("decision_price must be > 0")
    n = len(adverse)
    if n == 0:
        raise ValueError("adverse must not be empty")
    if not np.isfinite(adverse).all():
        raise ValueError("adverse must be finite")
    if len(closes) < n:
        raise ValueError("closes must be at least as long as adverse")
    if not np.isfinite(closes[: n + 1]).all():
        raise ValueError("closes must be finite")

    schedule: list[tuple[int, float, float, float]] = []
    carried = 0.0
    own_share = 1.0 / tranche_count
    for k in range(1, tranche_count + 1):
        sub_start = (k - 1) * n // tranche_count
        sub_end = k * n // tranche_count if k < tranche_count else n
        if k == 1:
            limit_price = decision_price
        else:
            anchor = float(closes[sub_start])
            limit_price = decision_price + side * (k - 1) / tranche_count * (anchor - decision_price)
        sub = adverse[sub_start:sub_end]
        if side == 1:
            crossed = (sub < limit_price) if require_strict else (sub <= limit_price)
        else:
            crossed = (sub > limit_price) if require_strict else (sub >= limit_price)
        if crossed.any():
            hit = int(np.argmax(crossed))
            schedule.append(
                (sub_start + hit, float(limit_price), float(spec.maker_fee_bps), carried + own_share)
            )
            carried = 0.0
        else:
            carried += own_share
    if carried > 0.0:
        fallback_close = float(closes[min(n, len(closes) - 1)])
        schedule.append(
            (n, fallback_close, float(spec.taker_fee_bps + spec.taker_slippage_bps), carried)
        )
    return schedule


def bar_funding_panel(
    funding_by_symbol: Mapping[str, pd.Series],
    grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Column-wise ``_align_funding_rates`` reuse over the decision grid.

    A symbol whose funding series cannot be causally aligned is excluded from
    the output: missing funding is never silently zero-filled.
    """
    cols: dict[str, pd.Series] = {}
    for sym, series in funding_by_symbol.items():
        try:
            cols[sym] = pd.Series(_align_funding_rates(series, grid), index=grid, dtype="float64")
        except DataIntegrityError:
            continue
    df = pd.DataFrame(cols, index=grid)
    # Sanitize internal alignment NaNs/Infs by forward filling and zero-filling
    return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


@dataclass(frozen=True, slots=True)
class ExecutionDataGap:
    """Deterministic provenance for one cache-required data gap.

    ``decision_time``/``signal_time`` carry the intent that would have traded
    through the gap (null when no intent applies, e.g. a held-position mark or
    funding gap). Terminal-window censoring is report telemetry and is never
    represented by this record.
    """

    code: _ExecutionGapCode
    symbol: str
    timestamp: pd.Timestamp
    decision_time: pd.Timestamp | None = None
    signal_time: pd.Timestamp | None = None
    execution_bound: str = "OHLCV_STRICT_PROXY"


@dataclass(frozen=True, slots=True)
class ExecutionReplayWindow:
    """One chronological execution window fed to ``replay_execution_windows``.

    ``columns`` is the canonical artifact column order (identical across every
    window); ``symbols`` is this window's active roster actually present in the
    ``highs``/``lows``/``closes``/``marks``/``bar_funding`` frames. The minute
    grid covers the strict timeout overlap of the window's final order plus the
    boundary bars the engine needs for decision-time funding and MTM.
    """

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    columns: tuple[str, ...]
    symbols: tuple[str, ...]
    minute_grid: pd.DatetimeIndex
    highs: pd.DataFrame
    lows: pd.DataFrame
    closes: pd.DataFrame
    marks: pd.DataFrame | None
    bar_funding: pd.DataFrame
    target_weights: pd.DataFrame
    signal_available_at: pd.DatetimeIndex


@dataclass(frozen=True, slots=True)
class ForwardExecutionObservation:
    """Phase 4B forward collection record for one signal intent.

    Every intent is recorded, including rejected, cancelled, unfilled, and
    partial-filled orders. This data calibrates proxy fill/cost bounds and
    gates Execution/Pilot/Scale; it must never alter an already frozen signal,
    stop, exit, or sizing architecture after final OOS.
    """

    symbol: str
    signal_time: pd.Timestamp
    intent_time: pd.Timestamp
    submit_time: pd.Timestamp | None
    fill_time: pd.Timestamp | None
    side: int
    requested_quantity: float
    filled_quantity: float
    limit_price: float | None
    fill_price: float | None
    best_bid: float | None
    best_ask: float | None
    top_n_depth_notional: float | None
    trade_print_notional: float | None
    reject_reason: str | None
    cancel_replace_count: int
    latency_ms: int | None


@dataclass(frozen=True, slots=True)
class SimulatedInventoryLedgerResult:
    """The only PnL source allowed for Research GO, OOS, and capital metrics."""

    equity: pd.Series
    net_returns: pd.Series
    simulated_units: pd.DataFrame | None
    mark_to_market_pnl: pd.Series
    funding_charge: pd.Series
    fee_charge: pd.Series
    fill_turnover: pd.Series
    fill_source: str
    mark_source: _MarkSource
    primary_valid: bool
    invalid_reasons: tuple[str, ...]
    equity_floor_breached_at: tuple[pd.Timestamp, ...] = ()
    data_gaps: tuple[ExecutionDataGap, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyExecutionReplayResult:
    """Outcome of one OHLCV proxy-execution replay over target weights."""

    simulated_fills: pd.DataFrame
    ledger: SimulatedInventoryLedgerResult
    simulated_units: pd.DataFrame
    simulated_notional_weights: pd.DataFrame
    fill_source: _ExecutionBound
    mark_source: _MarkSource
    submit_times: pd.Series
    fill_times: pd.Series
    fill_count: int
    unfilled_count: int
    fallback_count: int
    all_intent_shortfall_bps: float
    forced_exit_count: int
    forced_exit_notional: float
    termination_counts: Mapping[str, int]
    unsupported_assumptions: tuple[str, ...]
    elapsed_seconds: float
    data_gaps: tuple[ExecutionDataGap, ...] = ()
    event_snapshots_retained: bool = True
    notional_weighted_shortfall_bps: float = float("nan")


@dataclass(frozen=True, slots=True)
class IsolatedBoundFailure:
    bound_index: int
    execution_bound: str
    error_class: str
    message: str
    windows_consumed: int


@dataclass(frozen=True, slots=True)
class BatchReplayOutcome:
    results: tuple[StrategyExecutionReplayResult | None, ...]
    isolated_failures: tuple[IsolatedBoundFailure, ...]


def ruin_guard_equity(fill_track_equity: float, last_ledger_equity: float | None) -> float:
    if last_ledger_equity is None:
        return float(fill_track_equity)
    if not np.isfinite(last_ledger_equity):
        return float(fill_track_equity)
    return float(min(fill_track_equity, last_ledger_equity))


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
        "OHLCV_STRICT_PROXY", "OHLCV_TOUCH_PROXY", "OHLCV_IMMEDIATE_TAKER", "OHLCV_LADDERED_PROXY"
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
        mark_source: _MarkSource = "MARK_PRICE"
    else:
        marks = minute_closes
        mark_source = "OHLCV_CLOSE_FALLBACK"

    minute_grid = minute_closes.index
    # Normalize to nanoseconds regardless of the pandas index resolution so the
    # searchsorted bounds and timeout arithmetic use one canonical epoch unit.
    grid_ns = np.asarray(minute_grid, dtype="datetime64[ns]").astype("int64")
    n_grid = len(grid_ns)
    if not bar_funding.index.equals(minute_grid):
        raise DataIntegrityError("bar_funding must align exactly to the minute grid")
    symbols = list(target_weights.columns)
    n_cols = len(symbols)
    require_strict = execution_bound == "OHLCV_STRICT_PROXY"

    marks_values = marks[symbols].to_numpy(dtype="float64")
    highs_values = minute_highs[symbols].to_numpy(dtype="float64")
    lows_values = minute_lows[symbols].to_numpy(dtype="float64")
    closes_values = minute_closes[symbols].to_numpy(dtype="float64")
    close_finite = np.isfinite(closes_values)
    mark_valid = np.isfinite(marks_values) & (marks_values > 0.0)
    funding_matrix = np.stack([bar_funding[s].to_numpy(dtype="float64") for s in symbols], axis=1)

    last_reliable: dict[str, pd.Timestamp] = {}
    for sym in symbols:
        valid = minute_closes[sym].dropna()
        last_reliable[sym] = valid.index[-1] if len(valid) else minute_grid[0]

    units_arr = np.zeros(n_cols, dtype="float64")
    cash = float(initial_equity)
    last_prices_arr = np.full(n_cols, np.nan, dtype="float64")
    last_time_ns: int | None = None

    def _equity_at() -> float:
        return cash + float(np.sum(units_arr * np.nan_to_num(last_prices_arr, nan=0.0)))

    def _advance(target_ns: int, dpos: int, on_grid: bool) -> None:
        nonlocal cash, last_time_ns, last_prices_arr
        if last_time_ns is not None and target_ns < last_time_ns:
            raise DataIntegrityError("decision times must be monotonically increasing")
        if on_grid:
            m = marks_values[dpos]
            finite = np.isfinite(m)
            prev = last_prices_arr
            mark_changed = finite & np.isfinite(prev)
            if mark_changed.any():
                cash += float(np.sum(units_arr[mark_changed] * (m[mark_changed] - prev[mark_changed])))
            last_prices_arr = np.where(finite, m, prev)
        lo = np.searchsorted(grid_ns, last_time_ns, side="right") if last_time_ns is not None else 0
        hi = int(np.searchsorted(grid_ns, target_ns, side="right"))
        if lo < hi:
            rates_block = funding_matrix[lo:hi, :]
            priced = np.isfinite(last_prices_arr)
            cash -= float(np.sum(rates_block * units_arr * np.where(priced, last_prices_arr, 0.0)))
        last_time_ns = target_ns

    def _decision_price(col: int, on_grid: bool, dpos: int, spos: int) -> float | None:
        if on_grid and mark_valid[dpos, col]:
            return float(marks_values[dpos, col])
        j = spos - 1
        while j >= 0 and not close_finite[j, col]:
            j -= 1
        if j >= 0 and mark_valid[j, col]:
            return float(marks_values[j, col])
        if spos < n_grid and mark_valid[spos, col]:
            return float(marks_values[spos, col])
        return None

    fill_records: list[dict[str, object]] = []
    submit_times: list[pd.Timestamp] = []
    fill_times: list[pd.Timestamp] = []
    shortfalls: list[float] = []
    shortfall_notionals: list[float] = []
    fill_count = 0
    unfilled_count = 0
    fallback_count = 0
    termination_counts: dict[str, int] = {"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0}
    data_gaps: list[ExecutionDataGap] = []
    units_after_events: list[tuple[pd.Timestamp, np.ndarray]] = []
    notional_after_events: list[tuple[pd.Timestamp, np.ndarray]] = []

    decision_ns_all = np.asarray(target_weights.index, dtype="datetime64[ns]").astype("int64")
    signal_ns_all = np.asarray(signal_available_at, dtype="datetime64[ns]").astype("int64")
    spos_all = np.searchsorted(grid_ns, signal_ns_all, side="right")
    dpos_all = np.searchsorted(grid_ns, decision_ns_all, side="left")
    dpos_clipped = np.minimum(dpos_all, n_grid - 1)
    on_grid_all = np.where(
        dpos_all < n_grid, grid_ns[dpos_clipped] == decision_ns_all, False,
    )
    target_values = target_weights.to_numpy(dtype="float64")

    _t0 = time.perf_counter()
    for i, decision_time in enumerate(target_weights.index):
        dns = int(decision_ns_all[i])
        dpos = int(dpos_all[i])
        on_grid = bool(on_grid_all[i])
        _advance(dns, dpos, on_grid)
        equity = _equity_at()
        row = target_values[i]
        spos = int(spos_all[i])
        signal_time = signal_available_at[i]

        # Active columns: finite non-zero targets plus non-zero held units. The
        # held-units term preserves the existing zero-target close behavior for
        # inventory left over when a symbol leaves the target set.
        active = np.where(np.isfinite(row) & ((row != 0.0) | (units_arr != 0.0)))[0]
        for col in active.tolist():
            sym = symbols[col]
            weight = float(row[col])
            decision_price = _decision_price(col, on_grid, dpos, spos)
            if decision_price is None:
                termination_counts["MISSING_DATA"] += 1
                data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_DECISION_MARK", symbol=sym, timestamp=decision_time,
                        decision_time=decision_time, signal_time=signal_time,
                        execution_bound=execution_bound,
                    )
                )
                continue
            if not np.isfinite(last_prices_arr[col]):
                last_prices_arr[col] = decision_price
            desired_units = weight * equity / decision_price
            net_units = desired_units - units_arr[col]
            if abs(net_units) < 1e-12:
                continue
            side = 1 if net_units > 0 else -1

            if spos >= n_grid:
                termination_counts["MISSING_DATA"] += 1
                continue
            submit_pos = spos
            timeout_ns = grid_ns[spos] + int(spec.passive_timeout_minutes) * 60_000_000_000
            timeout_pos = int(np.searchsorted(grid_ns, timeout_ns, side="left"))
            timeout_close = float("nan")
            adverse = np.array([], dtype="float64")

            if execution_bound == "OHLCV_IMMEDIATE_TAKER":
                fill_pos = submit_pos
                fill_price = float(closes_values[fill_pos, col])
                if not np.isfinite(fill_price):
                    termination_counts["MISSING_DATA"] += 1
                    data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                            timestamp=minute_grid[fill_pos], decision_time=decision_time,
                            signal_time=signal_time, execution_bound=execution_bound,
                        )
                    )
                    continue
                fee_bps = spec.taker_fee_bps + spec.taker_slippage_bps
                reason = "timeout_taker"
            else:
                if execution_bound == "OHLCV_LADDERED_PROXY":
                    if timeout_pos <= spos:
                        termination_counts["MISSING_DATA"] += 1
                        data_gaps.append(
                            ExecutionDataGap(
                                code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                timestamp=minute_grid[spos], decision_time=decision_time,
                                signal_time=signal_time, execution_bound=execution_bound,
                            )
                        )
                        continue
                    adverse = (
                        lows_values[spos:timeout_pos, col]
                        if side == 1
                        else highs_values[spos:timeout_pos, col]
                    )
                    if not np.isfinite(adverse).all():
                        termination_counts["MISSING_DATA"] += 1
                        first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
                        data_gaps.append(
                            ExecutionDataGap(
                                code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                timestamp=minute_grid[first_bad], decision_time=decision_time,
                                signal_time=signal_time, execution_bound=execution_bound,
                            )
                        )
                        continue
                    closes_window = closes_values[spos:timeout_pos + 1, col]
                    if not np.isfinite(closes_window).all():
                        termination_counts["MISSING_DATA"] += 1
                        first_bad = spos + int(np.argmax(~np.isfinite(closes_window)))
                        data_gaps.append(
                            ExecutionDataGap(
                                code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                timestamp=minute_grid[first_bad], decision_time=decision_time,
                                signal_time=signal_time, execution_bound=execution_bound,
                            )
                        )
                        continue
                    for rel_pos, tranche_price, tranche_fee_bps, qty_fraction in laddered_fill_schedule(
                        decision_price, side, adverse,
                        closes_window,
                        spec.ladder_tranches, spec, True,
                    ):
                        fill_pos = spos + rel_pos
                        if rel_pos == len(adverse):
                            if timeout_pos >= n_grid or grid_ns[timeout_pos] != timeout_ns:
                                termination_counts["MISSING_DATA"] += 1
                                data_gaps.append(
                                    ExecutionDataGap(
                                        code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                        timestamp=minute_grid[spos], decision_time=decision_time,
                                        signal_time=signal_time, execution_bound=execution_bound,
                                    )
                                )
                                continue
                            timeout_close = float(closes_values[timeout_pos, col])
                            if not np.isfinite(timeout_close):
                                termination_counts["MISSING_DATA"] += 1
                                data_gaps.append(
                                    ExecutionDataGap(
                                        code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                        timestamp=minute_grid[timeout_pos], decision_time=decision_time,
                                        signal_time=signal_time, execution_bound=execution_bound,
                                    )
                                )
                                continue
                            unfilled_count += 1
                            fallback_count += 1
                            reason = "timeout_taker"
                        else:
                            fill_count += 1
                            reason = "passive_fill"
                        fill_price = float(tranche_price)
                        fee_bps = float(tranche_fee_bps)
                        qty = net_units * float(qty_fraction)
                        if reason == "passive_fill":
                            shortfall = side * (fill_price / decision_price - 1.0) * 1e4 + spec.maker_fee_bps
                        else:
                            shortfall = (
                                side * (fill_price / decision_price - 1.0) * 1e4
                                + spec.taker_fee_bps + spec.taker_slippage_bps
                            )
                        shortfalls.append(shortfall)
                        shortfall_notionals.append(abs(qty) * fill_price)
                        fill_time = minute_grid[fill_pos]
                        submit_time = minute_grid[submit_pos]
                        mark_price = float(marks_values[fill_pos, col])
                        if np.isfinite(last_prices_arr[col]):
                            if np.isfinite(mark_price):
                                cash += units_arr[col] * (mark_price - last_prices_arr[col])
                                last_prices_arr[col] = mark_price
                        elif np.isfinite(mark_price):
                            last_prices_arr[col] = mark_price
                        cash -= qty * fill_price
                        fee = fee_bps / 1e4 * abs(qty) * fill_price
                        cash -= fee
                        units_arr[col] += qty
                        pre_trade_equity = _equity_at()
                        fill_records.append(
                            {
                                "timestamp": fill_time,
                                "symbol": sym,
                                "quantity_delta": qty,
                                "fill_price": fill_price,
                                "fee_bps": fee_bps,
                                "reason": reason,
                                "pre_trade_equity": pre_trade_equity,
                            }
                        )
                        fill_times.append(fill_time)
                        submit_times.append(submit_time)
                        units_after_events.append((fill_time, units_arr.copy()))
                        notional_after_events.append(
                            (fill_time, units_arr * marks_values[fill_pos, :])
                        )
                    continue
                if timeout_pos <= spos:
                    termination_counts["MISSING_DATA"] += 1
                    data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                            timestamp=minute_grid[spos], decision_time=decision_time,
                            signal_time=signal_time, execution_bound=execution_bound,
                        )
                    )
                    continue
                adverse = (
                    lows_values[spos:timeout_pos, col]
                    if side == 1
                    else highs_values[spos:timeout_pos, col]
                )
                if not np.isfinite(adverse).all():
                    termination_counts["MISSING_DATA"] += 1
                    first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
                    data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                            timestamp=minute_grid[first_bad], decision_time=decision_time,
                            signal_time=signal_time, execution_bound=execution_bound,
                        )
                    )
                    continue
                if side == 1:
                    crossed = (
                        (adverse < decision_price)
                        if require_strict
                        else (adverse <= decision_price)
                    )
                else:
                    crossed = (
                        (adverse > decision_price)
                        if require_strict
                        else (adverse >= decision_price)
                    )
                if crossed.any():
                    hit = int(np.argmax(crossed))
                    fill_pos = spos + hit
                    fill_price = decision_price
                    fee_bps = spec.maker_fee_bps
                    reason = "passive_fill"
                else:
                    if timeout_pos >= n_grid or grid_ns[timeout_pos] != timeout_ns:
                        termination_counts["MISSING_DATA"] += 1
                        data_gaps.append(
                            ExecutionDataGap(
                                code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                timestamp=minute_grid[spos], decision_time=decision_time,
                                signal_time=signal_time, execution_bound=execution_bound,
                            )
                        )
                        continue
                    timeout_close = float(closes_values[timeout_pos, col])
                    if not np.isfinite(timeout_close):
                        termination_counts["MISSING_DATA"] += 1
                        data_gaps.append(
                            ExecutionDataGap(
                                code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                timestamp=minute_grid[timeout_pos], decision_time=decision_time,
                                signal_time=signal_time, execution_bound=execution_bound,
                            )
                        )
                        continue
                    unfilled_count += 1
                    fallback_count += 1
                    fill_pos = timeout_pos
                    fill_price = timeout_close
                    fee_bps = spec.taker_fee_bps + spec.taker_slippage_bps
                    reason = "timeout_taker"

            if reason == "passive_fill":
                fill_count += 1
            if execution_bound == "OHLCV_IMMEDIATE_TAKER":
                shortfall = side * (fill_price / decision_price - 1.0) * 1e4 + fee_bps
            else:
                shortfall = passive_fill_shortfall_bps(
                    decision_price, adverse, timeout_close, side, spec,
                )
            shortfalls.append(shortfall)
            shortfall_notionals.append(abs(net_units) * fill_price)

            fill_time = minute_grid[fill_pos]
            submit_time = minute_grid[submit_pos]

            mark_price = float(marks_values[fill_pos, col])
            if np.isfinite(last_prices_arr[col]):
                if np.isfinite(mark_price):
                    cash += units_arr[col] * (mark_price - last_prices_arr[col])
                    last_prices_arr[col] = mark_price
            elif np.isfinite(mark_price):
                last_prices_arr[col] = mark_price
            cash -= net_units * fill_price
            fee = fee_bps / 1e4 * abs(net_units) * fill_price
            cash -= fee
            units_arr[col] += net_units
            if reason in ("passive_fill", "timeout_taker"):
                pre_trade_equity = _equity_at()
                fill_records.append(
                    {
                        "timestamp": fill_time,
                        "symbol": sym,
                        "quantity_delta": net_units,
                        "fill_price": fill_price,
                        "fee_bps": fee_bps,
                        "reason": reason,
                        "pre_trade_equity": pre_trade_equity,
                    }
                )
                fill_times.append(fill_time)
                submit_times.append(submit_time)
                units_after_events.append((fill_time, units_arr.copy()))
                notional_after_events.append(
                    (fill_time, units_arr * marks_values[fill_pos, :])
                )

    # Persistent source-end gap with held units: UNKNOWN_TERMINATION forced exit.
    forced_exit_count = 0
    forced_exit_notional = 0.0
    grid_end = minute_grid[-1]
    for col in range(n_cols):
        sym = symbols[col]
        if units_arr[col] == 0.0:
            continue
        if last_reliable[sym] >= grid_end:
            continue
        exit_pos = int(np.searchsorted(grid_ns, last_reliable[sym].value, side="left"))
        exit_time = minute_grid[exit_pos]
        exit_price = float(closes_values[exit_pos, col])
        if not np.isfinite(exit_price) or exit_price <= 0:
            termination_counts["MISSING_DATA"] += 1
            data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_FORCED_EXIT_CLOSE", symbol=sym, timestamp=exit_time,
                    execution_bound=execution_bound,
                )
            )
            continue
        termination_counts["UNKNOWN_TERMINATION"] += 1
        forced_exit_count += 1
        forced_exit_notional += abs(units_arr[col] * exit_price)
        penalty = (
            TERMINATION_STRESS_PENALTY_BPS
            if execution_bound == "OHLCV_IMMEDIATE_TAKER"
            else 0.0
        )
        fee_bps = spec.taker_fee_bps + spec.taker_slippage_bps + penalty
        prev_price = (
            float(last_prices_arr[col])
            if np.isfinite(last_prices_arr[col])
            else exit_price
        )
        mark_price = float(marks_values[exit_pos, col])
        if np.isfinite(mark_price):
            cash -= units_arr[col] * (mark_price - prev_price)
            last_prices_arr[col] = mark_price
        cash -= -units_arr[col] * exit_price
        fee = fee_bps / 1e4 * abs(units_arr[col]) * exit_price
        cash -= fee
        fill_records.append(
            {
                "timestamp": exit_time,
                "symbol": sym,
                "quantity_delta": -units_arr[col],
                "fill_price": exit_price,
                "fee_bps": fee_bps,
                "reason": "forced_exit",
                "pre_trade_equity": _equity_at(),
            }
        )
        fill_times.append(exit_time)
        units_arr[col] = 0.0
        units_after_events.append((exit_time, units_arr.copy()))
    elapsed_seconds = time.perf_counter() - _t0

    simulated_fills = pd.DataFrame(
        fill_records,
        columns=[
            "timestamp", "symbol", "quantity_delta", "fill_price",
            "fee_bps", "reason", "pre_trade_equity",
        ],
    )
    if simulated_fills.empty:
        simulated_fills = simulated_fills.astype(
            {"quantity_delta": "float64", "fill_price": "float64", "fee_bps": "float64"}
        )

    ledger = simulated_inventory_ledger(
        simulated_fills, marks, bar_funding, initial_equity, execution_bound, mark_source,
        retain_simulated_units=False,
    )
    if units_after_events:
        events_index = pd.DatetimeIndex([t for t, _ in units_after_events])
        simulated_units = pd.DataFrame(
            [row for _t, row in units_after_events], index=events_index, columns=symbols,
        )
        notional_events_index = pd.DatetimeIndex([t for t, _ in notional_after_events])
        simulated_notional_weights = pd.DataFrame(
            [row for _t, row in notional_after_events],
            index=notional_events_index,
            columns=symbols,
        )
    else:
        simulated_units = pd.DataFrame(columns=symbols)
        simulated_notional_weights = pd.DataFrame(columns=symbols)

    all_intent_shortfall_bps = (
        float(np.mean(shortfalls)) if shortfalls else float("nan")
    )
    weighted_shortfall_bps = notional_weighted_shortfall_bps(
        shortfalls, shortfall_notionals
    )
    data_gaps.extend(ledger.data_gaps)
    data_gaps.sort(key=lambda g: (g.timestamp, g.code, g.symbol))
    return StrategyExecutionReplayResult(
        simulated_fills=simulated_fills,
        ledger=ledger,
        simulated_units=simulated_units,
        simulated_notional_weights=simulated_notional_weights,
        fill_source=execution_bound,
        mark_source=mark_source,
        submit_times=pd.Series(submit_times, dtype="datetime64[ns, UTC]"),
        fill_times=pd.Series(fill_times, dtype="datetime64[ns, UTC]"),
        fill_count=fill_count,
        unfilled_count=unfilled_count,
        fallback_count=fallback_count,
        all_intent_shortfall_bps=all_intent_shortfall_bps,
        forced_exit_count=forced_exit_count,
        forced_exit_notional=forced_exit_notional,
        termination_counts=termination_counts,
        unsupported_assumptions=(
            "partial_fill",
            "queue_position",
            "post_only_rejection",
            "cancel_replace_latency",
            "order_size_impact",
        ),
        elapsed_seconds=elapsed_seconds,
        data_gaps=tuple(data_gaps),
        notional_weighted_shortfall_bps=weighted_shortfall_bps,
    )


def mhs_ledger_pnl(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    one_way_bps: float,
    execution_delay_bars: int = 1,
    gap_carry: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Pinned target-weight pre-screen proxy delegating to ``run_xs_composite_ledger``.

    ``XsCompositeSpec.halflife_bars`` and ``no_trade_band`` are inert passthrough
    constants here (they only affect weight-construction call sites this module
    never calls). The rebalances target notional implicitly, so it must never be
    used for Research GO, OOS, capital metrics, or capacity claims.
    """
    half = one_way_bps / 2.0 * 1e-4
    spec = XsCompositeSpec(
        halflife_bars=0,
        no_trade_band=0.0,
        execution_delay_bars=execution_delay_bars,
        fee_rate=half,
        slippage_rate=half,
        gap_carry=gap_carry,
    )
    equity, turnover = run_xs_composite_ledger(weights, opens, bar_funding, spec)
    net = equity.pct_change().dropna()
    return net, turnover


def mhs_ledger_pnl_multi_tier(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    one_way_bps_list: Sequence[float],
    execution_delay_bars: int = 1,
    gap_carry: bool = True,
) -> list[tuple[pd.Series, pd.Series]]:
    """Single-pass multi-tier pre-screen proxy sharing the ledger arrays.

    Mirrors ``mhs_ledger_pnl`` exactly for each entry in ``one_way_bps_list``:
    the spec is built with ``fee_rate = slippage_rate = bps / 2.0 * 1e-4`` and
    the frozen round-trip rate ``half + half`` (IEEE doubling is exact, so it
    equals the single call's ``round_trip_cost_rate()`` bit-for-bit). The shared
    array construction means element ``i``'s ``(net, turnover)`` is bit-identical
    to ``mhs_ledger_pnl(weights, opens, bar_funding, bps_i)`` for the same
    index. Like ``mhs_ledger_pnl`` this is a pinned pre-screen proxy -- never
    Research GO, OOS, capital metrics, or capacity claims. Raises ``ValueError``
    on an empty list or any negative bps.
    """
    if not one_way_bps_list:
        raise ValueError("one_way_bps_list must not be empty")
    for bps in one_way_bps_list:
        if bps < 0.0:
            raise ValueError(f"one_way_bps must be >= 0, got {bps}")

    base_spec = XsCompositeSpec(
        halflife_bars=0,
        no_trade_band=0.0,
        execution_delay_bars=execution_delay_bars,
        fee_rate=0.0,
        slippage_rate=0.0,
        gap_carry=gap_carry,
    )
    cost_rates = [bps / 2.0 * 1e-4 + bps / 2.0 * 1e-4 for bps in one_way_bps_list]
    results = run_xs_composite_ledger_multi_tier(
        weights, opens, bar_funding, base_spec, cost_rates,
    )
    return [(equity.pct_change().dropna(), turnover) for equity, turnover in results]

class _BoundExecutionReplayAccumulator:
    """Private streaming accumulator for one execution bound.

    ``replay_execution_windows`` and ``replay_execution_window_pair`` share
    this bound-specific state machine. Windows are consumed one at a time:
    cash, units, last prices, the last finite-close mark provenance, and the
    streamed ledger carry into the next window, and a completed window's
    frames are released before the next is read. Each window's grid covers the
    strict timeout overlap of its final order plus the boundary bars needed
    for decision-time funding/MTM, so an order never crosses a window boundary
    unresolved. The six ledger series are computed per window in chronological
    order and concatenated once in ``finalize``, matching the single-panel
    oracle at ``rtol=atol=1e-12`` where the inputs are equal.

    ``retain_event_snapshots`` defaults to ``False`` for bounded memory: the
    dense per-fill ``simulated_units``/``simulated_notional_weights`` event
    tables are then empty (correctly columned) and ``event_snapshots_retained``
    is ``False``, so empty tables cannot be mistaken for no fills. Diagnostic
    callers that compare event snapshots (the single-panel oracle and
    equivalence tests) must explicitly opt in with ``True``; the ledger, fills,
    gaps, termination data, and numerical results are identical either way.
    """

    def __init__(
        self,
        first: ExecutionReplayWindow,
        initial_equity: float,
        execution_bound: _ExecutionBound,
        spec: ExecutionSpec,
        retain_event_snapshots: bool,
        min_equity_fraction: float | None = None,
    ) -> None:
        if initial_equity <= 0:
            raise DataIntegrityError("initial_equity must be > 0")
        if min_equity_fraction is not None and not (0.0 < min_equity_fraction < 1.0):
            raise ValueError(f"min_equity_fraction must be in (0.0, 1.0) when set, got {min_equity_fraction}")
        self.min_equity_fraction = min_equity_fraction
        self.initial_equity = float(initial_equity)
        self.equity_floor_breaches: list[pd.Timestamp] = []
        if execution_bound not in (
            "OHLCV_STRICT_PROXY", "OHLCV_TOUCH_PROXY", "OHLCV_IMMEDIATE_TAKER", "OHLCV_LADDERED_PROXY"
        ):
            raise ValueError(f"unknown execution_bound '{execution_bound}'")
        self.execution_bound = execution_bound
        self.require_strict = execution_bound == "OHLCV_STRICT_PROXY"
        self.spec = spec
        self.retain_event_snapshots = retain_event_snapshots
        self.timeout_ns_delta = int(spec.passive_timeout_minutes) * 60_000_000_000

        self.columns = tuple(first.columns)
        self.n_cols = len(self.columns)
        self.gpos_of = {sym: i for i, sym in enumerate(self.columns)}
        self.mark_source: _MarkSource = "MARK_PRICE" if first.marks is not None else "OHLCV_CLOSE_FALLBACK"
        self.first_grid = first.minute_grid

        self.units_arr = np.zeros(self.n_cols, dtype="float64")
        self.cash = float(initial_equity)
        self.last_prices_arr = np.full(self.n_cols, np.nan, dtype="float64")
        self.last_time_ns: int | None = None

        self.ledger_cash = float(initial_equity)
        self.ledger_units = np.zeros(self.n_cols, dtype="float64")
        self.last_valid_mark = np.full(self.n_cols, np.nan, dtype="float64")
        self.ledger_start_ns: int | None = None

        self.last_close_ts: dict[str, pd.Timestamp] = {}
        self.last_close_value: dict[str, float] = {}
        self.last_close_mark: dict[str, float] = {}

        self.fill_ts: list[pd.Timestamp] = []
        self.fill_symbol: list[str] = []
        self.fill_qty: list[float] = []
        self.fill_price: list[float] = []
        self.fill_fee_bps: list[float] = []
        self.fill_reason: list[str] = []
        self.fill_pre_trade_equity: list[float] = []
        self.submit_times: list[pd.Timestamp] = []
        self.fill_times: list[pd.Timestamp] = []
        self.shortfalls: list[float] = []
        self.shortfall_notionals: list[float] = []
        self.fill_count = 0
        self.unfilled_count = 0
        self.fallback_count = 0
        self.termination_counts: dict[str, int] = {"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0}
        self.data_gaps: list[ExecutionDataGap] = []
        self.units_after_events: list[tuple[pd.Timestamp, np.ndarray]] = []
        self.notional_after_events: list[tuple[pd.Timestamp, np.ndarray]] = []

        self.equity_chunks: list[np.ndarray] = []
        self.equity_times: list[pd.DatetimeIndex] = []
        self.mtm_chunks: list[np.ndarray] = []
        self.funding_chunks: list[np.ndarray] = []
        self.fee_chunks: list[np.ndarray] = []
        self.turnover_chunks: list[np.ndarray] = []
        self.ledger_valid = True
        self.invalid_reasons: set[str] = set()
        self.first_held_mark: tuple[str, pd.Timestamp] | None = None
        self.first_held_funding: tuple[str, pd.Timestamp] | None = None
        self.full_grid_end: pd.Timestamp = first.minute_grid[-1]
        self._t0 = time.perf_counter()

    def _equity_at(self, gpos: np.ndarray | None = None) -> float:
        if gpos is None:
            return self.cash + float(
                np.sum(self.units_arr * np.nan_to_num(self.last_prices_arr, nan=0.0))
            )
        return self.cash + float(
            np.sum(self.units_arr[gpos] * np.nan_to_num(self.last_prices_arr[gpos], nan=0.0))
        )

    def consume(self, w: ExecutionReplayWindow) -> None:
        columns = self.columns
        n_cols = self.n_cols
        gpos_of = self.gpos_of
        if w.columns != columns:
            raise DataIntegrityError("all execution windows must share an identical column order")
        local_cols = list(w.symbols)
        n_local = len(local_cols)
        gpos = np.asarray([gpos_of[s] for s in local_cols], dtype=np.intp)
        grid = w.minute_grid
        grid_ns = np.asarray(grid, dtype="datetime64[ns]").astype("int64")
        n_grid = len(grid_ns)
        if n_grid < 2:
            raise DataIntegrityError("an execution window must span at least two grid bars")
        if not w.bar_funding.index.equals(grid):
            raise DataIntegrityError("bar_funding must align exactly to the window minute grid")
        bar_ns = int(grid_ns[1] - grid_ns[0])
        self.full_grid_end = grid[-1]
        marks = w.marks if w.marks is not None else w.closes
        marks_values = marks[local_cols].to_numpy(dtype="float64")
        highs_values = w.highs[local_cols].to_numpy(dtype="float64")
        lows_values = w.lows[local_cols].to_numpy(dtype="float64")
        closes_values = w.closes[local_cols].to_numpy(dtype="float64")
        close_finite = np.isfinite(closes_values)
        mark_valid = np.isfinite(marks_values) & (marks_values > 0.0)
        if n_local:
            funding_matrix = np.stack(
                [w.bar_funding[s].to_numpy(dtype="float64") for s in local_cols], axis=1,
            )
        else:
            funding_matrix = np.zeros((n_grid, 0), dtype="float64")
        if not np.isfinite(funding_matrix).all():
            raise DataIntegrityError("bar_funding must be finite")
        finite_marks = marks_values[np.isfinite(marks_values)]
        if (finite_marks <= 0).any():
            raise DataIntegrityError("finite marks must be strictly positive")

        for j in range(n_local):
            idxs = np.flatnonzero(close_finite[:, j])
            if not len(idxs):
                continue
            pos = int(idxs[-1])
            ts = grid[pos]
            prev_ts = self.last_close_ts.get(local_cols[j])
            if prev_ts is None or ts > prev_ts:
                self.last_close_ts[local_cols[j]] = ts
                self.last_close_value[local_cols[j]] = float(closes_values[pos, j])
                self.last_close_mark[local_cols[j]] = float(marks_values[pos, j])

        def _advance(target_ns: int, dpos: int, on_grid: bool) -> None:
            if self.last_time_ns is not None and target_ns < self.last_time_ns:
                raise DataIntegrityError("decision times must be monotonically increasing")
            if on_grid:
                m = marks_values[dpos]
                finite = np.isfinite(m)
                prev = self.last_prices_arr[gpos]
                mark_changed = finite & np.isfinite(prev)
                if mark_changed.any():
                    self.cash += float(
                        np.sum(self.units_arr[gpos][mark_changed] * (m[mark_changed] - prev[mark_changed]))
                    )
                self.last_prices_arr[gpos] = np.where(finite, m, prev)
            lo = np.searchsorted(grid_ns, self.last_time_ns, side="right") if self.last_time_ns is not None else 0
            hi = int(np.searchsorted(grid_ns, target_ns, side="right"))
            if lo < hi:
                rates_block = funding_matrix[lo:hi, :]
                priced = np.isfinite(self.last_prices_arr[gpos])
                self.cash -= float(
                    np.sum(rates_block * self.units_arr[gpos] * np.where(priced, self.last_prices_arr[gpos], 0.0))
                )
            self.last_time_ns = target_ns

        # Per-window last-finite-close index table: ``last_close_idx[i, col]`` is
        # the largest ``j <= i`` with ``close_finite[j, col]`` True, else -1.
        # This makes the scalar ``_decision_price`` backward scan a single
        # vectorised lookup (bit-identical: it returns the same last finite
        # close position the while-loop would stop at).
        close_row = np.where(close_finite, np.arange(n_grid)[:, None], -1)
        last_close_idx = np.maximum.accumulate(close_row, axis=0)

        def _decision_price(col: int, on_grid: bool, dpos: int, spos: int) -> float | None:
            if on_grid and mark_valid[dpos, col]:
                return float(marks_values[dpos, col])
            j = int(last_close_idx[spos - 1, col]) if spos > 0 else -1
            if j >= 0 and mark_valid[j, col]:
                return float(marks_values[j, col])
            sym = local_cols[col]
            carried_ts = self.last_close_ts.get(sym)
            if carried_ts is not None:
                carried_mark = self.last_close_mark[sym]
                if np.isfinite(carried_mark) and carried_mark > 0.0:
                    return float(carried_mark)
            if spos < n_grid and mark_valid[spos, col]:
                return float(marks_values[spos, col])
            return None

        decision_ns_all = np.asarray(w.target_weights.index, dtype="datetime64[ns]").astype("int64")
        signal_ns_all = np.asarray(w.signal_available_at, dtype="datetime64[ns]").astype("int64")
        spos_all = np.searchsorted(grid_ns, signal_ns_all, side="right")
        dpos_all = np.searchsorted(grid_ns, decision_ns_all, side="left")
        dpos_clipped = np.minimum(dpos_all, n_grid - 1)
        on_grid_all = np.where(dpos_all < n_grid, grid_ns[dpos_clipped] == decision_ns_all, False)
        target_values = w.target_weights[local_cols].to_numpy(dtype="float64")

        fill_start = len(self.fill_ts)
        for i, decision_time in enumerate(w.target_weights.index):
            dns = int(decision_ns_all[i])
            dpos = int(dpos_all[i])
            on_grid = bool(on_grid_all[i])
            _advance(dns, dpos, on_grid)
            equity = self._equity_at(gpos)
            last_ledger_equity: float | None = None
            if self.equity_chunks:
                last_ledger_equity = float(self.equity_chunks[-1][-1])
            guard_equity = ruin_guard_equity(equity, last_ledger_equity)
            row = target_values[i]
            if self.min_equity_fraction is not None and guard_equity <= self.min_equity_fraction * self.initial_equity:
                if not self.equity_floor_breaches or self.equity_floor_breaches[-1] != decision_time:
                    self.equity_floor_breaches.append(decision_time)
                row = np.zeros_like(row)
            spos = int(spos_all[i])
            signal_time = w.signal_available_at[i]
            active = np.where(np.isfinite(row) & ((row != 0.0) | (self.units_arr[gpos] != 0.0)))[0]
            for col in active.tolist():
                gcol = int(gpos[col])
                sym = local_cols[col]
                weight = float(row[col])
                decision_price = _decision_price(col, on_grid, dpos, spos)
                if decision_price is None:
                    self.termination_counts["MISSING_DATA"] += 1
                    self.data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_DECISION_MARK", symbol=sym, timestamp=decision_time,
                            decision_time=decision_time, signal_time=signal_time,
                            execution_bound=self.execution_bound,
                        )
                    )
                    continue
                if not np.isfinite(self.last_prices_arr[gcol]):
                    self.last_prices_arr[gcol] = decision_price
                desired_units = weight * equity / decision_price
                net_units = desired_units - self.units_arr[gcol]
                if abs(net_units) < 1e-12:
                    continue
                side = 1 if net_units > 0 else -1
                if spos >= n_grid:
                    self.termination_counts["MISSING_DATA"] += 1
                    continue
                submit_pos = spos
                timeout_ns = grid_ns[spos] + self.timeout_ns_delta
                timeout_pos = int(np.searchsorted(grid_ns, timeout_ns, side="left"))
                timeout_close = float("nan")
                adverse = np.array([], dtype="float64")
                if self.execution_bound == "OHLCV_IMMEDIATE_TAKER":
                    fill_pos = submit_pos
                    fill_price = float(closes_values[fill_pos, col])
                    if not np.isfinite(fill_price):
                        self.termination_counts["MISSING_DATA"] += 1
                        self.data_gaps.append(
                            ExecutionDataGap(
                                code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                timestamp=grid[fill_pos], decision_time=decision_time,
                                signal_time=signal_time, execution_bound=self.execution_bound,
                            )
                        )
                        continue
                    fee_bps = self.spec.taker_fee_bps + self.spec.taker_slippage_bps
                    reason = "timeout_taker"
                else:
                    if self.execution_bound == "OHLCV_LADDERED_PROXY":
                        if timeout_pos <= spos:
                            self.termination_counts["MISSING_DATA"] += 1
                            self.data_gaps.append(
                                ExecutionDataGap(
                                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                    timestamp=grid[spos], decision_time=decision_time,
                                    signal_time=signal_time, execution_bound=self.execution_bound,
                                )
                            )
                            continue
                        adverse = (
                            lows_values[spos:timeout_pos, col]
                            if side == 1
                            else highs_values[spos:timeout_pos, col]
                        )
                        if not np.isfinite(adverse).all():
                            self.termination_counts["MISSING_DATA"] += 1
                            first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
                            self.data_gaps.append(
                                ExecutionDataGap(
                                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                    timestamp=grid[first_bad], decision_time=decision_time,
                                    signal_time=signal_time, execution_bound=self.execution_bound,
                                )
                            )
                            continue
                        closes_window = closes_values[spos:timeout_pos + 1, col]
                        if not np.isfinite(closes_window).all():
                            self.termination_counts["MISSING_DATA"] += 1
                            first_bad = spos + int(np.argmax(~np.isfinite(closes_window)))
                            self.data_gaps.append(
                                ExecutionDataGap(
                                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                    timestamp=grid[first_bad], decision_time=decision_time,
                                    signal_time=signal_time, execution_bound=self.execution_bound,
                                )
                            )
                            continue
                        for rel_pos, tranche_price, tranche_fee_bps, qty_fraction in laddered_fill_schedule(
                            decision_price, side, adverse,
                            closes_window,
                            self.spec.ladder_tranches, self.spec, True,
                        ):
                            fill_pos = spos + rel_pos
                            if rel_pos == len(adverse):
                                if timeout_pos >= n_grid or grid_ns[timeout_pos] != timeout_ns:
                                    self.termination_counts["MISSING_DATA"] += 1
                                    self.data_gaps.append(
                                        ExecutionDataGap(
                                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                            timestamp=grid[spos], decision_time=decision_time,
                                            signal_time=signal_time, execution_bound=self.execution_bound,
                                        )
                                    )
                                    continue
                                timeout_close = float(closes_values[timeout_pos, col])
                                if not np.isfinite(timeout_close):
                                    self.termination_counts["MISSING_DATA"] += 1
                                    self.data_gaps.append(
                                        ExecutionDataGap(
                                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                            timestamp=grid[timeout_pos], decision_time=decision_time,
                                            signal_time=signal_time, execution_bound=self.execution_bound,
                                        )
                                    )
                                    continue
                                self.unfilled_count += 1
                                self.fallback_count += 1
                                reason = "timeout_taker"
                            else:
                                self.fill_count += 1
                                reason = "passive_fill"
                            fill_price = float(tranche_price)
                            fee_bps = float(tranche_fee_bps)
                            qty = net_units * float(qty_fraction)
                            if reason == "passive_fill":
                                shortfall = (
                                    side * (fill_price / decision_price - 1.0) * 1e4
                                    + self.spec.maker_fee_bps
                                )
                            else:
                                shortfall = (
                                    side * (fill_price / decision_price - 1.0) * 1e4
                                    + self.spec.taker_fee_bps + self.spec.taker_slippage_bps
                                )
                            self.shortfalls.append(shortfall)
                            self.shortfall_notionals.append(abs(qty) * fill_price)
                            fill_time = grid[fill_pos]
                            submit_time = grid[submit_pos]
                            mark_price = float(marks_values[fill_pos, col])
                            if np.isfinite(self.last_prices_arr[gcol]):
                                if np.isfinite(mark_price):
                                    self.cash += self.units_arr[gcol] * (mark_price - self.last_prices_arr[gcol])
                                    self.last_prices_arr[gcol] = mark_price
                            elif np.isfinite(mark_price):
                                self.last_prices_arr[gcol] = mark_price
                            if not (np.isfinite(qty) and np.isfinite(fill_price)):
                                raise DataIntegrityError(
                                    "non-finite fill sizing breaches the capital accounting invariant "
                                    f"(symbol={sym!r} ts={fill_time!r} weight={weight!r} equity={equity!r} "
                                    f"decision_price={decision_price!r} qty={qty!r} fill_price={fill_price!r})"
                                )
                            self.cash -= qty * fill_price
                            fee = fee_bps / 1e4 * abs(qty) * fill_price
                            self.cash -= fee
                            self.units_arr[gcol] += qty
                            pre_trade_equity = self._equity_at(gpos)
                            self.fill_ts.append(fill_time)
                            self.fill_symbol.append(sym)
                            self.fill_qty.append(qty)
                            self.fill_price.append(fill_price)
                            self.fill_fee_bps.append(fee_bps)
                            self.fill_reason.append(reason)
                            self.fill_pre_trade_equity.append(pre_trade_equity)
                            self.fill_times.append(fill_time)
                            self.submit_times.append(submit_time)
                            if self.retain_event_snapshots:
                                marks_row = np.full(n_cols, np.nan, dtype="float64")
                                marks_row[gpos] = marks_values[fill_pos]
                                self.units_after_events.append((fill_time, self.units_arr.copy()))
                                self.notional_after_events.append((fill_time, self.units_arr * marks_row))
                        continue
                    if timeout_pos <= spos:
                        self.termination_counts["MISSING_DATA"] += 1
                        self.data_gaps.append(
                            ExecutionDataGap(
                                code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                timestamp=grid[spos], decision_time=decision_time,
                                signal_time=signal_time, execution_bound=self.execution_bound,
                            )
                        )
                        continue
                    adverse = (
                        lows_values[spos:timeout_pos, col]
                        if side == 1
                        else highs_values[spos:timeout_pos, col]
                    )
                    if not np.isfinite(adverse).all():
                        self.termination_counts["MISSING_DATA"] += 1
                        first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
                        self.data_gaps.append(
                            ExecutionDataGap(
                                code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                timestamp=grid[first_bad], decision_time=decision_time,
                                signal_time=signal_time, execution_bound=self.execution_bound,
                            )
                        )
                        continue
                    if side == 1:
                        crossed = (adverse < decision_price) if self.require_strict else (adverse <= decision_price)
                    else:
                        crossed = (adverse > decision_price) if self.require_strict else (adverse >= decision_price)
                    if crossed.any():
                        hit = int(np.argmax(crossed))
                        fill_pos = spos + hit
                        fill_price = decision_price
                        fee_bps = self.spec.maker_fee_bps
                        reason = "passive_fill"
                    else:
                        if timeout_pos >= n_grid or grid_ns[timeout_pos] != timeout_ns:
                            self.termination_counts["MISSING_DATA"] += 1
                            self.data_gaps.append(
                                ExecutionDataGap(
                                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                    timestamp=grid[spos], decision_time=decision_time,
                                    signal_time=signal_time, execution_bound=self.execution_bound,
                                )
                            )
                            continue
                        timeout_close = float(closes_values[timeout_pos, col])
                        if not np.isfinite(timeout_close):
                            self.termination_counts["MISSING_DATA"] += 1
                            self.data_gaps.append(
                                ExecutionDataGap(
                                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                                    timestamp=grid[timeout_pos], decision_time=decision_time,
                                    signal_time=signal_time, execution_bound=self.execution_bound,
                                )
                            )
                            continue
                        self.unfilled_count += 1
                        self.fallback_count += 1
                        fill_pos = timeout_pos
                        fill_price = timeout_close
                        fee_bps = self.spec.taker_fee_bps + self.spec.taker_slippage_bps
                        reason = "timeout_taker"
                if reason == "passive_fill":
                    self.fill_count += 1
                if self.execution_bound == "OHLCV_IMMEDIATE_TAKER":
                    shortfall = side * (fill_price / decision_price - 1.0) * 1e4 + fee_bps
                else:
                    shortfall = passive_fill_shortfall_bps(decision_price, adverse, timeout_close, side, self.spec)
                self.shortfalls.append(shortfall)
                self.shortfall_notionals.append(abs(net_units) * fill_price)
                fill_time = grid[fill_pos]
                submit_time = grid[submit_pos]
                mark_price = float(marks_values[fill_pos, col])
                if np.isfinite(self.last_prices_arr[gcol]):
                    if np.isfinite(mark_price):
                        self.cash += self.units_arr[gcol] * (mark_price - self.last_prices_arr[gcol])
                        self.last_prices_arr[gcol] = mark_price
                elif np.isfinite(mark_price):
                    self.last_prices_arr[gcol] = mark_price
                if not (np.isfinite(net_units) and np.isfinite(fill_price)):
                    raise DataIntegrityError(
                        "non-finite fill sizing breaches the capital accounting invariant "
                        f"(symbol={sym!r} ts={fill_time!r} weight={weight!r} equity={equity!r} "
                        f"decision_price={decision_price!r} qty={net_units!r} fill_price={fill_price!r})"
                    )
                self.cash -= net_units * fill_price
                fee = fee_bps / 1e4 * abs(net_units) * fill_price
                self.cash -= fee
                self.units_arr[gcol] += net_units
                if reason in ("passive_fill", "timeout_taker"):
                    pre_trade_equity = self._equity_at(gpos)
                    self.fill_ts.append(fill_time)
                    self.fill_symbol.append(sym)
                    self.fill_qty.append(net_units)
                    self.fill_price.append(fill_price)
                    self.fill_fee_bps.append(fee_bps)
                    self.fill_reason.append(reason)
                    self.fill_pre_trade_equity.append(pre_trade_equity)
                    self.fill_times.append(fill_time)
                    self.submit_times.append(submit_time)
                    if self.retain_event_snapshots:
                        marks_row = np.full(n_cols, np.nan, dtype="float64")
                        marks_row[gpos] = marks_values[fill_pos]
                        self.units_after_events.append((fill_time, self.units_arr.copy()))
                        self.notional_after_events.append((fill_time, self.units_arr * marks_row))

        # ---- streamed ledger chunk over [ledger_start_ns, grid end] ----
        p0 = 0 if self.ledger_start_ns is None else int(np.searchsorted(grid_ns, self.ledger_start_ns, side="left"))
        if p0 >= n_grid:
            raise DataIntegrityError("execution windows must not leave an uncovered grid gap")
        chunk_len = n_grid - p0
        if chunk_len:
            # Vectorized fill scatter over the (grid, symbol) plane: the scalar
            # per-fill ``local_cols.index`` / ``searchsorted`` / list append is
            # replaced by one searchsorted + one add.at over the window's fills.
            sym_to_local = {s: j for j, s in enumerate(local_cols)}
            n_fill = len(self.fill_ts) - fill_start
            turnover_pos_arr: np.ndarray
            turnover_qty_arr: np.ndarray
            turnover_price_arr: np.ndarray
            if n_fill:
                wf_ts = np.asarray(
                    [int(ts.value) for ts in self.fill_ts[fill_start:]], dtype="int64",
                )
                wf_pos = np.searchsorted(grid_ns, wf_ts, side="left")
                wf_exact = np.minimum(wf_pos, n_grid - 1)
                if (
                    np.any(wf_pos >= n_grid)
                    or np.any(grid_ns[wf_exact] != wf_ts)
                ):
                    raise DataIntegrityError("windowed fills must occur on the window minute grid")
                wf_j = np.asarray(
                    [sym_to_local[s] for s in self.fill_symbol[fill_start:]],
                    dtype=np.intp,
                )
                wf_qty = np.asarray(self.fill_qty[fill_start:], dtype="float64")
                wf_price = np.asarray(self.fill_price[fill_start:], dtype="float64")
                wf_fee = np.asarray(self.fill_fee_bps[fill_start:], dtype="float64")
                wf_fee_amt = wf_fee / 1e4 * np.abs(wf_qty) * wf_price
                d_flat = np.ravel_multi_index(
                    (wf_pos, wf_j), (n_grid, n_local),
                )
                d_matrix = np.zeros(n_grid * n_local, dtype="float64")
                np.add.at(d_matrix, d_flat, wf_qty)
                d_matrix = d_matrix.reshape((n_grid, n_local))
                fill_flow = np.zeros(n_grid, dtype="float64")
                fee_by_ts = np.zeros(n_grid, dtype="float64")
                np.add.at(fill_flow, wf_pos, -(wf_qty * wf_price + wf_fee_amt))
                np.add.at(fee_by_ts, wf_pos, wf_fee_amt)
                turnover_pos_arr = wf_pos
                turnover_qty_arr = wf_qty
                turnover_price_arr = wf_price
            else:
                d_matrix = np.zeros((n_grid, n_local), dtype="float64")
                fill_flow = np.zeros(n_grid, dtype="float64")
                fee_by_ts = np.zeros(n_grid, dtype="float64")
                turnover_pos_arr = np.empty(0, dtype=np.intp)
                turnover_qty_arr = np.empty(0, dtype="float64")
                turnover_price_arr = np.empty(0, dtype="float64")

            # Vectorized (grid, symbol) ledger pass.  Each column's arithmetic is
            # bit-identical to the scalar per-symbol loop; only the iteration
            # order changes (column-major collapse into one 2-D broadcast).
            sym_finite = np.isfinite(marks_values)
            units_state = np.cumsum(d_matrix, axis=0) + self.ledger_units[gpos][None, :]
            units_before = np.zeros_like(units_state)
            units_before[0] = self.ledger_units[gpos]
            units_before[1:] = units_state[:-1]

            # Vectorized forward-fill of the last valid mark per symbol
            # (replaces the ``for i in range(n_grid)`` carry loop).
            last_finite_idx = np.maximum.accumulate(
                np.where(sym_finite, np.arange(n_grid)[:, None], -1), axis=0,
            )
            m_ff = marks_values[last_finite_idx, np.arange(n_local)[None, :]]
            carry_row = np.asarray(self.last_valid_mark[gpos], dtype="float64")[None, :]
            m_ff = np.where(sym_finite, marks_values, np.where(last_finite_idx >= 0, m_ff, carry_row))
            self.last_valid_mark[gpos] = m_ff[-1]

            valuation = np.where(
                sym_finite | (units_state != 0.0),
                np.where(sym_finite, marks_values, m_ff),
                0.0,
            )

            held = units_before != 0.0
            joint = np.zeros_like(sym_finite, dtype=bool)
            joint[1:] = sym_finite[1:] & sym_finite[:-1]
            kept_region = np.arange(n_grid)[:, None] >= p0
            # Held-gap provenance is judged only on this window's kept chunk
            # region: bars before p0 belong to the previous chunk's ledger,
            # where the carried state (not this window's frames) is correct.
            held_mark_trigger = (held & ~joint) & kept_region
            if held_mark_trigger.any():
                self.ledger_valid = False
                self.invalid_reasons.add("MISSING_DATA")
                if self.first_held_mark is None:
                    col_hit = held_mark_trigger.any(axis=0)
                    j0 = int(np.argmax(col_hit))
                    mask_col = (held & ~joint)[:, j0]
                    trigger_pos = int(np.argmax(held_mark_trigger[:, j0][mask_col]))
                    self.first_held_mark = (local_cols[j0], grid[p0 + trigger_pos])

            delta_price = np.zeros_like(marks_values)
            delta_price[1:] = marks_values[1:] - marks_values[:-1]
            mtm_contrib = np.zeros_like(marks_values)
            mtm_contrib[1:] = np.where(
                joint[1:], units_before[1:] * delta_price[1:], 0.0,
            )
            # Sequential-order column sum (cumsum's last column), bit-identical
            # to the scalar ``arr += col`` accumulation order.  An empty roster
            # (n_local == 0) yields a zero contribution series.
            mtm_arr = (
                mtm_contrib.cumsum(axis=1)[:, -1]
                if n_local
                else np.zeros(n_grid, dtype="float64")
            )

            charged = funding_matrix * units_before * marks_values
            charged = np.where(sym_finite, charged, 0.0)
            held_funding_trigger = (~sym_finite & held & (funding_matrix != 0.0)) & kept_region
            if held_funding_trigger.any():
                self.ledger_valid = False
                self.invalid_reasons.add("MISSING_DATA")
                if self.first_held_funding is None:
                    col_hit = held_funding_trigger.any(axis=0)
                    j0 = int(np.argmax(col_hit))
                    mask_col = (~sym_finite & held & (funding_matrix != 0.0))[:, j0]
                    trigger_pos = int(np.argmax(held_funding_trigger[:, j0][mask_col]))
                    self.first_held_funding = (local_cols[j0], grid[p0 + trigger_pos])
            funding_arr = (
                charged.cumsum(axis=1)[:, -1]
                if n_local
                else np.zeros(n_grid, dtype="float64")
            )

            notional_arr = (
                (units_state * valuation).cumsum(axis=1)[:, -1]
                if n_local
                else np.zeros(n_grid, dtype="float64")
            )
            notional_before_arr = (
                (units_before * valuation).cumsum(axis=1)[:, -1]
                if n_local
                else np.zeros(n_grid, dtype="float64")
            )
            self.ledger_units[gpos] = units_state[-1]

            # The cash cumsum starts at the chunk's first bar (p0): positions
            # [0, p0) belong to the previous chunk's ledger and must not be
            # re-accumulated from the carried cash.
            chunk_flow = fill_flow[p0:] - funding_arr[p0:]
            cash_after = self.ledger_cash + np.cumsum(chunk_flow)
            cash_pre_fill = np.empty(chunk_len, dtype="float64")
            cash_pre_fill[0] = self.ledger_cash - funding_arr[p0]
            cash_pre_fill[1:] = cash_after[:-1] - funding_arr[p0 + 1 :]
            equity_arr = cash_after + notional_arr[p0:]
            turnover_arr = np.zeros(chunk_len, dtype="float64")
            if len(turnover_pos_arr):
                pre_trade_equity = (
                    cash_pre_fill[turnover_pos_arr - p0] + notional_before_arr[turnover_pos_arr]
                )
                if not np.isfinite(pre_trade_equity).all() or (pre_trade_equity <= 0).any():
                    bad = np.where(~np.isfinite(pre_trade_equity) | (pre_trade_equity <= 0))[0]
                    bad_pos = turnover_pos_arr[bad[0]]
                    raise DataIntegrityError(
                        f"pre-trade equity must be positive and finite "
                        f"(ts={grid[bad_pos]!r} pre_trade_equity={pre_trade_equity[bad[0]]!r})"
                    )
                np.add.at(
                    turnover_arr, turnover_pos_arr - p0,
                    np.abs(turnover_qty_arr * turnover_price_arr) / pre_trade_equity,
                )
            if not np.isfinite(equity_arr).all() or (equity_arr <= 0).any():
                raise DataIntegrityError("simulated inventory equity must be finite and strictly positive")
            self.equity_chunks.append(equity_arr)
            self.equity_times.append(grid[p0:])
            self.mtm_chunks.append(mtm_arr[p0:])
            self.funding_chunks.append(funding_arr[p0:])
            self.fee_chunks.append(fee_by_ts[p0:])
            self.turnover_chunks.append(turnover_arr)
            self.ledger_cash = float(cash_after[-1])
        self.ledger_start_ns = int(grid_ns[-1]) + bar_ns

    def finalize(self) -> StrategyExecutionReplayResult:
        columns = self.columns
        n_cols = self.n_cols

        # Persistent source-end gap with held units: UNKNOWN_TERMINATION forced exit.
        forced_exit_count = 0
        forced_exit_notional = 0.0
        grid_end = self.full_grid_end
        for col in range(n_cols):
            sym = columns[col]
            if abs(self.units_arr[col]) < 1e-12:
                continue
            if sym not in self.last_close_ts or self.last_close_ts[sym] >= grid_end:
                continue
            exit_ts = self.last_close_ts[sym]
            exit_price = self.last_close_value[sym]
            if not np.isfinite(exit_price) or exit_price <= 0:
                self.termination_counts["MISSING_DATA"] += 1
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_FORCED_EXIT_CLOSE", symbol=sym, timestamp=exit_ts,
                        execution_bound=self.execution_bound,
                    )
                )
                continue
            self.termination_counts["UNKNOWN_TERMINATION"] += 1
            forced_exit_count += 1
            forced_exit_notional += abs(self.units_arr[col] * exit_price)
            penalty = (
                TERMINATION_STRESS_PENALTY_BPS
                if self.execution_bound == "OHLCV_IMMEDIATE_TAKER"
                else 0.0
            )
            fee_bps = self.spec.taker_fee_bps + self.spec.taker_slippage_bps + penalty
            prev_price = (
                float(self.last_prices_arr[col]) if np.isfinite(self.last_prices_arr[col]) else exit_price
            )
            mark_price = self.last_close_mark.get(sym, float("nan"))
            if np.isfinite(mark_price):
                self.cash -= self.units_arr[col] * (mark_price - prev_price)
                self.last_prices_arr[col] = mark_price
            self.cash -= -self.units_arr[col] * exit_price
            fee = fee_bps / 1e4 * abs(self.units_arr[col]) * exit_price
            self.cash -= fee
            self.fill_ts.append(exit_ts)
            self.fill_symbol.append(sym)
            self.fill_qty.append(-self.units_arr[col])
            self.fill_price.append(exit_price)
            self.fill_fee_bps.append(fee_bps)
            self.fill_reason.append("forced_exit")
            self.fill_pre_trade_equity.append(self._equity_at())
            self.fill_times.append(exit_ts)
            self.units_arr[col] = 0.0
            if self.retain_event_snapshots:
                self.units_after_events.append((exit_ts, self.units_arr.copy()))
        elapsed_seconds = time.perf_counter() - self._t0

        simulated_fills = pd.DataFrame(
            {
                "timestamp": self.fill_ts,
                "symbol": self.fill_symbol,
                "quantity_delta": self.fill_qty,
                "fill_price": self.fill_price,
                "fee_bps": self.fill_fee_bps,
                "reason": self.fill_reason,
                "pre_trade_equity": self.fill_pre_trade_equity,
            }
        )[
            [
                "timestamp", "symbol", "quantity_delta", "fill_price",
                "fee_bps", "reason", "pre_trade_equity",
            ]
        ]
        if simulated_fills.empty:
            simulated_fills = simulated_fills.astype(
                {"quantity_delta": "float64", "fill_price": "float64", "fee_bps": "float64"}
            )

        if self.equity_chunks:
            full_index = self.equity_times[0].append(self.equity_times[1:]) if len(self.equity_times) > 1 else self.equity_times[0]
            equity_values_arr = np.concatenate(self.equity_chunks)
            mtm_arr = np.concatenate(self.mtm_chunks)
            funding_arr = np.concatenate(self.funding_chunks)
            fee_arr = np.concatenate(self.fee_chunks)
            turnover_arr = np.concatenate(self.turnover_chunks)
            del self.equity_chunks, self.equity_times, self.mtm_chunks, self.funding_chunks, self.fee_chunks, self.turnover_chunks
        else:
            full_index = self.first_grid
            equity_values_arr = np.array([], dtype="float64")
            mtm_arr = np.array([], dtype="float64")
            funding_arr = np.array([], dtype="float64")
            fee_arr = np.array([], dtype="float64")
            turnover_arr = np.array([], dtype="float64")
        equity = pd.Series(equity_values_arr, index=full_index, dtype="float64")
        if not np.isfinite(equity_values_arr).all() or (equity_values_arr <= 0).any():
            raise DataIntegrityError("simulated inventory equity must be finite and strictly positive")
        ledger = SimulatedInventoryLedgerResult(
            equity=equity,
            net_returns=equity.pct_change().dropna(),
            simulated_units=None,
            mark_to_market_pnl=pd.Series(mtm_arr, index=full_index, dtype="float64"),
            funding_charge=pd.Series(funding_arr, index=full_index, dtype="float64"),
            fee_charge=pd.Series(fee_arr, index=full_index, dtype="float64"),
            fill_turnover=pd.Series(turnover_arr, index=full_index, dtype="float64"),
            fill_source=self.execution_bound,
            mark_source=self.mark_source,
            primary_valid=self.ledger_valid,
            invalid_reasons=tuple(sorted(self.invalid_reasons)),
            equity_floor_breached_at=tuple(self.equity_floor_breaches),
        )
        if self.first_held_mark is not None:
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_HELD_MARK", symbol=self.first_held_mark[0],
                    timestamp=self.first_held_mark[1], execution_bound=self.execution_bound,
                )
            )
        if self.first_held_funding is not None:
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_HELD_FUNDING", symbol=self.first_held_funding[0],
                    timestamp=self.first_held_funding[1], execution_bound=self.execution_bound,
                )
            )
        self.data_gaps.sort(key=lambda g: (g.timestamp, g.code, g.symbol))

        if self.units_after_events:
            events_index = pd.DatetimeIndex([t for t, _ in self.units_after_events])
            simulated_units = pd.DataFrame(
                [row for _t, row in self.units_after_events], index=events_index, columns=list(columns),
            )
            notional_events_index = pd.DatetimeIndex([t for t, _ in self.notional_after_events])
            simulated_notional_weights = pd.DataFrame(
                [row for _t, row in self.notional_after_events],
                index=notional_events_index,
                columns=list(columns),
            )
        else:
            simulated_units = pd.DataFrame(columns=list(columns))
            simulated_notional_weights = pd.DataFrame(columns=list(columns))

        all_intent_shortfall_bps = (
            float(np.mean(self.shortfalls)) if self.shortfalls else float("nan")
        )
        weighted_shortfall_bps = notional_weighted_shortfall_bps(
            self.shortfalls, self.shortfall_notionals
        )
        return StrategyExecutionReplayResult(
            simulated_fills=simulated_fills,
            ledger=ledger,
            simulated_units=simulated_units,
            simulated_notional_weights=simulated_notional_weights,
            fill_source=self.execution_bound,
            mark_source=self.mark_source,
            submit_times=pd.Series(self.submit_times, dtype="datetime64[ns, UTC]"),
            fill_times=pd.Series(self.fill_times, dtype="datetime64[ns, UTC]"),
            fill_count=self.fill_count,
            unfilled_count=self.unfilled_count,
            fallback_count=self.fallback_count,
            all_intent_shortfall_bps=all_intent_shortfall_bps,
            forced_exit_count=forced_exit_count,
            forced_exit_notional=forced_exit_notional,
            termination_counts=self.termination_counts,
            unsupported_assumptions=(
                "partial_fill",
                "queue_position",
                "post_only_rejection",
                "cancel_replace_latency",
                "order_size_impact",
            ),
            elapsed_seconds=elapsed_seconds,
            data_gaps=tuple(self.data_gaps),
            event_snapshots_retained=self.retain_event_snapshots,
            notional_weighted_shortfall_bps=weighted_shortfall_bps,
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
    accumulator = _BoundExecutionReplayAccumulator(
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
        _BoundExecutionReplayAccumulator(
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