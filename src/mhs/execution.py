"""Execution layer: passive fills, funding panel, and the simulated inventory ledger.

The Research-GO PnL source is ``simulated_inventory_ledger`` fed by the
OHLCV strict-proxy ``strategy_aware_execution_replay``; ``mhs_ledger_pnl`` is
the pinned target-weight *pre-screen* proxy only and must never back Research
GO, OOS, capital, or capacity claims.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.contracts import ExecutionSpec
from src.research.baseline.backtest import _align_funding_rates
from src.research.technical_experts.cross_sectional import XsCompositeSpec, run_xs_composite_ledger

# Conservative extra settlement/slippage penalty applied to a stress-ledger
# (OHLCV_IMMEDIATE_TAKER) UNKNOWN_TERMINATION forced exit (spec §2.17/§7.5).
TERMINATION_STRESS_PENALTY_BPS = 50.0

_ExecutionBound = Literal["OHLCV_STRICT_PROXY", "OHLCV_TOUCH_PROXY", "OHLCV_IMMEDIATE_TAKER"]
_MarkSource = Literal["MARK_PRICE", "OHLCV_CLOSE_FALLBACK"]


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
    return pd.DataFrame(cols, index=grid)


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
    simulated_units: pd.DataFrame
    mark_to_market_pnl: pd.Series
    funding_charge: pd.Series
    fee_charge: pd.Series
    fill_turnover: pd.Series
    fill_source: str
    mark_source: _MarkSource
    primary_valid: bool
    invalid_reasons: tuple[str, ...]


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


def simulated_inventory_ledger(
    simulated_fills: pd.DataFrame,
    marks: pd.DataFrame,
    bar_funding: pd.DataFrame,
    initial_equity: float,
    fill_source: str,
    mark_source: _MarkSource,
) -> SimulatedInventoryLedgerResult:
    """Compound a timestamp-sorted proxy fill stream into a cash-and-inventory ledger.

    ``marks`` and ``bar_funding`` share an identical UTC index and ordered
    symbol columns. For every interval the units held since the preceding event
    are marked first, then funding published in the interval is charged against
    the pre-fill quantity times mark price, then timestamp-sorted fills and
    their fees are applied. A proxy fill cannot earn or lose PnL before its
    timestamp.
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
    n_cols = len(columns)
    col_index = {c: j for j, c in enumerate(columns)}

    fill_positions = np.searchsorted(grid, fill_ts)
    delta_units = np.zeros((n_grid, n_cols), dtype="float64")
    fill_flow = np.zeros(n_grid, dtype="float64")
    fee_by_ts = np.zeros(n_grid, dtype="float64")
    turnover_terms: dict[pd.Timestamp, list[tuple[float, float]]] = defaultdict(list)
    turnover_pos: dict[pd.Timestamp, int] = {}
    for k, row in enumerate(fills.itertuples(index=False)):
        pos = int(fill_positions[k])
        col = col_index[str(row.symbol)]
        qty = float(row.quantity_delta)
        price = float(row.fill_price)
        fee_bps = float(row.fee_bps)
        fee = fee_bps / 1e4 * abs(qty) * price
        delta_units[pos, col] += qty
        fill_flow[pos] += -(qty * price + fee)
        fee_by_ts[pos] += fee
        turnover_terms[row.timestamp].append((qty, price))
        turnover_pos[row.timestamp] = pos

    units_state = np.cumsum(delta_units, axis=0)
    units_before = np.zeros_like(units_state)
    units_before[1:] = units_state[:-1]

    finite = np.isfinite(marks_values)
    last_index = np.where(finite, np.arange(n_grid)[:, None], 0)
    last_index = np.maximum.accumulate(last_index, axis=0)
    valuation = marks_values[last_index, np.arange(n_cols)[None, :]]

    primary_valid = True
    invalid_reasons: set[str] = set()

    mtm = np.zeros(n_grid, dtype="float64")
    delta_price = np.zeros_like(marks_values)
    delta_price[1:] = marks_values[1:] - marks_values[:-1]
    joint = np.zeros_like(finite)
    joint[1:] = finite[1:] & finite[:-1]
    held = units_before != 0.0
    if np.any(held & ~joint):
        primary_valid = False
        invalid_reasons.add("MISSING_DATA")
    mtm[1:] = np.sum(
        np.where(joint[1:], units_before[1:] * delta_price[1:], 0.0), axis=1,
    )

    charged = funding_rates * units_before * marks_values
    charged = np.where(finite, charged, 0.0)
    if np.any(~finite & held & (funding_rates != 0.0)):
        primary_valid = False
        invalid_reasons.add("MISSING_DATA")
    funding_charge = np.sum(charged, axis=1)

    cash_after = initial_equity + np.cumsum(fill_flow - funding_charge)
    cash_pre_fill = np.empty(n_grid, dtype="float64")
    cash_pre_fill[0] = initial_equity - funding_charge[0]
    cash_pre_fill[1:] = cash_after[:-1] - funding_charge[1:]

    notional = np.sum(units_state * valuation, axis=1)
    notional_before = np.sum(units_before * valuation, axis=1)
    equity_values_arr = cash_after + notional

    turnover_arr = np.zeros(n_grid, dtype="float64")
    for ts, terms in turnover_terms.items():
        pos = turnover_pos[ts]
        pre_trade_equity = cash_pre_fill[pos] + notional_before[pos]
        if pre_trade_equity <= 0:
            raise DataIntegrityError("pre-trade equity must be positive")
        turnover_arr[pos] = sum(
            abs(qty * price) / pre_trade_equity for qty, price in terms
        )

    equity = pd.Series(equity_values_arr, index=grid, dtype="float64")
    if (equity <= 0).any():
        raise DataIntegrityError("simulated inventory equity reached zero or below")
    return SimulatedInventoryLedgerResult(
        equity=equity,
        net_returns=equity.pct_change().dropna(),
        simulated_units=pd.DataFrame(units_state, index=grid, columns=columns),
        mark_to_market_pnl=pd.Series(mtm, index=grid, dtype="float64"),
        funding_charge=pd.Series(funding_charge, index=grid, dtype="float64"),
        fee_charge=pd.Series(fee_by_ts, index=grid, dtype="float64"),
        fill_turnover=pd.Series(turnover_arr, index=grid, dtype="float64"),
        fill_source=fill_source,
        mark_source=mark_source,
        primary_valid=primary_valid,
        invalid_reasons=tuple(sorted(invalid_reasons)),
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

    Live forward collection (Phase 4B) records one ``ForwardExecutionObservation``
    per signal intent; this OHLCV replay cannot observe queue position, partial
    fills, or rejections, so those assumptions are reported as unsupported.
    """
    if initial_equity <= 0:
        raise DataIntegrityError("initial_equity must be > 0")
    if execution_bound not in ("OHLCV_STRICT_PROXY", "OHLCV_TOUCH_PROXY", "OHLCV_IMMEDIATE_TAKER"):
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
    minute_set = set(minute_grid)
    symbols = list(target_weights.columns)
    require_strict = execution_bound == "OHLCV_STRICT_PROXY"

    last_reliable: dict[str, pd.Timestamp] = {}
    for sym in symbols:
        valid = minute_closes[sym].dropna()
        last_reliable[sym] = valid.index[-1] if len(valid) else minute_grid[0]

    units = dict.fromkeys(symbols, 0.0)
    cash = float(initial_equity)
    last_prices: dict[str, float] = {}
    last_time: pd.Timestamp | None = None

    def _equity_at() -> float:
        return cash + sum(
            units[s] * last_prices[s] for s in symbols if s in last_prices
        )

    def _advance(target_time: pd.Timestamp) -> None:
        nonlocal cash, last_time
        if last_time is not None and target_time < last_time:
            raise DataIntegrityError("decision times must be monotonically increasing")
        if last_time is not None:
            between = minute_grid[(minute_grid > last_time) & (minute_grid <= target_time)]
        else:
            between = minute_grid[minute_grid <= target_time]
        for s in symbols:
            if target_time in minute_set:
                price = float(marks.loc[target_time, s])
                if np.isfinite(price):
                    if s in last_prices:
                        cash += units[s] * (price - last_prices[s])
                    last_prices[s] = price
        if len(between) and len(bar_funding):
            for t in between:
                for s in symbols:
                    rate = float(bar_funding.loc[t, s])
                    if rate != 0.0 and units[s] != 0.0 and s in last_prices:
                        cash -= rate * units[s] * last_prices[s]
        last_time = target_time

    def _decision_price(sym: str, decision_time: pd.Timestamp, signal_time: pd.Timestamp) -> float | None:
        if decision_time in minute_set:
            price = float(marks.loc[decision_time, sym])
            if np.isfinite(price) and price > 0:
                return price
        prior = minute_grid[(minute_grid <= signal_time) & minute_closes[sym].notna()]
        if len(prior):
            price = float(marks.loc[prior[-1], sym])
            if np.isfinite(price) and price > 0:
                return price
        after = minute_grid[minute_grid > signal_time]
        if len(after):
            price = float(marks.loc[after[0], sym])
            if np.isfinite(price) and price > 0:
                return price
        return None

    fill_records: list[dict[str, object]] = []
    submit_times: list[pd.Timestamp] = []
    fill_times: list[pd.Timestamp] = []
    shortfalls: list[float] = []
    fill_count = 0
    unfilled_count = 0
    fallback_count = 0
    termination_counts: dict[str, int] = {"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0}
    units_after_events: list[tuple[pd.Timestamp, list[float]]] = []
    notional_after_events: list[tuple[pd.Timestamp, list[float]]] = []

    for i, decision_time in enumerate(target_weights.index):
        signal_time = signal_available_at[i]
        _advance(decision_time)
        equity = _equity_at()
        row = target_weights.loc[decision_time]

        for sym in symbols:
            weight = float(row[sym]) if sym in row.index else 0.0
            if not np.isfinite(weight) or weight == 0.0:
                continue
            decision_price = _decision_price(sym, decision_time, signal_time)
            if decision_price is None:
                termination_counts["MISSING_DATA"] += 1
                continue
            if sym not in last_prices:
                last_prices[sym] = decision_price
            desired_units = weight * equity / decision_price
            net_units = desired_units - units[sym]
            if abs(net_units) < 1e-12:
                continue
            side = 1 if net_units > 0 else -1

            candidates = minute_grid[minute_grid > signal_time]
            if not len(candidates):
                termination_counts["MISSING_DATA"] += 1
                continue
            submit_time = candidates[0]
            timeout_bar = submit_time + pd.Timedelta(minutes=spec.passive_timeout_minutes)
            adverse = np.array([], dtype="float64")
            timeout_close = float("nan")

            if execution_bound == "OHLCV_IMMEDIATE_TAKER":
                fill_time = submit_time
                fill_price = float(minute_closes.loc[fill_time, sym])
                fee_bps = spec.taker_fee_bps + spec.taker_slippage_bps
                reason = "timeout_taker"
            else:
                window = minute_grid[
                    (minute_grid > signal_time) & (minute_grid < timeout_bar)
                ]
                if len(window):
                    adverse = minute_lows.loc[window, sym].to_numpy(dtype="float64")
                    if side == -1:
                        adverse = minute_highs.loc[window, sym].to_numpy(dtype="float64")
                if not len(window) or not np.isfinite(adverse).all():
                    termination_counts["MISSING_DATA"] += 1
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
                    fill_time = window[hit]
                    fill_price = decision_price
                    fee_bps = spec.maker_fee_bps
                    reason = "passive_fill"
                else:
                    if timeout_bar not in minute_set:
                        termination_counts["MISSING_DATA"] += 1
                        continue
                    timeout_close = float(minute_closes.loc[timeout_bar, sym])
                    if not np.isfinite(timeout_close):
                        termination_counts["MISSING_DATA"] += 1
                        continue
                    unfilled_count += 1
                    fallback_count += 1
                    fill_time = timeout_bar
                    fill_price = timeout_close
                    fee_bps = spec.taker_fee_bps + spec.taker_slippage_bps
                    reason = "timeout_taker"

            if reason == "passive_fill":
                fill_count += 1
            if execution_bound == "OHLCV_IMMEDIATE_TAKER":
                shortfall = side * (fill_price / decision_price - 1.0) * 1e4 + fee_bps
            else:
                timeout_price = timeout_close
                shortfall = passive_fill_shortfall_bps(
                    decision_price, adverse, timeout_price, side, spec,
                )
            shortfalls.append(shortfall)

            if fill_time in minute_set and sym in last_prices:
                mark_price = float(marks.loc[fill_time, sym])
                cash += units[sym] * (mark_price - last_prices[sym])
                last_prices[sym] = mark_price
            elif fill_time in minute_set:
                last_prices[sym] = float(marks.loc[fill_time, sym])
            cash -= net_units * fill_price
            fee = fee_bps / 1e4 * abs(net_units) * fill_price
            cash -= fee
            units[sym] += net_units
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
                units_after_events.append((fill_time, [units[s] for s in symbols]))
                notional_after_events.append(
                    (
                        fill_time,
                        [
                            units[s] * float(marks.loc[fill_time, s])
                            for s in symbols
                        ],
                    )
                )

    # Persistent source-end gap with held units: UNKNOWN_TERMINATION forced exit.
    forced_exit_count = 0
    forced_exit_notional = 0.0
    grid_end = minute_grid[-1]
    for sym in symbols:
        if units[sym] == 0.0:
            continue
        if last_reliable[sym] >= grid_end:
            continue
        exit_time = last_reliable[sym]
        exit_price = float(minute_closes.loc[exit_time, sym])
        if not np.isfinite(exit_price) or exit_price <= 0:
            termination_counts["MISSING_DATA"] += 1
            continue
        termination_counts["UNKNOWN_TERMINATION"] += 1
        forced_exit_count += 1
        forced_exit_notional += abs(units[sym] * exit_price)
        penalty = (
            TERMINATION_STRESS_PENALTY_BPS
            if execution_bound == "OHLCV_IMMEDIATE_TAKER"
            else 0.0
        )
        fee_bps = spec.taker_fee_bps + spec.taker_slippage_bps + penalty
        if exit_time in minute_set:
            mark_price = float(marks.loc[exit_time, sym])
            cash -= units[sym] * (mark_price - last_prices.get(sym, exit_price))
            last_prices[sym] = mark_price
        cash -= -units[sym] * exit_price
        fee = fee_bps / 1e4 * abs(units[sym]) * exit_price
        cash -= fee
        fill_records.append(
            {
                "timestamp": exit_time,
                "symbol": sym,
                "quantity_delta": -units[sym],
                "fill_price": exit_price,
                "fee_bps": fee_bps,
                "reason": "forced_exit",
                "pre_trade_equity": _equity_at(),
            }
        )
        fill_times.append(exit_time)
        units[sym] = 0.0
        units_after_events.append((exit_time, [units[s] for s in symbols]))

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
    )


def mhs_ledger_pnl(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    one_way_bps: float,
    execution_delay_bars: int = 1,
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
    )
    equity, turnover = run_xs_composite_ledger(weights, opens, bar_funding, spec)
    net = equity.pct_change().dropna()
    return net, turnover
