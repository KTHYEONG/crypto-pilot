"""OHLCV strict-proxy strategy-aware execution replay."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.types import ExecutionSpec

from . import TERMINATION_STRESS_PENALTY_BPS, _ExecutionBound, _MarkSource
from . import ledger as _ledger
from . import microstructure as _microstructure
from .contracts import (
    ExecutionDataGap,
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
                    for rel_pos, tranche_price, tranche_fee_bps, qty_fraction in _microstructure.laddered_fill_schedule(
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
                shortfall = _microstructure.passive_fill_shortfall_bps(
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

    ledger = _ledger.simulated_inventory_ledger(
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
    weighted_shortfall_bps = _microstructure.notional_weighted_shortfall_bps(
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
