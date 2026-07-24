"""Causal intrabar exit labeling shared by L1 candidate pipelines."""

from __future__ import annotations

import numpy as np

from src.domain.futures.forecast.contracts import ExitPathLabels, ExitPathRequest

_BPS = 10_000.0


def _label_kernel(request: ExitPathRequest) -> tuple[np.ndarray, ...]:
    n = request.entry_idx.size
    t_len = request.open_2d.shape[0]
    gross = np.full(n, np.nan)
    costs = np.full(n, np.nan)
    funding = np.full(n, np.nan)
    edge = np.full(n, np.nan)
    reasons = np.zeros(n, dtype=np.int8)
    exits = np.full(n, -1, dtype=np.int64)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    collisions = np.zeros(n, dtype=np.int8)
    for i in range(n):
        decision = int(request.decision_idx[i])
        entry = int(request.entry_idx[i])
        symbol = int(request.symbol_idx[i])
        horizon = max(int(request.horizon_bars[i]), 1)
        if decision < 0 or entry != decision + 1 or entry >= t_len or symbol < 0 or symbol >= request.open_2d.shape[1]:
            continue
        entry_price = request.open_2d[entry, symbol]
        atr = request.atr_2d[decision, symbol]
        if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(atr) or atr <= 0.0:
            continue
        side = 1.0 if request.side[i] > 0 else -1.0
        stop_distance = max(float(request.stop_atr_mult[i]), 1e-6) * atr
        target_distance = max(float(request.target_atr_mult[i]), 1e-6) * atr
        stop_price = entry_price - side * stop_distance
        target_price = entry_price + side * target_distance
        end = min(entry + horizon - 1, t_len - 1)
        scan_from = max(int(request.min_hold_bars[i]), 1)
        stop_bar = t_len + 1
        target_bar = t_len + 1
        for offset in range(scan_from, end - entry + 1):
            bar = entry + offset
            high = request.high_2d[bar, symbol]
            low = request.low_2d[bar, symbol]
            if not np.isfinite(high) or not np.isfinite(low):
                continue
            stop_hit = low <= stop_price if side > 0 else high >= stop_price
            target_hit = high >= target_price if side > 0 else low <= target_price
            if stop_hit and stop_bar > t_len:
                stop_bar = bar
            if target_hit and target_bar > t_len:
                target_bar = bar
        collision = stop_bar == target_bar and stop_bar <= t_len
        collisions[i] = 1 if collision else 0
        if stop_bar <= target_bar and stop_bar <= t_len:
            exit_bar = stop_bar
            exit_code = 1
            raw_open = request.open_2d[exit_bar, symbol]
            exit_price = min(raw_open, stop_price) if side > 0 else max(raw_open, stop_price)
        elif target_bar <= t_len:
            exit_bar = target_bar
            exit_code = 2
            exit_price = target_price
        else:
            exit_bar = end
            exit_code = 3
            next_open = min(exit_bar + 1, t_len - 1)
            exit_price = request.open_2d[next_open, symbol]
        if not np.isfinite(exit_price) or exit_price <= 0.0:
            continue
        gross_return = side * np.log(exit_price / entry_price)
        cost = request.cost_floor_bps[i]
        if not np.isfinite(cost):
            cost = request.cost_bps_2d[decision, symbol]
        if not np.isfinite(cost):
            cost = request.taker_round_trip_bps
        cost = max(cost, request.taker_round_trip_bps)
        funding_stop = exit_bar + 1
        realized_funding = 0.0
        for f in range(entry, min(funding_stop, request.funding_2d.shape[0])):
            value = request.funding_2d[f, symbol]
            if np.isfinite(value):
                realized_funding += side * value * _BPS
        gross[i] = gross_return * _BPS
        costs[i] = cost
        funding[i] = realized_funding
        edge[i] = gross[i] - costs[i] - realized_funding - request.hurdle_bps[i]
        path_end = min(exit_bar + 1, t_len)
        for bar in range(entry, path_end):
            close = request.close_2d[bar, symbol]
            if not np.isfinite(close) or close <= 0.0:
                continue
            path_return = side * np.log(close / entry_price) * _BPS
            if not np.isfinite(mae[i]) or path_return < mae[i]:
                mae[i] = path_return
            if not np.isfinite(mfe[i]) or path_return > mfe[i]:
                mfe[i] = path_return
        reasons[i] = exit_code
        exits[i] = exit_bar + 1 if exit_code == 3 else exit_bar
    return gross, costs, funding, edge, reasons, exits, mae, mfe, collisions


def label_exit_paths(request: ExitPathRequest) -> ExitPathLabels:
    """Label causal trade paths with conservative stop/target semantics."""
    arrays = (
        request.decision_idx,
        request.entry_idx,
        request.side,
        request.horizon_bars,
        request.stop_atr_mult,
        request.target_atr_mult,
        request.min_hold_bars,
        request.symbol_idx,
        request.cost_floor_bps,
        request.hurdle_bps,
    )
    lengths = {int(np.asarray(value).size) for value in arrays}
    if len(lengths) != 1:
        raise ValueError("event arrays must have equal length")
    if request.taker_round_trip_bps < 0.0:
        raise ValueError("taker_round_trip_bps must be non-negative")
    causal_mismatch = (request.decision_idx >= 0) & (request.entry_idx != request.decision_idx + 1)
    if bool(np.any(causal_mismatch)):
        raise ValueError("entry_idx must equal decision_idx + 1")
    symbols = request.open_2d.shape[1]
    expected = (request.high_2d, request.low_2d, request.close_2d, request.atr_2d, request.cost_bps_2d)
    if any(value.ndim != 2 for value in expected) or any(value.shape[1] != symbols for value in expected):
        raise ValueError("market arrays must be two-dimensional with a common symbol axis")
    if request.funding_2d.ndim != 2 or request.funding_2d.shape[1] != symbols:
        raise ValueError("funding_2d shape mismatch")
    result = _label_kernel(request)
    reason_map = np.array(["invalid", "stop_loss", "take_profit", "time_exit"], dtype=object)
    return ExitPathLabels(
        gross_bps=result[0], cost_bps=result[1], funding_bps=result[2], edge_bps=result[3],
        exit_reason=reason_map[result[4]], exit_idx=result[5], mae_bps=result[6], mfe_bps=result[7],
        same_bar_collision=result[8],
    )
