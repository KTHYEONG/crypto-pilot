"""Deterministic synthetic MHS replay workload for the resource benchmark.

Every fresh benchmark process builds exactly the same fixed 90-day, 64-symbol,
5-minute workload (25,920 bars and 360 decisions) so that wall time, peak RSS,
fill counts, and the result checksum can be compared across processes and
before/after refactor baselines.
"""

from __future__ import annotations

import hashlib
import resource
import time

import numpy as np
import pandas as pd

from src.mhs.execution import (
    ExecutionReplayWindow,
    ExecutionSpec,
    replay_execution_windows,
    strategy_aware_execution_replay,
)

N_SYMBOLS = 64
N_DAYS = 90
FREQ = "5min"
SEED = 20260807
N_DECISIONS = 360
ACTIVE_PER_DECISION = 20
N_WINDOWS = 3
WINDOW_MAX_DAYS = 31


def _partition_windows(
    grid, weights, signals, highs, lows, closes, marks, funding, spec,
) -> list[ExecutionReplayWindow]:
    """Split the fixed workload into contiguous execution windows with strict
    timeout overlap, matching the production planner's residency change."""
    full_ns = np.asarray(grid, dtype="datetime64[ns]").astype("int64")
    timeout = spec.passive_timeout_minutes * 60_000_000_000
    sig_ns = np.asarray(signals, dtype="datetime64[ns]").astype("int64")
    spos = np.searchsorted(full_ns, sig_ns, side="right")
    resolve = [None] * len(weights)
    for i in range(len(weights)):
        if spos[i] >= len(full_ns):
            continue
        tns = full_ns[spos[i]] + timeout
        tpos = int(np.searchsorted(full_ns, tns, side="left"))
        if tpos < len(full_ns) and full_ns[tpos] == tns:
            resolve[i] = pd.Timestamp(tns, unit="ns", tz="UTC")
    decision_times = pd.DatetimeIndex(weights.index)
    max_window = pd.Timedelta(days=WINDOW_MAX_DAYS)
    bounds: list[tuple[int, int]] = []
    i0 = 0
    while i0 < len(decision_times):
        i1 = i0 + 1
        while i1 < len(decision_times) and decision_times[i1] - decision_times[i0] <= max_window:
            i1 += 1
        bounds.append((i0, i1))
        i0 = i1
    out: list[ExecutionReplayWindow] = []
    for bi, (i0, i1) in enumerate(bounds):
        is_last = bi == len(bounds) - 1
        ws = weights.iloc[i0:i1]
        sg = signals[i0:i1]
        grid_start = grid[0] if bi == 0 else decision_times[i0 - 1]
        if is_last:
            grid_end = grid[-1]
        else:
            grid_end = max((resolve[i] for i in range(i0, i1) if resolve[i] is not None), default=decision_times[i1 - 1] + pd.Timedelta(hours=2))
        wgrid = pd.date_range(grid_start, grid_end, freq=FREQ, tz="UTC")
        out.append(
            ExecutionReplayWindow(
                window_start=grid_start,
                window_end=grid_end,
                columns=tuple(weights.columns),
                symbols=tuple(weights.columns),
                minute_grid=wgrid,
                highs=highs.loc[wgrid],
                lows=lows.loc[wgrid],
                closes=closes.loc[wgrid],
                marks=marks.loc[wgrid],
                bar_funding=funding.loc[wgrid],
                target_weights=ws,
                signal_available_at=sg,
            )
        )
    return out


def build_workload() -> dict[str, object]:
    """Fixed synthetic workload matching the spec's directional profiling harness."""
    grid = pd.date_range("2021-01-01", periods=N_DAYS * 24 * 12, freq=FREQ, tz="UTC")
    symbols = [f"SYM{i:03d}USDT" for i in range(N_SYMBOLS)]
    rng = np.random.default_rng(SEED)
    drift = rng.normal(0.0, 0.0001, N_SYMBOLS)
    closes = pd.DataFrame(
        {
            sym: 100.0 * np.exp(np.cumsum(drift[i] + rng.normal(0.0, 0.002, len(grid))))
            for i, sym in enumerate(symbols)
        },
        index=grid,
    )
    decision_grid = pd.date_range("2021-01-01", periods=N_DECISIONS, freq="6h", tz="UTC")
    weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
    rng_w = np.random.default_rng(SEED + 1)
    for ts in decision_grid:
        active = rng_w.choice(symbols, size=ACTIVE_PER_DECISION, replace=False)
        weights.loc[ts, active] = rng_w.uniform(0.01, 0.06, ACTIVE_PER_DECISION)
    return {
        "grid": grid,
        "symbols": symbols,
        "highs": closes * 1.001,
        "lows": closes * 0.999,
        "closes": closes,
        "marks": closes,
        "funding": pd.DataFrame(1.0e-5, index=grid, columns=symbols),
        "weights": weights,
        "signal_available_at": decision_grid + pd.Timedelta(hours=1),
    }


def run_sample() -> dict[str, object]:
    """Run strict + stress replay plus the windowed engine once and report
    timing, RSS, window profile, and checksums."""
    workload = build_workload()
    weights = workload["weights"]
    signal = workload["signal_available_at"]
    elapsed_start = time.perf_counter()
    strict = strategy_aware_execution_replay(
        weights, signal, workload["highs"], workload["lows"], workload["closes"],
        workload["marks"], workload["funding"], 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    stress = strategy_aware_execution_replay(
        weights, signal, workload["highs"], workload["lows"], workload["closes"],
        workload["marks"], workload["funding"], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
    )
    elapsed_seconds = time.perf_counter() - elapsed_start
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    fills = strict.simulated_fills.sort_values(["timestamp", "symbol"], kind="stable")
    checksum = hashlib.sha256(
        pd.util.hash_pandas_object(fills, index=False).to_numpy().tobytes()
        + strict.ledger.equity.to_numpy().tobytes()
        + str(sorted(strict.termination_counts.items())).encode()
        + str(sorted(stress.termination_counts.items())).encode()
    ).hexdigest()

    windows = _partition_windows(
        workload["grid"], weights, signal, workload["highs"], workload["lows"],
        workload["closes"], workload["marks"], workload["funding"], ExecutionSpec(),
    )
    window_elapsed_start = time.perf_counter()
    windowed = replay_execution_windows(
        windows, 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    window_elapsed_seconds = time.perf_counter() - window_elapsed_start
    windowed_fills = windowed.simulated_fills.sort_values(["timestamp", "symbol"], kind="stable")
    windowed_checksum = hashlib.sha256(
        pd.util.hash_pandas_object(windowed_fills, index=False).to_numpy().tobytes()
        + windowed.ledger.equity.to_numpy().tobytes()
        + str(sorted(windowed.termination_counts.items())).encode()
    ).hexdigest()
    windowed_matches_single_panel = bool(
        len(windowed_fills) == len(fills)
        and np.allclose(
            windowed.ledger.equity.to_numpy(), strict.ledger.equity.to_numpy(),
            rtol=1e-12, atol=1e-12,
        )
        and dict(windowed.termination_counts) == dict(strict.termination_counts)
    )
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_kb": peak_rss_kb,
        "grid_bars": len(workload["grid"]),
        "n_symbols": N_SYMBOLS,
        "decisions": len(weights),
        "fill_count": len(fills),
        "checksum": checksum,
        "n_windows": len(windows),
        "max_window_grid_bars": max(len(w.minute_grid) for w in windows),
        "window_elapsed_seconds": window_elapsed_seconds,
        "windowed_checksum": windowed_checksum,
        "windowed_matches_single_panel": windowed_matches_single_panel,
    }
