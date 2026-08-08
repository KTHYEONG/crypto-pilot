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

from src.mhs.execution import ExecutionSpec, strategy_aware_execution_replay

N_SYMBOLS = 64
N_DAYS = 90
FREQ = "5min"
SEED = 20260807
N_DECISIONS = 360
ACTIVE_PER_DECISION = 20


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
    """Run strict + stress replay once and report timing, RSS, and checksum."""
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
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_kb": peak_rss_kb,
        "grid_bars": len(workload["grid"]),
        "n_symbols": N_SYMBOLS,
        "decisions": len(weights),
        "fill_count": len(fills),
        "checksum": checksum,
    }
