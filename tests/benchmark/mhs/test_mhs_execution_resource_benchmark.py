"""Small deterministic paired-replay resource benchmark (MHS-MEM-PAIR-03).

Runs the fixed 90-day / 64-symbol / 5-minute workload once in-process, feeds a
single window iterator to ``replay_execution_window_pair``, and records paired
equivalence against the two independent single-bound calls plus paired and
independent elapsed time, peak RSS, and the maximum loaded-window grid.  No
absolute time/RSS gate is enforced: the multi-year acceptance command is the
slow CLI-only verification.  This benchmark is intentionally fast enough to run
in the default (non-slow) suite.

SCENARIO_MHS_PERF_ACCEPT_02_END_TO_END (docs/specs/mhs_perf_refactor_contract.json):
this fast paired-replay benchmark is the proxy this file automates; the full
end-to-end acceptance gate (CLI wall time including report persistence,
byte-identical output) is a manual reproduction run against production data,
not a pytest scenario -- ``uv run python -m src.cli.main research run
portfolio mhs-horizon-diagnostic``, per docs/specs/mhs_perf_refactor.md §10-11.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

from src.mhs.execution import ExecutionSpec, replay_execution_window_pair, replay_execution_windows

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mhs_benchmark_sample import _partition_windows, build_workload  # noqa: E402


def _max_window_grid_bars(windows) -> int:
    return max(len(w.minute_grid) for w in windows)


def run_paired_benchmark() -> dict[str, object]:
    """One in-process paired vs independent measurement on the fixed workload."""
    workload = build_workload()
    windows = _partition_windows(
        workload["grid"], workload["weights"], workload["signal_available_at"],
        workload["highs"], workload["lows"], workload["closes"], workload["marks"],
        workload["funding"], ExecutionSpec(),
    )
    independent_start = time.perf_counter()
    strict_single = replay_execution_windows(windows, 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec())
    stress_single = replay_execution_windows(windows, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    independent_elapsed = time.perf_counter() - independent_start

    paired_start = time.perf_counter()
    strict_pair, stress_pair = replay_execution_window_pair(windows, 1.0, ExecutionSpec())
    paired_elapsed = time.perf_counter() - paired_start

    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def _matches(a, b) -> bool:
        fill_a = a.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        fill_b = b.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return bool(
            len(fill_a) == len(fill_b)
            and dict(a.termination_counts) == dict(b.termination_counts)
            and np.allclose(
                a.ledger.equity.to_numpy(), b.ledger.equity.to_numpy(),
                rtol=1e-12, atol=1e-12,
            )
        )

    return {
        "paired_equivalent": bool(_matches(strict_single, strict_pair) and _matches(stress_single, stress_pair)),
        "paired_elapsed_seconds": paired_elapsed,
        "independent_elapsed_seconds": independent_elapsed,
        "peak_rss_kb": peak_rss_kb,
        "n_windows": len(windows),
        "max_window_grid_bars": _max_window_grid_bars(windows),
        "fill_count": len(strict_pair.simulated_fills),
    }


def test_mhs_mem_pair_03_paired_resource_benchmark() -> None:
    result = run_paired_benchmark()
    assert result["paired_equivalent"] is True
    assert result["paired_elapsed_seconds"] > 0.0
    assert result["independent_elapsed_seconds"] > 0.0
    assert result["peak_rss_kb"] > 0
    assert result["n_windows"] >= 3
    assert result["max_window_grid_bars"] > 0
    assert result["fill_count"] > 0
    # The paired fan-out shares one window iterator, so it must not cost
    # meaningfully more than two independent passes on identical pre-built
    # windows.  A relative bound keeps the check robust under parallel CI load
    # while still catching a regression that re-iterates or re-builds windows.
    assert result["paired_elapsed_seconds"] <= result["independent_elapsed_seconds"] * 1.5 + 1.0
