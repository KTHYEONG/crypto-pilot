"""Deterministic synthetic MHS discovery-gate workload for the benchmark.

Every fresh benchmark process builds exactly the same fixed multi-year,
64-symbol hourly panel so that wall time, the result checksum, and the
selected/admitted outcome can be compared across processes and before/after
refactor baselines.  The workload runs the densified 19-candidate momentum
grid (``DISCOVERY_MOMENTUM_CANDIDATES``) with the production
``tranche_count=8`` convention.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from src.mhs.discovery import select_horizon_by_discovery_qualification

N_SYMBOLS = 64
SEED = 20260811
COST_BPS = 2.64
TRANCHE_COUNT = 8

DISCOVERY_START = pd.Timestamp("2021-01-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
QUALIFICATION_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

MOMENTUM_CANDIDATES: tuple[int, ...] = (
    72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336,
    360, 384, 408, 432, 456, 480, 504,
)


def _build_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Fixed 4-year hourly panel spanning discovery 2021-2023 + qualification 2024."""
    idx = pd.date_range("2021-01-01", periods=4 * 365 * 24, freq="1h", tz="UTC")
    symbols = [f"SYM{i:03d}USDT" for i in range(N_SYMBOLS)]
    rng = np.random.default_rng(SEED)
    drift = rng.normal(0.0, 1e-4, N_SYMBOLS)
    incs = np.stack(
        [drift[i] + rng.normal(0.0, 1e-3, len(idx)) for i in range(N_SYMBOLS)],
        axis=1,
    )
    log_close = pd.DataFrame(np.cumsum(incs, axis=0), index=idx, columns=symbols)
    o2o = np.stack(
        [rng.normal(0.0, 1e-4, len(idx)) for _ in range(N_SYMBOLS)],
        axis=1,
    )
    opens = pd.DataFrame(
        100.0 * np.exp(np.cumsum(o2o, axis=0)), index=idx, columns=symbols,
    )
    bar_funding = pd.DataFrame(0.0, index=idx, columns=symbols)
    eligible = pd.DataFrame(True, index=idx, columns=symbols)
    return log_close, opens, bar_funding, eligible, idx


def run_sample() -> dict[str, object]:
    """Run the discovery gate once and report timing, workload shape, and
    deterministic outcome checksum."""
    log_close, opens, bar_funding, eligible, idx = _build_panel()
    elapsed_start = time.perf_counter()
    result = select_horizon_by_discovery_qualification(
        sign=1, horizon_candidates=MOMENTUM_CANDIDATES,
        log_close=log_close, eligible=eligible, opens=opens,
        bar_funding=bar_funding, grid_1h=idx,
        discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
        qualification_end=QUALIFICATION_END,
        tranche_count=TRANCHE_COUNT, cost_bps=COST_BPS,
    )
    elapsed_seconds = time.perf_counter() - elapsed_start
    discovery_scores = tuple(result.discovery_scores)
    checksum = hashlib.sha256(
        str(result.selected_horizon).encode()
        + str(result.admitted).encode()
        + repr(discovery_scores).encode()
    ).hexdigest()
    return {
        "elapsed_seconds": elapsed_seconds,
        "n_symbols": N_SYMBOLS,
        "n_candidates": len(MOMENTUM_CANDIDATES),
        "grid_bars": len(idx),
        "selected_horizon": result.selected_horizon,
        "admitted": result.admitted,
        "discovery_scores": discovery_scores,
        "checksum": checksum,
    }
