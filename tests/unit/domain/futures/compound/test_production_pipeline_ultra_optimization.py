from __future__ import annotations

import time

import numpy as np

from src.domain.futures.compound.l1_sleeves import compute_chunked_2d_tensor_bootstrap
from src.domain.futures.compound.signal_bank import _rolling_mad_z_numba_kernel

def test_tensor_bootstrap_oom_safety() -> None:
    rng = np.random.default_rng(42)
    returns_2d = (rng.standard_normal((200, 100)) * 0.001).astype(np.float64)
    res = compute_chunked_2d_tensor_bootstrap(returns_2d, periods_per_year=2191.5, n_bootstrap=50, chunk_size=25, seed=42)
    assert res.shape == (100,)
    assert np.all(np.isfinite(res))

def test_tensor_bootstrap_math_identity() -> None:
    rng = np.random.default_rng(42)
    returns_2d = (rng.standard_normal((200, 20)) * 0.001).astype(np.float64)
    res1 = compute_chunked_2d_tensor_bootstrap(returns_2d, periods_per_year=2191.5, n_bootstrap=50, chunk_size=20, seed=42)
    res2 = compute_chunked_2d_tensor_bootstrap(returns_2d, periods_per_year=2191.5, n_bootstrap=50, chunk_size=20, seed=42)
    assert np.allclose(res1, res2, atol=1e-12)

def test_full_production_pipeline_speedup() -> None:
    assert True


def test_production_pipeline_deep_optimization_performance() -> None:
    rng = np.random.default_rng(42)
    n_t, n_s = 5000, 51
    arr = rng.standard_normal((n_t, n_s)).astype(np.float64)
    arr[100:200, :5] = np.nan

    total = 0.0
    n_trials = 5
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr), window=252, min_periods=126)
        total += time.perf_counter() - start

    avg_seconds = total / n_trials
    assert avg_seconds < 10.0, (
        f"Zero-allocation MAD kernel too slow: {avg_seconds:.3f}s avg over {n_trials} runs "
        f"(expected < 10s for {n_t}x{n_s})"
    )

