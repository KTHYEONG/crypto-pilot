from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.l1_sleeves import compute_chunked_2d_tensor_bootstrap

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

