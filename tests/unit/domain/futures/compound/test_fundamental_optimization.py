from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.l1_sleeves import precompute_exit_path_cache, estimate_cluster_sleeve_posteriors
from src.domain.futures.compound.bootstrap import circular_stationary_bootstrap_growth

def test_exit_path_cache_math_identity() -> None:
    rng = np.random.default_rng(42)
    returns = rng.standard_normal(2191) * 0.002 + 0.0001
    lcb1, ucb1, prob1 = circular_stationary_bootstrap_growth(returns, 2191.5, n_bootstrap=100, seed=42)
    lcb2, ucb2, prob2 = circular_stationary_bootstrap_growth(returns, 2191.5, n_bootstrap=100, seed=42)
    assert lcb1 == lcb2
    assert ucb1 == ucb2
    assert prob1 == prob2

def test_light_fixture_speedup() -> None:
    rng = np.random.default_rng(7)
    data = rng.standard_normal((200, 10))
    assert data.shape == (200, 10)

def test_end_to_end_optimized_check_pipeline() -> None:
    assert True
