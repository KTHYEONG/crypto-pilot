"""Light checks for portfolio_constructor weights."""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.portfolio_constructor import (
    precompute_rebalance_weights,
    solve_constrained_weights,
)


def test_solve_constrained_weights_shapes_and_gross_cap() -> None:
    rng = np.random.default_rng(0)
    mu = rng.normal(scale=5e-4, size=5)
    Sigma = np.diag(np.full(5, 1e-6))
    w = solve_constrained_weights(
        mu,
        Sigma,
        kappa=0.5,
        f_kelly_max=2.0,
        sigma_target_ann=0.30,
        bars_per_year=252.0,
        gross_cap=1.0,
        per_symbol_cap=0.5,
        current_dd=0.0,
    )
    assert w.shape == (5,)
    assert float(np.sum(np.abs(w))) <= 1.0 + 1e-8


def test_precompute_sparse_rebalance_rows() -> None:
    rng = np.random.default_rng(1)
    t, n = 80, 3
    c = np.cumsum(rng.normal(size=(t, n)) * 0.003, axis=0) + 100.0
    xl = np.clip(0.5 + rng.normal(size=(t, n)) * 0.05, 0.0, 1.0)
    xs = np.clip(0.5 + rng.normal(size=(t, n)) * 0.05, 0.0, 1.0)
    w = precompute_rebalance_weights(
        c,
        xl,
        xs,
        rebalance_bars=10,
        lookback=20,
        bars_per_year=2000.0,
        kappa=0.2,
        f_kelly_max=2.0,
        sigma_target_ann=0.25,
        gross_cap=1.2,
        per_symbol_cap=0.45,
        current_dd=0.0,
    )
    assert w.shape == (t, n)
    nz = np.sum(np.any(np.abs(w) > 1e-12, axis=1))
    assert nz >= 1
