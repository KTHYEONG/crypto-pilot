from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.portfolio.portfolio_constructor import precompute_rebalance_weights
from src.domain.futures.portfolio.portfolio_optimizer import PortfolioPolicyInputs


def _base_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bars = 3
    n_syms = 2
    close_2d = np.array(
        [
            [100.0, 101.0],
            [100.5, 101.5],
            [101.0, 102.0],
        ],
        dtype=np.float64,
    )
    xs_long = np.full((n_bars, n_syms), 1.0, dtype=np.float64)
    xs_short = np.zeros((n_bars, n_syms), dtype=np.float64)
    sigma_3d = np.repeat(np.eye(n_syms, dtype=np.float64)[None, :, :] * 1e-4, n_bars, axis=0)
    return close_2d, xs_long, xs_short, sigma_3d


# --- From test_portfolio_constructor.py ---


def test_portfolio_constructor_basic() -> None:
    close_2d, xs_long, xs_short, sigma_3d = _base_inputs()

    # Minimal inputs to just verify it runs without error
    w = precompute_rebalance_weights(
        close_2d,
        xs_long,
        xs_short,
        rebalance_bars=1,
        lookback=2,
        bars_per_year=365.0 * 24.0,
        kappa=0.3,
        f_kelly_max=1.0,
        sigma_target_ann=0.2,
        gross_cap=1.0,
        per_symbol_cap=1.0,
        sigma_3d=sigma_3d,
    )

    assert w.shape == (3, 2)
    assert np.isfinite(w).all()


def test_portfolio_constructor_cost_suppresses_target_weights() -> None:
    close_2d, xs_long, xs_short, sigma_3d = _base_inputs()

    low_cost_inputs = PortfolioPolicyInputs(
        mu_long_2d=np.full_like(xs_long, 0.004),
        mu_short_2d=np.zeros_like(xs_short),
        risk_sigma_3d=sigma_3d,
        cost_fraction_2d=np.zeros_like(xs_long),
        cost_bps_2d=np.zeros_like(xs_long),
        cost_source="per_symbol",
    )
    high_cost_inputs = PortfolioPolicyInputs(
        mu_long_2d=np.full_like(xs_long, 0.0002),
        mu_short_2d=np.zeros_like(xs_short),
        risk_sigma_3d=sigma_3d,
        cost_fraction_2d=np.full_like(xs_long, 0.0004),
        cost_bps_2d=np.full_like(xs_long, 4.0),
        cost_source="per_symbol",
    )

    w_low = precompute_rebalance_weights(
        close_2d,
        xs_long,
        xs_short,
        rebalance_bars=1,
        lookback=2,
        bars_per_year=365.0 * 24.0,
        kappa=0.3,
        f_kelly_max=1.0,
        sigma_target_ann=0.2,
        gross_cap=1.0,
        per_symbol_cap=0.8,
        sigma_3d=sigma_3d,
        policy_inputs=low_cost_inputs,
    )
    w_high = precompute_rebalance_weights(
        close_2d,
        xs_long,
        xs_short,
        rebalance_bars=1,
        lookback=2,
        bars_per_year=365.0 * 24.0,
        kappa=0.3,
        f_kelly_max=1.0,
        sigma_target_ann=0.2,
        gross_cap=1.0,
        per_symbol_cap=0.8,
        sigma_3d=sigma_3d,
        policy_inputs=high_cost_inputs,
    )

    assert float(np.sum(np.abs(w_low[-1]))) >= float(np.sum(np.abs(w_high[-1])))


def test_portfolio_constructor_caps_and_beta_remain_enforced() -> None:
    close_2d, xs_long, xs_short, sigma_3d = _base_inputs()
    n_bars, n_syms = xs_long.shape
    beta_2d = np.ones((n_bars, n_syms), dtype=np.float64)

    policy_inputs = PortfolioPolicyInputs(
        mu_long_2d=np.full_like(xs_long, 0.01),
        mu_short_2d=np.zeros_like(xs_short),
        risk_sigma_3d=sigma_3d,
        risk_beta_2d=beta_2d,
    )
    w = precompute_rebalance_weights(
        close_2d,
        xs_long,
        xs_short,
        rebalance_bars=1,
        lookback=2,
        bars_per_year=365.0 * 24.0,
        kappa=0.8,
        f_kelly_max=2.0,
        sigma_target_ann=0.2,
        gross_cap=0.6,
        per_symbol_cap=0.2,
        sigma_3d=sigma_3d,
        btc_beta_2d=beta_2d,
        policy_inputs=policy_inputs,
    )
    last = w[-1]
    assert np.max(np.abs(last)) <= 0.200001
    assert float(np.sum(np.abs(last))) <= 0.600001
    assert abs(float(np.dot(last, beta_2d[-1]))) <= 0.500001


def test_precompute_rebalance_weights_ignores_static_current_dd_scaling() -> None:
    close_2d, xs_long, xs_short, sigma_3d = _base_inputs()
    common_kwargs = dict(
        close_2d=close_2d,
        xs_long=xs_long,
        xs_short=xs_short,
        rebalance_bars=1,
        lookback=2,
        bars_per_year=365.0 * 24.0,
        kappa=0.3,
        f_kelly_max=1.0,
        sigma_target_ann=0.2,
        gross_cap=1.0,
        per_symbol_cap=1.0,
        sigma_3d=sigma_3d,
    )
    w_dd0 = precompute_rebalance_weights(current_dd=0.0, **common_kwargs)
    w_dd20 = precompute_rebalance_weights(current_dd=0.2, **common_kwargs)
    np.testing.assert_allclose(w_dd0, w_dd20, rtol=0.0, atol=0.0)
