"""Tests for portfolio covariance estimation and portfolio Kelly solver.

Covers: compute_log_returns_2d, ledoit_wolf_shrink, active_covariance, solve_portfolio_kelly.
Scenarios: S1-S8 (unit), S9-S10 (integration via build_candidate_target_weights).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.portfolio.covariance import (
    active_covariance,
    compute_log_returns_2d,
    ledoit_wolf_shrink,
    solve_portfolio_kelly,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corr_cov(sigmas: list[float], rho: float) -> np.ndarray:
    """Build 2x2 covariance matrix with given std-devs and correlation."""
    s = np.array(sigmas)
    corr = np.array([[1.0, rho], [rho, 1.0]])
    D = np.diag(s)
    return np.asarray(D @ corr @ D, dtype=np.float64)


# ---------------------------------------------------------------------------
# S1 — Happy: positive correlation → concentration penalty reduces total exposure
# ---------------------------------------------------------------------------

def test_solve_portfolio_kelly_positive_correlation_reduces_exposure() -> None:
    # Arrange
    sigma_vals = [0.02, 0.02]
    cov_high_rho = _make_corr_cov(sigma_vals, rho=0.8)
    cov_diag = np.diag(np.array(sigma_vals) ** 2)  # zero correlation baseline
    mu = np.array([0.5, 0.5])

    # Act — large cap so clip doesn't mask the difference
    w_corr = solve_portfolio_kelly(mu, cov_high_rho, kelly_fraction=0.25, ridge_eps=1e-3, per_symbol_cap=1000.0)
    w_diag = solve_portfolio_kelly(mu, cov_diag, kelly_fraction=0.25, ridge_eps=1e-3, per_symbol_cap=1000.0)

    # Assert: high correlation → smaller total exposure (risk concentration penalty)
    assert np.sum(np.abs(w_corr)) < np.sum(np.abs(w_diag))
    assert np.all(w_corr >= 0.0), "sign of mu must be preserved"


# ---------------------------------------------------------------------------
# S2 — Diagonal equivalence (backward-compatible)
# ---------------------------------------------------------------------------

def test_solve_portfolio_kelly_diagonal_covariance_matches_scalar_formula() -> None:
    # Arrange
    var1, var2 = 0.0004, 0.0009  # sigma = 2%, 3%
    cov_diag = np.diag([var1, var2])
    mu = np.array([0.3, 0.6])
    f_k = 0.25

    # Act — cap large enough to avoid clipping (expected ~187, ~166)
    w = solve_portfolio_kelly(mu, cov_diag, kelly_fraction=f_k, ridge_eps=0.0, per_symbol_cap=500.0)

    # Assert: w_i ≈ f_k * mu_i / sigma_i²  (no ridge since ridge_eps=0)
    expected = f_k * mu / np.array([var1, var2])
    assert w == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# S3 — Negative correlation → more aggressive (lower portfolio variance)
# ---------------------------------------------------------------------------

def test_solve_portfolio_kelly_negative_correlation_increases_exposure() -> None:
    # Arrange
    sigma_vals = [0.02, 0.02]
    cov_neg = _make_corr_cov(sigma_vals, rho=-0.7)
    cov_pos = _make_corr_cov(sigma_vals, rho=0.8)
    mu = np.array([0.5, 0.5])

    # Act — large cap so clip doesn't mask the difference
    w_neg = solve_portfolio_kelly(mu, cov_neg, kelly_fraction=0.25, ridge_eps=1e-3, per_symbol_cap=1000.0)
    w_pos = solve_portfolio_kelly(mu, cov_pos, kelly_fraction=0.25, ridge_eps=1e-3, per_symbol_cap=1000.0)

    # Assert: negative correlation → larger total weight (diversification benefit)
    assert np.sum(np.abs(w_neg)) > np.sum(np.abs(w_pos))
    # Signs preserved
    assert np.all(w_neg >= 0.0)


# ---------------------------------------------------------------------------
# S4 — Insufficient observations → None
# ---------------------------------------------------------------------------

def test_active_covariance_returns_none_when_insufficient_obs() -> None:
    # Arrange
    T, N = 100, 3
    logret = np.random.default_rng(0).standard_normal((T, N)) * 0.01
    active_idx = np.array([0, 1], dtype=np.int64)

    # Act: t=50 with min_obs=60 → only 50 observations available
    result = active_covariance(logret, t=50, active_idx=active_idx, window=200, shrinkage=None, min_obs=60)

    # Assert
    assert result is None


# ---------------------------------------------------------------------------
# S5 — Single active symbol: portfolio path not entered (integration via build)
# ---------------------------------------------------------------------------

def test_solve_portfolio_kelly_single_asset_still_returns_valid() -> None:
    # Arrange — k=1 edge case
    cov_1d = np.array([[0.0004]])
    mu_1d = np.array([0.5])

    # Act
    w = solve_portfolio_kelly(mu_1d, cov_1d, kelly_fraction=0.25, ridge_eps=1e-3, per_symbol_cap=1.0)

    # Assert: finite, positive
    assert np.isfinite(w[0])
    assert w[0] > 0.0


# ---------------------------------------------------------------------------
# S6 — Near-singular covariance: ridge prevents LinAlgError
# ---------------------------------------------------------------------------

def test_solve_portfolio_kelly_near_singular_no_exception() -> None:
    # Arrange: perfect correlation → singular matrix
    sigma = 0.02
    cov_singular = np.array([[sigma**2, sigma**2], [sigma**2, sigma**2]])  # rho=1
    mu = np.array([0.5, 0.5])

    # Act (should NOT raise)
    w = solve_portfolio_kelly(mu, cov_singular, kelly_fraction=0.25, ridge_eps=1e-2, per_symbol_cap=1.0)

    # Assert: finite values
    assert np.all(np.isfinite(w))


# ---------------------------------------------------------------------------
# S7 — Ledoit-Wolf boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("intensity", "desc"),
    [
        (0.0, "no shrinkage → equals sample_cov"),
        (1.0, "full shrinkage → diagonal"),
    ],
)
def test_ledoit_wolf_shrink_boundary_intensities(intensity: float, desc: str) -> None:
    # Arrange
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((50, 3))
    S = np.cov(raw, rowvar=False)

    # Act
    result = ledoit_wolf_shrink(S, intensity)

    if intensity == 0.0:
        assert result == pytest.approx(S, rel=1e-10), desc
    else:
        diag_S = np.diag(np.diag(S))
        assert result == pytest.approx(diag_S, rel=1e-10), desc


def test_ledoit_wolf_shrink_auto_produces_spd() -> None:
    # Arrange
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((30, 4)) * 0.01
    S = np.cov(raw, rowvar=False)

    # Act
    result = ledoit_wolf_shrink(S, intensity=None)

    # Assert: SPD (all positive eigenvalues)
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > 0.0), f"Eigenvalues not all positive: {eigvals}"


def test_ledoit_wolf_shrink_auto_intensity_in_unit_interval() -> None:
    # Arrange: verify analytic δ is between 0 and 1 (SPD implied)
    rng = np.random.default_rng(99)
    S = np.cov(rng.standard_normal((60, 5)), rowvar=False)

    # Act
    result = ledoit_wolf_shrink(S, intensity=None)

    # Verify: result ≠ S (some shrinkage applied) and result has diag matching S
    assert not np.allclose(result, S), "auto should shrink toward diagonal"
    assert result.diagonal() == pytest.approx(S.diagonal(), rel=1e-10)


# ---------------------------------------------------------------------------
# S8 — Look-ahead defense: future close spike must not affect result at t
# ---------------------------------------------------------------------------

def test_active_covariance_no_lookahead() -> None:
    # Arrange
    T, N = 300, 3
    rng = np.random.default_rng(123)
    close = 100.0 + np.cumsum(rng.standard_normal((T, N)) * 0.5, axis=0)
    logret = compute_log_returns_2d(close)
    active_idx = np.array([0, 1], dtype=np.int64)
    t = 200
    min_obs = 60

    baseline = active_covariance(logret, t, active_idx, window=180, shrinkage=0.1, min_obs=min_obs)

    # Inject spike at t and beyond (future data)
    logret_future = logret.copy()
    logret_future[t:, 0] *= 100.0  # massive spike from t onwards

    spike_result = active_covariance(logret_future, t, active_idx, window=180, shrinkage=0.1, min_obs=min_obs)

    # Assert: results identical (window [t-180, t) excludes bar t)
    assert baseline is not None
    assert spike_result is not None
    assert baseline == pytest.approx(spike_result, rel=1e-10)


def test_active_covariance_past_change_affects_result() -> None:
    # Arrange
    T, N = 300, 3
    rng = np.random.default_rng(456)
    close = 100.0 + np.cumsum(rng.standard_normal((T, N)) * 0.5, axis=0)
    logret = compute_log_returns_2d(close)
    active_idx = np.array([0, 1], dtype=np.int64)
    t = 200

    baseline = active_covariance(logret, t, active_idx, window=180, shrinkage=0.1, min_obs=60)

    # Modify past data (within the window)
    logret_past = logret.copy()
    logret_past[t - 10, 0] *= 50.0

    past_result = active_covariance(logret_past, t, active_idx, window=180, shrinkage=0.1, min_obs=60)

    # Assert: must differ (past data participates in covariance window)
    assert baseline is not None
    assert past_result is not None
    assert not np.allclose(baseline, past_result)


# ---------------------------------------------------------------------------
# S9/S10 — Integration tests via build_candidate_target_weights
# ---------------------------------------------------------------------------

def _make_minimal_cfg(use_portfolio_kelly: bool = False, cov_window: int = 30) -> CandidateStrategyConfig:
    """Build a minimal CandidateStrategyConfig for integration tests."""
    from dataclasses import replace

    base = CandidateStrategyConfig()
    return replace(
        base,
        sizing_mode="calibrated_event_kelly",
        use_portfolio_kelly=use_portfolio_kelly,
        cov_window=cov_window,
        cov_min_obs=10,
        gross_cap=999.0,
        net_cap=999.0,
        beta_cap=999.0,
        target_ann_vol=999.0,
        double_scaling_guard=True,
    )


def _make_selected_events(symbols: list[str], entry_idx: int) -> pd.DataFrame:
    """Build minimal selected_events DataFrame for two symbols."""
    return pd.DataFrame(
        {
            "symbol": symbols,
            "entry_idx": [entry_idx] * len(symbols),
            "side": [1.0] * len(symbols),
            "mu_net_decision_bps": [30.0] * len(symbols),
            "q10_net_bps": [-10.0] * len(symbols),
            "q90_net_bps": [70.0] * len(symbols),
            "risk_unit_bps": [25.0] * len(symbols),
            "p_pass": [1.0] * len(symbols),
            "expected_holding_bars": [5] * len(symbols),
            "overlay_mult": [1.0] * len(symbols),
        }
    )


def test_build_candidate_target_weights_portfolio_kelly_fallback_when_short_panel() -> None:
    """S10 — short panel (< cov_min_obs) → fallback, no exception, finite weights."""
    from src.domain.futures.strategy.candidate_portfolio import build_candidate_target_weights

    # Arrange: only 5 bars < cov_min_obs=10
    T, N = 5, 2
    rng = np.random.default_rng(0)
    close_2d = 100.0 + np.cumsum(rng.standard_normal((T, N)) * 0.5, axis=0)
    symbols = ("BTCUSDT", "ETHUSDT")
    selected = _make_selected_events(list(symbols), entry_idx=1)
    cfg = _make_minimal_cfg(use_portfolio_kelly=True, cov_window=30)

    # Act
    weights = build_candidate_target_weights(
        selected_events=selected,
        close_2d=close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )

    # Assert: no exception, finite weights
    assert np.all(np.isfinite(weights))
    assert weights.shape == (T, N)


def test_build_candidate_target_weights_portfolio_kelly_toggle_finite() -> None:
    """S9 — portfolio_kelly True/False both produce finite weights for same input."""
    from src.domain.futures.strategy.candidate_portfolio import build_candidate_target_weights

    # Arrange: zero-correlation close data (independent assets)
    T = 200
    rng = np.random.default_rng(42)
    returns_a = rng.standard_normal(T) * 0.01
    returns_b = rng.standard_normal(T) * 0.01  # independent
    close_a = 100.0 * np.exp(np.cumsum(returns_a))
    close_b = 100.0 * np.exp(np.cumsum(returns_b))
    close_2d = np.column_stack([close_a, close_b])
    symbols = ("BTCUSDT", "ETHUSDT")
    selected = _make_selected_events(list(symbols), entry_idx=100)

    cfg_diag = _make_minimal_cfg(use_portfolio_kelly=False, cov_window=50)
    cfg_port = _make_minimal_cfg(use_portfolio_kelly=True, cov_window=50)

    # Act
    w_diag = build_candidate_target_weights(
        selected_events=selected, close_2d=close_2d, symbols=symbols,
        beta_2d=None, sigma_3d=None, cfg=cfg_diag,
    )
    w_port = build_candidate_target_weights(
        selected_events=selected, close_2d=close_2d, symbols=symbols,
        beta_2d=None, sigma_3d=None, cfg=cfg_port,
    )

    # Assert: both produce finite, non-negative (same side) weights
    assert np.all(np.isfinite(w_diag))
    assert np.all(np.isfinite(w_port))
    # Both should have positive weights at entry (signal has positive side+mu)
    assert w_diag[100].sum() > 0.0
    assert w_port[100].sum() > 0.0


# ---------------------------------------------------------------------------
# compute_log_returns_2d basic sanity
# ---------------------------------------------------------------------------

def test_compute_log_returns_2d_first_row_zero() -> None:
    close = np.array([[100.0, 200.0], [110.0, 210.0], [105.0, 215.0]])
    logret = compute_log_returns_2d(close)
    assert logret[0] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_compute_log_returns_2d_correct_values() -> None:
    close = np.array([[100.0], [110.0]])
    logret = compute_log_returns_2d(close)
    expected = np.log1p((110.0 - 100.0) / 100.0)
    assert float(logret[1, 0]) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Extra coverage: OAS fallback, scalar ndim=0, LinAlgError fallback
# ---------------------------------------------------------------------------

def test_ledoit_wolf_shrink_oas_near_zero_denominator_fallback() -> None:
    # Force near-zero denominator in OAS: perfectly homogeneous matrix → tr(S²) ≈ tr(S)²/k
    # Using an identity-like matrix scaled so numerator/denominator blows up
    k = 2
    S = np.eye(k) * 1.0  # symmetric, tr(S²)=k, tr(S)=k → denominator=(k - k²/k)=0
    result = ledoit_wolf_shrink(S, intensity=None)
    assert np.all(np.isfinite(result))
    assert result.shape == (k, k)


def test_active_covariance_k1_scalar_path() -> None:
    # k=1 with sufficient obs triggers np.cov scalar (ndim=0) path
    T = 100
    logret = np.random.default_rng(7).standard_normal((T, 3)) * 0.01
    active_idx = np.array([0], dtype=np.int64)  # single asset → scalar cov
    result = active_covariance(logret, t=80, active_idx=active_idx, window=60, shrinkage=0.1, min_obs=10)
    assert result is not None
    assert result.shape == (1, 1)
    assert np.isfinite(result[0, 0])


def test_solve_portfolio_kelly_linalg_error_fallback() -> None:
    # Construct a matrix that is exactly singular after ridge (force by making ridge_eps=0 and cov=zeros)
    # With ridge_eps>0 and mean_var=0 → eps=0 → may still solve; use monkeypatch instead
    import unittest.mock as mock

    mu = np.array([0.3, 0.4])
    cov = np.eye(2) * 0.001
    with mock.patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError("forced")):
        w = solve_portfolio_kelly(mu, cov, kelly_fraction=0.25, ridge_eps=1e-3, per_symbol_cap=1000.0)
    # Diagonal fallback: w_i = f_k * mu_i / diag_var_i
    expected = 0.25 * mu / np.diag(cov)
    assert w == pytest.approx(expected, rel=1e-5)
