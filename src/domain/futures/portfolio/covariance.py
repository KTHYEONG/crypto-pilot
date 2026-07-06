"""Portfolio covariance estimation and Kelly weight solver.

Implements trailing-window Ledoit-Wolf shrinkage covariance and
portfolio Kelly w = f_k · (Σ̂ + εI)⁻¹ μ for active sub-universes.

Memory design: no dense [T,N,N] allocation — active sub-matrix computed on-demand per bar.
Look-ahead safety: window is always [t-window, t), never includes bar t.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


def compute_log_returns_2d(close_2d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert close prices to log-returns.

    Args:
        close_2d: close prices, shape [T, N], float64.

    Returns:
        log-returns shape [T, N]; row 0 is 0.0 (no prior bar).
    """
    out = np.zeros_like(close_2d)
    prev = np.maximum(close_2d[:-1], 1e-12)
    out[1:] = np.log1p((close_2d[1:] - close_2d[:-1]) / prev)
    return out


def ledoit_wolf_shrink(
    sample_cov: NDArray[np.float64],
    intensity: float | None,
) -> NDArray[np.float64]:
    """Shrink sample covariance toward its diagonal (Oracle Approximating Shrinkage).

    Shrinkage target: F = diag(sample_cov) — preserves marginal variances, shrinks correlations.
    S_hat = delta*F + (1-delta)*S, delta in [0,1].

    Args:
        sample_cov: [k, k] symmetric PSD sample covariance.
        intensity: None → analytic OAS intensity; float → fixed δ ∈ [0,1].

    Returns:
        [k, k] shrunk covariance (SPD when sample_cov is PSD and δ > 0).
    """
    k = sample_cov.shape[0]
    target = np.diag(np.diag(sample_cov))  # diagonal target

    if intensity is not None:
        delta = float(np.clip(intensity, 0.0, 1.0))
    else:
        # Analytic Oracle Approximating Shrinkage (OAS) estimator
        # Chen, Wiesel, Eldar, Hero (2010): δ* = ((1-2/k) tr(S²) + tr(S)²) / ((n+1-2/k)(tr(S²) - tr(S)²/k))
        # We use a simplified form safe for small k and small n.
        tr_s = float(np.trace(sample_cov))
        tr_s2 = float(np.trace(sample_cov @ sample_cov))
        tr_s_sq = tr_s * tr_s
        numerator = (1.0 - 2.0 / k) * tr_s2 + tr_s_sq
        denominator = (tr_s2 - tr_s_sq / k) if k > 1 else 1.0
        if abs(denominator) < 1e-15 or not np.isfinite(numerator / denominator):
            delta = 0.2  # safe fallback
        else:
            delta = float(np.clip(numerator / denominator, 0.0, 1.0))

    return delta * target + (1.0 - delta) * sample_cov


def active_covariance(
    logret_2d: NDArray[np.float64],
    t: int,
    active_idx: NDArray[np.int64],
    window: int,
    shrinkage: float | None,
    min_obs: int,
) -> NDArray[np.float64] | None:
    """Compute trailing-window shrunk covariance over active sub-universe.

    Uses [t-window, t) — strictly causal, bar t excluded.

    Args:
        logret_2d: [T, N] log-returns.
        t: current bar index (excluded from window).
        active_idx: [k] column indices of active symbols.
        window: trailing window length in bars.
        shrinkage: None → analytic OAS; float → fixed δ.
        min_obs: minimum required observations; returns None if fewer available.

    Returns:
        [k, k] SPD shrunk covariance, or None if insufficient observations.
    """
    t_start = max(0, t - window)
    n_obs = t - t_start  # excludes bar t itself
    if n_obs < min_obs:
        return None

    sub = logret_2d[t_start:t, :][:, active_idx]  # [n_obs, k]
    # sample covariance with ddof=1; minimum variance floor for stability
    sample_cov = np.cov(sub, rowvar=False, ddof=1) if sub.shape[0] > 1 else np.eye(len(active_idx)) * 1e-6
    if sample_cov.ndim == 0:
        # scalar case (k=1)
        sample_cov = np.array([[float(sample_cov)]])

    return ledoit_wolf_shrink(sample_cov, shrinkage)


def solve_portfolio_kelly(
    mu_s: NDArray[np.float64],
    cov_s: NDArray[np.float64],
    kelly_fraction: float,
    ridge_eps: float,
    per_symbol_cap: float,
) -> NDArray[np.float64]:
    """Solve w = f_k · (Σ̂ + ε·mean(diag)·I)⁻¹ μ_s.

    Uses np.linalg.solve (no explicit matrix inverse). Falls back to diagonal
    Kelly on LinAlgError.

    Args:
        mu_s: [k] signed expected return in risk-unit space.
        cov_s: [k, k] SPD covariance.
        kelly_fraction: fractional Kelly multiplier f_k ∈ (0, 0.25].
        ridge_eps: ridge coefficient; regularisation = ridge_eps · mean(diag(Σ̂)).
        per_symbol_cap: per-symbol weight cap; result clipped to ±per_symbol_cap.

    Returns:
        [k] signed weights clipped to ±per_symbol_cap.
    """
    k = len(mu_s)
    mean_var = float(np.mean(np.diag(cov_s)))
    eps = ridge_eps * max(mean_var, 1e-12)
    reg_cov = cov_s + eps * np.eye(k)

    try:
        w_raw = kelly_fraction * np.linalg.solve(reg_cov, mu_s)
    except np.linalg.LinAlgError:
        _logger.warning("portfolio Kelly solve failed (LinAlgError); using diagonal fallback")
        diag_var = np.maximum(np.diag(cov_s), 1e-12)
        w_raw = kelly_fraction * mu_s / diag_var

    # Sign guard: prevent covariance structure from flipping signal direction
    sign_mu = np.sign(mu_s)
    sign_mu[sign_mu == 0] = 1.0
    w_raw = np.where(np.sign(w_raw) == sign_mu, w_raw, 0.0)

    return np.clip(w_raw, -per_symbol_cap, per_symbol_cap)
