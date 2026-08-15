from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import FactorRiskConfig

_logger = logging.getLogger(__name__)


def _ewm_cov(x: NDArray[np.float64], half_life: float) -> NDArray[np.float64]:
    n = x.shape[0]
    lam = np.exp(-np.log(2) / half_life)
    w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
    w /= float(w.sum())
    if x.ndim == 1:
        mean = np.average(x, weights=w)
        centered = x - mean
        cov_val = np.average(centered * centered, weights=w)
        return np.array([[cov_val]], dtype=np.float64)
    w_2d = w[:, None]
    mean_2d = np.average(x, axis=0, weights=w)
    centered = x - mean_2d
    cov_mat: NDArray[np.float64] = np.dot((centered * w_2d).T, centered) / float(1 - np.sum(w * w))
    return cov_mat


def estimate_causal_factor_covariance(
    *,
    daily_returns_2d: NDArray[np.float64],
    end_exclusive: int,
    cluster_ids_1d: NDArray[np.int16],
    config: FactorRiskConfig,
) -> NDArray[np.float64]:
    n_syms = daily_returns_2d.shape[1]
    n = daily_returns_2d.shape[0] if end_exclusive > daily_returns_2d.shape[0] else end_exclusive

    if n < 10:
        return np.eye(n_syms, dtype=np.float64) * config.variance_floor

    returns = daily_returns_2d[:n]
    btc_idx = 0
    btc_ret = returns[:, btc_idx] if n_syms > 0 else np.zeros(n)

    market_loadings = np.full(n_syms, np.nan, dtype=np.float64)
    for i in range(n_syms):
        mask = np.isfinite(returns[:, i]) & np.isfinite(btc_ret)
        if np.sum(mask) < 10:
            market_loadings[i] = 0.0
            continue
        rx = returns[mask, i]
        rm = btc_ret[mask]
        cov = np.cov(rm, rx)[0, 1]
        var_m = np.var(rm)
        market_loadings[i] = cov / var_m if var_m > 1e-15 else 0.0

    market_loadings = np.where(np.isfinite(market_loadings), market_loadings, 0.0)
    f_market = _ewm_cov(btc_ret.reshape(-1, 1), float(config.ewm_half_life_days))
    f_market_val = float(f_market[0, 0])

    unique_clusters = np.unique(cluster_ids_1d[cluster_ids_1d >= 0])
    n_clusters = min(len(unique_clusters), config.max_cluster_factors)
    cluster_matrix = np.zeros((n_syms, n_clusters), dtype=np.float64)
    for j, cid in enumerate(unique_clusters[:n_clusters]):
        cluster_matrix[:, j] = (cluster_ids_1d == cid).astype(np.float64)

    f_cluster = np.eye(n_clusters, dtype=np.float64) * f_market_val * 0.5

    try:
        market_cov = np.outer(market_loadings, market_loadings) * f_market_val
    except Exception:
        market_cov = np.zeros((n_syms, n_syms), dtype=np.float64)

    if n_clusters > 0:
        cluster_cov = cluster_matrix @ f_cluster @ cluster_matrix.T
    else:
        cluster_cov = np.zeros((n_syms, n_syms), dtype=np.float64)

    residual_var = np.full(n_syms, config.variance_floor, dtype=np.float64)
    for i in range(n_syms):
        resid = returns[:, i] - market_loadings[i] * btc_ret
        mask = np.isfinite(resid)
        if np.sum(mask) > 5:
            rv = np.var(resid[mask])
            residual_var[i] = max(rv, config.variance_floor)

    sigma = market_cov + cluster_cov + np.diag(residual_var)
    sigma = (sigma + sigma.T) / 2.0

    try:
        np.linalg.cholesky(sigma)
        return sigma
    except np.linalg.LinAlgError:
        _logger.warning("Covariance non-PSD, falling back to diagonal")
        return np.diag(residual_var)
