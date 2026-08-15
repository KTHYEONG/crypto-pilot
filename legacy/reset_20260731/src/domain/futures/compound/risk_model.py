from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import RiskModelConfig
from src.domain.futures.compound.contracts import CovariancePath

_logger = logging.getLogger(__name__)


def _ewm_cov_4h(
    returns: NDArray[np.float64],
    half_life: int,
    variance_floor: float,
) -> NDArray[np.float64]:
    n_bars, n_syms = returns.shape
    if n_bars < 2:
        return np.eye(n_syms, dtype=np.float64) * variance_floor
    lam = np.exp(-np.log(2) / half_life)
    w = lam ** np.arange(n_bars - 1, -1, -1, dtype=np.float64)
    w /= float(w.sum())
    mean = np.average(returns, axis=0, weights=w)
    centered = returns - mean
    w_2d = w[:, None]
    cov: NDArray[np.float64] = np.dot((centered * w_2d).T, centered) / float(1 - np.sum(w * w))
    return cov


def _shrink_covariance(
    cov_ewma: NDArray[np.float64],
    delta: float,
    variance_floor: float,
) -> NDArray[np.float64]:
    n_syms = cov_ewma.shape[0]
    variances = np.diag(cov_ewma)
    stds = np.sqrt(np.maximum(variances, variance_floor))

    corr = cov_ewma / np.outer(stds, stds)
    np.fill_diagonal(corr, 1.0)
    mean_corr = float((np.sum(corr) - n_syms) / max(n_syms * (n_syms - 1), 1))

    diag_target = np.diag(variances)
    const_corr_target = np.outer(stds, stds) * mean_corr
    np.fill_diagonal(const_corr_target, variances)

    target = 0.5 * diag_target + 0.5 * const_corr_target

    sigma = (1 - delta) * cov_ewma + delta * target
    sigma = (sigma + sigma.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    eigenvalues = np.maximum(eigenvalues, variance_floor)
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    sigma = (reconstructed + reconstructed.T) / 2.0

    return sigma  # type: ignore[no-any-return]


def estimate_covariance_path(
    returns_2d: NDArray[np.float32],
    valid_2d: NDArray[np.bool_],
    config: RiskModelConfig,
) -> CovariancePath:
    if returns_2d.ndim != 2 or valid_2d.ndim != 2:
        raise ValueError("inputs must be 2D")
    if returns_2d.shape != valid_2d.shape:
        raise ValueError("returns_2d and valid_2d must have the same shape")
    if not np.all(np.isfinite(returns_2d[valid_2d])):
        raise ValueError("non-finite values in returns_2d where valid")

    n_bars, n_syms = returns_2d.shape
    timestamps_ns = np.arange(n_bars, dtype=np.int64)

    cov_3d = np.zeros((n_bars, n_syms, n_syms), dtype=np.float64)
    investable_2d = np.zeros((n_bars, n_syms), dtype=np.bool_)

    for t in range(n_bars):
        cum_valid = np.cumsum(valid_2d[:t + 1].astype(np.int32), axis=0)
        history_count = cum_valid[-1]
        investable_2d[t] = history_count >= config.min_history_bars

        if t < config.min_history_bars:
            cov_3d[t] = np.eye(n_syms, dtype=np.float64) * config.variance_floor
            continue

        window_start = max(0, t - config.min_history_bars * 2)
        window_ret = returns_2d[window_start:t + 1].astype(np.float64)
        window_valid = valid_2d[window_start:t + 1]

        filled_ret = np.where(window_valid, window_ret, 0.0)

        cov_ewma = _ewm_cov_4h(filled_ret, config.ewm_half_life_bars, config.variance_floor)
        sigma = _shrink_covariance(cov_ewma, config.shrink_delta, config.variance_floor)
        cov_3d[t] = sigma

    return CovariancePath(
        decision_timestamps_ns=timestamps_ns,
        covariance_3d=cov_3d,
        investable_2d=investable_2d,
    )
