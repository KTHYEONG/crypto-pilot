"""Deflated Sharpe primitives and growth sampling statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common.errors import DataIntegrityError
from src.core.bootstrap import (
    iter_stationary_bootstrap_index_chunks,
    stationary_bootstrap_max_blocks,
)
from src.quant.evaluation.reliability import derive_block_size

_EULER_GAMMA = 0.577215664901532860606512090082402431


def psr_radicand(observed_sr: float, skew: float, kurtosis: float) -> float:
    """Variance term of the PSR denominator."""
    return 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado): normal CDF of the
    probability that the true (non-annualized) Sharpe exceeds ``benchmark_sr``.

    All Sharpe inputs are per-observation (non-annualized): the raw ``mean/std``
    of the return series, never scaled by ``sqrt(periods_per_year)``.
    ``kurtosis`` is the full (non-excess) fourth standardized moment, so pass
    ``excess_kurtosis + 3.0``. Returns NaN (never ``inf`` or a complex value)
    when the denominator radicand is not strictly positive.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    radicand = psr_radicand(observed_sr, skew, kurtosis)
    if radicand <= 0.0:
        return float("nan")
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1.0) / math.sqrt(radicand)
    return float(norm.cdf(z))


def expected_max_trial_sr(trial_sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe over ``n_trials`` zero-edge trials."""
    sd = math.sqrt(trial_sr_variance)
    return float(
        sd
        * (
            (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
            + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    )


def deflated_sharpe_ratio(
    observed_sr: float,
    trial_sr_variance: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado): PSR against the expected
    maximum Sharpe over ``n_trials`` independent trials under the null.

    The benchmark is ``sqrt(trial_sr_variance) * ((1 - gamma) * Phi_inv(1 - 1/N)
    + gamma * Phi_inv(1 - 1/(N*e)))`` with ``gamma`` the Euler-Mascheroni
    constant. All Sharpe inputs are per-observation (non-annualized);
    ``trial_sr_variance`` is the variance of the per-observation Sharpe across
    trials. With zero trial dispersion the benchmark collapses to zero and the
    result equals the plain PSR against a zero benchmark.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    if trial_sr_variance < 0.0:
        raise ValueError(f"trial_sr_variance must be >= 0, got {trial_sr_variance}")
    if trial_sr_variance == 0.0 or n_trials == 1:
        return probabilistic_sharpe_ratio(observed_sr, 0.0, n_obs, skew, kurtosis)
    benchmark_sr = expected_max_trial_sr(trial_sr_variance, n_trials)
    return probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs, skew, kurtosis)


def sharpe_sampling_variance(sharpe: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """Asymptotic variance of the per-observation Sharpe estimate (Mertens/Lo with
    skew and full kurtosis). Used as the floor of the cross-trial Sharpe variance so DSR is always computable and never
    more lenient than sampling noise alone."""
    for name, value in (("sharpe", sharpe), ("skew", skew), ("kurtosis", kurtosis)):
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise DataIntegrityError(f"{name} must be finite, got {value!r}")
    if isinstance(n_obs, bool) or not isinstance(n_obs, (int, np.integer)) or int(n_obs) < 2:
        raise DataIntegrityError(f"n_obs must be an integer >= 2, got {n_obs!r}")
    radicand = psr_radicand(float(sharpe), float(skew), float(kurtosis))
    if radicand <= 0.0:
        raise DataIntegrityError(f"PSR radicand must be positive, got {radicand!r}")
    return float(radicand / (int(n_obs) - 1))


def _require_finite_daily(daily_returns: pd.Series) -> np.ndarray:
    if not isinstance(daily_returns, pd.Series) or daily_returns.empty:
        raise DataIntegrityError("daily_returns must be a non-empty Series")
    values = np.asarray(daily_returns.to_numpy(dtype="float64"), dtype="float64")
    if not bool(np.isfinite(values).all()):
        raise DataIntegrityError("daily_returns must be finite (NaN/inf fail closed)")
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        raise DataIntegrityError("daily_returns must carry a DatetimeIndex")
    index = daily_returns.index
    if index.tz is None or str(index.tz) != "UTC" or index.hasnans or not index.is_unique:
        raise DataIntegrityError("daily_returns must have a unique UTC index")
    if not index.is_monotonic_increasing:
        raise DataIntegrityError("daily_returns must be monotonic in time")
    if bool((values < -1.0).any()):
        raise DataIntegrityError("daily_returns cannot fall below -100%")
    return values


def log_growth_lcb(daily_returns: pd.Series, *, alpha: float, n_paths: int, seed: int) -> float:
    """Lower `alpha` quantile of annualized mean log growth (365 days) under a stationary block
    bootstrap of daily log returns (block length from `derive_block_size`). Fat tails and serial dependence are kept by
    resampling blocks of observed days rather than assuming normality."""
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise DataIntegrityError(f"alpha must be in (0, 1), got {alpha!r}")
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or int(n_paths) < 1:
        raise DataIntegrityError(f"n_paths must be a positive integer, got {n_paths!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise DataIntegrityError(f"seed must be a non-negative integer, got {seed!r}")
    values = _require_finite_daily(daily_returns)
    log_returns = np.log1p(values)
    if not bool(np.isfinite(log_returns).all()):
        raise DataIntegrityError("log returns must be finite")
    n = int(values.size)
    mean_block = max(1, int(derive_block_size(values)))
    max_blocks = stationary_bootstrap_max_blocks(n, mean_block)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_paths), dtype="float64")
    filled = 0
    chunk_size = min(int(n_paths), 500)
    for chunk in iter_stationary_bootstrap_index_chunks(
        rng,
        source_len=n,
        path_len=n,
        n_replicates=int(n_paths),
        mean_block=mean_block,
        chunk_size=chunk_size,
        max_blocks=max_blocks,
    ):
        paths = log_returns[chunk.indices]
        for row in range(paths.shape[0]):
            means[filled] = float(paths[row].mean() * 365.0)
            filled += 1
    return float(np.quantile(means, float(alpha)))
