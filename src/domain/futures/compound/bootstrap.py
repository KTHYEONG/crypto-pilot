from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


def politis_white_block_length(returns: NDArray[np.float64]) -> float:
    n = len(returns)
    if n < 30:
        raise ValueError(f"politis_white_block_length requires n>=30, got n={n}")

    max_block = float(np.ceil(3.0 * float(n) ** (1.0 / 3.0)))
    r = returns - np.mean(returns)
    var = np.mean(r ** 2)
    if var < 1e-15:
        _logger.info("[EVAL] pw_block=fallback")
        return 5.0

    m = min(int(np.floor(1.2 * n ** 0.5)), n - 2)
    if m < 2:
        _logger.info("[EVAL] pw_block=fallback")
        return 5.0

    acf = np.zeros(m)
    for k in range(1, m + 1):
        acf[k - 1] = float(np.mean(r[:-k] * r[k:])) / var

    threshold = np.sqrt(2.0 * np.log(m)) / np.sqrt(n)
    max_abs_acf = float(np.max(np.abs(acf)))
    if max_abs_acf < threshold:
        _logger.info("[EVAL] pw_block=fallback")
        return 5.0

    rho1_sq = acf[0] ** 2
    if rho1_sq > 1e-6:
        block = (4.0 * float(n) * rho1_sq / max((1.0 - acf[0] ** 2) ** 2, 1e-15)) ** (1.0 / 3.0)
    else:
        block = (2.0 * float(n) / float(m)) ** (1.0 / 3.0)
    block = max(1.0, min(block, max_block))
    _logger.info("[EVAL] pw_block=%.2f", block)
    return float(block)


def _circular_block_sample(
    r: NDArray[np.float64], block_size: float, n: int, rng: np.random.Generator,
) -> NDArray[np.float64]:
    boot = np.empty(n, dtype=np.float64)
    idx = 0
    while idx < n:
        start = int(rng.integers(0, n))
        block_len = int(min(rng.geometric(1.0 / block_size), n - start))
        block_len = int(min(block_len, n - idx))
        end = start + block_len
        if end <= n:
            boot[idx: idx + block_len] = r[start:end]
        else:
            wrap = end - n
            boot[idx: idx + (block_len - wrap)] = r[start:]
            boot[idx + (block_len - wrap): idx + block_len] = r[:wrap]
        idx += block_len
    return boot


def circular_stationary_bootstrap_growth(
    returns: NDArray[np.float64], periods_per_year: float, *,
    n_bootstrap: int = 1000, block_size: float | None = None, seed: int = 42,
) -> tuple[float, float, float]:
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 10:
        return (0.0, 0.0, 0.5)

    bs = block_size if block_size is not None else politis_white_block_length(r)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        boot = _circular_block_sample(r, bs, n, rng)
        log_boot = np.log1p(np.where(np.isfinite(boot), boot, 0.0))
        samples[i] = float(periods_per_year * np.mean(log_boot))

    lcb = float(np.percentile(samples, 10))
    ucb = float(np.percentile(samples, 90))
    prob_positive = float(np.mean(samples > 0.0))
    return (lcb, ucb, prob_positive)


def circular_stationary_bootstrap_sharpe(
    returns: NDArray[np.float64], periods_per_year: float, *,
    n_bootstrap: int = 2000, block_size: float | None = None, seed: int = 42,
) -> tuple[float, float]:
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 10:
        return (0.0, 0.5)

    std_obs = float(np.std(r, ddof=1))
    if std_obs < 1e-12:
        return (0.0, 0.5)

    bs = block_size if block_size is not None else politis_white_block_length(r)
    rng = np.random.default_rng(seed)
    sharpe_samples = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        boot = _circular_block_sample(r, bs, n, rng)
        mean_r = float(np.mean(boot))
        std_r = float(np.std(boot, ddof=1))
        if std_r < 1e-12:
            if abs(mean_r) < 1e-12:
                sharpe_samples[i] = 0.0
            elif mean_r > 0:
                sharpe_samples[i] = 10.0 * math.sqrt(periods_per_year)
            else:
                sharpe_samples[i] = -10.0 * math.sqrt(periods_per_year)
        else:
            sharpe_samples[i] = float(mean_r / std_r * math.sqrt(periods_per_year))

    obs_sharpe = float(np.mean(r) / std_obs * math.sqrt(periods_per_year))
    prob_positive = float(np.mean(sharpe_samples > 0.0))
    return (obs_sharpe, prob_positive)


def stepwise_spa_pvalue(
    strategy_daily: NDArray[np.float64], controls_2d: NDArray[np.float64], *,
    block_size: float | None = None, n_bootstrap: int = 1000, seed: int = 42,
) -> float:
    finite_mask = np.isfinite(strategy_daily)
    s = strategy_daily[finite_mask].copy()
    n_strat = len(s)
    if controls_2d.ndim != 2:
        raise ValueError(f"controls_2d must be 2-D, got shape {controls_2d.shape}")
    if controls_2d.shape[1] != finite_mask.sum():
        raise ValueError(
            f"controls_2d length ({controls_2d.shape[1]}) must match finite strategy_daily ({n_strat})"
        )
    n_controls = controls_2d.shape[0]
    if n_controls == 0 or n_strat < 10:
        return 1.0

    n = n_strat
    d = np.zeros((n_controls, n), dtype=np.float64)
    for k in range(n_controls):
        ck = controls_2d[k, finite_mask].copy()
        d[k] = s - ck

    d_mean = np.mean(d, axis=1)
    d_var = np.var(d, axis=1, ddof=1)
    d_var = np.maximum(d_var, 1e-15)
    t_stat = d_mean / np.sqrt(d_var / n)
    t_max = float(np.max(t_stat))

    rng = np.random.default_rng(seed)
    t_boot = np.empty((n_bootstrap, n_controls), dtype=np.float64)

    c_centered = np.zeros((n_controls, n), dtype=np.float64)
    for k in range(n_controls):
        ck = controls_2d[k, finite_mask].copy()
        c_centered[k] = ck - np.mean(ck)

    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, n, size=n)
        d_boot = np.zeros((n_controls, n), dtype=np.float64)
        for k in range(n_controls):
            d_boot[k] = (s[boot_idx] - np.mean(s)) - c_centered[k, boot_idx]
        t_boot[b] = np.mean(d_boot, axis=1) / (np.std(d_boot, axis=1, ddof=1) / np.sqrt(n) + 1e-15)

    t_boot_max = np.max(t_boot, axis=1)
    p_value = float(np.mean(t_boot_max >= t_max))

    if not np.isfinite(p_value):
        _logger.warning("[EVAL] spa_pvalue non-finite, returning 1.0 (fail-closed)")
        return 1.0

    return float(min(max(p_value, 0.0), 1.0))
