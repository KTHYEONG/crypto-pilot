from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.stats import norm

_logger = logging.getLogger(__name__)

_EULER_GAMMA: float = 0.5772156649


@dataclass(slots=True, frozen=True)
class TrialMultiplicity:
    n_trials: int
    effective_trials: float
    sigma_sharpe: float

    def __post_init__(self) -> None:
        if self.n_trials < 0:
            raise ValueError("n_trials must be >= 0")
        if self.effective_trials < 1.0:
            raise ValueError("effective_trials must be >= 1.0")
        if self.sigma_sharpe < 0.0 or not np.isfinite(self.sigma_sharpe):
            raise ValueError("sigma_sharpe must be finite and >= 0")


def build_candidate_trial_returns(
    *, z_3d: NDArray[np.float32], valid_3d: NDArray[np.bool_],
    close_2d: NDArray[np.float32], timestamps_ns: NDArray[np.int64],
    start_idx: int, end_idx: int,
) -> NDArray[np.float64]:
    if z_3d.ndim != 3:
        raise ValueError(f"z_3d must be 3-D, got shape {z_3d.shape}")
    if z_3d.shape[:2] != valid_3d.shape[:2]:
        raise ValueError("z_3d and valid_3d shape mismatch")
    t_4h, _n_sym, n_trial = z_3d.shape
    if end_idx > t_4h or start_idx < 0 or start_idx >= end_idx:
        raise ValueError(f"invalid window [{start_idx}, {end_idx}) for t_4h={t_4h}")

    n_step = end_idx - start_idx
    trial_rets = np.zeros((n_trial, n_step), dtype=np.float64)

    for k in range(n_trial):
        port_rets_4h = np.zeros(n_step, dtype=np.float64)
        max_t = min(t_4h, close_2d.shape[0]) - 1
        for t in range(start_idx, min(end_idx, max_t + 1)):
            z_t = z_3d[t, :, k]
            v_t = valid_3d[t, :, k]
            abs_sum = float(np.sum(np.abs(z_t)))
            if abs_sum < 1e-12:
                continue
            w_t = z_t / abs_sum
            prev_close = close_2d[t].astype(np.float64)
            curr_close = close_2d[t + 1].astype(np.float64)
            ret_mask = (prev_close > 0) & np.isfinite(prev_close) & (curr_close > 0) & np.isfinite(curr_close) & v_t
            sym_rets = np.where(ret_mask, curr_close / prev_close - 1.0, 0.0)
            port_rets_4h[t - start_idx] = float(np.dot(w_t, sym_rets))
        trial_rets[k] = port_rets_4h

    return trial_rets


def _aggregate_4h_to_daily_compound(
    returns_4h: NDArray[np.float64],
) -> NDArray[np.float64]:
    n = len(returns_4h)
    n_days = n // 6
    if n_days == 0:
        return np.array([], dtype=np.float64)
    daily = np.empty(n_days, dtype=np.float64)
    for d in range(n_days):
        block = returns_4h[d * 6:(d + 1) * 6]
        daily[d] = float(np.expm1(np.sum(np.log1p(block))))
    return daily


def compute_trial_multiplicity(trial_daily_returns_2d: NDArray[np.float64]) -> TrialMultiplicity:
    n_trial, n_day = trial_daily_returns_2d.shape
    if n_trial == 0 or n_day < 10:
        return TrialMultiplicity(n_trials=n_trial, effective_trials=1.0, sigma_sharpe=0.0)

    corr = np.corrcoef(trial_daily_returns_2d)
    np.nan_to_num(corr, nan=0.0, copy=False)
    eigenvalues = linalg.eigh(corr, eigvals_only=True)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sum_eig = float(np.sum(eigenvalues))
    sum_eig2 = float(np.sum(eigenvalues ** 2))
    k_eff = sum_eig ** 2 / max(sum_eig2, 1e-15)
    k_eff = min(max(k_eff, 1.0), float(n_trial))

    sharpe_per_trial = np.empty(n_trial, dtype=np.float64)
    for k in range(n_trial):
        r = trial_daily_returns_2d[k]
        valid = r[np.isfinite(r)]
        if len(valid) < 5:
            sharpe_per_trial[k] = 0.0
        else:
            mean_r = float(np.mean(valid))
            std_r = float(np.std(valid, ddof=1))
            sharpe_per_trial[k] = mean_r / max(std_r, 1e-12) * math.sqrt(365.25)
    sigma_sharpe = max(float(np.std(sharpe_per_trial, ddof=1)), 0.0)

    return TrialMultiplicity(
        n_trials=n_trial,
        effective_trials=k_eff,
        sigma_sharpe=sigma_sharpe,
    )


def deflated_sharpe_probability(
    *, observed_sharpe: float, multiplicity: TrialMultiplicity,
    excess_returns: NDArray[np.float64], periods_per_year: float = 365.25,
) -> float:
    k_eff = multiplicity.effective_trials
    sigma_sr = multiplicity.sigma_sharpe
    n_obs = excess_returns.shape[0]

    if k_eff <= 1.0 or n_obs < 30:
        return 0.5

    sigma_sr = max(sigma_sr, 1e-12)

    r = excess_returns[np.isfinite(excess_returns)]
    if len(r) < 5:
        return 0.5

    gamma3 = float(np.mean((r - np.mean(r)) ** 3) / max(np.std(r, ddof=1) ** 3, 1e-15))
    gamma4 = float(np.mean((r - np.mean(r)) ** 4) / max(np.std(r, ddof=1) ** 4, 1e-15))

    inv_k = 1.0 / k_eff
    inv_ke = 1.0 / (k_eff * math.e)
    z1 = norm.ppf(1.0 - inv_k)
    z2 = norm.ppf(1.0 - inv_ke)
    sr_0 = sigma_sr * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)

    denominator = math.sqrt(max(1.0 - gamma3 * observed_sharpe + (gamma4 - 1.0) / 4.0 * observed_sharpe ** 2, 0.0))
    if denominator <= 0.0:
        return 0.5

    z_stat = (observed_sharpe - sr_0) * math.sqrt(n_obs - 1) / denominator
    dsr = float(norm.cdf(z_stat))
    return min(max(dsr, 0.0), 1.0)
