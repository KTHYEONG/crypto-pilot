from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import BaselineAllocConfig
from src.domain.futures.compound.contracts import CovariancePath

_logger = logging.getLogger(__name__)


def solve_baseline_weights(
    mu_2d: NDArray[np.float32],
    cov_path: CovariancePath,
    config: BaselineAllocConfig,
    mode: Literal["inverse_vol", "risk_scaled"],
) -> NDArray[np.float64]:
    if mode not in ("inverse_vol", "risk_scaled"):
        raise ValueError(f"unknown mode: {mode}")
    if mu_2d.ndim != 2:
        raise ValueError("mu_2d must be 2D")
    if mu_2d.shape[0] != cov_path.covariance_3d.shape[0] or mu_2d.shape[1] != cov_path.covariance_3d.shape[1]:
        raise ValueError("mu_2d shape does not match cov_path")

    n_bars, n_syms = mu_2d.shape
    bars_per_year = 2190.0
    weights = np.zeros((n_bars, n_syms), dtype=np.float64)

    for t in range(n_bars):
        mu_t = mu_2d[t]
        cov_t = cov_path.covariance_3d[t]
        investable_t = cov_path.investable_2d[t]
        sigmas = np.sqrt(np.maximum(np.diag(cov_t), 1e-30))

        sign = np.where(mu_t > 0, 1.0, np.where(mu_t < 0, -1.0, 0.0))
        raw_w = sign / sigmas
        raw_w[np.abs(mu_t) < 1e-15] = 0.0
        raw_w[~investable_t] = 0.0

        gross = float(np.sum(np.abs(raw_w)))
        if gross < 1e-15:
            weights[t] = 0.0
            continue

        raw_w = raw_w / gross * config.gross_cap
        raw_w = np.clip(raw_w, -config.per_symbol_cap, config.per_symbol_cap)

        if mode == "inverse_vol":
            port_var_4h = float(np.sum(raw_w ** 2 * np.diag(cov_t)))
        else:
            port_var_4h = float(raw_w @ cov_t @ raw_w)

        port_vol_4h = np.sqrt(max(port_var_4h, 1e-30))
        scale = config.target_ann_vol / (port_vol_4h * np.sqrt(bars_per_year))
        scaled_w = raw_w * scale

        # Re-project after vol-target scaling: caps are the final binding constraint,
        # never a side effect of the vol scale (see [LIMIT-05], no double vol adjustment).
        scaled_gross = float(np.sum(np.abs(scaled_w)))
        if scaled_gross > config.gross_cap:
            scaled_w = scaled_w / scaled_gross * config.gross_cap
        scaled_w = np.clip(scaled_w, -config.per_symbol_cap, config.per_symbol_cap)

        weights[t] = scaled_w

    return weights
