from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import CalibrationConfig
from src.domain.futures.compound.contracts import (
    CalibrationTarget,
    CausalFold,
    CausalityError,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalCalibration,
)

_logger = logging.getLogger(__name__)


def build_folds_4h(
    n_bars: int, config: CalibrationConfig, max_target_horizon_bars: int = 0,
) -> tuple[CausalFold, ...]:
    n_folds = config.n_folds
    purge = max(config.purge_bars, max_target_horizon_bars)
    embargo = config.embargo_bars
    if n_bars < n_folds * (purge + embargo + 2):
        raise CausalityError(
            f"n_bars={n_bars} insufficient for {n_folds} folds with purge={purge}, embargo={embargo}",
        )
    fold_size = (n_bars - purge - embargo) // n_folds
    folds: list[CausalFold] = []
    for i in range(n_folds):
        fit_start = 0
        fit_end = (i + 1) * fold_size
        cal_start = max(0, fit_end - purge)
        oos_start = fit_end + purge
        oos_end = min(oos_start + fold_size, n_bars)
        if i == n_folds - 1:
            oos_end = n_bars - embargo
        folds.append(CausalFold(
            fold_id=i,
            fit_start=fit_start,
            fit_end_exclusive=fit_end,
            calibration_start=cal_start,
            calibration_end_exclusive=fit_end,
            oos_start=oos_start,
            oos_end_exclusive=oos_end,
            purge_bars=purge,
            embargo_bars=embargo,
        ))
    return tuple(folds)


def build_calibration_target(
    bars: MultiTimeframeBars, sigma_2d: NDArray[np.float32],
    horizon_bars: int = 1,
) -> CalibrationTarget:
    cube_4h = bars.cubes["4h"]
    close_4h = cube_4h.close_2d.astype(np.float64)
    ts = bars.decision_timestamps_ns
    n_t = ts.size
    n_syms = sigma_2d.shape[1]

    if sigma_2d.shape != (n_t, n_syms):
        raise ValueError(
            f"sigma_2d shape {sigma_2d.shape} != 4h grid ({n_t}, {n_syms})",
        )

    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")

    # y[t] must be the FORWARD horizon_bars return realized after decision time t
    # (log_ret[t] = log(close[t+horizon_bars]/close[t])); the last horizon_bars bars
    # have no known future close yet, so they are masked invalid.
    log_ret = np.full((n_t, n_syms), np.nan, dtype=np.float64)
    if n_t > horizon_bars:
        log_ret[:-horizon_bars] = np.log(close_4h[horizon_bars:] / close_4h[:-horizon_bars])

    y = np.where(
        np.isfinite(log_ret) & (sigma_2d > 0),
        log_ret / sigma_2d.astype(np.float64),
        np.nan,
    ).astype(np.float32)

    valid = np.isfinite(y)
    valid[-horizon_bars:, :] = False

    return CalibrationTarget(
        decision_timestamps_ns=ts,
        y_2d=y,
        valid_2d=valid,
    )


def build_multi_horizon_targets(
    bars: MultiTimeframeBars, sigma_2d: NDArray[np.float32],
    horizons_hours: tuple[int, ...],
) -> dict[int, CalibrationTarget]:
    for h in horizons_hours:
        if h % 4 != 0:
            raise ValueError(f"each horizon_hours must be a multiple of 4, got {h}")
    return {h: build_calibration_target(bars, sigma_2d, horizon_bars=h // 4) for h in horizons_hours}


def _pooled_ridge_beta(
    z: NDArray[np.float32], y: NDArray[np.float32],
    valid: NDArray[np.bool_], ridge_lambda: float,
) -> tuple[float, float, int]:
    mask = valid
    z_vals = z[mask].astype(np.float64)
    y_vals = y[mask].astype(np.float64)
    n = mask.sum()
    if n < 1:
        return 0.0, 0.0, 0
    z2 = np.sum(z_vals ** 2)
    zy = np.sum(z_vals * y_vals)
    denom = z2 + ridge_lambda * n
    if denom <= 0:
        return 0.0, 0.0, n
    beta = zy / denom
    residual = y_vals - beta * z_vals
    sigma2 = np.mean(residual ** 2) if n > 1 else 1.0
    se = math.sqrt(sigma2 / denom)
    return beta, se, n


def calibrate_signals(
    panel: RawSignalPanel, targets: dict[int, CalibrationTarget],
    folds: tuple[CausalFold, ...], config: CalibrationConfig,
) -> tuple[SignalCalibration, ...]:
    _ = panel.z_3d.shape[0]
    _ = panel.z_3d.shape[1]
    n_cat = panel.z_3d.shape[2]
    n_folds = len(folds)

    raw_beta = np.zeros((n_cat, n_folds), dtype=np.float64)
    raw_se = np.zeros((n_cat, n_folds), dtype=np.float64)
    raw_n = np.zeros((n_cat, n_folds), dtype=np.int64)

    for k in range(n_cat):
        signal_id = panel.descriptors[k].signal_id
        tgt_horizon = panel.descriptors[k].target_horizon_hours
        if tgt_horizon not in targets:
            raise ValueError(f"missing target for horizon={tgt_horizon}h")
        target = targets[tgt_horizon]
        y = target.y_2d
        target_valid = target.valid_2d

        for fi, fold in enumerate(folds):
            fit_slice = slice(fold.fit_start, fold.fit_end_exclusive)
            z_k = panel.z_3d[fit_slice, :, k]
            y_fit = y[fit_slice, :]
            v = panel.valid_3d[fit_slice, :, k] & target_valid[fit_slice, :]

            ridge_lambda = config.ridge_lambda_scale * 1.0
            b_k, se_k, n_k = _pooled_ridge_beta(z_k, y_fit, v, ridge_lambda)

            if n_k < config.min_fold_obs:
                _logger.debug(
                    "[ALGO] signal=%s fold=%d: n_obs=%d < min=%d, beta=0",
                    signal_id, fold.fold_id, n_k, config.min_fold_obs,
                )
                b_k = 0.0
                se_k = 0.0

            raw_beta[k, fi] = b_k
            raw_se[k, fi] = se_k
            raw_n[k, fi] = n_k

    calibrations: list[SignalCalibration] = []
    for k in range(n_cat):
        family = panel.descriptors[k].family
        family_indices = [j for j in range(n_cat) if panel.descriptors[j].family == family]
        shrunk_beta: list[float] = []
        for fi in range(n_folds):
            fmean = float(np.mean(raw_beta[family_indices, fi]))
            b_raw = float(raw_beta[k, fi])
            b_shrunk = (1.0 - config.family_shrink) * b_raw + config.family_shrink * fmean
            shrunk_beta.append(b_shrunk)

        calibrations.append(SignalCalibration(
            signal_id=panel.descriptors[k].signal_id,
            beta_by_fold=tuple(shrunk_beta),
            beta_se_by_fold=tuple(float(raw_se[k, fi]) for fi in range(n_folds)),
            n_obs_by_fold=tuple(int(raw_n[k, fi]) for fi in range(n_folds)),
        ))

    return tuple(calibrations)
