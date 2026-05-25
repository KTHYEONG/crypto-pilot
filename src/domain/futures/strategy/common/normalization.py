from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def winsorized_cs_zscore(
    values: NDArray[np.float64], clip_z: float, min_symbols: int
) -> NDArray[np.float64]:
    """Compute robust cross-sectional z-score by timestamp."""
    if values.ndim != 2:
        raise ValueError("values must be 2D")
    t_len, n_len = values.shape
    out = np.zeros((t_len, n_len), dtype=np.float64)
    for t in range(t_len):
        row = values[t]
        m = np.isfinite(row)
        if int(m.sum()) < min_symbols:
            continue
        valid_vals = row[m]
        med = np.median(valid_vals)
        mad = np.median(np.abs(valid_vals - med))
        scale = 1.4826 * mad
        if scale < 1e-12:
            continue
        out[t, m] = np.clip((row[m] - med) / scale, -clip_z, clip_z)
    return out


def cross_sectional_rank(
    values: NDArray[np.float64], mask: NDArray[np.bool_], min_symbols: int
) -> NDArray[np.float64]:
    """Compute [0, 1] percentile rank inside each timestamp."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    t_len, _n_len = values.shape
    for t in range(t_len):
        eligible = mask[t] & np.isfinite(values[t])
        idx = np.flatnonzero(eligible)
        if idx.size < min_symbols:
            continue
        row = values[t, idx]
        if np.allclose(row, row[0], equal_nan=False):
            out[t, idx] = 0.5
            continue
        order = np.argsort(np.argsort(row))
        denom = max(idx.size - 1, 1)
        out[t, idx] = order.astype(np.float64) / float(denom)
    return np.asarray(out, dtype=np.float64)


@dataclass(slots=True, frozen=True)
class RobustBounds:
    """Per-feature clipping bounds fit on train split only."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class MissingValueImputer:
    """Per-feature train-only imputation values."""

    feature_medians: NDArray[np.float64]


def fit_robust_bounds(train_values: NDArray[np.float64], clip_quantile: float) -> RobustBounds:
    """Fit per-feature lower/upper quantiles on train-only tensor [T, N, F]."""
    if train_values.ndim != 3:
        raise ValueError("train_values must be [T, N, F]")
    if not (0.5 < clip_quantile < 1.0):
        raise ValueError("clip_quantile must satisfy 0.5 < q < 1.0")
    flat = train_values.reshape(-1, train_values.shape[2])
    lower_q = 1.0 - clip_quantile
    lower = np.nanquantile(flat, lower_q, axis=0).astype(np.float64)
    upper = np.nanquantile(flat, clip_quantile, axis=0).astype(np.float64)
    return RobustBounds(lower=lower, upper=upper)


def fit_missing_value_imputer(train_values: NDArray[np.float64]) -> MissingValueImputer:
    """Fit per-feature median imputer on train-only tensor [T, N, F]."""
    if train_values.ndim != 3:
        raise ValueError("train_values must be [T, N, F]")
    medians = np.nanmedian(train_values.reshape(-1, train_values.shape[2]), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float64)
    return MissingValueImputer(feature_medians=medians)


def apply_missing_value_imputer(
    values: NDArray[np.float64],
    imputer: MissingValueImputer,
) -> NDArray[np.float64]:
    """Apply train-only missing-value imputation to tensor [T, N, F]."""
    if values.ndim != 3:
        raise ValueError("values must be [T, N, F]")
    if imputer.feature_medians.shape[0] != values.shape[2]:
        raise ValueError("imputer feature dimension mismatch")
    return np.where(
        np.isfinite(values),
        values,
        imputer.feature_medians[np.newaxis, np.newaxis, :],
    ).astype(np.float64, copy=False)


def apply_robust_bounds(values: NDArray[np.float64], bounds: RobustBounds) -> NDArray[np.float64]:
    """Clip feature tensor [T, N, F] using pre-fit bounds."""
    if values.ndim != 3:
        raise ValueError("values must be [T, N, F]")
    if bounds.lower.shape[0] != values.shape[2]:
        raise ValueError("bounds feature dimension mismatch")
    clipped = np.clip(
        values,
        bounds.lower[np.newaxis, np.newaxis, :],
        bounds.upper[np.newaxis, np.newaxis, :],
    )
    return clipped.astype(np.float64, copy=False)
