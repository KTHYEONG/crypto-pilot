from __future__ import annotations

import numpy as np


def winsorized_cs_zscore(
    sig_2d: np.ndarray,
    *,
    clip_z: float = 3.0,
    min_symbols: int = 5,
) -> np.ndarray:
    """Computes cross-sectional robust z-score using Median Absolute Deviation (MAD).

    Args:
        sig_2d: Raw signals array of shape [T, N].
        clip_z: Z-score threshold for clipping.
        min_symbols: Minimum required non-NaN symbols in a row.

    Returns:
        Standardized signals of shape [T, N] with values clipped to [-clip_z, clip_z].

    """
    if sig_2d.ndim != 2:
        raise ValueError("sig_2d must be 2D")

    t_len, n_syms = sig_2d.shape
    out = np.zeros((t_len, n_syms), dtype=np.float64)

    for t in range(t_len):
        row = sig_2d[t]
        m = np.isfinite(row)
        n_valid = int(m.sum())

        if n_valid < min_symbols:
            continue

        valid_vals = row[m]
        med = np.median(valid_vals)
        mad = np.median(np.abs(valid_vals - med))
        scale = 1.4826 * mad

        if scale < 1e-12:
            # If scale is zero (e.g. all values are identical), keep them 0 to avoid /0
            continue

        z = np.clip((row[m] - med) / scale, -clip_z, clip_z)
        out[t, m] = z

    return out


def to_return_units(
    z_2d: np.ndarray,
    sigma_fwd_2d: np.ndarray,
    ic_lagged: np.ndarray | float,
) -> np.ndarray:
    """Calibrates standardized z-score to expected per-bar return units using Grinold forecast.

    Forecast: alpha_hat[t, i] = ic_lagged[t] * sigma_fwd[t, i] * z[t, i]

    Args:
        z_2d: Standardized score of shape [T, N].
        sigma_fwd_2d: Volatility prediction of shape [T, N].
        ic_lagged: Chronologically lagged sleeve IC (shape [T] or scalar).

    Returns:
        Calibrated alpha forecast in simple return units of shape [T, N].

    """
    if z_2d.shape != sigma_fwd_2d.shape:
        raise ValueError("z_2d and sigma_fwd_2d must have the same shape")

    t_len, n_syms = z_2d.shape

    # Align ic_lagged to [T, 1] for broadcasting
    if isinstance(ic_lagged, np.ndarray):
        if ic_lagged.ndim == 1:
            if len(ic_lagged) != t_len:
                raise ValueError("Length of ic_lagged array must match the time dimension T")
            ic_expanded = ic_lagged[:, np.newaxis]
        elif ic_lagged.ndim == 2:
            if ic_lagged.shape != (t_len, 1):
                raise ValueError("2D ic_lagged must have shape [T, 1]")
            ic_expanded = ic_lagged
        else:
            raise ValueError("ic_lagged must be a scalar, 1D array, or 2D [T, 1] array")
    else:
        ic_expanded = np.full((t_len, 1), ic_lagged, dtype=np.float64)

    # Grinold forecast
    alpha_hat = ic_expanded * sigma_fwd_2d * z_2d

    # Replace any potential NaNs or Infs with zero
    alpha_hat = np.where(np.isfinite(alpha_hat), alpha_hat, 0.0)
    return alpha_hat
