"""Causal scaling utilities for regime features.

These helpers avoid full-sample fit/transform leakage by using
rolling (past-only) robust statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_robust_zscore(
    df: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    clip: float | None = 5.0,
) -> pd.DataFrame:
    """Return rolling median/IQR z-score per column.

    Uses only current/past rows within the rolling window.
    """
    if df.empty:
        return df.copy()
    if window <= 1:
        out = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    clean = df.replace([np.inf, -np.inf], np.nan).astype(np.float64)
    mp = min_periods if min_periods is not None else max(8, window // 8)

    med = clean.rolling(window=window, min_periods=min(window, mp)).median()
    q75 = clean.rolling(window=window, min_periods=min(window, mp)).quantile(0.75)
    q25 = clean.rolling(window=window, min_periods=min(window, mp)).quantile(0.25)
    iqr = (q75 - q25).replace(0.0, np.nan)

    z = (clean - med) / (iqr + 1e-12)

    # Stabilize warmup periods with expanding fallback.
    exp_med = clean.expanding(min_periods=max(2, min(16, mp))).median()
    exp_q75 = clean.expanding(min_periods=max(2, min(16, mp))).quantile(0.75)
    exp_q25 = clean.expanding(min_periods=max(2, min(16, mp))).quantile(0.25)
    exp_iqr = (exp_q75 - exp_q25).replace(0.0, np.nan)
    z_exp = (clean - exp_med) / (exp_iqr + 1e-12)
    z = z.where(np.isfinite(z), z_exp)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if clip is not None and clip > 0:
        z = z.clip(-clip, clip)
    return z.astype(np.float64)


def causal_log_robust_zscore(
    df: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    clip: float | None = 5.0,
) -> pd.DataFrame:
    """Apply log1p(non-negative) then causal robust z-score."""
    if df.empty:
        return df.copy()
    logged = np.log1p(np.maximum(df.astype(np.float64), 0.0))
    return causal_robust_zscore(
        logged,
        window=window,
        min_periods=min_periods,
        clip=clip,
    )
