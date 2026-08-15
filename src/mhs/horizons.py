"""Horizon-feature builders: log return, realized vol, efficiency ratio.

``efficiency_ratio`` conditions the FAST band's reversal strength and must
never be added into any trend score (spec §2.2): high ER is measured as
stronger reversal, not stronger trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def horizon_log_return(log_price: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Forward-free log return over ``horizon_bars`` (shifted lookback)."""
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    return log_price - log_price.shift(horizon_bars)


def realized_vol(log_price: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Rolling std of one-bar log differences on the horizon's own scale.

    ``min_periods`` equals ``horizon_bars`` so a short window fails closed to
    NaN rather than reporting an unreliable estimate.
    """
    if horizon_bars < 2:
        raise ValueError(f"horizon_bars must be >= 2, got {horizon_bars}")
    return (
        log_price.diff().rolling(horizon_bars, min_periods=horizon_bars).std()
        * np.sqrt(horizon_bars)
    )

def vol_normalized_horizon_signal(log_price: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Horizon log-return scaled by its own realized vol (risk-adjusted momentum).

    Avoids raw-return cross-sectional rank being dominated by high-realized-vol symbols
    whose large moves carry no proportionally larger persistence signal.
    """
    raw = horizon_log_return(log_price, horizon_bars)
    vol = realized_vol(log_price, horizon_bars)
    r = raw.to_numpy(dtype="float64")
    v = vol.to_numpy(dtype="float64")
    out = np.full_like(r, np.nan)
    valid = np.isfinite(v) & (v > 0)
    np.divide(r, v, out=out, where=valid)
    return pd.DataFrame(out, index=raw.index, columns=raw.columns)


def efficiency_ratio(log_price: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Efficiency ratio: |net move| / sum(|one-bar moves|) over the window.

    A monotone path returns 1.0, a closed round trip 0.0, and a flat path NaN
    (never 0/0): ``np.divide`` is used with an explicit ``out`` array and
    ``where=denominator>0``.
    """
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    gross = log_price.diff().abs().rolling(horizon_bars, min_periods=horizon_bars).sum()
    net = (log_price - log_price.shift(horizon_bars)).abs()
    g = gross.to_numpy(dtype="float64")
    n = net.to_numpy(dtype="float64")
    out = np.full_like(g, np.nan)
    np.divide(n, g, out=out, where=g > 0)
    return pd.DataFrame(out, index=net.index, columns=net.columns)
