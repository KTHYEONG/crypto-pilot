"""Causal NumPy helpers for futures signals (vectorized rolling)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ema_numpy(close: np.ndarray, period: int) -> np.ndarray:
    if len(close) == 0:
        return np.array([], dtype=np.float64)
    ema = pd.Series(close).ewm(span=max(2, period), adjust=False).mean().to_numpy(dtype=np.float64)
    return np.asarray(ema, dtype=np.float64)


def compute_atr_numpy(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    if len(close) == 0:
        return np.array([], dtype=np.float64)
    p = max(2, int(period))
    tr = np.empty(len(close), dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    atr = pd.Series(tr).ewm(span=p, adjust=False).mean().to_numpy(dtype=np.float64)
    return np.asarray(atr, dtype=np.float64)


def compute_adx_numpy(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    if len(close) == 0:
        return np.array([], dtype=np.float64)
    p = max(2, int(period))
    alpha = 1.0 / p
    h_diff = np.diff(high, prepend=high[0])
    l_diff = np.diff(low, prepend=low[0])
    l_diff_neg = -l_diff
    plus_dm = np.where((h_diff > l_diff_neg) & (h_diff > 0.0), h_diff, 0.0)
    minus_dm = np.where((l_diff_neg > h_diff) & (l_diff_neg > 0.0), l_diff_neg, 0.0)
    tr = np.empty(len(close), dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    s_tr = np.asarray(
        pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    s_pdm = np.asarray(
        pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    s_mdm = np.asarray(
        pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(s_tr > 1e-12, 100.0 * s_pdm / s_tr, 0.0)
        mdi = np.where(s_tr > 1e-12, 100.0 * s_mdm / s_tr, 0.0)
        dx = np.where((pdi + mdi) > 1e-12, 100.0 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
    return np.asarray(np.clip(adx, 0.0, 100.0), dtype=np.float64)
