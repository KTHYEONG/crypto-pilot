"""Shared causal NumPy/Pandas indicator helpers for spot signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _rolling_ema_winsorize_jit(volume: np.ndarray, span: int, k: float) -> np.ndarray:
    """Causal EMA + EMA-of-absolute-deviation cap; clip volume to cap (no future leak)."""
    n = volume.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (float(span) + 1.0)
    ema = np.empty(n, dtype=np.float64)
    mad_ema = np.empty(n, dtype=np.float64)
    ema[0] = volume[0]
    mad_ema[0] = 0.0
    for i in range(1, n):
        ema[i] = alpha * volume[i] + (1.0 - alpha) * ema[i - 1]
        deviation = abs(volume[i] - ema[i - 1])
        mad_ema[i] = alpha * deviation + (1.0 - alpha) * mad_ema[i - 1]
    for i in range(n):
        cap = ema[i] + k * mad_ema[i]
        v = volume[i]
        if v > cap:
            out[i] = cap
        else:
            out[i] = v
    return out


def rolling_ema_winsorize_volume(volume: np.ndarray, span: int, k: float = 3.0) -> np.ndarray:
    v = np.asarray(volume, dtype=np.float64).ravel()
    if v.size == 0:
        return v.copy()
    sp = max(2, int(span))
    return _rolling_ema_winsorize_jit(v, sp, float(k))


def compute_ema_numpy(close: np.ndarray, period: int) -> np.ndarray:
    if len(close) == 0:
        return np.array([], dtype=np.float64)
    return (
        pd.Series(close)
        .ewm(span=max(2, period), adjust=False)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def compute_rsi_numpy(close: np.ndarray, period: int) -> np.ndarray:
    if len(close) == 0:
        return np.array([], dtype=np.float64)
    p = max(2, period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    alpha = 1.0 / p
    avg_gain = pd.Series(gain).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
    avg_loss = pd.Series(loss).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 1e-12, avg_gain / np.maximum(avg_loss, 1e-12), 100.0)
    rsi = np.clip(100.0 - (100.0 / (1.0 + rs)), 0.0, 100.0)
    return np.where(np.isfinite(rsi), rsi, 50.0)


def compute_atr_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    if len(close) == 0:
        return np.array([], dtype=np.float64)
    p = max(2, int(period))
    tr = np.empty(len(close), dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    return pd.Series(tr).ewm(span=p, adjust=False).mean().to_numpy(dtype=np.float64)


def compute_adx_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
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
    s_tr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
    s_pdm = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
    s_mdm = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(s_tr > 1e-12, 100.0 * s_pdm / s_tr, 0.0)
        mdi = np.where(s_tr > 1e-12, 100.0 * s_mdm / s_tr, 0.0)
        dx = np.where((pdi + mdi) > 1e-12, 100.0 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)
    return np.clip(adx, 0.0, 100.0)
