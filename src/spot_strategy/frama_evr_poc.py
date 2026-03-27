"""
FRAMA + EvR POC: optional alternative signal layer (does not replace production SuperTrend).

FRAMA: Ehlers FRAMA recurrence (Numba inner loop; O(T) per window).
EvR: effort-vs-result = |close-open| / volume (vectorized).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_logger: logging.Logger = logging.getLogger(__name__)

try:
    from numba import njit

    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

    def njit(*args: Any, **kwargs: Any) -> Any:
        def _wrap(f: Any) -> Any:
            return f

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


@njit(cache=True)
def _frama_recurrence(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: int,
) -> np.ndarray:
    """Ehlers FRAMA: fractal dimension from N/2 price ranges (John Ehlers)."""
    length = close.shape[0]
    out = np.empty(length, dtype=np.float64)
    half = n // 2
    if half < 2:
        for i in range(length):
            out[i] = close[i]
        return out

    log2 = np.log(2.0)
    for i in range(length):
        if i < n - 1:
            out[i] = close[i]
            continue
        hh1 = high[i - n + 1]
        ll1 = low[i - n + 1]
        for j in range(1, half):
            j0 = i - n + 1 + j
            if high[j0] > hh1:
                hh1 = high[j0]
            if low[j0] < ll1:
                ll1 = low[j0]
        n1 = hh1 - ll1
        if n1 < 1e-12:
            n1 = 1e-12

        hh2 = high[i - half + 1]
        ll2 = low[i - half + 1]
        for j in range(1, half):
            j0 = i - half + 1 + j
            if high[j0] > hh2:
                hh2 = high[j0]
            if low[j0] < ll2:
                ll2 = low[j0]
        n2 = hh2 - ll2
        if n2 < 1e-12:
            n2 = 1e-12

        hh3 = high[i - n + 1]
        ll3 = low[i - n + 1]
        for j in range(1, n):
            j0 = i - n + 1 + j
            if high[j0] > hh3:
                hh3 = high[j0]
            if low[j0] < ll3:
                ll3 = low[j0]
        n3 = hh3 - ll3
        if n3 < 1e-12:
            n3 = 1e-12

        d = (np.log(n1 + n2) - np.log(n3)) / log2
        if d < 1.0:
            d = 1.0
        if d > 2.0:
            d = 2.0
        alpha = np.exp(-4.6 * (d - 1.0))
        prev = out[i - 1]
        out[i] = alpha * close[i] + (1.0 - alpha) * prev
    return out


def compute_frama_series(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    period: int,
) -> np.ndarray:
    """Ehlers FRAMA (POC)."""
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = max(8, int(period))
    if n % 2 == 1:
        n += 1
    if not _NUMBA_OK:
        _logger.warning("numba unavailable; FRAMA POC uses slower Python path.")
    return np.asarray(_frama_recurrence(h, lo, c, n), dtype=np.float64)


def compute_evr_series(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    eps: float = 1e-9,
    *,
    directional_bull: bool = False,
) -> np.ndarray:
    """Effort vs Result core series; optional bullish-directional body."""
    o = np.asarray(open_, dtype=np.float64)
    cl = np.asarray(close, dtype=np.float64)
    v = np.maximum(np.asarray(volume, dtype=np.float64), eps)
    body = np.maximum(cl - o, 0.0) if directional_bull else np.abs(cl - o)
    rng = np.maximum(np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64), eps)
    return (body / v * rng).astype(np.float64)


def compute_evr_zscore(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
    *,
    eps: float = 1e-9,
    directional_bull: bool = False,
) -> np.ndarray:
    """Rolling z-score of EvR (causal rolling mean/std). NaN -> 0, clipped to [-5, 5]."""
    evr = compute_evr_series(
        open_,
        high,
        low,
        close,
        volume,
        eps=eps,
        directional_bull=directional_bull,
    )
    w = max(10, int(window))
    min_periods = max(3, min(w // 3, w))
    ser = pd.Series(evr, copy=False)
    mu = ser.rolling(window=w, min_periods=min_periods).mean()
    sig = ser.rolling(window=w, min_periods=min_periods).std(ddof=0)
    sig = sig.replace(0.0, eps)
    z = (ser - mu) / sig
    out = z.fillna(0.0).to_numpy(dtype=np.float64)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)


def compute_supertrend_direction_poc(
    df: pd.DataFrame,
    period: int,
    mult: float,
) -> np.ndarray:
    """Minimal SuperTrend direction (+1 bull / -1 bear) for A/B agreement (POC)."""
    from src.futures_strategy.indicators_advanced_futures import calculate_supertrend

    st = calculate_supertrend(df, period=period, multiplier=mult)
    return np.asarray(st, dtype=np.int32)


@dataclass(frozen=True)
class FramaEvrAbSummary:
    frama_bull_share: float
    st_bull_share: float
    direction_agreement: float
    mean_evr: float
    n_bars: int


def run_frama_evr_ab_summary(
    df: pd.DataFrame,
    *,
    frama_period: int = 16,
    supertrend_period: int = 10,
    supertrend_mult: float = 3.0,
) -> FramaEvrAbSummary:
    """
    Compare FRAMA trend vs SuperTrend direction (POC; does not modify production signals).

    Agreement = mean( (frama>lag) == (st==1) ) with FRAMA compared to prior bar.
    """
    req = ("open", "high", "low", "close", "volume")
    for col in req:
        if col not in df.columns:
            raise ValueError(f"run_frama_evr_ab_summary: missing column {col}")

    frama = compute_frama_series(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        frama_period,
    )
    frama_bull = frama > np.roll(frama, 1)
    frama_bull[0] = False

    st_dir = compute_supertrend_direction_poc(
        df[["open", "high", "low", "close"]].copy(),
        supertrend_period,
        supertrend_mult,
    )
    st_bull = st_dir == 1

    agree = np.mean(frama_bull == st_bull)
    evr = compute_evr_series(
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        df["volume"].to_numpy(),
    )

    return FramaEvrAbSummary(
        frama_bull_share=float(np.mean(frama_bull)),
        st_bull_share=float(np.mean(st_bull)),
        direction_agreement=float(agree),
        mean_evr=float(np.nanmean(evr)),
        n_bars=int(len(df)),
    )
