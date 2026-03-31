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
        n1 = (hh1 - ll1) / half
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
        n2 = (hh2 - ll2) / half
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
        n3 = (hh3 - ll3) / n
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


@njit(cache=True)
def _rsi_wilder_core(
    gain: np.ndarray,
    loss: np.ndarray,
    period: int,
    out: np.ndarray,
) -> None:
    """In-place Wilder RSI into out; bars 0..period-1 unchanged (caller fills neutral)."""
    n = gain.shape[0]
    if n <= period or period < 1:
        return
    ag = 0.0
    al = 0.0
    for j in range(1, period + 1):
        ag += gain[j]
        al += loss[j]
    ag /= float(period)
    al /= float(period)
    if al > 1e-18:
        rs = ag / al
        out[period] = 100.0 - (100.0 / (1.0 + rs))
    else:
        out[period] = 100.0 if ag > 1e-18 else 50.0
    pm1 = float(period - 1)
    for i in range(period + 1, n):
        ag = (ag * pm1 + gain[i]) / float(period)
        al = (al * pm1 + loss[i]) / float(period)
        if al > 1e-18:
            rs = ag / al
            out[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            out[i] = 100.0 if ag > 1e-18 else 50.0


def compute_williams_fractals(
    high: np.ndarray,
    low: np.ndarray,
    *,
    lookback: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Williams fractals with confirmation at center_index + lookback (causal).
    Returns confirmed flags and prices at the confirmation bar only.
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    n = int(h.shape[0])
    lb = max(1, int(lookback))
    fractal_low_c = np.ones(n, dtype=np.bool_)
    fractal_high_c = np.ones(n, dtype=np.bool_)
    for k in range(1, lb + 1):
        left_l = np.empty(n, dtype=np.float64)
        right_l = np.empty(n, dtype=np.float64)
        left_l[:k] = -np.inf
        left_l[k:] = l[:-k]
        right_l[n - k :] = -np.inf
        right_l[: n - k] = l[k:]
        fractal_low_c &= (l < left_l) & (l < right_l)
        left_h = np.empty(n, dtype=np.float64)
        right_h = np.empty(n, dtype=np.float64)
        left_h[:k] = np.inf
        left_h[k:] = h[:-k]
        right_h[n - k :] = np.inf
        right_h[: n - k] = h[k:]
        fractal_high_c &= (h > left_h) & (h > right_h)
    fractal_low_c[:lb] = False
    fractal_low_c[n - lb :] = False
    fractal_high_c[:lb] = False
    fractal_high_c[n - lb :] = False

    fractal_low = np.zeros(n, dtype=np.bool_)
    fractal_high = np.zeros(n, dtype=np.bool_)
    low_price = np.full(n, np.nan, dtype=np.float64)
    high_price = np.full(n, np.nan, dtype=np.float64)

    i_centers = np.flatnonzero(fractal_low_c)
    j_low = i_centers + lb
    m_ok = j_low < n
    jj = j_low[m_ok]
    fractal_low[jj] = True
    low_price[jj] = l[i_centers[m_ok]]

    i_hi = np.flatnonzero(fractal_high_c)
    j_hi = i_hi + lb
    m_h = j_hi < n
    jh = j_hi[m_h]
    fractal_high[jh] = True
    high_price[jh] = h[i_hi[m_h]]

    return fractal_low, fractal_high, low_price, high_price


def compute_rsi2_series(
    close: np.ndarray,
    *,
    period: int = 2,
) -> np.ndarray:
    """Wilder RSI; early bars filled with 50.0; output clipped to [0, 100]."""
    c = np.asarray(close, dtype=np.float64)
    n = int(c.shape[0])
    out = np.full(n, 50.0, dtype=np.float64)
    if n < 2:
        return out
    delta = np.empty(n, dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = c[1:] - c[:-1]
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    p = max(1, int(period))
    if n > p:
        _rsi_wilder_core(gain, loss, p, out)
    return np.clip(np.nan_to_num(out, nan=50.0, posinf=100.0, neginf=0.0), 0.0, 100.0)


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
