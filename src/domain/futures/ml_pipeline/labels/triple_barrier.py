"""Triple barrier labels using 4h entry and 1m path scan."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from src.core.indicators.numpy_ops_futures import compute_atr_numpy


@njit(cache=True)  # type: ignore[untyped-decorator]
def _tbm_inner_loop_numba(
    dt4: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    high4: np.ndarray,
    low4: np.ndarray,
    m1_ts: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    tp_atr_mult: float,
    sl_atr_mult: float,
    time_stop_bars: int,
) -> np.ndarray:
    """Numba-accelerated inner loop for TBM labeling (legacy unidirectional)."""
    n4 = len(dt4)
    labels = np.zeros(n4, dtype=np.float64)
    for i in range(n4 - 1):
        entry = open_[i]
        a = atr[i]
        if not np.isfinite(a):
            a = abs(high4[i] - low4[i])

        tp = entry + tp_atr_mult * a
        sl = entry - sl_atr_mult * a
        t0 = dt4[i]
        t1 = dt4[i + 1]

        idx_start = np.searchsorted(m1_ts, t0)
        idx_end = np.searchsorted(m1_ts, t1)

        if idx_start >= idx_end:
            continue

        max_scan = min(idx_end, idx_start + time_stop_bars)

        hit_tp = False
        hit_sl = False
        for j in range(idx_start, max_scan):
            hi = m1_high[j]
            lo = m1_low[j]
            if lo <= sl:
                hit_sl = True
                break
            if hi >= tp:
                hit_tp = True
                break

        if hit_tp and not hit_sl:
            labels[i] = 1.0

    return labels


@njit(cache=True)  # type: ignore[untyped-decorator]
def _tbm_bidirectional_numba(
    dt4: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    high4: np.ndarray,
    low4: np.ndarray,
    m1_ts: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    tp_atr_mult: float,
    sl_atr_mult: float,
    time_stop_bars: int,
) -> np.ndarray:
    """Bidirectional TBM: simultaneously evaluates Long and Short barriers.

    Returns labels:
      +1.0 = Long TP hit first  → calib_prob_long training target
      -1.0 = Short TP hit first → calib_prob_short training target (= Long SL hit)
       0.0 = Time barrier (neutral; no directional edge confirmed)

    Asymmetric TP/SL (tp_atr_mult > sl_atr_mult) increases positive-class
    ratio for each direction classifier, mitigating class imbalance.
    """
    n4 = len(dt4)
    labels = np.zeros(n4, dtype=np.float64)
    for i in range(n4 - 1):
        entry = open_[i]
        a = atr[i]
        if not np.isfinite(a):
            a = abs(high4[i] - low4[i])

        long_tp = entry + tp_atr_mult * a
        long_sl = entry - sl_atr_mult * a  # = Short direction's TP

        t0 = dt4[i]
        t1 = dt4[i + 1]
        idx_start = np.searchsorted(m1_ts, t0)
        idx_end = np.searchsorted(m1_ts, t1)

        if idx_start >= idx_end:
            continue

        max_scan = min(idx_end, idx_start + time_stop_bars)

        for j in range(idx_start, max_scan):
            hi = m1_high[j]
            lo = m1_low[j]
            # Long SL hit first → price moved down → Short direction edge confirmed
            if lo <= long_sl:
                labels[i] = -1.0
                break
            # Long TP hit first → Long direction edge confirmed
            if hi >= long_tp:
                labels[i] = 1.0
                break
        # Time barrier: labels[i] remains 0.0 (no edge)

    return labels


def label_triple_barrier(
    df_1h: pd.DataFrame,
    df_1m: pd.DataFrame,
    atr_period: int = 14,
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    time_stop_bars: int = 1440,
    bidirectional: bool = True,
    vol_scale_window: int = 24,
) -> pd.Series:
    """Per 1h bar: entry at open; TP/SL from ATR on 1h; scan 1m candles until hit or time stop.

    Args:
        df_1h: 1h timeframe DataFrame containing high, low, close, and optionally open.
        df_1m: 1m timeframe DataFrame containing high, low, and datetime for path scanning.
        atr_period: Period for ATR calculation (default 14).
        tp_atr_mult: TP width multiplier (default 1.5).
        sl_atr_mult: SL width multiplier (default 1.0).
        time_stop_bars: 1m bars before time barrier (default 1440 ≈ 24h on 1m).
        bidirectional: If True, returns +1/-1/0 labels for independent Long/Short classifiers.
        vol_scale_window: rolling mean ATR window; ATR_effective = ATR * clip(ATR / ATR_ma).

    Returns:
        pd.Series containing triple barrier labels.

    """
    if df_1h.empty or df_1m.empty or len(df_1h) < atr_period + 2:
        return pd.Series(dtype=np.float64)

    df1 = df_1h.sort_values("datetime").reset_index(drop=True)
    m1 = df_1m.sort_values("datetime").reset_index(drop=True)

    m1_ts = m1["datetime"].astype("int64").to_numpy()
    m1_high = m1["high"].to_numpy(dtype=np.float64)
    m1_low = m1["low"].to_numpy(dtype=np.float64)

    dt1 = df1["datetime"].astype("int64").to_numpy()
    high1 = df1["high"].to_numpy(dtype=np.float64)
    low1 = df1["low"].to_numpy(dtype=np.float64)
    close1 = df1["close"].to_numpy(dtype=np.float64)
    open1 = df1["open"].to_numpy(dtype=np.float64) if "open" in df1.columns else close1

    atr1 = compute_atr_numpy(high1, low1, close1, atr_period)
    vw = max(4, int(vol_scale_window))
    atr_ma = (
        pd.Series(atr1)
        .rolling(vw, min_periods=max(2, vw // 3))
        .mean()
        .bfill()
        .to_numpy(dtype=np.float64)
    )
    vol_ratio = atr1 / np.maximum(atr_ma, 1e-12)
    vol_ratio = np.clip(vol_ratio, 0.75, 1.45)
    atr_eff = atr1 * vol_ratio

    if bidirectional:
        labels = _tbm_bidirectional_numba(
            dt1, open1, atr_eff, high1, low1,
            m1_ts, m1_high, m1_low,
            float(tp_atr_mult), float(sl_atr_mult), int(time_stop_bars),
        )
    else:
        labels = _tbm_inner_loop_numba(
            dt1, open1, atr_eff, high1, low1,
            m1_ts, m1_high, m1_low,
            float(tp_atr_mult), float(sl_atr_mult), int(time_stop_bars),
        )

    return pd.Series(labels, index=df1["datetime"], name="tbm_label")
