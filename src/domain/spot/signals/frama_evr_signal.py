from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from numba import njit

from src.domain.spot.signals.base import SignalOutput
from src.domain.spot.signals.registry import register_signal


@njit(cache=True)
def compute_frama_series(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    n = len(close)
    frama = np.copy(close)
    if n < period:
        return frama

    half_p = period // 2
    for i in range(period, n):
        # First half
        h1 = np.max(high[i - period : i - half_p])
        l1 = np.min(low[i - period : i - half_p])
        n1 = (h1 - l1) / half_p

        # Second half
        h2 = np.max(high[i - half_p : i])
        l2 = np.min(low[i - half_p : i])
        n2 = (h2 - l2) / half_p

        # Total
        h3 = np.max(high[i - period : i])
        l3 = np.min(low[i - period : i])
        n3 = (h3 - l3) / period

        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
        else:
            d = 1.0

        w = -4.6 * (d - 1.0)
        alpha = np.exp(w)
        alpha = min(max(alpha, 0.01), 1.0)
        frama[i] = alpha * close[i] + (1.0 - alpha) * frama[i - 1]
    return frama


@njit(cache=True)
def compute_evr_zscore(
    open_p: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    # Elastic Volume Ratio (Proxy) - Ratio of returns weighted by volume spikes
    n = len(close)
    evr_z = np.zeros(n)
    if n < window:
        return evr_z

    # Simple z-score of (Volume * Range / StdDev)
    price_range = np.abs(high - low)
    raw_evr = volume * price_range

    for i in range(window, n):
        win = raw_evr[i - window : i]
        mean = np.mean(win)
        std = np.std(win)
        if std > 1e-12:
            evr_z[i] = (raw_evr[i] - mean) / std
        else:
            evr_z[i] = 0.0
    return evr_z


@register_signal
class FramaEvrSignal:
    name: ClassVar[str] = "FRAMA_EVR"
    param_space: ClassVar[dict[str, Any]] = {
        "FRAMA_PERIOD": {"type": "int", "low": 8, "high": 32, "step": 4},
        "EVR_WINDOW": {"type": "int", "low": 20, "high": 80, "step": 10},
        "EVR_Z_ENTRY": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "FRAMA_KILL_LAG": {"type": "int", "low": 2, "high": 8, "step": 1},
    }

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalOutput:
        frama_p = int(params.get("FRAMA_PERIOD", 16))
        evr_w = int(params.get("EVR_WINDOW", 40))
        evr_z_entry = float(params.get("EVR_Z_ENTRY", 1.0))
        kill_lag = int(params.get("FRAMA_KILL_LAG", 4))

        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        open_ = df["open"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

        frama = compute_frama_series(high, low, close, frama_p)
        frama_prev = np.roll(frama, 1)
        frama_prev[0] = frama[0]
        frama_bull = (close > frama) & (frama > frama_prev)

        evr_z = compute_evr_zscore(open_, high, low, close, volume, evr_w)
        entry = frama_bull & (evr_z > evr_z_entry)

        frama_kill_ref = np.roll(frama, kill_lag)
        frama_kill_ref[:kill_lag] = frama[:kill_lag]
        kill = close < frama_kill_ref

        rank = np.clip(evr_z, -5.0, 5.0)
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=rank.astype(np.float64),
        )
