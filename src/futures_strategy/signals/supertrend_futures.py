"""Bidirectional SuperTrend: trend==1 long, trend==-1 short."""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.futures_strategy.signals.base import FuturesSignalOutput
from src.futures_strategy.signals.numpy_ops import compute_atr_numpy
from src.futures_strategy.signals.registry import register_futures_signal


def _supertrend_line_trend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(close.size)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32)
    p = max(2, int(period))
    m = float(mult)
    atr = compute_atr_numpy(high, low, close, p)
    hl2 = (high + low) * 0.5
    bu = hl2 + m * atr
    bl = hl2 - m * atr
    final_upper = np.zeros(n, dtype=np.float64)
    final_lower = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int32)
    upper = float(bu[0])
    lower = float(bl[0])
    curr_trend = 1
    final_upper[0] = upper
    final_lower[0] = lower
    trend[0] = curr_trend
    for i in range(1, n):
        c = float(close[i])
        c_prev = float(close[i - 1])
        if bu[i] < upper or c_prev > upper:
            upper = float(bu[i])
        if bl[i] > lower or c_prev < lower:
            lower = float(bl[i])
        if curr_trend == 1:
            curr_trend = -1 if c < lower else 1
        else:
            curr_trend = 1 if c > upper else -1
        final_upper[i] = upper
        final_lower[i] = lower
        trend[i] = curr_trend
    st_line = np.where(trend == 1, final_lower, final_upper).astype(np.float64)
    return st_line, trend


@register_futures_signal
class SuperTrendFuturesSignal:
    name: ClassVar[str] = "SUPERTREND"
    param_space: ClassVar[Dict[str, Any]] = {
        "ST_ATR_PERIOD": {"type": "int", "low": 7, "high": 21, "step": 7},
        "ST_MULT": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.5},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        p = int(params.get("ST_ATR_PERIOD", 10))
        mult = float(params.get("ST_MULT", 3.0))
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        st_line, trend = _supertrend_line_trend(high, low, close, p, mult)
        n = close.size
        if n == 0:
            z = np.array([], dtype=np.float64)
            return FuturesSignalOutput(
                np.array([], dtype=np.bool_),
                np.array([], dtype=np.bool_),
                z,
                z,
                z,
            )
        atr = compute_atr_numpy(high, low, close, max(2, p))
        st_prev = np.roll(st_line, 1)
        st_prev[0] = st_line[0]
        long_e = (trend == 1) & (close > st_prev)
        short_e = (trend == -1) & (close < st_prev)
        kill_l = (close < st_line).astype(np.float64)
        kill_s = (close > st_line).astype(np.float64)
        safe_atr = np.maximum(atr, 1e-9)
        rank = (close - st_line) / safe_atr
        rank = np.clip(np.nan_to_num(rank, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
        return FuturesSignalOutput(
            long_entry=long_e.astype(np.bool_),
            short_entry=short_e.astype(np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=rank.astype(np.float64),
        )
