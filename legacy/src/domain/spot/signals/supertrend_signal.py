"""SuperTrend-style signal using the same state machine as `IndicatorEngine.calculate_supertrend`."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from src.core.indicators.numpy_ops_spot import compute_atr_numpy
from src.domain.spot.signals.base import SignalOutput
from src.domain.spot.signals.registry import register_signal


def _supertrend_line_trend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal SuperTrend line and direction (+1 bull / -1 bear)."""
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


@register_signal
class SuperTrendSignal:
    name: ClassVar[str] = "SUPERTREND"
    param_space: ClassVar[dict[str, Any]] = {
        "ST_ATR_PERIOD": {"type": "int", "low": 7, "high": 21, "step": 7},
        "ST_MULT": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.5},
    }

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalOutput:
        p = int(params.get("ST_ATR_PERIOD", 10))
        mult = float(params.get("ST_MULT", 3.0))
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        st_line, trend = _supertrend_line_trend(high, low, close, p, mult)
        n = close.size
        if n == 0:
            return SignalOutput(
                entry_signal=np.array([], dtype=np.bool_),
                kill_signal=np.array([], dtype=np.float64),
                rank_score=np.array([], dtype=np.float64),
            )
        atr = compute_atr_numpy(high, low, close, max(2, p))
        st_prev = np.roll(st_line, 1)
        st_prev[0] = st_line[0]
        supertrend_bull = trend == 1
        entry = supertrend_bull & (close > st_prev)
        kill = close < st_line
        safe_atr = np.maximum(atr, 1e-9)
        rank = (close - st_line) / safe_atr
        rank = np.clip(np.nan_to_num(rank, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=rank.astype(np.float64),
        )
