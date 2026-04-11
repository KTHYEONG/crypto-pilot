from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_spot import compute_ema_numpy
from src.domain.spot.signals.base import SignalOutput
from src.domain.spot.signals.registry import register_signal


@register_signal
class MacdHistDivSignal:
    name: ClassVar[str] = "MACD_HIST_DIV"
    param_space: ClassVar[Dict[str, Any]] = {
        "MACD_FAST": {"type": "int", "low": 8, "high": 14, "step": 2},
        "MACD_SLOW": {"type": "int", "low": 20, "high": 30, "step": 2},
        "MACD_SIGNAL": {"type": "int", "low": 7, "high": 11, "step": 1},
        "MACD_DIV_LAG": {"type": "int", "low": 3, "high": 8, "step": 1},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        fast_p = int(params.get("MACD_FAST", 12))
        slow_p = int(params.get("MACD_SLOW", 26))
        sig_p = int(params.get("MACD_SIGNAL", 9))
        lag = int(params.get("MACD_DIV_LAG", 4))
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            return SignalOutput(
                entry_signal=np.array([], dtype=np.bool_),
                kill_signal=np.array([], dtype=np.float64),
                rank_score=np.array([], dtype=np.float64),
            )
        ema_f = compute_ema_numpy(close, max(2, fast_p))
        ema_s = compute_ema_numpy(close, max(slow_p, fast_p + 1))
        macd = ema_f - ema_s
        sig = (
            pd.Series(macd).ewm(span=max(2, sig_p), adjust=False).mean().to_numpy(dtype=np.float64)
        )
        hist = macd - sig
        if lag >= n:
            lag = max(1, n // 4)
        price_ll = close < np.roll(close, lag)
        price_ll[:lag] = False
        hist_hl = hist > np.roll(hist, lag)
        hist_hl[:lag] = False
        entry = (hist > 0.0) & price_ll & hist_hl & np.isfinite(hist)
        kill = hist < np.roll(hist, 1)
        kill[0] = False
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=hist.astype(np.float64),
        )
