"""MACD histogram divergence: bullish div -> long, bearish div -> short."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_futures import compute_ema_numpy
from src.domain.futures.signals.base import FuturesSignalOutput
from src.domain.futures.signals.registry import register_futures_signal


@register_futures_signal
class MacdHistDivFuturesSignal:
    name: ClassVar[str] = "MACD_HIST_DIV"
    param_space: ClassVar[Dict[str, Any]] = {
        "MACD_FAST": {"type": "int", "low": 8, "high": 14, "step": 2},
        "MACD_SLOW": {"type": "int", "low": 20, "high": 30, "step": 2},
        "MACD_SIGNAL": {"type": "int", "low": 7, "high": 11, "step": 1},
        "MACD_DIV_LAG": {"type": "int", "low": 3, "high": 8, "step": 1},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        fast_p = int(params.get("MACD_FAST", 12))
        slow_p = int(params.get("MACD_SLOW", 26))
        sig_p = int(params.get("MACD_SIGNAL", 9))
        lag = int(params.get("MACD_DIV_LAG", 4))
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            z = np.array([], dtype=np.float64)
            return FuturesSignalOutput(
                np.array([], dtype=np.bool_),
                np.array([], dtype=np.bool_),
                z,
                z,
                z,
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
        price_hh = close > np.roll(close, lag)
        price_hh[:lag] = False
        hist_hl = hist > np.roll(hist, lag)
        hist_hl[:lag] = False
        hist_lh = hist < np.roll(hist, lag)
        hist_lh[:lag] = False
        long_e = (hist > 0.0) & price_ll & hist_hl & np.isfinite(hist)
        short_e = (hist < 0.0) & price_hh & hist_lh & np.isfinite(hist)
        kill_l = (hist < np.roll(hist, 1)).astype(np.float64)
        kill_l[0] = 0.0
        kill_s = (hist > np.roll(hist, 1)).astype(np.float64)
        kill_s[0] = 0.0
        rank = hist.astype(np.float64)
        return FuturesSignalOutput(
            long_entry=long_e.astype(np.bool_),
            short_entry=short_e.astype(np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=rank,
        )
