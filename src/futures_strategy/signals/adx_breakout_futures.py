"""ADX + Keltner: breakout above KC -> long, below KC -> short."""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.futures_strategy.signals.base import FuturesSignalOutput
from src.futures_strategy.signals.numpy_ops import compute_adx_numpy, compute_atr_numpy, compute_ema_numpy
from src.futures_strategy.signals.registry import register_futures_signal


@register_futures_signal
class AdxBreakoutFuturesSignal:
    name: ClassVar[str] = "ADX_BREAKOUT"
    param_space: ClassVar[Dict[str, Any]] = {
        "ADX_KC_PERIOD": {"type": "int", "low": 15, "high": 40, "step": 5},
        "ADX_LONG_KC_MULT": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.25},
        "ADX_SHORT_KC_MULT": {"type": "float", "low": 1.0, "high": 3.5, "step": 0.25},
        "ADX_THRESHOLD": {"type": "float", "low": 15.0, "high": 35.0, "step": 5.0},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        kc_p = int(params.get("ADX_KC_PERIOD", 20))
        kc_m_long = float(params.get("ADX_LONG_KC_MULT", 2.0))
        kc_m_short = float(params.get("ADX_SHORT_KC_MULT", 2.0))
        adx_threshold = float(params.get("ADX_THRESHOLD", 20.0))
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
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
        atr = compute_atr_numpy(high, low, close, kc_p)
        ema_kc = compute_ema_numpy(close, kc_p)
        kc_upper = ema_kc + kc_m_long * atr
        kc_lower = ema_kc - kc_m_short * atr
        adx = compute_adx_numpy(high, low, close, kc_p)
        long_e = (close > kc_upper) & (adx > adx_threshold)
        short_e = (close < kc_lower) & (adx > adx_threshold)
        kill_l = (close < kc_lower).astype(np.float64)
        kill_s = (close > kc_upper).astype(np.float64)
        return FuturesSignalOutput(
            long_entry=long_e.astype(np.bool_),
            short_entry=short_e.astype(np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=adx.astype(np.float64),
        )
