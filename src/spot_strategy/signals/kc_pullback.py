from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.spot_strategy.signals.base import SignalOutput
from src.spot_strategy.signals.numpy_ops import compute_atr_numpy, compute_ema_numpy, compute_rsi_numpy
from src.spot_strategy.signals.registry import register_signal


@register_signal
class KCPullbackSignal:
    name: ClassVar[str] = "KC_PULLBACK"
    param_space: ClassVar[Dict[str, Any]] = {
        "EMA_SLOW_PERIOD": {"type": "int", "low": 100, "high": 250, "step": 10},
        "KC_PERIOD": {"type": "int", "low": 15, "high": 40, "step": 5},
        "KC_MULT": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "RSI_PERIOD": {"type": "int", "low": 10, "high": 20, "step": 2},
        "RSI_LOW_THRESH": {"type": "float", "low": 20.0, "high": 40.0, "step": 2.5},
        "TP_MEAN_PERIOD": {"type": "int", "low": 10, "high": 50, "step": 10},
        "EMA_SLOPE_LAG": {"type": "int", "low": 5, "high": 20, "step": 5},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        slope_lag = int(params.get("EMA_SLOPE_LAG", 10))
        ema_slow_p = int(params.get("EMA_SLOW_PERIOD", 200))
        kc_p = int(params.get("KC_PERIOD", 20))
        kc_m = float(params.get("KC_MULT", 2.0))
        rsi_p = int(params.get("RSI_PERIOD", 14))
        rsi_low = float(params.get("RSI_LOW_THRESH", 30.0))
        tp_mean_p = int(params.get("TP_MEAN_PERIOD", 20))
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        ema_slow = compute_ema_numpy(close, ema_slow_p)
        ema_slow_prev = np.roll(ema_slow, slope_lag)
        ema_slow_prev[:slope_lag] = ema_slow[:slope_lag]
        atr = compute_atr_numpy(high, low, close, kc_p)
        kc_lower = compute_ema_numpy(close, kc_p) - kc_m * atr
        rsi = compute_rsi_numpy(close, rsi_p)
        macro_bull = (close > ema_slow) & (ema_slow > ema_slow_prev)
        dip = (rsi < rsi_low) | (close < kc_lower)
        entry = macro_bull & dip
        ema_fast = compute_ema_numpy(close, tp_mean_p)
        kill = close > ema_fast
        # rank_score: deeper oversold (lower RSI) => higher priority for long dip entries
        rank = 100.0 - rsi
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=rank.astype(np.float64),
        )
