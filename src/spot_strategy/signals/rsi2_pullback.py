from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.spot_strategy.signals.base import SignalOutput
from src.spot_strategy.signals.numpy_ops import compute_ema_numpy, compute_rsi_numpy
from src.spot_strategy.signals.registry import register_signal


@register_signal
class RSI2PullbackSignal:
    name: ClassVar[str] = "RSI2_PULLBACK"
    param_space: ClassVar[Dict[str, Any]] = {
        "RSI2_TREND_EMA": {"type": "int", "low": 150, "high": 250, "step": 25},
        "RSI2_OVERSOLD": {"type": "float", "low": 5.0, "high": 15.0, "step": 2.5},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        trend_p = int(params.get("RSI2_TREND_EMA", 200))
        os_level = float(params.get("RSI2_OVERSOLD", 10.0))
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            return SignalOutput(
                entry_signal=np.array([], dtype=np.bool_),
                kill_signal=np.array([], dtype=np.float64),
                rank_score=np.array([], dtype=np.float64),
            )
        ema_t = compute_ema_numpy(close, max(20, trend_p))
        rsi2 = compute_rsi_numpy(close, 2)
        uptrend = np.isfinite(ema_t) & (close > ema_t)
        entry = uptrend & (rsi2 < os_level)
        kill = rsi2 > 70.0
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=(os_level - rsi2).astype(np.float64),
        )
