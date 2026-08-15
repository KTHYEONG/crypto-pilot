from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_spot import (
    compute_adx_numpy,
    compute_atr_numpy,
    compute_ema_numpy,
)
from src.domain.spot.signals.base import SignalOutput
from src.domain.spot.signals.registry import register_signal


@register_signal
class ADXBreakoutSignal:
    name: ClassVar[str] = "ADX_BREAKOUT"
    param_space: ClassVar[dict[str, Any]] = {
        "KC_PERIOD": {"type": "int", "low": 15, "high": 40, "step": 5},
        "KC_MULT": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "ADX_THRESHOLD": {"type": "float", "low": 15.0, "high": 35.0, "step": 5.0},
    }

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalOutput:
        kc_p = int(params.get("KC_PERIOD", 20))
        kc_m = float(params.get("KC_MULT", 2.0))
        adx_threshold = float(params.get("ADX_THRESHOLD", 20.0))
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        atr = compute_atr_numpy(high, low, close, kc_p)
        ema_kc = compute_ema_numpy(close, kc_p)
        kc_upper = ema_kc + kc_m * atr
        kc_lower = ema_kc - kc_m * atr
        adx = compute_adx_numpy(high, low, close, kc_p)
        entry = (close > kc_upper) & (adx > adx_threshold)
        kill = close < kc_lower
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=adx.astype(np.float64),
        )
