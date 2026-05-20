from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_spot import compute_rsi_numpy
from src.domain.spot.signals.base import SignalOutput
from src.domain.spot.signals.registry import register_signal


@register_signal
class StochRSICrossSignal:
    name: ClassVar[str] = "STOCHRSI_CROSS"
    param_space: ClassVar[dict[str, Any]] = {
        "STOCHRSI_RSI_P": {"type": "int", "low": 10, "high": 21, "step": 2},
        "STOCHRSI_LEN": {"type": "int", "low": 10, "high": 20, "step": 2},
        "STOCHRSI_OS": {"type": "float", "low": 0.1, "high": 0.35, "step": 0.05},
    }

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalOutput:
        rsi_p = int(params.get("STOCHRSI_RSI_P", 14))
        st_len = int(params.get("STOCHRSI_LEN", 14))
        os_level = float(params.get("STOCHRSI_OS", 0.2))
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            return SignalOutput(
                entry_signal=np.array([], dtype=np.bool_),
                kill_signal=np.array([], dtype=np.float64),
                rank_score=np.array([], dtype=np.float64),
            )
        rsi = compute_rsi_numpy(close, max(2, rsi_p))
        rsi_min = pd.Series(rsi).rolling(max(2, st_len)).min().to_numpy(dtype=np.float64)
        rsi_max = pd.Series(rsi).rolling(max(2, st_len)).max().to_numpy(dtype=np.float64)
        denom = np.maximum(rsi_max - rsi_min, 1e-12)
        stoch = (rsi - rsi_min) / denom
        k = pd.Series(stoch).rolling(3).mean().to_numpy(dtype=np.float64)
        d = pd.Series(k).rolling(3).mean().to_numpy(dtype=np.float64)
        k_prev = np.roll(k, 1)
        d_prev = np.roll(d, 1)
        k_prev[0] = np.nan
        d_prev[0] = np.nan
        cross_up = (k > d) & (k_prev <= d_prev) & np.isfinite(k_prev) & np.isfinite(d_prev)
        entry = cross_up & (k < os_level) & np.isfinite(k)
        kill = k > 0.85
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=(os_level - k).astype(np.float64),
        )
