from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.spot.signals.base import SignalOutput
from src.domain.spot.signals.registry import register_signal


@register_signal
class VIXFixSignal:
    name: ClassVar[str] = "VIX_FIX"
    param_space: ClassVar[Dict[str, Any]] = {
        "WVF_PERIOD": {"type": "int", "low": 16, "high": 30, "step": 2},
        "WVF_LOOKBACK": {"type": "int", "low": 80, "high": 150, "step": 10},
        "WVF_PERCENTILE": {"type": "float", "low": 0.95, "high": 0.99, "step": 0.01},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        p = int(params.get("WVF_PERIOD", 22))
        lb = int(params.get("WVF_LOOKBACK", 100))
        pct = float(params.get("WVF_PERCENTILE", 0.975))
        close = df["close"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            return SignalOutput(
                entry_signal=np.array([], dtype=np.bool_),
                kill_signal=np.array([], dtype=np.float64),
                rank_score=np.array([], dtype=np.float64),
            )
        roll_max = pd.Series(close).rolling(max(2, p)).max().to_numpy(dtype=np.float64)
        safe_rm = np.maximum(roll_max, 1e-12)
        wvf = (roll_max - low) / safe_rm
        wvf = np.clip(wvf, 0.0, 1.5)
        thr = (
            pd.Series(wvf)
            .rolling(max(lb, p + 5))
            .quantile(pct)
            .to_numpy(dtype=np.float64)
        )
        spike = wvf > thr
        rev = np.zeros(n, dtype=np.bool_)
        rev[1:] = close[1:] > close[:-1]
        entry = spike & rev
        kill = wvf < np.roll(wvf, 1)
        kill[0] = False
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=wvf.astype(np.float64),
        )
