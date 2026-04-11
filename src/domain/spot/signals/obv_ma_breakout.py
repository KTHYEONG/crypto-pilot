from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_spot import compute_ema_numpy, rolling_ema_winsorize_volume
from src.domain.spot.signals.base import SignalOutput
from src.domain.spot.signals.registry import register_signal


@register_signal
class ObvMaBreakoutSignal:
    name: ClassVar[str] = "OBV_MA"
    param_space: ClassVar[Dict[str, Any]] = {
        "OBV_WINSOR_SPAN": {"type": "int", "low": 36, "high": 60, "step": 6},
        "OBV_EMA_PERIOD": {"type": "int", "low": 14, "high": 30, "step": 2},
        "OBV_RANK_WINDOW": {"type": "int", "low": 14, "high": 30, "step": 2},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        wspan = int(params.get("OBV_WINSOR_SPAN", 48))
        ema_p = int(params.get("OBV_EMA_PERIOD", 20))
        rw = int(params.get("OBV_RANK_WINDOW", 20))
        close = df["close"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            return SignalOutput(
                entry_signal=np.array([], dtype=np.bool_),
                kill_signal=np.array([], dtype=np.float64),
                rank_score=np.array([], dtype=np.float64),
            )
        vclip = rolling_ema_winsorize_volume(volume, wspan, 3.0)
        chg = np.diff(close, prepend=close[0])
        sign = np.sign(chg)
        obv = np.cumsum(sign * vclip)
        obv_ma = compute_ema_numpy(obv, max(2, ema_p))
        obv_prev = np.roll(obv, 1)
        ma_prev = np.roll(obv_ma, 1)
        obv_prev[0] = obv[0]
        ma_prev[0] = obv_ma[0]
        cross_up = (obv > obv_ma) & (obv_prev <= ma_prev)
        entry = cross_up & np.isfinite(obv_ma)
        kill = obv < obv_ma
        resid = obv - obv_ma
        roll_std = pd.Series(resid).rolling(max(2, rw)).std(ddof=0).to_numpy(dtype=np.float64)
        z = np.where(roll_std > 1e-12, resid / roll_std, 0.0)
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=z.astype(np.float64),
        )
