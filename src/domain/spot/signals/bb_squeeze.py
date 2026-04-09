from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.spot.signals.base import SignalOutput
from src.core.indicators.numpy_ops_spot import compute_atr_numpy, compute_ema_numpy
from src.domain.spot.signals.registry import register_signal


@register_signal
class BBSqueezeSignal:
    name: ClassVar[str] = "BB_SQUEEZE"
    param_space: ClassVar[Dict[str, Any]] = {
        "BB_SQ_PERIOD": {"type": "int", "low": 14, "high": 24, "step": 2},
        "BB_SQ_STD": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.25},
        "BB_SQ_KC_MULT": {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        p = int(params.get("BB_SQ_PERIOD", 20))
        n_std = float(params.get("BB_SQ_STD", 2.0))
        kc_m = float(params.get("BB_SQ_KC_MULT", 1.5))
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            return SignalOutput(
                entry_signal=np.array([], dtype=np.bool_),
                kill_signal=np.array([], dtype=np.float64),
                rank_score=np.array([], dtype=np.float64),
            )
        ma_s = pd.Series(close).rolling(max(2, p)).mean()
        std_s = pd.Series(close).rolling(max(2, p)).std(ddof=0)
        bb_u = (ma_s + n_std * std_s).to_numpy(dtype=np.float64)
        bb_l = (ma_s - n_std * std_s).to_numpy(dtype=np.float64)
        atr = compute_atr_numpy(high, low, close, p)
        ema_c = compute_ema_numpy(close, p)
        kc_u = ema_c + kc_m * atr
        kc_l = ema_c - kc_m * atr
        squeeze = (bb_l > kc_l) & (bb_u < kc_u)
        sq_prev = np.roll(squeeze, 1)
        sq_prev[0] = False
        released = sq_prev & (~squeeze)
        mom = np.zeros(n, dtype=np.bool_)
        mom[1:] = close[1:] > close[:-1]
        entry = released & mom
        kill = close < ema_c
        width = np.where(np.isfinite(bb_u - bb_l), bb_u - bb_l, 0.0)
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=width.astype(np.float64),
        )
