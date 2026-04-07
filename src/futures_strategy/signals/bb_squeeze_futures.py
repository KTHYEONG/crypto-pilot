"""BB squeeze release: long if momentum up, short if momentum down."""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.futures_strategy.signals.base import FuturesSignalOutput
from src.futures_strategy.signals.numpy_ops import compute_atr_numpy, compute_ema_numpy
from src.futures_strategy.signals.registry import register_futures_signal


@register_futures_signal
class BbSqueezeFuturesSignal:
    name: ClassVar[str] = "BB_SQUEEZE"
    param_space: ClassVar[Dict[str, Any]] = {
        "BB_SQ_PERIOD": {"type": "int", "low": 14, "high": 24, "step": 2},
        "BB_SQ_STD": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.25},
        "BB_SQ_KC_MULT": {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        p = int(params.get("BB_SQ_PERIOD", 20))
        n_std = float(params.get("BB_SQ_STD", 2.0))
        kc_m = float(params.get("BB_SQ_KC_MULT", 1.5))
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
        mom_dn = np.zeros(n, dtype=np.bool_)
        mom_dn[1:] = close[1:] < close[:-1]
        long_e = released & mom
        short_e = released & mom_dn
        kill_l = (close < ema_c).astype(np.float64)
        kill_s = (close > ema_c).astype(np.float64)
        width = np.where(np.isfinite(bb_u - bb_l), bb_u - bb_l, 0.0)
        rank = np.where(long_e, width, np.where(short_e, -width, 0.0)).astype(np.float64)
        return FuturesSignalOutput(
            long_entry=long_e.astype(np.bool_),
            short_entry=short_e.astype(np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=rank,
        )
