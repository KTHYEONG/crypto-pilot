from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.futures_strategy.signals.numpy_ops import compute_atr_numpy
from src.futures_strategy.sizing.registry import register_futures_sizing


@register_futures_sizing
class InvVolParityFuturesSizing:
    name: ClassVar[str] = "inv_vol_parity"
    param_space: ClassVar[Dict[str, Any]] = {}

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        atr_p = int(params.get("ATR_PERIOD", 14))
        risk = float(params.get("RISK_PER_TRADE", 0.02))
        max_exp = float(params.get("MAX_EXPOSURE", 1.0))
        atr = compute_atr_numpy(high, low, close, atr_p)
        atr_pct = np.where(close > 1e-12, atr / close, 0.01)
        raw = risk / (atr_pct + 1e-9)
        mean_atr_pct = float(np.nanmean(atr_pct))
        if mean_atr_pct < 1e-9:
            mean_atr_pct = 0.01
        parity_scale = mean_atr_pct / (atr_pct + 1e-9)
        size = np.clip(raw * parity_scale, 0.0, max_exp)
        return size.astype(np.float64)
