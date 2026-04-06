from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.spot_strategy.signals.numpy_ops import compute_atr_numpy
from src.spot_strategy.sizing.registry import register_sizing


@register_sizing
class VolTargetSizing:
    name: ClassVar[str] = "vol_target"
    param_space: ClassVar[Dict[str, Any]] = {
        "VOL_SCALE": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        atr_p = int(params.get("ATR_PERIOD", 14))
        atr = compute_atr_numpy(high, low, close, atr_p)
        risk_per_trade = float(params.get("RISK_PER_TRADE", 0.02))
        max_exp = float(params.get("MAX_EXPOSURE", 1.0))
        vol_scale = float(params.get("VOL_SCALE", 1.0))
        atr_pct = np.where(close > 1e-12, atr / close, 0.01)
        size = np.clip(risk_per_trade * vol_scale / (atr_pct + 1e-9), 0.0, max_exp)
        return size.astype(np.float64)
