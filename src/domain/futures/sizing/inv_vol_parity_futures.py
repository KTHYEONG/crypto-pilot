from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_futures import compute_atr_numpy
from src.domain.futures.sizing.registry import register_futures_sizing


@register_futures_sizing
class InvVolParityFuturesSizing:
    name: ClassVar[str] = "inv_vol_parity"
    param_space: ClassVar[Dict[str, Any]] = {
        "PARITY_ATR_PERIOD": {"type": "int", "low": 10, "high": 40, "step": 10},
        "PARITY_MAX_MULT": {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """
        [IMPROVED] Normalized Parity Weight based on relative volatility.
        Parameterized for optimization discovery.
        """
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        atr_p = int(params.get("PARITY_ATR_PERIOD", 20))
        max_mult = float(params.get("PARITY_MAX_MULT", 1.0))
        
        atr = compute_atr_numpy(high, low, close, atr_p)
        atr_pct = np.where(close > 1e-12, atr / close, 0.01)
        
        # Use a rolling mean of ATR% for a more stable baseline than global mean
        mean_atr_pct = pd.Series(atr_pct).rolling(window=100, min_periods=20).mean().to_numpy()
        mean_atr_pct = np.nan_to_num(mean_atr_pct, nan=0.02)
            
        # Parity Weight: If current vol is HIGHER than average, reduce weight relative to mean.
        with np.errstate(divide="ignore", invalid="ignore"):
            parity_weight = mean_atr_pct / np.maximum(atr_pct, 1e-9)
        
        parity_weight = np.nan_to_num(parity_weight, nan=1.0)
        
        # We clip to [0.1, max_mult]
        return np.clip(parity_weight, 0.1, max_mult).astype(np.float64)
