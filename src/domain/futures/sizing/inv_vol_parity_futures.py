from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_futures import compute_atr_numpy
from src.domain.futures.sizing.registry import register_futures_sizing


@register_futures_sizing
class InvVolParityFuturesSizing:
    name: ClassVar[str] = "inv_vol_parity"
    param_space: ClassVar[Dict[str, Any]] = {}

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """
        [FIX] Double-Penalty Removed.
        Engine handles absolute inverse-volatility sizing.
        This module now returns a Normalized Parity Weight based on relative volatility.
        """
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        atr_p = int(params.get("ATR_PERIOD", 14))
        
        atr = compute_atr_numpy(high, low, close, atr_p)
        atr_pct = np.where(close > 1e-12, atr / close, 0.01)
        
        mean_atr_pct = float(np.nanmean(atr_pct)) if atr_pct.size > 0 else 0.02
        if mean_atr_pct < 1e-9:
            mean_atr_pct = 0.02
            
        # Parity Weight: If current vol is HIGHER than average, reduce weight relative to mean.
        # This provides a secondary smooth dampening without the square-penalty.
        parity_weight = mean_atr_pct / (atr_pct + 1e-9)
        
        # We clip to [0.1, 1.0] to maintain a minimum bet and prevent over-leveraging
        return np.clip(parity_weight, 0.1, 1.0).astype(np.float64)
