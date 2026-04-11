from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_futures import compute_atr_numpy
from src.domain.futures.sizing.registry import register_futures_sizing


@register_futures_sizing
class VolTargetFuturesSizing:
    name: ClassVar[str] = "vol_target"
    param_space: ClassVar[Dict[str, Any]] = {
        "VOL_SCALE": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """
        [FIX] Double-Penalty Bug Removed.
        The engine already implements Inverse Volatility Sizing (Risk / Stop_Distance).
        This module now purely returns a Confidence Multiplier [0.0, 1.0].
        """
        vol_scale = float(params.get("VOL_SCALE", 1.0))
        n = len(df)
        
        # Vol scale represents the model's target conviction level.
        # We clip it to [0.05, 1.0] to prevent zero-sizing and over-exposure.
        conf_mult = np.full(n, np.clip(vol_scale, 0.05, 1.0), dtype=np.float64)
        
        return conf_mult
