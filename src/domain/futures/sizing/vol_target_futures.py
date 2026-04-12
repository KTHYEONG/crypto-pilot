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
        # Institutional target vol range: 20% to 40% annualized
        "TARGET_ANN_VOL": {"type": "float", "low": 0.20, "high": 0.40, "step": 0.05},
        "VOL_LOOKBACK": {"type": "int", "low": 20, "high": 60, "step": 10},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """
        [REDESIGN] Actual Volatility Targeting.
        size_mult = target_vol / realized_vol
        Annualized for consistent risk across timeframes.
        """
        target_ann_vol = float(params.get("TARGET_ANN_VOL", 0.30))
        lookback = int(params.get("VOL_LOOKBACK", 30))
        
        # Calculate log returns
        close = df["close"].to_numpy(dtype=np.float64)
        if len(close) < lookback:
            return np.ones(len(df), dtype=np.float64) * 0.5
            
        returns = np.log(close[1:] / close[:-1])
        returns = np.insert(returns, 0, 0.0) # Match length
        
        # Rolling realized volatility (annualized)
        # 4H data -> 6 bars per day -> 2190 bars per year
        bars_per_year = 365 * 6
        
        rolling_std = pd.Series(returns).rolling(window=lookback, min_periods=lookback//2).std().to_numpy()
        realized_ann_vol = rolling_std * np.sqrt(bars_per_year)
        
        # Avoid division by zero and extreme values
        # We cap the multiplier to [0.1, 1.5] to prevent over-betting in low-vol and death-spirals in high-vol
        with np.errstate(divide="ignore", invalid="ignore"):
            size_mult = target_ann_vol / np.maximum(realized_ann_vol, 1e-6)
            
        size_mult = np.nan_to_num(size_mult, nan=0.5)
        return np.clip(size_mult, 0.1, 1.5).astype(np.float64)
