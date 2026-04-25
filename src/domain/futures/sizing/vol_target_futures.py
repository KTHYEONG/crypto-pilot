from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.sizing.registry import register_futures_sizing


@register_futures_sizing
class VolTargetFuturesSizing:
    name: ClassVar[str] = "vol_target"
    param_space: ClassVar[Dict[str, Any]] = {
        "TARGET_ANN_VOL": {"type": "float", "low": 0.50, "high": 1.25, "step": 0.25},
        "VOL_LOOKBACK": {"type": "int", "low": 20, "high": 40, "step": 10},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        target_ann_vol = float(params.get("TARGET_ANN_VOL", 0.75))
        lookback = int(params.get("VOL_LOOKBACK", 30))

        close = df["close"].to_numpy(dtype=np.float64)
        if len(close) < lookback:
            return np.ones(len(df), dtype=np.float64) * 0.5

        returns = np.log(close[1:] / close[:-1])
        returns = np.insert(returns, 0, 0.0)

        # Derive bars_per_year from index frequency; fall back to 1h.
        if len(df) > 1 and isinstance(df.index, pd.DatetimeIndex):
            median_td = (
                pd.Series(df.index).diff().dropna().dt.total_seconds().median()
            )
            bars_per_year = int(365 * 24 * 3600 / max(median_td, 1))
        else:
            bars_per_year = 365 * 24

        rolling_std = (
            pd.Series(returns)
            .rolling(window=lookback, min_periods=lookback // 2)
            .std()
            .to_numpy()
        )
        realized_ann_vol = rolling_std * np.sqrt(bars_per_year)

        with np.errstate(divide="ignore", invalid="ignore"):
            size_mult = target_ann_vol / np.maximum(realized_ann_vol, 1e-6)

        size_mult = np.nan_to_num(size_mult, nan=0.5)
        return np.clip(size_mult, 0.1, 1.5).astype(np.float64)
