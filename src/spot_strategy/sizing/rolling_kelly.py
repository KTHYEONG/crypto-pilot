from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.spot_strategy.sizing.registry import register_sizing


@register_sizing
class RollingKellySizing:
    name: ClassVar[str] = "rolling_kelly"
    param_space: ClassVar[Dict[str, Any]] = {
        "KELLY_WINDOW": {"type": "int", "low": 30, "high": 120, "step": 10},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        window = int(max(5, params.get("KELLY_WINDOW", 60)))
        kelly_frac = float(params.get("KELLY_FRACTION", 0.5))
        max_exp = float(params.get("MAX_EXPOSURE", 1.0))
        r = pd.Series(close).pct_change().to_numpy(dtype=np.float64)
        s = pd.Series(r)
        mu = s.rolling(window, min_periods=window).mean().to_numpy(dtype=np.float64)
        var = s.rolling(window, min_periods=window).var().to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(var > 1e-12, (mu / var) * kelly_frac, 0.0)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        entry_mask = (
            df["long_entry_signal"].to_numpy(dtype=np.float64)
            if "long_entry_signal" in df.columns
            else np.ones(len(f), dtype=np.float64)
        )
        floored = np.where(entry_mask > 0, np.maximum(f, 0.0), f)
        return np.clip(floored, 0.0, max_exp).astype(np.float64)
