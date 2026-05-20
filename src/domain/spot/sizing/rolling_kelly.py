from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.domain.spot.sizing.registry import register_sizing


@register_sizing
class RollingKellySizing:
    name: ClassVar[str] = "rolling_kelly"
    param_space: ClassVar[dict[str, Any]] = {
        "KELLY_WINDOW": {"type": "int", "low": 30, "high": 120, "step": 10},
    }

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        window = int(max(10, params.get("KELLY_WINDOW", 60)))
        # Fractional Kelly: Default to Quarter Kelly (0.25) to mitigate estimation error.
        kelly_frac = float(params.get("KELLY_FRACTION", 0.25))
        # Hard Cap: 30% individual coin limit for wealth preservation.
        max_exp = float(min(params.get("MAX_EXPOSURE", 0.3), 0.3))

        r_raw = pd.Series(close).pct_change().fillna(0).to_numpy(dtype=np.float64)
        entry_mask = (
            df["long_entry_signal"].to_numpy(dtype=np.float64)
            if "long_entry_signal" in df.columns
            else np.ones(n, dtype=np.float64)
        )
        r_conditional = pd.Series(np.where(entry_mask > 0.5, r_raw, np.nan), dtype=np.float64)
        min_periods = max(5, window // 6)
        mu = (
            r_conditional.ewm(span=window, min_periods=min_periods)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        var = (
            r_conditional.ewm(span=window, min_periods=min_periods)
            .var()
            .fillna(1e-8)
            .to_numpy(dtype=np.float64)
        )

        # Regularization to prevent division by zero in flat periods.
        eps = 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(var > eps, (mu / var) * kelly_frac, 0.0)

        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        # Apply fractional size only on long signals.
        floored = np.where(entry_mask > 0.5, np.maximum(f, 0.0), 0.0)
        return np.clip(floored, 0.0, max_exp).astype(np.float64)
