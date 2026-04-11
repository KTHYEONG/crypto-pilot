"""
Funding rate z-score regime: extreme positive favors shorts; extreme negative favors longs.
Uses merged `funding_rate` column (causal rolling stats).
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.regimes.registry import register_regime


def compute_funding_z_mult(
    funding_rate: np.ndarray,
    window: int,
    z_threshold: float,
    neutral_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    w = max(2, int(window))
    s = pd.Series(funding_rate.astype(np.float64))
    mu = s.rolling(window=w, min_periods=max(2, w // 2)).mean()
    sig = s.rolling(window=w, min_periods=max(2, w // 2)).std().replace(0.0, 1e-12)
    z = ((s - mu) / sig).to_numpy(dtype=np.float64)
    zt = float(z_threshold)
    nt = float(neutral_threshold)

    long_mult = np.where(z > zt, 0.5, np.where(z < -zt, 1.0, 1.0))
    short_mult = np.where(z < -zt, 0.5, np.where(z > zt, 1.0, 1.0))

    is_neutral = (z >= -nt) & (z <= nt)
    long_mult = np.where(is_neutral, 0.0, long_mult)
    short_mult = np.where(is_neutral, 0.0, short_mult)

    return long_mult.astype(np.float64), short_mult.astype(np.float64)


@register_regime
class FundingRateRegime:
    name: ClassVar[str] = "FUNDING_RATE"
    param_space: ClassVar[Dict[str, Any]] = {
        "FUNDING_Z_WINDOW": {"type": "int", "low": 20, "high": 60, "step": 10},
        "FUNDING_Z_THRESHOLD": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.5},
        "NEUTRAL_THRESHOLD": {"type": "float", "low": 0.0, "high": 0.8, "step": 0.2},
    }

    def compute_long_short_mult(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        if "funding_rate" not in df.columns:
            ones = np.ones(n, dtype=np.float64)
            return ones, ones
        fr = df["funding_rate"].to_numpy(dtype=np.float64)
        fr = np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)
        return compute_funding_z_mult(
            fr,
            window=int(params.get("FUNDING_Z_WINDOW", 40)),
            z_threshold=float(params.get("FUNDING_Z_THRESHOLD", 1.5)),
            neutral_threshold=float(params.get("NEUTRAL_THRESHOLD", 0.0)),
        )
