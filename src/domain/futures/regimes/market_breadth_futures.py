"""
Single-asset breadth proxy: rolling fraction of up closes vs EMA(W) -> long_mult / short_mult.
Causal; multi-symbol basket not required.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_futures import compute_ema_numpy
from src.domain.futures.regimes.registry import register_regime


@register_regime
class MarketBreadthFuturesRegime:
    name: ClassVar[str] = "MARKET_BREADTH"
    param_space: ClassVar[Dict[str, Any]] = {
        "MB_W_SIGNAL": {"type": "int", "low": 5, "high": 20, "step": 5},
        "MB_FLOOR": {"type": "float", "low": 0.35, "high": 0.65, "step": 0.05},
    }

    def compute_long_short_mult(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            z = np.array([], dtype=np.float64)
            return z, z
        w = max(2, int(params.get("MB_W_SIGNAL", params.get("W_SIGNAL", 10))))
        floor = float(np.clip(params.get("MB_FLOOR", 0.5), 0.05, 0.95))
        ema = compute_ema_numpy(close, w)
        bull = (close > ema).astype(np.float64)
        mb = (
            pd.Series(bull)
            .rolling(window=w, min_periods=max(2, w // 2))
            .mean()
            .to_numpy(dtype=np.float64)
        )
        mb = np.nan_to_num(mb, nan=floor, posinf=1.0, neginf=0.0)
        long_mult = np.clip((mb - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
        short_mult = np.clip(((1.0 - mb) - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
        return long_mult.astype(np.float64), short_mult.astype(np.float64)
