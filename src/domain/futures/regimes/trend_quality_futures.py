"""Trend quality: separate long vs short quality using EMA stack + ADX (causal)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_futures import compute_adx_numpy, compute_ema_numpy
from src.domain.futures.regimes.registry import register_regime


@register_regime
class TrendQualityFuturesRegime:
    name: ClassVar[str] = "TREND_QUALITY"
    param_space: ClassVar[Dict[str, Any]] = {
        "TQ_EMA_FAST": {"type": "int", "low": 20, "high": 50, "step": 5},
        "TQ_EMA_SLOW": {"type": "int", "low": 100, "high": 200, "step": 20},
        "TQ_ADX_PERIOD": {"type": "int", "low": 10, "high": 20, "step": 5},
        "TQ_ADX_THRESHOLD": {"type": "float", "low": 20.0, "high": 35.0, "step": 5.0},
    }

    def compute_long_short_mult(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            z = np.array([], dtype=np.float64)
            return z, z

        ema_fast_n = int(params.get("TQ_EMA_FAST", 30))
        ema_slow_n = int(params.get("TQ_EMA_SLOW", 150))
        adx_period = int(params.get("TQ_ADX_PERIOD", 14))
        adx_th = float(params.get("TQ_ADX_THRESHOLD", 25.0))

        ema_f = compute_ema_numpy(close, ema_fast_n)
        ema_s = compute_ema_numpy(close, ema_slow_n)
        adx = compute_adx_numpy(high, low, close, adx_period)
        long_ok = (close > ema_s) & (ema_f > ema_s) & (adx > adx_th)
        short_ok = (close < ema_s) & (ema_f < ema_s) & (adx > adx_th)
        
        # [MODIFIED] Use 0.0 instead of 0.5 for poor quality to enforce strict gating
        long_mult = np.where(long_ok, 1.0, 0.0).astype(np.float64)
        short_mult = np.where(short_ok, 1.0, 0.0).astype(np.float64)
        return long_mult, short_mult
