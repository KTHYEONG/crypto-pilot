"""
Triple-filter trend quality regime on the reference symbol (causal).

Active when: close > EMA_slow AND EMA_fast > EMA_slow AND ADX > threshold.
Returns {0.0, 1.0} as float64 risk multiplier series.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.spot_strategy.regimes.registry import register_regime
from src.spot_strategy.signals.numpy_ops import compute_adx_numpy, compute_ema_numpy


@register_regime
class TrendQualityRegime:
    name: ClassVar[str] = "TREND_QUALITY"
    param_space: ClassVar[Dict[str, Any]] = {
        "TQ_EMA_FAST": {"type": "int", "low": 20, "high": 50, "step": 5},
        "TQ_EMA_SLOW": {"type": "int", "low": 100, "high": 200, "step": 20},
        "TQ_ADX_PERIOD": {"type": "int", "low": 10, "high": 20, "step": 5},
        "TQ_ADX_THRESHOLD": {"type": "float", "low": 20.0, "high": 35.0, "step": 5.0},
    }

    def compute(self, data_maps: Dict[str, Dict[str, Any]], params: Dict[str, Any]) -> np.ndarray:
        tf = str(params.get("TIMEFRAME", "4h"))
        symbols = sorted(s for s in data_maps if tf in data_maps[s] and data_maps[s][tf] is not None)
        if not symbols:
            raise ValueError("trend_quality regime: empty data_maps")
        ref = "KRW-BTC" if "KRW-BTC" in data_maps and data_maps["KRW-BTC"].get(tf) is not None else symbols[0]
        df: pd.DataFrame = data_maps[ref][tf]
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            return np.array([], dtype=np.float64)

        ema_fast_n = int(params.get("TQ_EMA_FAST", 30))
        ema_slow_n = int(params.get("TQ_EMA_SLOW", 150))
        adx_period = int(params.get("TQ_ADX_PERIOD", 14))
        adx_th = float(params.get("TQ_ADX_THRESHOLD", 25.0))

        ema_f = compute_ema_numpy(close, ema_fast_n)
        ema_s = compute_ema_numpy(close, ema_slow_n)
        adx = compute_adx_numpy(high, low, close, adx_period)
        active = (close > ema_s) & (ema_f > ema_s) & (adx > adx_th)
        return np.where(active, 1.0, 0.0).astype(np.float64)
