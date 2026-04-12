"""Adaptive Volatility Regime: Realized Volatility ratio for stress detection."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.regimes.registry import register_regime


@register_regime
class AdaptiveVolRegime:
    name: ClassVar[str] = "ADAPTIVE_VOL"
    param_space: ClassVar[Dict[str, Any]] = {
        "AV_FAST_WINDOW": {"type": "int", "low": 12, "high": 36, "step": 6},
        "AV_SLOW_WINDOW": {"type": "int", "low": 60, "high": 200, "step": 20},
        "AV_SPIKE_RATIO": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
        "AV_STRESS_MULT": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
    }

    def compute_long_short_mult(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        fast_w = int(params.get("AV_FAST_WINDOW", 24))
        slow_w = int(params.get("AV_SLOW_WINDOW", 120))
        spike_ratio = float(params.get("AV_SPIKE_RATIO", 2.0))
        stress_mult = float(params.get("AV_STRESS_MULT", 0.25))

        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)

        if n < slow_w:
            ones = np.ones(n, dtype=np.float64)
            return ones, ones

        # Calculate log returns
        returns = np.log(close[1:] / close[:-1])
        returns = np.insert(returns, 0, 0.0)
        ret_s = pd.Series(returns)

        # Realized Volatility (unscaled std is enough for ratio)
        rv_fast = ret_s.rolling(window=fast_w, min_periods=fast_w // 2).std().to_numpy()
        rv_slow = ret_s.rolling(window=slow_w, min_periods=slow_w // 2).std().to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            vol_ratio = rv_fast / np.maximum(rv_slow, 1e-12)
        
        vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)

        # If ratio > spike_ratio, it's a stress regime -> reduce size to stress_mult
        # If ratio <= 1.0, it's stable -> full size (1.0)
        # In between, linear interpolation (optional, but let's keep it simple first)
        
        # If ratio > spike_ratio, it's a stress regime -> reduce size to stress_mult
        # If ratio <= 1.0, it's stable -> full size (1.0)
        mult = np.where(vol_ratio >= spike_ratio, stress_mult, 1.0)
        
        # Smoothen the transition: Let's provide a "weak stable" zone
        mult = np.where((vol_ratio > 1.2) & (vol_ratio < spike_ratio), 0.75, mult)

        return mult.astype(np.float64), mult.astype(np.float64)
