"""VWAP Mean Reversion: Mean reversion around volume-weighted average price."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.signals.base import FuturesSignalOutput
from src.domain.futures.signals.registry import register_futures_signal


@register_futures_signal
class VwapMeanReversionFuturesSignal:
    name: ClassVar[str] = "VWAP_MR"
    param_space: ClassVar[Dict[str, Any]] = {
        "MR_VWAP_WINDOW": {"type": "int", "low": 12, "high": 48, "step": 6},  # bars
        "MR_ZSCORE_THRESHOLD": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
        "MR_EXIT_THRESHOLD": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        window = int(params.get("MR_VWAP_WINDOW", 24))
        z_threshold = float(params.get("MR_ZSCORE_THRESHOLD", 2.0))
        exit_threshold = float(params.get("MR_EXIT_THRESHOLD", 0.5))

        close = df["close"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)
        n = len(close)

        if n < window:
            z = np.zeros(n, dtype=np.float64)
            return FuturesSignalOutput(
                long_entry=np.zeros(n, dtype=np.bool_),
                short_entry=np.zeros(n, dtype=np.bool_),
                kill_long=z,
                kill_short=z,
                rank_score=z,
            )

        # Vectorized VWAP calculation
        pv = close * volume
        sum_pv = pd.Series(pv).rolling(window=window, min_periods=window // 2).sum().to_numpy()
        sum_v = pd.Series(volume).rolling(window=window, min_periods=window // 2).sum().to_numpy()
        
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = sum_pv / np.maximum(sum_v, 1e-12)
        
        # VWAP standard deviation (rolling)
        diff_sq = (close - vwap)**2
        var_vwap = pd.Series(diff_sq).rolling(window=window, min_periods=window // 2).mean().to_numpy()
        std_vwap = np.sqrt(np.maximum(var_vwap, 1e-12))
        
        # Z-score
        with np.errstate(divide="ignore", invalid="ignore"):
            z_score = (close - vwap) / std_vwap
        
        z_score = np.nan_to_num(z_score, nan=0.0)

        # Signals
        long_e = (z_score < -z_threshold)
        short_e = (z_score > z_threshold)
        
        # Kill signals (Exit when returning to mean)
        kill_l = (z_score > -exit_threshold).astype(np.float64)
        kill_s = (z_score < exit_threshold).astype(np.float64)

        return FuturesSignalOutput(
            long_entry=long_e.astype(np.bool_),
            short_entry=short_e.astype(np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=-z_score, # Higher score for oversold
        )
