from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.spot_strategy.signals.base import SignalOutput
from src.spot_strategy.signals.registry import register_signal


@register_signal
class RSMomentumSignal:
    name: ClassVar[str] = "RS_MOMENTUM"
    param_space: ClassVar[Dict[str, Any]] = {
        "MOMENTUM_PERIOD": {"type": "int", "low": 10, "high": 50, "step": 5},
        "MOMENTUM_LOOKBACK": {"type": "int", "low": 40, "high": 120, "step": 20},
        "MOMENTUM_THRESHOLD": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        p = int(params.get("MOMENTUM_PERIOD", 20))
        lookback = int(params.get("MOMENTUM_LOOKBACK", 60))
        threshold = float(params.get("MOMENTUM_THRESHOLD", 1.0))
        close = df["close"].to_numpy(dtype=np.float64)
        ret = pd.Series(close).pct_change(p)
        vol = pd.Series(close).pct_change().rolling(p).std()
        momentum_score = (ret / (vol + 1e-9)).to_numpy(dtype=np.float64)
        momentum_score = np.nan_to_num(momentum_score, nan=0.0, posinf=0.0, neginf=0.0)
        min_periods = max(10, lookback // 3)
        rolling_mean = (
            pd.Series(momentum_score)
            .rolling(lookback, min_periods=min_periods)
            .mean()
            .to_numpy(dtype=np.float64)
        )
        rolling_std = (
            pd.Series(momentum_score)
            .rolling(lookback, min_periods=min_periods)
            .std()
            .to_numpy(dtype=np.float64)
        )
        z_score = (momentum_score - rolling_mean) / np.where(rolling_std > 1e-9, rolling_std, 1e-9)
        z_score = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)
        entry = z_score > threshold
        kill = z_score < -threshold
        rank = -z_score
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=rank.astype(np.float64),
        )
