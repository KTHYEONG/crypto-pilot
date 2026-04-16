"""ML calibrated probability gate: uses ml_calib_prob / ml_side_strength columns on TF bars."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.signals.base import FuturesSignalOutput
from src.domain.futures.signals.registry import register_futures_signal


@register_futures_signal
class MlCalibProbFuturesSignal:
    name: ClassVar[str] = "ML_CALIB_PROB"
    param_space: ClassVar[Dict[str, Any]] = {
        "ENTRY_THRESHOLD": {"type": "float", "low": 0.7, "high": 0.95, "step": 0.05},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        thr = float(params.get("ENTRY_THRESHOLD", 0.75))
        p = (
            df["ml_calib_prob"].to_numpy(dtype=np.float64)
            if "ml_calib_prob" in df.columns
            else np.full(len(df), 0.5)
        )
        side = (
            df["ml_side_strength"].to_numpy(dtype=np.float64)
            if "ml_side_strength" in df.columns
            else np.zeros(len(df))
        )
        long_e = (p >= thr) & (side > 0.05)
        short_e = (p >= thr) & (side < -0.05)
        n = len(df)
        kill_l = np.zeros(n, dtype=np.float64)
        kill_s = np.zeros(n, dtype=np.float64)
        rank = np.clip((p - 0.5) * 2.0, -1.0, 1.0)
        return FuturesSignalOutput(
            long_entry=np.asarray(long_e, dtype=np.bool_),
            short_entry=np.asarray(short_e, dtype=np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=rank,
        )
