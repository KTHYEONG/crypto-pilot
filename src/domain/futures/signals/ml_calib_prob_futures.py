"""ML calibrated probability gate: dual Long/Short directional signals.

Design (Q3 decision):
  - long_entry  = ml_calib_prob_long  >= thr   (Long MetaLabeler approval)
  - short_entry = ml_calib_prob_short >= thr   (Short MetaLabeler approval)
  - No ml_side_strength dependency: direction is inherent in each probability.
  - If both exceed threshold → higher probability wins (argmax).
"""

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
        "ENTRY_THRESHOLD": {"type": "float", "low": 0.10, "high": 0.70, "step": 0.05},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        thr = float(params.get("ENTRY_THRESHOLD", 0.60))

        # [Q3] Independent directional probabilities from dual MetaLabeler
        p_long = (
            df["ml_calib_prob_long"].to_numpy(dtype=np.float64)
            if "ml_calib_prob_long" in df.columns
            else (
                # Legacy fallback: single calib prob → Long only
                df["ml_calib_prob"].to_numpy(dtype=np.float64)
                if "ml_calib_prob" in df.columns
                else np.full(len(df), 0.5)
            )
        )
        p_short = (
            df["ml_calib_prob_short"].to_numpy(dtype=np.float64)
            if "ml_calib_prob_short" in df.columns
            else np.full(len(df), 0.5)
        )

        # [Q3] If both exceed threshold, pick higher-probability side (argmax), not suppress both
        direction = p_long >= p_short
        long_e = direction & (p_long >= thr)
        short_e = (~direction) & (p_short >= thr)

        n = len(df)
        kill_l = np.zeros(n, dtype=np.float64)
        kill_s = np.zeros(n, dtype=np.float64)

        # rank_score: long prob - short prob (positive = long bias, negative = short bias)
        rank = np.clip(p_long - p_short, -1.0, 1.0)

        return FuturesSignalOutput(
            long_entry=np.asarray(long_e, dtype=np.bool_),
            short_entry=np.asarray(short_e, dtype=np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=rank,
        )
