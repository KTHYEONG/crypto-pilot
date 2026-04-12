"""Cross-Sectional Momentum Signal: Long leaders, Short laggards."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.signals.base import FuturesSignalOutput
from src.domain.futures.signals.registry import register_futures_signal


@register_futures_signal
class CrossSectionalMomentumFuturesSignal:
    name: ClassVar[str] = "CS_MOMENTUM"
    param_space: ClassVar[Dict[str, Any]] = {
        "CSM_LOOKBACK": {"type": "int", "low": 12, "high": 72, "step": 12},  # bars
        "CSM_LONG_RANK": {"type": "float", "low": 0.6, "high": 0.9, "step": 0.05},
        "CSM_SHORT_RANK": {"type": "float", "low": 0.1, "high": 0.4, "step": 0.05},
        "CSM_RANK_COL": {"type": "categorical", "choices": ["cs_mom_rank"]}, # Internal use
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FuturesSignalOutput:
        """
        Expects 'cs_mom_rank' column injected by objective_futures in multi-symbol mode.
        If missing (single mode), falls back to time-series return rank.
        """
        long_th = float(params.get("CSM_LONG_RANK", 0.75))
        short_th = float(params.get("CSM_SHORT_RANK", 0.25))
        lookback = int(params.get("CSM_LOOKBACK", 24))
        
        # 1. Access rank data
        if "cs_mom_rank" in df.columns:
            # Multi-symbol cross-sectional rank [0, 1]
            rank = df["cs_mom_rank"].to_numpy(dtype=np.float64)
        else:
            # Fallback for single symbol: Time-series momentum percentile (approximate)
            # Use rolling return and calculate its percentile relative to its own history
            ret = df["close"].pct_change(periods=lookback)
            rank = ret.rolling(window=200, min_periods=lookback).rank(pct=True).to_numpy()
            rank = np.nan_to_num(rank, nan=0.5)

        # 2. Signals
        long_e = (rank >= long_th)
        short_e = (rank <= short_th)
        
        # Kill signals: Exit if no longer in extreme ranks
        # Long exit if rank drops below 0.5; Short exit if rank rises above 0.5
        kill_l = (rank < 0.5).astype(np.float64)
        kill_s = (rank > 0.5).astype(np.float64)

        return FuturesSignalOutput(
            long_entry=long_e.astype(np.bool_),
            short_entry=short_e.astype(np.bool_),
            kill_long=kill_l,
            kill_short=kill_s,
            rank_score=rank,
        )
