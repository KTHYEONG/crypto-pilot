from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.spot_strategy.frama_evr_poc import compute_evr_zscore, compute_frama_series
from src.spot_strategy.signals.base import SignalOutput
from src.spot_strategy.signals.registry import register_signal


@register_signal
class FramaEvrSignal:
    name: ClassVar[str] = "FRAMA_EVR"
    param_space: ClassVar[Dict[str, Any]] = {
        "FRAMA_PERIOD": {"type": "int", "low": 8, "high": 32, "step": 4},
        "EVR_WINDOW": {"type": "int", "low": 20, "high": 80, "step": 10},
        "EVR_Z_ENTRY": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "FRAMA_KILL_LAG": {"type": "int", "low": 2, "high": 8, "step": 1},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        frama_p = int(params.get("FRAMA_PERIOD", 16))
        evr_w = int(params.get("EVR_WINDOW", 40))
        evr_z_entry = float(params.get("EVR_Z_ENTRY", 1.0))
        kill_lag = int(params.get("FRAMA_KILL_LAG", 4))

        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        open_ = df["open"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

        frama = compute_frama_series(high, low, close, frama_p)
        frama_prev = np.roll(frama, 1)
        frama_prev[0] = frama[0]
        frama_bull = (close > frama) & (frama > frama_prev)

        evr_z = compute_evr_zscore(open_, high, low, close, volume, evr_w)
        entry = frama_bull & (evr_z > evr_z_entry)

        frama_kill_ref = np.roll(frama, kill_lag)
        frama_kill_ref[:kill_lag] = frama[:kill_lag]
        kill = close < frama_kill_ref

        rank = np.clip(evr_z, -5.0, 5.0)
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill.astype(np.float64),
            rank_score=rank.astype(np.float64),
        )
