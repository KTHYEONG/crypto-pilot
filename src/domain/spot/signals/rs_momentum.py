"""Relative strength momentum vs BTC/ETH anchor basket (vol-adjusted return spread, z-scored)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.spot.signals.base import SignalOutput
from src.domain.spot.signals.registry import register_signal


def _vol_adj_momentum(close: np.ndarray, period: int) -> np.ndarray:
    ret = pd.Series(close).pct_change(period)
    vol = pd.Series(close).pct_change().rolling(period).std()
    raw = (ret / (vol + 1e-9)).to_numpy(dtype=np.float64)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


@register_signal
class RSMomentumSignal:
    name: ClassVar[str] = "RS_MOMENTUM"
    param_space: ClassVar[Dict[str, Any]] = {
        "MOMENTUM_PERIOD": {"type": "int", "low": 10, "high": 50, "step": 5},
        "MOMENTUM_LOOKBACK": {"type": "int", "low": 40, "high": 120, "step": 20},
        "MOMENTUM_THRESHOLD": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
        "RS_PEAK_WINDOW": {"type": "int", "low": 5, "high": 25, "step": 5},
        "RS_PEAK_DROP": {"type": "float", "low": 0.35, "high": 0.8, "step": 0.05},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        p = int(params.get("MOMENTUM_PERIOD", 20))
        lookback = int(params.get("MOMENTUM_LOOKBACK", 60))
        threshold = float(params.get("MOMENTUM_THRESHOLD", 1.0))
        peak_win = max(3, int(params.get("RS_PEAK_WINDOW", max(5, p // 2))))
        peak_drop = float(params.get("RS_PEAK_DROP", 0.5))

        close = df["close"].to_numpy(dtype=np.float64)
        sym_m = _vol_adj_momentum(close, p)

        if "btc_close" in df.columns:
            btc = df["btc_close"].to_numpy(dtype=np.float64)
            b_m = _vol_adj_momentum(btc, p)
            if "eth_close" in df.columns:
                eth = df["eth_close"].to_numpy(dtype=np.float64)
                e_m = _vol_adj_momentum(eth, p)
                bench = 0.5 * (b_m + e_m)
            else:
                bench = b_m
        else:
            bench = np.zeros_like(sym_m)

        rel = sym_m - bench
        min_periods = max(10, lookback // 3)
        rolling_mean = (
            pd.Series(rel)
            .rolling(lookback, min_periods=min_periods)
            .mean()
            .to_numpy(dtype=np.float64)
        )
        rolling_std = (
            pd.Series(rel)
            .rolling(lookback, min_periods=min_periods)
            .std()
            .to_numpy(dtype=np.float64)
        )
        z_score = (rel - rolling_mean) / np.where(rolling_std > 1e-9, rolling_std, 1e-9)
        z_score = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)

        entry = z_score > threshold

        z_prev = np.empty_like(z_score)
        z_prev[0] = z_score[0]
        z_prev[1:] = z_score[:-1]
        roll_max_prev = (
            pd.Series(z_score).rolling(peak_win, min_periods=max(3, peak_win // 2)).max().shift(1)
        )
        rmp = np.nan_to_num(
            roll_max_prev.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        peak_fail = z_score < (rmp - peak_drop)
        zero_cross = (z_score < 0.0) & (z_prev >= 0.0)
        sym_kill = (z_score < -threshold) | peak_fail | zero_cross
        kill = sym_kill.astype(np.float64)

        rank = z_score
        return SignalOutput(
            entry_signal=entry.astype(np.bool_),
            kill_signal=kill,
            rank_score=rank.astype(np.float64),
        )
