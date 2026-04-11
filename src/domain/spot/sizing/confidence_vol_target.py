"""Vol-target sizing scaled by normalized slot_rank_score (tmp.md confidence_vol_target)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_spot import compute_atr_numpy
from src.domain.spot.sizing.registry import register_sizing


@register_sizing
class ConfidenceVolTargetSizing:
    name: ClassVar[str] = "confidence_vol_target"
    param_space: ClassVar[Dict[str, Any]] = {
        "VOL_SCALE": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "RANK_CONF_WIDTH": {"type": "float", "low": 2.0, "high": 8.0, "step": 0.5},
    }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        atr_p = int(params.get("ATR_PERIOD", 14))
        atr = compute_atr_numpy(high, low, close, atr_p)
        risk_per_trade = float(params.get("RISK_PER_TRADE", 0.02))
        max_exp = float(params.get("MAX_EXPOSURE", 1.0))
        vol_scale = float(params.get("VOL_SCALE", 1.0))
        width = float(params.get("RANK_CONF_WIDTH", 4.0))
        width = max(width, 0.5)
        atr_pct = np.where(close > 1e-12, atr / close, 0.01)
        base = np.clip(risk_per_trade * vol_scale / (atr_pct + 1e-9), 0.0, max_exp)
        st = str(params.get("SIGNAL_TYPE", "")).upper()
        if st == "RS_MOMENTUM":
            med_atr = (
                pd.Series(atr_pct)
                .rolling(max(32, atr_p * 3), min_periods=max(8, atr_p))
                .median()
                .to_numpy(dtype=np.float64)
            )
            med_atr = np.where(
                np.isfinite(med_atr) & (med_atr > 1e-12), med_atr, np.median(atr_pct) + 1e-12
            )
            compress = np.clip(atr_pct / med_atr, 0.45, 1.0)
            base = np.clip(base * compress, 0.0, max_exp * 0.92)
        if "slot_rank_score" not in df.columns:
            return base.astype(np.float64)
        rs = df["slot_rank_score"].to_numpy(dtype=np.float64)
        rs = np.nan_to_num(rs, nan=0.0, posinf=0.0, neginf=0.0)
        w = rolling_mad_norm(rs, window=max(16, atr_p * 2))
        conf = 1.0 + np.clip(w, -1.0, 1.0) * (0.5 / width)
        conf = np.clip(conf, 0.65, 1.35)
        out = np.clip(base * conf, 0.0, max_exp)
        return out.astype(np.float64)


def rolling_mad_norm(x: np.ndarray, *, window: int) -> np.ndarray:
    """Causal robust z-score using rolling median and MAD."""
    n = x.size
    if n == 0:
        return x
    s = pd.Series(x)
    med = s.rolling(window, min_periods=max(3, window // 4)).median().to_numpy(dtype=np.float64)
    dev = np.abs(x - med)
    mad = (
        pd.Series(dev)
        .rolling(window, min_periods=max(3, window // 4))
        .median()
        .to_numpy(dtype=np.float64)
    )
    scale = np.where(mad > 1e-12, 1.4826 * mad, 1.0)
    z = (x - med) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
