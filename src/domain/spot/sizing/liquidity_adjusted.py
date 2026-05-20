from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_spot import compute_atr_numpy
from src.domain.spot.sizing.registry import register_sizing


@register_sizing
class LiquidityAdjustedSizing:
    name: ClassVar[str] = "liquidity_adjusted"
    param_space: ClassVar[dict[str, Any]] = {
        "VOL_SCALE": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "LIQUIDITY_REF_NOTIONAL_KRW": {"type": "float", "low": 1e7, "high": 1e8, "step": 5e6},
    }

    def compute(self, df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)
        n = len(close)
        if n == 0:
            return np.array([], dtype=np.float64)
        atr_p = int(params.get("ATR_PERIOD", 14))
        atr = compute_atr_numpy(high, low, close, atr_p)
        risk_per_trade = float(params.get("RISK_PER_TRADE", 0.02))
        max_exp = float(params.get("MAX_EXPOSURE", 1.0))
        vol_scale = float(params.get("VOL_SCALE", 1.0))
        kelly_f = float(params.get("KELLY_FRACTION", 0.5))
        max_part = float(params.get("MAX_PARTICIPATION_RATE", 0.02))
        ref_n = float(params.get("LIQUIDITY_REF_NOTIONAL_KRW", 5e7))
        atr_pct = np.where(close > 1e-12, atr / close, 0.01)
        inv_atr_norm = risk_per_trade * vol_scale / (atr_pct + 1e-9)
        dv = volume * close
        adv_short = pd.Series(dv).ewm(span=24, adjust=False).mean().to_numpy(dtype=np.float64)
        adv_mid = pd.Series(dv).ewm(span=72, adjust=False).mean().to_numpy(dtype=np.float64)
        adv_sm = np.maximum(adv_short, adv_mid)
        max_allowed = adv_sm * max_part
        participation_cap = np.minimum(1.0, max_allowed / (ref_n + 1e-8))
        size = np.clip(inv_atr_norm * participation_cap * kelly_f, 0.0, max_exp)
        return np.where(np.isfinite(size), size, 0.0).astype(np.float64)
