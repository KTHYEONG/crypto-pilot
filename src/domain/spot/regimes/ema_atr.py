"""
EMA × ATR volatility-trend regime (EATF): labels 0–3 from trend × vol-high axes.

All computations are causal (no lookahead).
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.spot.regimes.registry import register_regime


def _atr_pct(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int,
) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        hl = float(high[i] - low[i])
        hc = abs(float(high[i] - close[i - 1]))
        lc = abs(float(low[i] - close[i - 1]))
        tr[i] = max(hl, hc, lc)
    atr = (
        pd.Series(tr).ewm(span=max(2, int(period)), adjust=False).mean().to_numpy(dtype=np.float64)
    )
    safe_close = np.where(close > 1e-12, close, 1.0)
    return atr / safe_close


def compute_ema_atr_regime_labels(
    df: pd.DataFrame,
    ema_slow: int,
    atr_period: int,
    vol_pct_window: int = 60,
    vol_quantile: float = 0.60,
) -> np.ndarray:
    """Integer labels in {0,1,2,3}: regime = 2*trend_bull + vol_high."""
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    ema = (
        pd.Series(close)
        .ewm(span=max(2, int(ema_slow)), adjust=False)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    trend_bull = (close > ema).astype(np.int32)

    atr_pct = _atr_pct(close, high, low, atr_period)
    rolling_q = (
        pd.Series(atr_pct)
        .rolling(window=max(2, int(vol_pct_window)), min_periods=1)
        .quantile(float(np.clip(vol_quantile, 0.01, 0.99)))
        .to_numpy(dtype=np.float64)
    )
    vol_high = (atr_pct >= rolling_q).astype(np.int32)

    return (2 * trend_bull + vol_high).astype(np.int32)


@register_regime
class EmaAtrRegime:
    name: ClassVar[str] = "EMA_ATR"
    param_space: ClassVar[Dict[str, Any]] = {
        "EMA_ATR_REGIME_SLOW": {"type": "int", "low": 50, "high": 200, "step": 10},
        "ATR_REGIME_PERIOD": {"type": "int", "low": 10, "high": 30, "step": 5},
        "VOL_PCT_WINDOW": {"type": "int", "low": 40, "high": 120, "step": 10},
        "VOL_QUANTILE": {"type": "float", "low": 0.45, "high": 0.75, "step": 0.05},
        "VOV_WINDOW": {"type": "int", "low": 20, "high": 60, "step": 20},
    }

    def compute(self, data_maps: Dict[str, Dict[str, Any]], params: Dict[str, Any]) -> np.ndarray:
        tf = str(params.get("TIMEFRAME", "4h"))
        symbols = sorted(s for s in data_maps if tf in data_maps[s])
        if not symbols:
            raise ValueError("ema_atr regime: empty data_maps")
        ref = (
            "KRW-BTC"
            if "KRW-BTC" in data_maps and data_maps["KRW-BTC"].get(tf) is not None
            else symbols[0]
        )
        df = data_maps[ref][tf]
        atr_period = int(params.get("ATR_REGIME_PERIOD", 14))
        labels = compute_ema_atr_regime_labels(
            df,
            ema_slow=int(params.get("EMA_ATR_REGIME_SLOW", 200)),
            atr_period=atr_period,
            vol_pct_window=int(params.get("VOL_PCT_WINDOW", 60)),
            vol_quantile=float(params.get("VOL_QUANTILE", 0.60)),
        )
        mult = np.where(labels < 2, 0.0, np.where(labels == 3, 0.5, 1.0))
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        atr_pct = _atr_pct(close, high, low, atr_period)
        vov_w = max(2, int(params.get("VOV_WINDOW", 40)))
        vov = (
            pd.Series(atr_pct).rolling(window=vov_w, min_periods=1).std().to_numpy(dtype=np.float64)
        )
        roll_q = max(vov_w * 3, 60)
        vov_q75 = (
            pd.Series(vov)
            .rolling(window=roll_q, min_periods=vov_w)
            .quantile(0.75)
            .to_numpy(dtype=np.float64)
        )
        high_vov = np.isfinite(vov) & np.isfinite(vov_q75) & (vov >= vov_q75)
        mult = np.where(high_vov, mult * 0.5, mult)
        return mult.astype(np.float64)
