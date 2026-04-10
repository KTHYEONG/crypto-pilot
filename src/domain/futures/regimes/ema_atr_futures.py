"""
EMA x ATR volatility-trend regime for USDT perpetuals: directional size multipliers in [0, 1].
Causal only (no lookahead).
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from src.domain.futures.regimes.registry import register_regime


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
        pd.Series(tr)
        .ewm(span=max(2, int(period)), adjust=False)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    safe_close = np.where(close > 1e-12, close, 1.0)
    return atr / safe_close


def compute_ema_atr_regime_labels(
    df: pd.DataFrame,
    ema_slow: int,
    atr_period: int,
    vol_pct_window: int,
    vol_quantile: float,
) -> np.ndarray:
    """Labels in {0,1,2,3} = 2*trend_bull + vol_high."""
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


def labels_to_long_short_mult(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map EMA-ATR labels to (long_mult, short_mult) per tmp.md."""
    lab = labels.astype(np.int32, copy=False)
    long_mult = np.where(lab == 3, 0.7, np.where(lab == 2, 0.4, 0.0))
    short_mult = np.where(lab == 1, 0.7, np.where(lab == 0, 0.4, 0.0))
    return long_mult.astype(np.float64), short_mult.astype(np.float64)


@register_regime
class EmaAtrFuturesRegime:
    name: ClassVar[str] = "EMA_ATR"
    param_space: ClassVar[Dict[str, Any]] = {
        "EMA_ATR_REGIME_SLOW": {"type": "int", "low": 50, "high": 200, "step": 10},
        "ATR_REGIME_PERIOD": {"type": "int", "low": 10, "high": 30, "step": 5},
        "VOL_PCT_WINDOW": {"type": "int", "low": 40, "high": 120, "step": 10},
        "VOL_QUANTILE": {"type": "float", "low": 0.45, "high": 0.75, "step": 0.05},
    }

    def compute_long_short_mult(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        atr_period = int(params.get("ATR_REGIME_PERIOD", 14))
        labels = compute_ema_atr_regime_labels(
            df,
            ema_slow=int(params.get("EMA_ATR_REGIME_SLOW", 120)),
            atr_period=atr_period,
            vol_pct_window=int(params.get("VOL_PCT_WINDOW", 60)),
            vol_quantile=float(params.get("VOL_QUANTILE", 0.60)),
        )
        return labels_to_long_short_mult(labels)
