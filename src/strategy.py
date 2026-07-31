from __future__ import annotations

import pandas as pd

from src.config import StrategySpec


def donchian_upper(high: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if len(high) == 0:
        raise ValueError("series must not be empty")
    return high.rolling(period).max().shift(1)


def donchian_lower(low: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if len(low) == 0:
        raise ValueError("series must not be empty")
    return low.rolling(period).min().shift(1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if not (high.index.equals(low.index) and high.index.equals(close.index)):
        raise ValueError("high, low, close must have matching indexes")
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def generate_signals(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")
    min_rows = max(spec.ema_period, spec.entry_period, spec.atr_period) + 1
    if len(df) < min_rows:
        raise ValueError(f"df must have at least {min_rows} rows, got {len(df)}")

    out = df.copy()
    out["upper"] = donchian_upper(out["high"], spec.entry_period)
    out["exit_lower"] = donchian_lower(out["low"], spec.exit_period)
    out["ema"] = out["close"].ewm(span=spec.ema_period, adjust=False).mean()
    out["atr"] = atr(out["high"], out["low"], out["close"], spec.atr_period)
    out["entry_signal"] = (
        (out["close"] > out["upper"])
        & (out["close"] > out["ema"])
        & out["atr"].notna()
        & (out["atr"] > 0)
    )
    return out
