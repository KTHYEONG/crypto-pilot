from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.contracts import StrategySpec


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
    if spec.min_taker_buy_ratio is not None:
        if "taker_buy_ratio" not in out.columns:
            raise ValueError(
                "min_taker_buy_ratio set but df lacks 'taker_buy_ratio' column"
            )
        ratio = out["taker_buy_ratio"]
        ratio_ok = (
            ratio.notna()
            & ratio.between(0.0, 1.0)
            & (ratio >= spec.min_taker_buy_ratio)
        )
        out["entry_signal"] = out["entry_signal"] & ratio_ok
    return out


def _settled_funding_rates(
    funding_rates: pd.Series,
    bar_index: pd.DatetimeIndex,
) -> pd.Series:
    """Last funding rate settled at or before each completed decision bar.

    A bar with no settled funding event at or before it is ``NaN``, so the
    directional components are disabled there and are never zero-filled. The
    event stream must be UTC, monotonic, finite, and duplicate-free; anything
    else raises ``DataIntegrityError``. A future funding event cannot veto an
    earlier decision because only events at or before the bar close are read.
    """
    if not isinstance(bar_index, pd.DatetimeIndex) or len(bar_index) < 1:
        raise DataIntegrityError("bar index must be a non-empty DatetimeIndex")
    if funding_rates is None or len(funding_rates) == 0:
        raise DataIntegrityError("funding_rates must be a non-empty series")
    ts = pd.DatetimeIndex(
        pd.to_datetime(funding_rates.index, utc=True, errors="coerce")
    )
    if ts.hasnans:
        raise DataIntegrityError("funding_rates index must contain datetimes")
    rates = pd.to_numeric(funding_rates, errors="coerce")
    if rates.isna().any():
        raise DataIntegrityError("funding_rates must be finite")
    if not ts.is_monotonic_increasing:
        raise DataIntegrityError("funding_rates must be monotonic in time")
    if ts.has_duplicates:
        raise DataIntegrityError("funding_rates must be duplicate-free in time")

    series = pd.Series(rates.to_numpy(dtype=np.float64), index=ts).sort_index()
    pos = series.index.searchsorted(bar_index, side="right") - 1
    settled = pd.Series(np.nan, index=bar_index, dtype=np.float64)
    valid = pos >= 0
    if valid.any():
        settled.iloc[np.flatnonzero(valid)] = series.to_numpy(dtype=np.float64)[
            pos[valid]
        ]
    return settled


def generate_directional_funding_signals(
    df: pd.DataFrame,
    spec: StrategySpec,
    funding_rates: pd.Series,
) -> pd.DataFrame:
    """Build mutually exclusive long/short Donchian signals gated by funding.

    ``long_entry_signal`` requires the baseline long breakout with a settled
    funding rate ``<= 0``; ``short_entry_signal`` requires the mirror breakdown
    (``close < entry-period lower Donchian`` and ``close < EMA``) with a settled
    funding rate ``>= 0``. Only the last funding event settled at or before the
    completed decision bar may be read; a missing settled event disables the
    components for that bar and never creates an entry. The existing
    ``generate_signals`` surface is preserved unchanged.
    """
    out = generate_signals(df, spec).copy()
    out["entry_lower"] = donchian_lower(out["low"], spec.entry_period)
    out["exit_upper"] = donchian_upper(out["high"], spec.exit_period)
    settled = _settled_funding_rates(funding_rates, out.index)
    out["settled_funding"] = settled
    mirror_breakdown = (
        (out["close"] < out["entry_lower"])
        & (out["close"] < out["ema"])
        & out["atr"].notna()
        & (out["atr"] > 0)
    )
    out["long_entry_signal"] = (
        out["entry_signal"].astype(bool)
        & settled.notna()
        & (settled <= 0.0)
    )
    out["short_entry_signal"] = (
        mirror_breakdown
        & settled.notna()
        & (settled >= 0.0)
    )
    return out
