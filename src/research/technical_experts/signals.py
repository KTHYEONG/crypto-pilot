"""Causal entry/exit event generation for the frozen technical candidate families.

Each completed 4h bar produces boolean entry/exit events from indicator values
computed only on completed bars at or before the decision index. Ichimoku's
cloud is read at its current index, never forward-shifted, and every event is
stored as one of four boolean columns. A LONG candidate emits only long-side
events and a SHORT candidate only short-side events.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.technical_experts.contracts import TechnicalCandidate
from src.research.technical_experts.indicators import (
    adx_di,
    bollinger,
    cci,
    ema,
    ichimoku,
    macd_histogram,
    mfi,
    rsi,
    stochastic,
)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_EVENT_COLUMNS = ("long_entry", "short_entry", "long_exit", "short_exit")


def _validate_ohlcv_frame(frame: pd.DataFrame, min_history_bars: int) -> pd.DatetimeIndex:
    """Fail-closed integrity gate for the causal OHLCV/volume input frame."""
    if not isinstance(frame, pd.DataFrame):
        raise DataIntegrityError(f"frame must be a DataFrame, got {type(frame).__name__}")
    missing = set(_OHLCV_COLUMNS) - set(frame.columns)
    if missing:
        raise DataIntegrityError(f"frame missing columns: {sorted(missing)}")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise DataIntegrityError("frame index must be a DatetimeIndex")
    if index.tz is None:
        raise DataIntegrityError("frame index must be tz-aware UTC")
    if index.has_duplicates:
        raise DataIntegrityError("frame index must not contain duplicates")
    if not index.is_monotonic_increasing:
        raise DataIntegrityError("frame index must be monotonic increasing")
    if len(frame) < min_history_bars:
        raise DataIntegrityError(
            f"frame must contain at least {min_history_bars} bars, got {len(frame)}"
        )
    for col in _OHLCV_COLUMNS:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise DataIntegrityError(f"frame {col} must be finite")
        if col in ("open", "high", "low", "close") and (values <= 0.0).any():
            raise DataIntegrityError(f"frame {col} must be strictly positive")
    return index


def _ema_alignment_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    fast = ema(close, int(config["fast"]))
    mid = ema(close, int(config["mid"]))
    slow = ema(close, int(config["slow"]))
    return {
        "long_entry": (fast > mid) & (mid > slow) & (close > fast),
        "short_entry": (fast < mid) & (mid < slow) & (close < fast),
        "long_exit": (close < mid) | (fast < mid) | (mid < slow),
        "short_exit": (close > mid) | (fast > mid) | (mid > slow),
    }


def _macd_histogram_regime_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    hist = macd_histogram(
        close, int(config["fast"]), int(config["slow"]), int(config["signal"]),
    )
    slow = ema(close, int(config["regime"]))
    return {
        "long_entry": (hist > 0.0) & (hist.shift() <= 0.0) & (close > slow),
        "short_entry": (hist < 0.0) & (hist.shift() >= 0.0) & (close < slow),
        "long_exit": hist < 0.0,
        "short_exit": hist > 0.0,
    }


def _adx_di_regime_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    adx_series, plus_di, minus_di = adx_di(
        frame["high"], frame["low"], frame["close"], int(config["period"]),
    )
    return {
        "long_entry": (
            (adx_series >= 25.0)
            & (plus_di > minus_di)
            & (plus_di.shift() <= minus_di.shift())
        ),
        "short_entry": (
            (adx_series >= 25.0)
            & (plus_di < minus_di)
            & (plus_di.shift() >= minus_di.shift())
        ),
        "long_exit": (plus_di < minus_di) | (adx_series < 25.0),
        "short_exit": (plus_di > minus_di) | (adx_series < 25.0),
    }


def _ichimoku_cloud_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    conversion, base, span_b = ichimoku(
        frame["high"], frame["low"],
        int(config["tenkan"]), int(config["kijun"]), int(config["span"]),
    )
    close = frame["close"].astype("float64")
    return {
        "long_entry": (close > conversion) & (conversion > base) & (close > span_b),
        "short_entry": (close < conversion) & (conversion < base) & (close < span_b),
        "long_exit": close < base,
        "short_exit": close > base,
    }


def _bb_squeeze_breakout_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    middle, upper, lower, bandwidth = bollinger(
        close, int(config["period"]), float(config["mult"]),
    )
    floor = bandwidth.rolling(int(config["squeeze_window"])).quantile(
        float(config["squeeze_percentile"])
    )
    slow = ema(close, int(config["regime"]))
    return {
        "long_entry": (bandwidth <= floor) & (close > upper) & (close > slow),
        "short_entry": (bandwidth <= floor) & (close < lower) & (close < slow),
        "long_exit": close < middle,
        "short_exit": close > middle,
    }


def _rsi_trend_pullback_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    rsi_series = rsi(close, int(config["period"]))
    lower = float(config["lower"])
    upper = float(config["upper"])
    return {
        "long_entry": (
            (close > slow) & (rsi_series > lower) & (rsi_series.shift() <= lower)
        ),
        "short_entry": (
            (close < slow) & (rsi_series < upper) & (rsi_series.shift() >= upper)
        ),
        "long_exit": rsi_series > upper,
        "short_exit": rsi_series < lower,
    }


def _stochastic_trend_pullback_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    slow_k, slow_d = stochastic(
        frame["high"], frame["low"], close,
        int(config["k_period"]), int(config["d_period"]), int(config["smooth"]),
    )
    lower = float(config["lower"])
    upper = float(config["upper"])
    return {
        "long_entry": (
            (close > slow)
            & (slow_k > slow_d)
            & (slow_k.shift() <= slow_d.shift())
            & (slow_k < lower)
        ),
        "short_entry": (
            (close < slow)
            & (slow_k < slow_d)
            & (slow_k.shift() >= slow_d.shift())
            & (slow_k > upper)
        ),
        "long_exit": slow_k < slow_d,
        "short_exit": slow_k > slow_d,
    }


def _cci_trend_pullback_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    cci_series = cci(frame["high"], frame["low"], close, int(config["period"]))
    lower = float(config["lower"])
    upper = float(config["upper"])
    return {
        "long_entry": (
            (close > slow) & (cci_series > lower) & (cci_series.shift() <= lower)
        ),
        "short_entry": (
            (close < slow) & (cci_series < upper) & (cci_series.shift() >= upper)
        ),
        "long_exit": cci_series < 0.0,
        "short_exit": cci_series > 0.0,
    }


def _mfi_trend_pullback_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    mfi_series = mfi(
        frame["high"], frame["low"], close,
        frame["volume"].astype("float64"), int(config["period"]),
    )
    lower = float(config["lower"])
    upper = float(config["upper"])
    return {
        "long_entry": (
            (close > slow) & (mfi_series > lower) & (mfi_series.shift() <= lower)
        ),
        "short_entry": (
            (close < slow) & (mfi_series < upper) & (mfi_series.shift() >= upper)
        ),
        "long_exit": mfi_series < 50.0,
        "short_exit": mfi_series > 50.0,
    }


_FAMILY_SIGNALS: dict[str, Callable[[pd.DataFrame, Mapping[str, int | float]], dict[str, pd.Series]]] = {
    "ema_alignment": _ema_alignment_conditions,
    "macd_histogram_regime": _macd_histogram_regime_conditions,
    "adx_di_regime": _adx_di_regime_conditions,
    "ichimoku_cloud": _ichimoku_cloud_conditions,
    "bb_squeeze_breakout": _bb_squeeze_breakout_conditions,
    "rsi_trend_pullback": _rsi_trend_pullback_conditions,
    "stochastic_trend_pullback": _stochastic_trend_pullback_conditions,
    "cci_trend_pullback": _cci_trend_pullback_conditions,
    "mfi_trend_pullback": _mfi_trend_pullback_conditions,
}


def generate_signal_events(
    frame: pd.DataFrame,
    candidate: TechnicalCandidate,
) -> pd.DataFrame:
    """Return indexed boolean entry/exit events for one directional candidate.

    The returned frame shares the input's index and carries exactly
    ``long_entry``, ``short_entry``, ``long_exit`` and ``short_exit`` boolean
    columns. A LONG candidate's events contain only the long entry/exit of its
    family; the opposite side is always False, so the candidate creates only its
    stated entry side. Missing, non-finite, non-UTC or non-monotonic OHLCV/
    volume, or an insufficient history, fails closed.
    """
    grid = _validate_ohlcv_frame(frame, candidate.min_history_bars)
    try:
        family_fn = _FAMILY_SIGNALS[candidate.family]
    except KeyError as exc:
        raise ValueError(f"unknown technical family '{candidate.family}'") from exc
    conditions = family_fn(frame, candidate.config)

    long_side = candidate.side == "LONG"
    events = pd.DataFrame(
        {
            "long_entry": conditions["long_entry"] & long_side,
            "short_entry": conditions["short_entry"] & (not long_side),
            "long_exit": conditions["long_exit"] & long_side,
            "short_exit": conditions["short_exit"] & (not long_side),
        },
        index=grid,
    )
    assert list(events.columns) == list(_EVENT_COLUMNS)
    return events


def _check_contract() -> None:
    """Executable assertions locking the family registry and event surface."""
    assert set(_FAMILY_SIGNALS) == {
        "ema_alignment",
        "macd_histogram_regime",
        "adx_di_regime",
        "ichimoku_cloud",
        "bb_squeeze_breakout",
        "rsi_trend_pullback",
        "stochastic_trend_pullback",
        "cci_trend_pullback",
        "mfi_trend_pullback",
    }
    assert _EVENT_COLUMNS == ("long_entry", "short_entry", "long_exit", "short_exit")


_check_contract()
