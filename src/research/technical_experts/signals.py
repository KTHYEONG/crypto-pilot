"""Causal entry/exit event generation for the frozen technical candidate families.

Each completed 4h bar produces boolean entry/exit events from indicator values
computed only on completed bars at or before the decision index. Ichimoku's
cloud is read at its current index, never forward-shifted, and every event is
stored as one of four boolean columns. A LONG candidate emits only long-side
events and a SHORT candidate only short-side events.

The frozen family condition functions are `_ema_alignment_conditions`,
`_macd_histogram_regime_conditions`, `_adx_di_regime_conditions`,
`_ichimoku_cloud_conditions`, `_bb_squeeze_breakout_conditions`,
`_rsi_trend_pullback_conditions`, `_stochastic_trend_pullback_conditions`,
`_cci_trend_pullback_conditions`, `_mfi_trend_pullback_conditions` plus the
new candidates `_supertrend_conditions, _parabolic_sar_conditions, _keltner_channel_breakout_conditions`;
every one is dispatched from `generate_signal_events()` through `_FAMILY_SIGNALS`
exactly the way `_bb_squeeze_breakout_conditions` is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.technical_experts.contracts import TechnicalCandidate
from src.research.technical_experts.indicators import (
    adx_di,
    aroon,
    atr,
    bollinger,
    cci,
    donchian,
    ema,
    hull_moving_average,
    ichimoku,
    keltner_channel,
    macd_histogram,
    mfi,
    parabolic_sar,
    regression_slope_tstat,
    rsi,
    sma,
    stochastic,
    supertrend,
    vortex,
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


def _supertrend_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    _line, long_trend = supertrend(
        frame["high"], frame["low"], close,
        int(config["period"]), float(config["mult"]),
    )
    return {
        "long_entry": long_trend & ~long_trend.shift(fill_value=False) & (close > slow),
        "short_entry": (~long_trend) & long_trend.shift(fill_value=False) & (close < slow),
        "long_exit": ~long_trend,
        "short_exit": long_trend,
    }


def _parabolic_sar_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    sar = parabolic_sar(
        frame["high"], frame["low"], float(config["step"]), float(config["max_step"]),
    )
    above = close > sar
    return {
        "long_entry": above & ~above.shift(fill_value=False) & (close > slow),
        "short_entry": (~above) & above.shift(fill_value=False) & (close < slow),
        "long_exit": ~above,
        "short_exit": above,
    }


def _keltner_channel_breakout_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    middle, upper, lower = keltner_channel(
        frame["high"], frame["low"], close,
        int(config["period"]), float(config["mult"]),
    )
    return {
        "long_entry": (
            (close > upper) & (close.shift() <= upper.shift()) & (close > slow)
        ),
        "short_entry": (
            (close < lower) & (close.shift() >= lower.shift()) & (close < slow)
        ),
        "long_exit": close < middle,
        "short_exit": close > middle,
    }


def _donchian_breakout_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    entry_upper, entry_lower = donchian(
        frame["high"], frame["low"], int(config["entry"]),
    )
    exit_upper, exit_lower = donchian(
        frame["high"], frame["low"], int(config["exit"]),
    )
    return {
        "long_entry": (close > entry_upper.shift()) & (close > slow),
        "short_entry": (close < entry_lower.shift()) & (close < slow),
        "long_exit": close < exit_lower,
        "short_exit": close > exit_upper,
    }


def _chandelier_trend_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    period = int(config["period"])
    atr_series = atr(frame["high"], frame["low"], close, period)
    longest_high = frame["high"].astype("float64").rolling(period).max()
    longest_low = frame["low"].astype("float64").rolling(period).min()
    stop_long = longest_high - float(config["mult"]) * atr_series
    stop_short = longest_low + float(config["mult"]) * atr_series
    return {
        "long_entry": (
            (close > stop_long) & (close.shift() <= stop_long.shift()) & (close > slow)
        ),
        "short_entry": (
            (close < stop_short) & (close.shift() >= stop_short.shift()) & (close < slow)
        ),
        "long_exit": close < stop_long,
        "short_exit": close > stop_short,
    }


def _aroon_trend_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    aroon_up, aroon_down = aroon(
        frame["high"], frame["low"], int(config["period"]),
    )
    oscillator = aroon_up - aroon_down
    return {
        "long_entry": (
            (oscillator > 0.0) & (oscillator.shift() <= 0.0) & (close > slow)
        ),
        "short_entry": (
            (oscillator < 0.0) & (oscillator.shift() >= 0.0) & (close < slow)
        ),
        "long_exit": oscillator < 0.0,
        "short_exit": oscillator > 0.0,
    }


def _vortex_trend_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    vi_plus, vi_minus = vortex(
        frame["high"], frame["low"], close, int(config["period"]),
    )
    return {
        "long_entry": (
            (vi_plus > vi_minus) & (vi_plus.shift() <= vi_minus.shift()) & (close > slow)
        ),
        "short_entry": (
            (vi_plus < vi_minus) & (vi_plus.shift() >= vi_minus.shift()) & (close < slow)
        ),
        "long_exit": vi_plus < vi_minus,
        "short_exit": vi_plus > vi_minus,
    }


def _hull_moving_average_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    hma = hull_moving_average(close, int(config["period"]))
    return {
        "long_entry": (hma > hma.shift()) & (close > slow) & (close > hma),
        "short_entry": (hma < hma.shift()) & (close < slow) & (close < hma),
        "long_exit": (close < hma) | (hma < hma.shift()),
        "short_exit": (close > hma) | (hma > hma.shift()),
    }


def _regression_slope_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    tstat = regression_slope_tstat(close, int(config["period"]))
    return {
        "long_entry": (tstat > 0.0) & (tstat.shift() <= 0.0) & (close > slow),
        "short_entry": (tstat < 0.0) & (tstat.shift() >= 0.0) & (close < slow),
        "long_exit": tstat < 0.0,
        "short_exit": tstat > 0.0,
    }


def _atr_volatility_breakout_conditions(
    frame: pd.DataFrame, config: Mapping[str, int | float],
) -> dict[str, pd.Series]:
    close = frame["close"].astype("float64")
    slow = ema(close, int(config["regime"]))
    period = int(config["period"])
    atr_series = atr(frame["high"], frame["low"], close, period)
    upper, lower = donchian(frame["high"], frame["low"], period)
    expanded = (upper - lower) > float(config["mult"]) * atr_series
    middle = sma(close, period)
    return {
        "long_entry": (close > upper.shift()) & expanded & (close > slow),
        "short_entry": (close < lower.shift()) & expanded & (close < slow),
        "long_exit": close < middle,
        "short_exit": close > middle,
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
    "supertrend": _supertrend_conditions,
    "parabolic_sar": _parabolic_sar_conditions,
    "keltner_channel_breakout": _keltner_channel_breakout_conditions,
    "donchian_breakout": _donchian_breakout_conditions,
    "chandelier_trend": _chandelier_trend_conditions,
    "aroon_trend": _aroon_trend_conditions,
    "vortex_trend": _vortex_trend_conditions,
    "hull_moving_average": _hull_moving_average_conditions,
    "regression_slope": _regression_slope_conditions,
    "atr_volatility_breakout": _atr_volatility_breakout_conditions,
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
        "supertrend",
        "parabolic_sar",
        "keltner_channel_breakout",
        "donchian_breakout",
        "chandelier_trend",
        "aroon_trend",
        "vortex_trend",
        "hull_moving_average",
        "regression_slope",
        "atr_volatility_breakout",
    }
    assert _EVENT_COLUMNS == ("long_entry", "short_entry", "long_exit", "short_exit")


_check_contract()
