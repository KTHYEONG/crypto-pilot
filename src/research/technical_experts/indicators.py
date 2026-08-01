"""Vectorized completed-bar indicator primitives for the frozen candidate families.

Every function consumes only observations at or before the current bar; none
shifts a value into the future and Ichimoku's cloud is deliberately not
forward-shifted, so a decision at a completed bar never reads a future index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average computed causally on completed closes."""
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    return close.ewm(span=span, adjust=False).mean()


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average computed causally on completed closes."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return close.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI on completed closes; a zero loss streak is a NaN zone."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def macd_histogram(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD line minus its signal line (the histogram)."""
    if not slow > fast >= 1:
        raise ValueError(f"requires slow > fast >= 1, got fast={fast} slow={slow}")
    if signal < 1:
        raise ValueError(f"signal must be >= 1, got {signal}")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def adx_di(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder (ADX, +DI, -DI) directional movement on completed bars."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0.0), 0.0)
    minus_dm = down.where((down > up) & (down > 0.0), 0.0)
    tr = pd.concat(
        (high - low, (high - close.shift()).abs(), (low - close.shift()).abs()),
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_series = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx_series, plus_di, minus_di


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Slow stochastic %K and %D on completed bars."""
    if min(k_period, d_period, smooth) < 1:
        raise ValueError("k_period, d_period and smooth must all be >= 1")
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    raw_k = 100.0 * (close - lowest) / (highest - lowest).replace(0.0, np.nan)
    slow_k = raw_k.rolling(smooth).mean()
    slow_d = slow_k.rolling(d_period).mean()
    return slow_k, slow_d


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index on completed typical prices."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    typical = (high + low + close) / 3.0
    mean = typical.rolling(period).mean()
    deviation = typical.rolling(period).apply(
        lambda values: float(np.mean(np.abs(values - float(np.mean(values))))), raw=True,
    )
    return (typical - mean) / (0.015 * deviation).replace(0.0, np.nan)


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index on completed typical-price flow."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    positive = money_flow.where(typical.diff() > 0.0, 0.0).rolling(period).sum()
    negative = money_flow.where(typical.diff() < 0.0, 0.0).rolling(period).sum()
    return 100.0 - 100.0 / (1.0 + positive / negative.replace(0.0, np.nan))


def bollinger(
    close: pd.Series,
    period: int = 20,
    mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger middle/upper/lower and the raw bandwidth series."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if mult <= 0.0:
        raise ValueError(f"mult must be > 0, got {mult}")
    middle = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = middle + mult * std
    lower = middle - mult * std
    bandwidth = (upper - lower) / middle.replace(0.0, np.nan)
    return middle, upper, lower, bandwidth


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    span: int = 52,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Conversion/base lines and the current (non-forward) 52-bar cloud edge."""
    if min(tenkan, kijun, span) < 1:
        raise ValueError("tenkan, kijun and span must all be >= 1")
    conversion = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    base = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    span_b = (high.rolling(span).max() + low.rolling(span).min()) / 2.0
    return conversion, base, span_b


def _check_contract() -> None:
    """Executable assertions locking the causal indicator surface."""
    index = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    base = np.linspace(0.0, 50.0, 300)
    close = pd.Series(
        100.0 + base + 5.0 * np.sin(np.arange(300) / 5.0), index=index, dtype="float64",
    )
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(np.full(300, 1000.0), index=index, dtype="float64")

    assert float(ema(close, 20).iloc[-1]) > 100.0
    assert float(sma(close, 20).iloc[-1]) > 100.0
    assert np.isfinite(float(rsi(close, 14).iloc[-1]))
    assert np.isfinite(float(macd_histogram(close, 12, 26, 9).iloc[-1]))
    adx_series, plus_di, minus_di = adx_di(high, low, close, 14)
    assert np.isfinite(float(adx_series.iloc[-1]))
    assert np.isfinite(float(plus_di.iloc[-1]))
    assert np.isfinite(float(minus_di.iloc[-1]))
    slow_k, slow_d = stochastic(high, low, close, 14, 3, 3)
    assert np.isfinite(float(slow_k.iloc[-1]))
    assert np.isfinite(float(slow_d.iloc[-1]))
    assert np.isfinite(float(cci(high, low, close, 20).iloc[-1]))
    assert np.isfinite(float(mfi(high, low, close, volume, 14).iloc[-1]))
    _middle, upper, lower, bandwidth = bollinger(close, 20, 2.0)
    assert np.isfinite(float(bandwidth.iloc[-1]))
    assert upper.iloc[-1] > lower.iloc[-1]
    conversion, _base, span_b = ichimoku(high, low, 9, 26, 52)
    assert np.isfinite(float(span_b.iloc[-1]))
    assert np.isfinite(float(conversion.iloc[-1]))


_check_contract()
