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


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder Average True Range on completed bars."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    tr = pd.concat(
        (high - low, (high - close.shift()).abs(), (low - close.shift()).abs()),
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0.0, np.nan)


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    mult: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """ATR-based SuperTrend trailing-stop line and its uptrend flag.

    ``long_trend`` is True on bars whose trend state is bullish. The recursive
    final-band state is evaluated in a sequential loop (the band state is
    inherently path-dependent); only completed bars at or before the current
    index are read.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if mult <= 0.0:
        raise ValueError(f"mult must be > 0, got {mult}")
    hl2 = (high + low) / 2.0
    atr_series = atr(high, low, close, period)
    basic_upper = hl2 + mult * atr_series
    basic_lower = hl2 - mult * atr_series
    n = len(close)
    if n == 0:
        return pd.Series(dtype="float64"), pd.Series(dtype=bool)
    final_upper = np.empty(n, dtype=np.float64)
    final_lower = np.empty(n, dtype=np.float64)
    line = np.empty(n, dtype=np.float64)
    long_trend = np.empty(n, dtype=bool)
    final_upper[0] = float(basic_upper.iloc[0])
    final_lower[0] = float(basic_lower.iloc[0])
    long_trend[0] = True
    line[0] = float(final_lower[0])
    for i in range(1, n):
        price = float(close.iloc[i])
        if price <= final_upper[i - 1]:
            final_upper[i] = min(float(basic_upper.iloc[i]), final_upper[i - 1])
        else:
            final_upper[i] = float(basic_upper.iloc[i])
        if price >= final_lower[i - 1]:
            final_lower[i] = max(float(basic_lower.iloc[i]), final_lower[i - 1])
        else:
            final_lower[i] = float(basic_lower.iloc[i])
        if price > final_upper[i - 1]:
            long_trend[i] = True
            line[i] = final_lower[i]
        else:
            long_trend[i] = False
            line[i] = final_upper[i]
    index = close.index
    return (
        pd.Series(line, index=index, dtype="float64"),
        pd.Series(long_trend, index=index, dtype=bool),
    )


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    step: float = 0.02,
    max_step: float = 0.2,
) -> pd.Series:
    """Parabolic Stop-And-Reverse line with Wildered acceleration.

    Sequential state (SAR, extreme point, acceleration factor) is path
    dependent, so the series is built in a loop over completed bars; the result
    shares the input index.
    """
    if step <= 0.0:
        raise ValueError(f"step must be > 0, got {step}")
    if max_step < step:
        raise ValueError(
            f"max_step must be >= step, got max_step={max_step} step={step}"
        )
    n = len(high)
    if n == 0:
        return pd.Series(dtype="float64")
    sar = np.empty(n, dtype=np.float64)
    ep = np.empty(n, dtype=np.float64)
    af = np.empty(n, dtype=np.float64)
    long_ = np.empty(n, dtype=bool)
    sar[0] = float(low.iloc[0])
    ep[0] = float(high.iloc[0])
    af[0] = step
    long_[0] = True
    for i in range(1, n):
        hi = float(high.iloc[i])
        lo = float(low.iloc[i])
        if long_[i - 1]:
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            sar[i] = min(sar[i], float(low.iloc[i - 1]))
            if lo < sar[i]:
                long_[i] = False
                sar[i] = ep[i - 1]
                ep[i] = lo
                af[i] = step
            else:
                long_[i] = True
                ep[i] = hi if hi > ep[i - 1] else ep[i - 1]
                af[i] = min(af[i - 1] + step, max_step) if hi > ep[i - 1] else af[i - 1]
        else:
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            sar[i] = max(sar[i], float(high.iloc[i - 1]))
            if hi > sar[i]:
                long_[i] = True
                sar[i] = ep[i - 1]
                ep[i] = hi
                af[i] = step
            else:
                long_[i] = False
                ep[i] = lo if lo < ep[i - 1] else ep[i - 1]
                af[i] = min(af[i - 1] + step, max_step) if lo < ep[i - 1] else af[i - 1]
    return pd.Series(sar, index=high.index, dtype="float64")


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner middle/upper/lower channel around an EMA of typical price."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if mult <= 0.0:
        raise ValueError(f"mult must be > 0, got {mult}")
    typical = (high + low + close) / 3.0
    middle = typical.ewm(span=period, adjust=False).mean()
    atr_series = atr(high, low, close, period)
    upper = middle + mult * atr_series
    lower = middle - mult * atr_series
    return middle, upper, lower


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
