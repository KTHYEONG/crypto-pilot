"""Point-in-time market observation builder for Binance USDT perpetual futures."""

from __future__ import annotations

import logging
from datetime import timedelta  # noqa: F401  # re-exported for callers

import numpy as np
import pandas as pd

from src.domain.futures.universe.contracts import DataConfidence

__all__ = ["build_pit_market_observations"]

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS: float = 1e-12
_BARS_PER_HOUR: int = 1  # 4h bar resolution → 1 bar per 4 hours
_BARS_PER_DAY: int = 6  # 24 / 4 = 6 bars per calendar day
_ANNUALISE_4H: float = float(np.sqrt(6 * 365))  # sqrt(bars_per_day * days_per_year)

_OUTPUT_COLS: list[str] = [
    "instrument_id",
    "metric",
    "observed_at",
    "available_at",
    "value",
    "source",
    "confidence",
]

_SOURCE_KLINES: str = "klines"
_SOURCE_FUNDING: str = "funding"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pit_market_observations(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    availability_lag: pd.Timedelta,
    min_observations: int = 20,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Build point-in-time market observation ledger from klines and funding.

    Args:
        klines: OHLCV data with columns
            [symbol, timestamp (UTC), open, high, low, close, volume, quote_volume].
            ``timestamp`` must be timezone-aware UTC.
        funding: Funding-rate events with columns
            [symbol, funding_time (UTC), funding_rate].
            ``funding_time`` must be timezone-aware UTC.
        availability_lag: Lag applied uniformly to all ``observed_at`` values to
            derive ``available_at``.  Must be non-negative (``>= pd.Timedelta(0)``).
        min_observations: Minimum number of valid daily observations required
            before rolling statistics are non-NaN.  Defaults to 20.
        lookback_days: Rolling window length in calendar days for ADV and
            Amihud medians.  Defaults to 30.

    Returns:
        DataFrame with columns
        ``[instrument_id, metric, observed_at, available_at, value, source,
        confidence]``.  Missing values remain ``NaN``; they are never replaced
        with ``0``.

    Raises:
        ValueError: If ``klines`` contains duplicate or non-monotonic timestamps
            for any symbol, or if ``available_at`` would precede ``observed_at``.
    """
    if availability_lag < pd.Timedelta(0):
        raise ValueError("available_at precedes observed_at")

    rows: list[pd.DataFrame] = []

    # -----------------------------------------------------------------------
    # Process per-symbol klines
    # -----------------------------------------------------------------------
    for symbol, sym_klines in klines.groupby("symbol", sort=False):
        sym_rows = _process_symbol_klines(
            symbol=str(symbol),
            sym_klines=sym_klines,
            availability_lag=availability_lag,
            min_observations=min_observations,
            lookback_days=lookback_days,
        )
        if sym_rows is not None:
            rows.append(sym_rows)

    # -----------------------------------------------------------------------
    # Process funding events
    # -----------------------------------------------------------------------
    funding_rows = _process_funding(
        funding=funding,
        availability_lag=availability_lag,
    )
    if funding_rows is not None:
        rows.append(funding_rows)

    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    result = pd.concat(rows, ignore_index=True)

    # Global available_at >= observed_at guard (catches any residual logic error)
    if (result["available_at"] < result["observed_at"]).any():
        raise ValueError("available_at precedes observed_at")

    return result[_OUTPUT_COLS]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_timestamps(sym_klines: pd.DataFrame, symbol: str) -> None:
    """Raise ValueError if timestamps are not unique and strictly monotonic.

    Args:
        sym_klines: Per-symbol klines sorted candidate.
        symbol: Symbol name for error messaging.

    Raises:
        ValueError: On duplicate or non-monotonic timestamps.
    """
    ts = sym_klines["timestamp"]
    if ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError(f"market timestamps must be unique and monotonic (symbol={symbol!r})")


def _process_symbol_klines(
    *,
    symbol: str,
    sym_klines: pd.DataFrame,
    availability_lag: pd.Timedelta,
    min_observations: int,
    lookback_days: int,
) -> pd.DataFrame | None:
    """Compute all kline-derived observations for one symbol.

    Args:
        symbol: Binance futures symbol string, used as ``instrument_id``.
        sym_klines: Subset of klines DataFrame for this symbol.
        availability_lag: Lag to add to ``observed_at``.
        min_observations: Rolling min_periods for median windows.
        lookback_days: Window size in days for ADV / Amihud medians.

    Returns:
        Long-format DataFrame of observations, or ``None`` if empty.
    """
    sym_klines = sym_klines.sort_values("timestamp").reset_index(drop=True)
    _validate_timestamps(sym_klines, symbol)

    if sym_klines.empty:
        return None

    result_parts: list[pd.DataFrame] = []

    # -- 4h bar volatility (vol30) ------------------------------------------
    vol_rows = _compute_vol30(
        symbol=symbol,
        sym_klines=sym_klines,
        availability_lag=availability_lag,
        min_observations=min_observations,
    )
    if vol_rows is not None:
        result_parts.append(vol_rows)

    # -- Daily aggregates (ADV30, Amihud30) ------------------------------------
    daily_rows = _compute_daily_metrics(
        symbol=symbol,
        sym_klines=sym_klines,
        availability_lag=availability_lag,
        min_observations=min_observations,
        lookback_days=lookback_days,
    )
    if daily_rows is not None:
        result_parts.append(daily_rows)

    if not result_parts:
        return None

    return pd.concat(result_parts, ignore_index=True)


def _compute_vol30(
    *,
    symbol: str,
    sym_klines: pd.DataFrame,
    availability_lag: pd.Timedelta,
    min_observations: int,
) -> pd.DataFrame | None:
    """Compute rolling 30-day annualised volatility from 4h bars.

    Math: ``sigma30_t = std(log(C_i/C_{i-1}) for i in window) * sqrt(6*365)``
    Window: ``min_observations * 6`` bars (min_observations days x 6 bars/day).

    Args:
        symbol: Instrument identifier.
        sym_klines: Per-symbol kline rows with columns ``timestamp``, ``close``.
        availability_lag: Lag for ``available_at``.
        min_observations: Minimum number of valid days (x 6 bars) before
            the window emits a non-NaN value.

    Returns:
        Long-format observation DataFrame, or ``None`` if insufficient data.

    Time Complexity: O(T) with rolling std.
    Space Complexity: O(T).
    """
    close = sym_klines["close"].to_numpy(dtype=np.float64)
    timestamps = sym_klines["timestamp"]

    # log-returns using only finite close prices
    finite_mask = np.isfinite(close)
    log_returns = np.full(len(close), np.nan, dtype=np.float64)
    if finite_mask.sum() >= 2:
        # compute pairwise log-return; NaN where either price is non-finite
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_lr = np.where(
                finite_mask & np.isfinite(prev_close) & (prev_close > 0),
                np.log(close / prev_close),
                np.nan,
            )
        log_returns = raw_lr

    min_periods_bars = min_observations * _BARS_PER_DAY
    lr_series = pd.Series(log_returns, index=timestamps)
    # Rolling std with NaN exclusion; min_periods enforces min_observations days
    vol = lr_series.rolling(window=min_periods_bars, min_periods=min_periods_bars).std(ddof=1).mul(_ANNUALISE_4H)

    # observed_at = bar close timestamp (each 4h bar)
    observed_at = timestamps.reset_index(drop=True)
    available_at = observed_at + availability_lag

    df = pd.DataFrame(
        {
            "instrument_id": symbol,
            "metric": "vol30",
            "observed_at": observed_at,
            "available_at": available_at,
            "value": vol.to_numpy(dtype=np.float64),
            "source": _SOURCE_KLINES,
            "confidence": DataConfidence.OBSERVED.value,
        }
    )
    # Drop rows that are fully NaN to save memory while preserving NaN semantics
    # for partial windows (kept but value=NaN)
    return df


def _compute_daily_metrics(
    *,
    symbol: str,
    sym_klines: pd.DataFrame,
    availability_lag: pd.Timedelta,
    min_observations: int,
    lookback_days: int,
) -> pd.DataFrame | None:
    """Compute ADV30 and Amihud30 daily observations.

    Math:
        V_d   = sum(quote_volume_b) for closed bars b in day d
        A_d   = abs(log(C_d / C_{d-1})) / max(V_d, eps)
        ADV30_d     = median(V_{d-29:d}),   NaN before min_observations valid days
        Amihud30_d  = median(A_{d-29:d}),   NaN before min_observations valid days

    observed_at for daily metrics = start of next day UTC (day+1 00:00 UTC).

    Args:
        symbol: Instrument identifier.
        sym_klines: Per-symbol klines with ``timestamp``, ``close``,
            ``quote_volume`` columns.
        availability_lag: Lag for ``available_at``.
        min_observations: Minimum valid days before rolling median is non-NaN.
        lookback_days: Rolling window length in calendar days.

    Returns:
        Long-format observation DataFrame, or ``None`` if insufficient data.

    Time Complexity: O(T/6 * lookback_days) ≈ O(T).
    Space Complexity: O(T/6).
    """
    kl = sym_klines.copy()
    # Normalise timestamp to UTC-aware for groupby
    ts_col = kl["timestamp"]
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize("UTC")
    kl["_date"] = ts_col.dt.normalize()  # day-floor in UTC

    # Daily last close and total quote_volume
    daily = (
        kl.groupby("_date", sort=True)
        .agg(
            close_last=("close", "last"),
            v_d=("quote_volume", "sum"),
        )
        .reset_index()
    )
    daily.rename(columns={"_date": "date"}, inplace=True)

    if daily.empty:
        return None

    # Amihud daily ratio A_d = |log(C_d/C_{d-1})| / max(V_d, eps)
    prev_close = daily["close_last"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret_daily = np.where(
            daily["close_last"].notna() & prev_close.notna() & (prev_close > 0),
            np.abs(np.log(daily["close_last"].to_numpy(dtype=np.float64) / prev_close.to_numpy(dtype=np.float64))),
            np.nan,
        )
    denom = np.where(daily["v_d"].to_numpy(dtype=np.float64) > 0, daily["v_d"].to_numpy(dtype=np.float64), _EPS)
    a_d = np.where(np.isfinite(log_ret_daily), log_ret_daily / denom, np.nan)
    daily["a_d"] = a_d

    # Rolling median over lookback_days with min_periods=min_observations
    adv30 = daily["v_d"].rolling(window=lookback_days, min_periods=min_observations).median()
    amihud30 = daily["a_d"].rolling(window=lookback_days, min_periods=min_observations).median()

    # observed_at = start of NEXT day UTC (day + 1 00:00 UTC)
    observed_at = daily["date"] + pd.Timedelta(days=1)
    available_at = observed_at + availability_lag

    rows: list[pd.DataFrame] = []
    for metric_name, values in [("adv30", adv30), ("amihud30", amihud30)]:
        df = pd.DataFrame(
            {
                "instrument_id": symbol,
                "metric": metric_name,
                "observed_at": observed_at.reset_index(drop=True),
                "available_at": available_at.reset_index(drop=True),
                "value": values.to_numpy(dtype=np.float64),
                "source": _SOURCE_KLINES,
                "confidence": DataConfidence.OBSERVED.value,
            }
        )
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def _process_funding(
    *,
    funding: pd.DataFrame,
    availability_lag: pd.Timedelta,
) -> pd.DataFrame | None:
    """Convert funding rate events to long-format observation rows.

    Each funding event is stored as a separate observation row.
    observed_at = funding_time (actual application time).

    Args:
        funding: DataFrame with columns
            [symbol, funding_time (UTC), funding_rate].
        availability_lag: Lag for ``available_at``.

    Returns:
        Long-format observation DataFrame, or ``None`` if empty.
    """
    if funding.empty:
        return None

    df = pd.DataFrame(
        {
            "instrument_id": funding["symbol"].astype(str),
            "metric": "funding_rate",
            "observed_at": funding["funding_time"].reset_index(drop=True),
            "available_at": (funding["funding_time"].reset_index(drop=True) + availability_lag),
            "value": funding["funding_rate"].to_numpy(dtype=np.float64),
            "source": _SOURCE_FUNDING,
            "confidence": DataConfidence.OBSERVED.value,
        }
    )
    return df
