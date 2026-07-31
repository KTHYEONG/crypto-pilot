from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_logger = logging.getLogger("DataLoader")


class DataIntegrityError(ValueError):
    pass


def _taker_buy_quote_series(df: pd.DataFrame) -> pd.Series:
    """Quote-notional taker-buy flow per source bar.

    Prefers ``taker_buy_quote``; falls back to ``taker_buy_quote_volume`` only
    where the former is null. Returns NaN where no flow is available.
    """
    if "taker_buy_quote" in df.columns:
        primary = pd.to_numeric(df["taker_buy_quote"], errors="coerce").astype("float64")
        if "taker_buy_quote_volume" in df.columns:
            fallback = pd.to_numeric(df["taker_buy_quote_volume"], errors="coerce").astype("float64")
            return primary.where(primary.notna(), fallback)
        return primary
    if "taker_buy_quote_volume" in df.columns:
        return pd.to_numeric(df["taker_buy_quote_volume"], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def load_ohlcv_1h_as_4h(
    path: Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load a canonical 1h kline parquet and explicitly resample to an exact 4h grid.

    The 1h source is fail-closed validated (tz-aware UTC, strictly monotonic,
    no gaps) before resampling; only 4h buckets with exactly four source bars
    are retained. ``start``/``end`` bound the returned grid.
    """
    df = pd.read_parquet(path)

    if "datetime" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})

    if "timestamp" not in df.columns:
        raise DataIntegrityError("parquet must contain a 'timestamp' column")

    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        raise DataIntegrityError("timestamp column must be numeric")

    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df.index.name = "ts"

    if df.index.tz is None:
        raise DataIntegrityError("index must be tz-aware UTC")

    dupe = df.index.duplicated()
    if dupe.any():
        raise DataIntegrityError(f"duplicate timestamps found: {df.index[dupe].tolist()}")

    if not df.index.is_monotonic_increasing:
        raise DataIntegrityError("index is not monotonic increasing")

    diff = df.index.to_series().diff().dropna()
    expected = pd.Timedelta(hours=1)
    gaps = diff[diff != expected]
    if not gaps.empty:
        first_gap = gaps.index[0]
        raise DataIntegrityError(f"missing 1h bars detected at {first_gap}")

    float_cols = ["open", "high", "low", "close"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    working = df.copy()
    if "quote_vol" in working.columns:
        working["quote_vol"] = pd.to_numeric(working["quote_vol"], errors="coerce").astype("float64")
    else:
        working["quote_vol"] = np.nan
    working["taker_buy_quote"] = _taker_buy_quote_series(working)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_vol": "sum",
        "taker_buy_quote": "sum",
    }
    resampled = working.resample("4h", label="left", closed="left").agg(agg)

    source_counts = working.resample("4h", label="left", closed="left").size()
    full_buckets = source_counts[source_counts == 4].index
    resampled = resampled.loc[full_buckets].copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = resampled["taker_buy_quote"] / resampled["quote_vol"]
    resampled["taker_buy_ratio"] = ratio.where(resampled["quote_vol"] > 0)

    if start is not None:
        start_ts = pd.to_datetime(start, utc=True) if isinstance(start, str) else start
        resampled = resampled[resampled.index >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end, utc=True) if isinstance(end, str) else end
        resampled = resampled[resampled.index <= end_ts]

    resampled = resampled[["open", "high", "low", "close", "volume", "quote_vol", "taker_buy_quote", "taker_buy_ratio"]]
    resampled.index.name = "ts"

    _logger.info(
        "load_ohlcv_1h_as_4h path=%s rows=%d start=%s end=%s",
        path, len(resampled),
        resampled.index[0] if not resampled.empty else "N/A",
        resampled.index[-1] if not resampled.empty else "N/A",
    )

    return resampled


load_ohlcv_4h = load_ohlcv_1h_as_4h  # compatibility alias for non-carry callers


def load_funding_rates(path: str | Path) -> pd.Series:
    """Load a published-funding parquet into a monotonic UTC rate Series."""
    p = Path(path)
    if not p.exists():
        raise DataIntegrityError(f"funding path does not exist: {path}")
    df = pd.read_parquet(p)
    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    else:
        raise DataIntegrityError("funding parquet must contain a 'datetime' or 'timestamp' column")
    if "funding_rate" not in df.columns:
        raise DataIntegrityError("funding parquet must contain a 'funding_rate' column")
    rates = pd.to_numeric(df["funding_rate"], errors="coerce")
    series = pd.Series(rates.to_numpy(dtype="float64"), index=pd.DatetimeIndex(ts))
    return series[series.index.notna()].sort_index()
