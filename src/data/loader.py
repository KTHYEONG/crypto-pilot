from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

_logger = logging.getLogger("DataLoader")


class DataIntegrityError(ValueError):
    pass


def load_ohlcv_4h(
    path: Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
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

    resampled = df.resample("4h", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    source_counts = df.resample("4h", label="left", closed="left").size()
    full_buckets = source_counts[source_counts == 4].index
    resampled = resampled.loc[full_buckets].copy()

    if start is not None:
        start_ts = pd.to_datetime(start, utc=True) if isinstance(start, str) else start
        resampled = resampled[resampled.index >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end, utc=True) if isinstance(end, str) else end
        resampled = resampled[resampled.index <= end_ts]

    resampled.columns = ["open", "high", "low", "close", "volume"]
    resampled.index.name = "ts"

    _logger.info(
        "load_ohlcv_4h path=%s rows=%d start=%s end=%s",
        path, len(resampled),
        resampled.index[0] if not resampled.empty else "N/A",
        resampled.index[-1] if not resampled.empty else "N/A",
    )

    return resampled
