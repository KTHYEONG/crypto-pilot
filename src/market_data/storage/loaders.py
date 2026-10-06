"""Published-funding parquet loader shared by live signal refresh and MHS marks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError


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
