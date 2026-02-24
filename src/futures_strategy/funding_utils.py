"""
Merge per-symbol funding rate parquet (8h Binance schedule) into OHLCV DataFrames
so BacktestEngineFast can apply bar-aligned funding fees.
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)


def _safe_symbol(symbol: str) -> str:
    """Convert symbol to parquet filename segment (e.g. BTC/USDT -> BTC_USDT)."""
    return symbol.replace("/", "_")


def merge_funding_into_ohlcv(
    symbol: str,
    df: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Merge funding_rate from {data_dir}/{symbol}_funding.parquet into the given
    OHLCV DataFrame using as-of backward merge on timestamp (ms).
    Engine applies funding only at UTC 0/8/16; this assigns the effective rate
    per bar (last settled funding rate at or before bar time).

    Parquet expected columns: timestamp (ms), funding_rate [, datetime].
    If file is missing or empty, returns df unchanged (engine uses constant fallback).

    Returns a copy of df with 'funding_rate' column added when file exists.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    safe = _safe_symbol(symbol)
    path = Path(data_dir) / f"{safe}_funding.parquet"
    if not path.exists():
        return df.copy()

    try:
        fr_df = pd.read_parquet(path)
    except Exception as e:
        _logger.warning("Failed to load funding parquet %s: %s", path, e)
        return df.copy()

    if fr_df.empty or "funding_rate" not in fr_df.columns:
        return df.copy()

    # Normalize timestamp to int64 ms
    ts_col = "timestamp" if "timestamp" in fr_df.columns else "datetime"
    if ts_col not in fr_df.columns:
        _logger.warning("Funding parquet %s has no timestamp/datetime column", path)
        return df.copy()

    fr_ts = fr_df[ts_col]
    if fr_ts.dtype == object or fr_ts.dtype.kind == "U":
        fr_ts = pd.to_numeric(fr_ts, errors="coerce").astype("int64")
    else:
        fr_ts = fr_ts.astype("int64")
    fr_df = fr_df.assign(timestamp=fr_ts)[["timestamp", "funding_rate"]].copy()
    fr_df = fr_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    out = df.copy()
    bar_ts = out["timestamp"] if "timestamp" in out.columns else None
    if bar_ts is None and "datetime" in out.columns:
        bar_ts = pd.to_datetime(out["datetime"]).astype("int64") // 10**6
        out = out.assign(timestamp=bar_ts)
    else:
        bar_ts = out["timestamp"]
    if bar_ts.dtype == object or bar_ts.dtype.kind == "U":
        bar_ts = pd.to_numeric(bar_ts, errors="coerce").astype("int64")
        out["timestamp"] = bar_ts

    out["_orig_idx"] = np.arange(len(out))
    out_sorted = out.sort_values("timestamp")
    merged = pd.merge_asof(
        out_sorted,
        fr_df,
        on="timestamp",
        direction="backward",
    )
    merged = merged.sort_values("_orig_idx")
    out["funding_rate"] = merged["funding_rate"].values
    out.drop(columns=["_orig_idx"], inplace=True)
    return out
