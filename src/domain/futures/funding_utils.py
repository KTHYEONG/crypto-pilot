"""
Merge per-symbol funding rate parquet (8h Binance schedule) into OHLCV DataFrames
so BacktestEngineFast can apply causal, bar-aligned funding fees.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


def _safe_symbol(symbol: str) -> str:
    """Convert symbol to parquet filename segment (e.g. BTC/USDT -> BTC_USDT)."""
    return symbol.replace("/", "_")


def _infer_bar_interval_ms(df: pd.DataFrame) -> int:
    """Infer a stable bar interval from sorted OHLCV timestamps."""
    if "timestamp" not in df.columns or len(df) < 2:
        return 0
    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().astype("int64").to_numpy()
    if ts.size < 2:
        return 0
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0
    return int(np.median(diffs))


def merge_funding_into_ohlcv(
    symbol: str,
    df: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Merge funding information from {data_dir}/{symbol}_funding.parquet into the
    given OHLCV DataFrame.

    Two views are attached:
    - funding_rate: last settled rate at or before bar open (legacy compatibility)
    - funding_event_count / funding_rate_sum: exact funding events that occur
      within the bar interval [bar_open, bar_open + bar_interval)

    Parquet expected columns: timestamp (ms), funding_rate [, datetime].
    If file is missing or empty, returns df unchanged (engine uses constant fallback).

    Returns a copy of df with funding columns added when file exists.
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

    bar_interval_ms = _infer_bar_interval_ms(out)

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

    if bar_interval_ms > 0:
        bar_ts_np = out_sorted["timestamp"].to_numpy(dtype="int64")
        bar_end_np = bar_ts_np + int(bar_interval_ms)
        fr_ts_np = fr_df["timestamp"].to_numpy(dtype="int64")
        fr_rate_np = fr_df["funding_rate"].to_numpy(dtype=np.float64)

        left_idx = np.searchsorted(fr_ts_np, bar_ts_np, side="left")
        right_idx = np.searchsorted(fr_ts_np, bar_end_np, side="left")
        event_counts = (right_idx - left_idx).astype(np.int32)

        rate_prefix = np.concatenate(([0.0], np.cumsum(fr_rate_np, dtype=np.float64)))
        rate_sums = rate_prefix[right_idx] - rate_prefix[left_idx]

        ordered_counts = np.zeros(len(out), dtype=np.int32)
        ordered_rate_sums = np.zeros(len(out), dtype=np.float64)
        ordered_counts[merged["_orig_idx"].to_numpy(dtype=np.int64)] = event_counts
        ordered_rate_sums[merged["_orig_idx"].to_numpy(dtype=np.int64)] = rate_sums
        out["funding_event_count"] = ordered_counts
        out["funding_rate_sum"] = ordered_rate_sums
    else:
        out["funding_event_count"] = 0
        out["funding_rate_sum"] = 0.0

    out.drop(columns=["_orig_idx"], inplace=True)
    return out
