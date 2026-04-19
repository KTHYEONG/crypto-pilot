"""
Merge per-symbol statistics metrics (OI, LSR) from parquet into OHLCV DataFrames.
Used for GP alpha mining and strategy signals.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")

def merge_metrics_into_ohlcv(
    symbol: str,
    df: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Merge metrics (sum_open_interest, count_toptrader_long_short_ratio_accounts)
    into the OHLCV DataFrame.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    safe = _safe_symbol(symbol)
    path = Path(data_dir) / f"{safe}_metrics.parquet"
    
    if not path.exists():
        return out

    try:
        metrics_df = pd.read_parquet(path)
    except Exception as e:
        _logger.warning(f"Failed to load metrics parquet {path}: {e}")
        return out

    if metrics_df.empty:
        return out

    # Ensure timestamp columns are comparable
    if "timestamp" not in metrics_df.columns and "datetime" in metrics_df.columns:
        metrics_df["timestamp"] = pd.to_datetime(metrics_df["datetime"]).view("int64") // 10**6
    
    if "timestamp" not in out.columns and "datetime" in out.columns:
        out["timestamp"] = pd.to_datetime(out["datetime"]).view("int64") // 10**6

    metrics_df = metrics_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    
    # Identify metrics columns to merge
    exclude = ["timestamp", "datetime", "create_time", "symbol"]
    feature_cols = [c for c in metrics_df.columns if c not in exclude]
    
    if not feature_cols:
        return out

    # merge_asof for causal alignment (backward)
    out = out.sort_values("timestamp")
    merged = pd.merge_asof(
        out,
        metrics_df[["timestamp"] + feature_cols],
        on="timestamp",
        direction="backward"
    )
    
    return merged
