"""Market data retention pruning with non-destructive guards."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.mhs.params import SIGNAL_PANEL_WINDOW_DAYS

MARKET_DATA_MIN_RETENTION_DAYS: int = SIGNAL_PANEL_WINDOW_DAYS + 30
MARKET_DATA_MIN_KEPT_ROWS: int = 24

_logger = logging.getLogger(__name__)


def prune_market_data(
    futures_root: Path, retention_days: int, *, now: pd.Timestamp
) -> dict[str, dict[str, int]]:
    if retention_days < MARKET_DATA_MIN_RETENTION_DAYS:
        raise ValueError(
            f"data_retention_days must be >= {MARKET_DATA_MIN_RETENTION_DAYS}"
        )
    cutoff_ms = int(
        (pd.Timestamp(now).tz_convert("UTC") - pd.Timedelta(days=retention_days)).timestamp()
        * 1000
    )
    result: dict[str, dict[str, int]] = {}
    for rel in ("ohlcv/1h", "markPriceKlines/1h", "funding"):
        d = Path(futures_root) / rel
        files_pruned = 0
        rows_removed = 0
        files_skipped = 0
        if not d.is_dir():
            result[rel] = {
                "files_pruned": 0,
                "rows_removed": 0,
                "files_skipped": 0,
            }
            continue
        for p in sorted(d.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
            except Exception:
                files_skipped += 1
                continue
            if "timestamp" not in df.columns or not pd.api.types.is_integer_dtype(
                df["timestamp"]
            ):
                files_skipped += 1
                continue
            n0 = len(df)
            kept = df[df["timestamp"] >= cutoff_ms]
            if len(kept) == n0:
                continue
            if kept.empty or len(kept) < MARKET_DATA_MIN_KEPT_ROWS:
                files_skipped += 1
                continue
            tmp = p.with_suffix(".prune.tmp.parquet")
            kept.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(p)
            files_pruned += 1
            rows_removed += n0 - len(kept)
        result[rel] = {
            "files_pruned": files_pruned,
            "rows_removed": rows_removed,
            "files_skipped": files_skipped,
        }
    _logger.info(
        "[DATA] stage=prune_market_data retention_days=%d cutoff_ms=%d result=%s",
        retention_days,
        cutoff_ms,
        result,
    )
    return result


def prune_orderbook_history(
    orderbook_dir: Path, retention_days: int, *, now: pd.Timestamp
) -> int:
    if retention_days < 1:
        raise ValueError("orderbook_retention_days must be >= 1")
    cutoff = (
        pd.Timestamp(now).tz_convert("UTC") - pd.Timedelta(days=retention_days)
    ).strftime("%Y%m%d")
    d = Path(orderbook_dir)
    if not d.is_dir():
        return 0
    removed = 0
    for p in sorted(d.glob("live_orderbook_*.parquet")):
        tag = p.stem.removeprefix("live_orderbook_")
        if len(tag) == 8 and tag.isdigit() and tag < cutoff:
            p.unlink()
            removed += 1
    return removed
