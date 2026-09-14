"""Operator-invoked OHLCV repair for quarantined symbols."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError

logger = logging.getLogger("LiveDataRepair")

REPAIR_REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume", "quote_vol", "taker_buy_quote")


@dataclass(frozen=True, slots=True)
class RepairResult:
    symbol: str
    status: str
    moved_to: Path | None
    rows: int


def _validated_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        names = set(pq.read_schema(path).names)
        rows = int(pq.read_metadata(path).num_rows)
    except (OSError, pa.ArrowException):
        return None
    if not set(REPAIR_REQUIRED_COLUMNS) <= names or rows == 0:
        return None
    return rows


def repair_ohlcv_file(symbol: str, *, now: pd.Timestamp, lookback_days: int, futures_root: Path, collector: Any, timeframe: str = "1h") -> RepairResult:
    path = Path(futures_root) / "ohlcv" / timeframe / f"{symbol}.parquet"
    rows = _validated_rows(path)
    if rows is not None:
        return RepairResult(symbol, "healthy", None, rows)
    moved_to = None
    if path.exists():
        moved_to = path.with_name(f"{path.stem}.corrupt-{now.tz_convert('UTC'):%Y%m%dT%H%M%SZ}")
        path.rename(moved_to)
    collector.ensure_ohlcv_data(symbol, timeframe, str(now - pd.Timedelta(days=lookback_days)), str(now))
    rows = _validated_rows(path)
    if rows is None:
        raise DataIntegrityError(f"ohlcv repair did not produce a valid file: {path}")
    logger.info("[DATA] stage=repair_ohlcv symbol=%s status=repaired moved_to=%s rows=%d", symbol, moved_to, rows)
    return RepairResult(symbol, "repaired", moved_to, rows)
