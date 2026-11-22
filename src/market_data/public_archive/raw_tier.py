from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa

from src.common.errors import DataIntegrityError
from src.market_data.public_archive import catalog, fetch, store


@dataclass(frozen=True, slots=True)
class RawStaged:
    path: Path | None  # None when a day has no trade rows: the raw tier is complete without a file
    report: store.RawWriteReport


def stage_raw(dataset: str, day: date, zip_bytes: bytes, sha: str, staging: Path) -> RawStaged | None:
    """Write and verify one day's raw tier inside the month staging dir; nothing is published yet.

    None when the typed trade reader cannot represent the day (the derived path still handles it), so the day
    is stored without a raw tier rather than failing.
    """
    if dataset == "aggTrades":
        path = staging / f"raw-{day.isoformat()}.parquet"
        try:
            report = store.write_raw_agg_trades(
                path, fetch.iter_agg_trade_batches(zip_bytes), source_zip_sha256=sha
            )
        except pa.ArrowInvalid:
            return None
        if report.rows == 0:
            return RawStaged(path=None, report=report)
        store.verify_raw_agg_trades(path, fetch.iter_agg_trade_batches(zip_bytes))
        return RawStaged(path=path, report=report)
    csv_bytes = fetch.read_single_csv_bytes(zip_bytes)
    path = staging / f"raw-{day.isoformat()}.csv.zst"
    report = store.write_raw_csv(path, csv_bytes, source_zip_sha256=sha)
    store.verify_raw_csv(path, csv_bytes)
    return RawStaged(path=path, report=report)


class RawBudget:
    """Per-UTC-day cap on new raw bytes; the spend persists in one status file across runs."""

    def __init__(self, root: Path, now: pd.Timestamp, cap_gib: int | None = None) -> None:
        if cap_gib is not None and cap_gib < 1:
            raise ValueError("raw budget cap must be at least 1 GiB")
        self._path = Path(root) / "_status" / "raw_backfill_budget.json"
        self._day = now.strftime("%Y-%m-%d")
        gib = catalog.RAW_BACKFILL_MAX_GIB_PER_DAY if cap_gib is None else cap_gib
        self._cap = int(gib) * (1 << 30)
        self._spent = 0
        self._guard = threading.Lock()
        if self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise DataIntegrityError("malformed raw backfill budget") from exc
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("utc_day"), str)
                or type(payload.get("bytes")) is not int
                or payload["bytes"] < 0
            ):
                raise DataIntegrityError("malformed raw backfill budget")
            self._spent = payload["bytes"] if payload["utc_day"] == self._day else 0

    def exhausted(self) -> bool:
        with self._guard:
            return self._spent >= self._cap

    def spend(self, nbytes: int) -> None:
        with self._guard:
            self._spent += nbytes
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"utc_day": self._day, "bytes": self._spent}, sort_keys=True), encoding="utf-8")
            tmp.replace(self._path)
