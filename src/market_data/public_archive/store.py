from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import socket
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.parquet_io import write_parquet_atomic


class ArchiveBusyError(Exception):
    """Another process holds the archive root lock."""

    def __init__(self, holder: str = "") -> None:
        self.holder = holder
        super().__init__(f"Another process holds the archive root lock: {holder}")


_held_roots: set[str] = set()
_held_guard = threading.Lock()


def _lock_path(root: Path) -> Path:
    return Path(root) / "_status" / "archive.lock"


def _busy_error(path: Path) -> ArchiveBusyError:
    try:
        holder = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        holder = ""
    return ArchiveBusyError(holder)


@contextlib.contextmanager
def archive_lock(root: Path, *, wait_s: float, holder: str = "") -> Iterator[None]:
    """Hold the exclusive cross-process lock of an archive root; raises ArchiveBusyError after wait_s."""
    if not math.isfinite(wait_s) or wait_s < 0:
        raise ValueError("wait_s must be a finite non-negative number")
    lock_path = _lock_path(root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    with _held_guard:
        if key in _held_roots:
            raise _busy_error(lock_path)
        _held_roots.add(key)
    fd = -1
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if now >= deadline:
                    raise _busy_error(lock_path) from None
                time.sleep(min(0.05, max(0.001, deadline - now)))
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": datetime.now(UTC).isoformat(),
            "command": holder,
        }
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
        os.fsync(fd)
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
                fd = -1
    finally:
        with _held_guard:
            _held_roots.discard(key)
        if fd != -1:
            with contextlib.suppress(OSError):
                os.close(fd)


def sealed_months(
    root: Path, dataset: str, coverage: Mapping[tuple[str, date], CoverageRow]
) -> frozenset[tuple[str, str, str]]:
    """(kind, symbol, YYYYMM) months that coverage proves were written but whose local partition is absent."""
    have: dict[tuple[str, str], bool] = {}
    for (symbol, day), row in coverage.items():
        if row.status != "ok" or row.rows <= 0:
            continue
        have[(symbol, day.strftime("%Y%m"))] = True
    sealed: set[tuple[str, str, str]] = set()
    for symbol, month in have:
        if not partition_path(root, dataset, symbol, month, kind="derived").exists():
            sealed.add(("derived", symbol, month))
        if dataset == "aggTrades" and not partition_path(
            root, dataset, symbol, month, kind="slice"
        ).exists():
            sealed.add(("slice", symbol, month))
    return frozenset(sealed)


@dataclass(frozen=True, slots=True)
class CoverageRow:
    dataset: str
    symbol: str
    day: date
    status: Literal["ok", "absent", "failed"]
    raw_sha256: str | None
    rows: int
    recorded_at: pd.Timestamp
    detail: str | None


def partition_path(
    root: Path,
    dataset: str,
    symbol: str,
    month: str,
    *,
    kind: Literal["derived", "slice"] = "derived",
) -> Path:
    base = dataset if kind == "derived" else f"{dataset}_slice"
    return Path(root) / base / symbol / f"{month}.parquet"


def _sort_key(kind: str) -> str:
    return "transact_time_ms" if kind == "slice" else "timestamp"


def _as_utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _row_month(value: object, kind: str) -> str:
    stamp = pd.Timestamp(cast(int, value), unit="ms", tz="UTC") if kind == "slice" else _as_utc(value)
    result: str = stamp.strftime("%Y%m")
    return result


def _row_day(value: object, kind: str) -> date:
    if kind == "slice":
        day: date = pd.Timestamp(cast(int, value), unit="ms", tz="UTC").date()
        return day
    result: date = _as_utc(value).date()
    return result


def merge_partition(
    root: Path,
    dataset: str,
    symbol: str,
    day: date,
    frame: pd.DataFrame,
    *,
    kind: Literal["derived", "slice"] = "derived",
) -> int:
    """Merge one day into its month partition."""
    month = day.strftime("%Y%m")
    path = partition_path(root, dataset, symbol, month, kind=kind)
    key = _sort_key(kind)
    if len(frame):
        if key not in frame.columns:
            raise DataIntegrityError(f"missing sort column: {key}")
        for value in frame[key].tolist():
            if _row_month(value, kind) != month:
                raise DataIntegrityError("frame row outside partition month")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(path) if path.exists() else None
    if existing is not None and len(existing) and key not in existing.columns:
        raise DataIntegrityError(f"partition missing sort column: {key}")
    if len(frame) == 0 and existing is None:
        return 0
    if existing is not None and len(existing):
        keep = [_row_day(v, kind) != day for v in existing[key].tolist()]
        kept = existing.loc[keep]
    else:
        kept = None
    if frame.empty and existing is not None:
        frame = existing.iloc[:0]
    merged = frame if kept is None or len(kept) == 0 else pd.concat([kept, frame], ignore_index=True)
    if len(merged) and any(_row_month(v, kind) != month for v in merged[key].tolist()):
        raise DataIntegrityError("partition row outside partition month")
    merged = merged.sort_values(key, kind="mergesort").reset_index(drop=True)
    write_parquet_atomic(merged, path, compression="zstd")
    return len(merged)


def _coverage_path(root: Path, dataset: str) -> Path:
    return Path(root) / "_coverage" / f"{dataset}.jsonl"


def _row_to_json(row: CoverageRow) -> str:
    payload = {
        "dataset": row.dataset,
        "day": row.day.isoformat(),
        "detail": row.detail,
        "raw_sha256": row.raw_sha256,
        "recorded_at": _as_utc(row.recorded_at).isoformat(),
        "rows": row.rows,
        "status": row.status,
        "symbol": row.symbol,
    }
    return json.dumps(payload, sort_keys=True)


def append_coverage(root: Path, row: CoverageRow) -> None:
    """Append one coverage record with an exclusive lock."""
    path = _coverage_path(root, row.dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (_row_to_json(row) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line)
        os.fsync(fd)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _parse_coverage_line(line: str) -> CoverageRow:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"malformed coverage line: {line!r}") from exc
    if not isinstance(obj, dict):
        raise DataIntegrityError(f"malformed coverage line: {line!r}")
    try:
        required = {"dataset", "symbol", "day", "status", "raw_sha256", "rows", "recorded_at", "detail"}
        if (
            set(obj) != required
            or obj["status"] not in ("ok", "absent", "failed")
            or type(obj["rows"]) is not int
            or obj["rows"] < 0
            or not isinstance(obj["dataset"], str)
            or not isinstance(obj["symbol"], str)
        ):
            raise ValueError("invalid coverage schema")
        return CoverageRow(
            dataset=str(obj["dataset"]),
            symbol=str(obj["symbol"]),
            day=date.fromisoformat(str(obj["day"])),
            status=obj["status"],
            raw_sha256=None if obj.get("raw_sha256") is None else str(obj["raw_sha256"]),
            rows=int(obj["rows"]),
            recorded_at=_as_utc(obj["recorded_at"]),
            detail=None if obj.get("detail") is None else str(obj.get("detail")),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed coverage line: {line!r}") from exc


def load_coverage(root: Path, dataset: str) -> dict[tuple[str, date], CoverageRow]:
    """Load the last record per (symbol, day)."""
    path = _coverage_path(root, dataset)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    result: dict[tuple[str, date], CoverageRow] = {}
    for raw_line in text.splitlines():
        row = _parse_coverage_line(raw_line)
        if row.dataset != dataset or pd.isna(row.recorded_at):
            raise DataIntegrityError("coverage dataset or timestamp mismatch")
        result[(row.symbol, row.day)] = row
    return result
