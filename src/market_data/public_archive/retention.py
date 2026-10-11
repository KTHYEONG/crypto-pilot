from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.market_data.streams.retention import BackupStatus

_ARCHIVE_DATASETS: tuple[str, ...] = ("metrics", "bookDepth", "aggTrades", "aggTrades_slice")
_MONTH_RE = re.compile(r"^(\d{4})(\d{2})\.parquet$")
_STATUS_MAX_AGE_S: float = 72 * 3600.0


@dataclass(frozen=True, slots=True)
class ArchivePruneReport:
    """Outcome of one backup-gated archive prune pass."""

    prune_blocked: bool
    blocked_reason: str | None
    pruned_files: int
    kept_files: int


def _now_utc(now: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(now)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _covered_month(name: str) -> tuple[int, int] | None:
    match = _MONTH_RE.match(name)
    if match is None:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    return (year, month) if 1 <= month <= 12 else None


def _next_month_start(year: int, month: int) -> pd.Timestamp:
    if month == 12:
        return pd.Timestamp(year + 1, 1, 1, tz="UTC")
    return pd.Timestamp(year, month + 1, 1, tz="UTC")


def _backed_up(path: Path, started_ns: int) -> bool:
    return path.stat().st_mtime_ns < started_ns


def prune_backed_up_archive(
    root: Path,
    *,
    status: BackupStatus | None,
    now: pd.Timestamp,
    keep_days: int = 45,
) -> ArchivePruneReport:
    """Delete month partitions provably held by Drive and older than ``keep_days``."""
    if keep_days < 31:
        raise ValueError("keep_days must be >= 31")
    now_utc = _now_utc(now)
    if status is None:
        return ArchivePruneReport(prune_blocked=True, blocked_reason="status_missing", pruned_files=0, kept_files=0)
    if (now_utc - status.finished_at).total_seconds() > _STATUS_MAX_AGE_S:
        return ArchivePruneReport(prune_blocked=True, blocked_reason="status_stale", pruned_files=0, kept_files=0)
    prev = now_utc.replace(day=1) - pd.Timedelta(days=1)
    protected = {(now_utc.year, now_utc.month), (prev.year, prev.month)}
    started_ns = int(status.started_at.value)
    pruned = 0
    kept = 0
    base = Path(root)
    for dataset in _ARCHIVE_DATASETS:
        ds_root = base / dataset
        if not ds_root.is_dir():
            continue
        for path in sorted(ds_root.rglob("*.parquet")):
            rel = path.relative_to(ds_root)
            if len(rel.parts) != 2:
                kept += 1
                continue
            ym = _covered_month(path.name)
            if ym is None or ym in protected:
                kept += 1
                continue
            if _next_month_start(*ym) + pd.Timedelta(days=keep_days) >= now_utc:
                kept += 1
                continue
            if not _backed_up(path, started_ns):
                kept += 1
                continue
            path.unlink()
            pruned += 1
        for symbol_dir in sorted(ds_root.iterdir()):
            if not symbol_dir.is_dir():
                continue
            if not any(symbol_dir.iterdir()):
                symbol_dir.rmdir()
    return ArchivePruneReport(prune_blocked=False, blocked_reason=None, pruned_files=pruned, kept_files=kept)
