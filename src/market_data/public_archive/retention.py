from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.market_data.streams.retention import BackupStatus

if TYPE_CHECKING:
    from src.market_data.public_archive.store import CoverageRow

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
    sealed_after: int = 0


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


def _coverage_dataset(archive_dataset: str) -> str:
    return "aggTrades" if archive_dataset == "aggTrades_slice" else archive_dataset


def _sealed_snapshot(base: Path, coverage: Mapping[str, Mapping[tuple[str, date], CoverageRow]]) -> set[tuple[str, ...]]:
    from src.market_data.public_archive import store as store_mod

    return {
        (dataset, *sealed)
        for dataset, rows in coverage.items()
        for sealed in store_mod.sealed_months(base, dataset, rows)
    }


def _retained_months(
    base: Path, coverage: Mapping[str, Mapping[tuple[str, date], CoverageRow]], now_date: date
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str]]]:
    """(dataset, symbol, YYYYMM) months that still owe a retry, and (dataset, YYYYMM) months a pending job spans.

    Sealing such a month would make the collector skip its missing days forever, so they stay local.
    """
    from src.market_data.public_archive import catalog as catalog_mod
    from src.market_data.public_archive import jobs as jobs_mod

    horizon = catalog_mod.ARCHIVE_RETRY_HORIZON_DAYS
    retry = {
        (dataset, symbol, day.strftime("%Y%m"))
        for dataset, rows in coverage.items()
        for (symbol, day), row in rows.items()
        if row.status == "failed" or (row.status != "ok" and 0 <= (now_date - day).days <= horizon)
    }
    job_months = {
        (job.dataset, month.strftime("%Y%m"))
        for job in jobs_mod.pending_jobs(base)
        for month in pd.date_range(job.start.replace(day=1), job.end, freq="MS")
    }
    return retry, job_months


def prune_backed_up_archive(
    root: Path,
    *,
    status: BackupStatus | None,
    now: pd.Timestamp,
    keep_days: int = 45,
) -> ArchivePruneReport:
    """Delete month partitions provably held by Drive, older than ``keep_days`` and owing no retry or backfill."""
    from src.market_data.public_archive import catalog as catalog_mod
    from src.market_data.public_archive import store as store_mod

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
    base = Path(root)
    coverage = {dataset: store_mod.load_coverage(base, dataset) for dataset in catalog_mod.DEFAULT_DATASETS}
    sealed_before = _sealed_snapshot(base, coverage)
    retry, job_months = _retained_months(base, coverage, now_utc.date())

    def _prunable(dataset: str, symbol: str, path: Path) -> bool:
        ym = _covered_month(path.name)
        if ym is None or ym in protected:
            return False
        if _next_month_start(*ym) + pd.Timedelta(days=keep_days) >= now_utc or not _backed_up(path, started_ns):
            return False
        cov_dataset = _coverage_dataset(dataset)
        return (cov_dataset, symbol, path.stem) not in retry and (cov_dataset, path.stem) not in job_months

    pruned = 0
    kept = 0
    for dataset in _ARCHIVE_DATASETS:
        ds_root = base / dataset
        if not ds_root.is_dir():
            continue
        for path in sorted(ds_root.rglob("*.parquet")):
            rel = path.relative_to(ds_root)
            if len(rel.parts) != 2 or not _prunable(dataset, rel.parts[0], path):
                kept += 1
                continue
            path.unlink()
            pruned += 1
        for symbol_dir in sorted(ds_root.iterdir()):
            if symbol_dir.is_dir() and not any(symbol_dir.iterdir()):
                symbol_dir.rmdir()
    return ArchivePruneReport(
        prune_blocked=False,
        blocked_reason=None,
        pruned_files=pruned,
        kept_files=kept,
        sealed_after=len(_sealed_snapshot(base, coverage) - sealed_before),
    )
