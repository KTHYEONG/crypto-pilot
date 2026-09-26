"""Backup-gated retention: prune only what a successful Drive backup already holds.

The Drive backup is copy-only, so a local day unit is removable once a backup run that started
after the unit's last modification completed successfully. Without such evidence nothing is
pruned and the block is reported, because unbounded local growth is recoverable while deleting
an unbacked-up live-only day is not. Temporary files from interrupted atomic writes are swept
by age.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.market_data.streams.compaction import ARCHIVE_STREAMS

if TYPE_CHECKING:
    from src.market_data.streams.normalizer import NormalizerConfig

_logger = logging.getLogger(__name__)

SNAPSHOT_DATASETS: tuple[str, str] = ("book_ticker", "premium_index")


@dataclass(frozen=True, slots=True)
class BackupStatus:
    """Last successful host backup as written by spec 03's backup script."""

    started_at: pd.Timestamp
    finished_at: pd.Timestamp


def read_backup_status(path: Path) -> BackupStatus | None:
    """Decode ``last_success.json``; ``None`` when absent, undecodable, ``rc != 0`` or timestamps naive."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("rc") != 0:
        return None
    try:
        started = pd.Timestamp(raw["started_at"])
        finished = pd.Timestamp(raw["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if pd.isna(started) or pd.isna(finished) or started.tzinfo is None or finished.tzinfo is None:
        return None
    return BackupStatus(started_at=started.tz_convert("UTC"), finished_at=finished.tz_convert("UTC"))


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """Outcome of one retention pass."""

    prune_blocked: bool
    blocked_reason: str | None
    pruned_files: int
    swept_partials: int
    removed_empty_dirs: int


def _older_than(path: Path, cutoff_ns: int) -> bool:
    try:
        return path.stat().st_mtime_ns < cutoff_ns
    except OSError:
        return False


def sweep_partials(capture_root: Path, liquidations_dir: Path, *, now: pd.Timestamp, max_age_s: float) -> int:
    """Delete abandoned temp files older than ``max_age_s``.

    Covers ``*.partial`` under ``raw/`` and ``.<name>.<pid>.<tid>.tmp`` parquet siblings left by an
    interrupted ``write_parquet_atomic`` under the derived trees.
    """
    now_utc = pd.Timestamp(now, tz="UTC") if pd.Timestamp(now).tzinfo is None else pd.Timestamp(now).tz_convert("UTC")
    cutoff_ns = int(now_utc.value) - int(max_age_s * 1_000_000_000)
    swept = 0
    raw_dir = Path(capture_root) / "raw"
    if raw_dir.is_dir():
        for path in raw_dir.rglob("*.partial"):
            if path.is_file() and _older_than(path, cutoff_ns):
                try:
                    path.unlink()
                    swept += 1
                except OSError:
                    continue
    for tree in (Path(capture_root) / "book_ticker", Path(capture_root) / "premium_index", Path(liquidations_dir)):
        if not tree.is_dir():
            continue
        for path in tree.rglob(".*.tmp"):
            if path.is_file() and _older_than(path, cutoff_ns):
                try:
                    path.unlink()
                    swept += 1
                except OSError:
                    continue
    return swept


def _archive_units(capture_root: Path) -> list[tuple[str, str, Path, Path]]:
    """``(stream, day, archive, manifest)`` pairs currently on disk."""
    units: list[tuple[str, str, Path, Path]] = []
    for stream in ARCHIVE_STREAMS:
        stream_dir = Path(capture_root) / "raw" / "archive" / stream
        if not stream_dir.is_dir():
            continue
        for archive in sorted(stream_dir.glob("*.jsonl.xz")):
            day = archive.name[: -len(".jsonl.xz")]
            if len(day) != 8 or not day.isdigit():
                continue
            units.append((stream, day, archive, stream_dir / f"{day}.manifest.json"))
    return units


def _snapshot_day_dirs(capture_root: Path) -> list[tuple[str, str, Path]]:
    """``(dataset, day, dir)`` derived day partitions currently on disk."""
    units: list[tuple[str, str, Path]] = []
    for dataset in SNAPSHOT_DATASETS:
        base = Path(capture_root) / dataset
        if not base.is_dir():
            continue
        units.extend(
            (dataset, day_dir.name, day_dir)
            for day_dir in sorted(base.iterdir())
            if day_dir.is_dir() and len(day_dir.name) == 8 and day_dir.name.isdigit()
        )
    return units


def _liquidation_day_files(liquidations_dir: Path) -> dict[str, list[Path]]:
    """Group liquidation partitions (hourly and legacy daily) by UTC day."""
    groups: dict[str, list[Path]] = {}
    base = Path(liquidations_dir)
    if not base.is_dir():
        return groups
    for path in sorted(base.glob("liquidations_*.parquet")):
        stem = path.name[: -len(".parquet")]
        rest = stem[len("liquidations_") :]
        day = rest[:8] if len(rest) >= 8 and rest[:8].isdigit() else None
        if day is None:
            continue
        groups.setdefault(day, []).append(path)
    return groups


def _unit_backed_up(paths: list[Path], started_at_ns: int) -> bool:
    """Every file of the unit was last modified before the backup run started."""
    for path in paths:
        try:
            if path.stat().st_mtime_ns >= started_at_ns:
                return False
        except OSError:
            return False
    return True


def _remove_empty_dirs(roots: list[Path]) -> int:
    """Remove empty day directories bottom-up, never the dataset roots themselves."""
    removed = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), reverse=True):
            if path == root or not path.is_dir():
                continue
            try:
                if not any(path.iterdir()):
                    path.rmdir()
                    removed += 1
            except OSError:
                continue
    return removed


def prune_backed_up(
    capture_root: Path,
    liquidations_dir: Path,
    *,
    status: BackupStatus | None,
    config: NormalizerConfig,
    now: pd.Timestamp,
    backup_status_path: Path | None = None,
) -> RetentionReport:
    """Delete local day units that are older than their retention window and provably on Drive.

    The Drive backup is copy-only, so a file is safely removable locally once a backup run that
    started after the file's last modification completed successfully. Without such evidence (no
    status, a stale status, or a non-zero rc), nothing is pruned and the block is reported, because
    unbounded local growth is recoverable while deleting an unbacked-up live-only day is not.

    Args:
        backup_status_path: Lets the report distinguish a missing status file
            (``status_missing``) from a present-but-unreadable one (``status_invalid``).
    """
    now_utc = pd.Timestamp(now, tz="UTC") if pd.Timestamp(now).tzinfo is None else pd.Timestamp(now).tz_convert("UTC")
    today = now_utc.normalize()
    if status is None:
        if backup_status_path is not None and Path(backup_status_path).exists():
            reason = "status_invalid"
        else:
            reason = "status_missing"
        return RetentionReport(
            prune_blocked=True, blocked_reason=reason, pruned_files=0, swept_partials=0, removed_empty_dirs=0
        )
    max_age_ns = int(config.backup_status_max_age_h * 3_600_000_000_000)
    if int(now_utc.value) - int(status.finished_at.value) > max_age_ns:
        return RetentionReport(
            prune_blocked=True, blocked_reason="status_stale", pruned_files=0, swept_partials=0, removed_empty_dirs=0
        )
    started_ns = int(status.started_at.value)
    pruned = 0
    archive_cutoff = today - pd.Timedelta(days=config.raw_archive_local_retention_days)
    for _stream, day, archive, manifest in _archive_units(capture_root):
        if pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}T00:00:00Z") >= archive_cutoff:
            continue
        targets = [archive] + ([manifest] if manifest.is_file() else [])
        if not _unit_backed_up(targets, started_ns):
            continue
        for path in targets:
            try:
                path.unlink()
                pruned += 1
            except OSError:
                break
    parquet_cutoff = today - pd.Timedelta(days=config.parquet_local_retention_days)
    for _, day, day_dir in _snapshot_day_dirs(capture_root):
        if pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}T00:00:00Z") >= parquet_cutoff:
            continue
        files = sorted(day_dir.rglob("*.parquet"))
        if not files or not _unit_backed_up(files, started_ns):
            continue
        for path in files:
            try:
                path.unlink()
                pruned += 1
            except OSError:
                continue
    for day, files in _liquidation_day_files(liquidations_dir).items():
        if pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}T00:00:00Z") >= parquet_cutoff:
            continue
        if not _unit_backed_up(files, started_ns):
            continue
        for path in files:
            try:
                path.unlink()
                pruned += 1
            except OSError:
                continue
    removed = _remove_empty_dirs(
        [
            Path(capture_root) / "raw" / "hot",
            Path(capture_root) / "raw" / "archive",
            Path(capture_root) / "book_ticker",
            Path(capture_root) / "premium_index",
        ]
    )
    return RetentionReport(
        prune_blocked=False, blocked_reason=None, pruned_files=pruned, swept_partials=0, removed_empty_dirs=removed
    )
