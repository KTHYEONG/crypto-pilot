from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.public_archive import catalog


@dataclass(frozen=True, slots=True)
class BackfillJob:
    """A persisted request to collect one dataset over a closed historical range."""

    job_id: str
    dataset: str
    start: date
    end: date
    census_root: str
    breadth: int | None
    preroll_days: int | None
    created_at: pd.Timestamp

    def __post_init__(self) -> None:
        if self.dataset not in catalog.DATASETS:
            raise ValueError(f"unknown archive dataset: {self.dataset!r}")
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        if self.start < catalog.MASK_POLICIES[self.dataset].start:
            raise ValueError(f"start {self.start} precedes mask policy start")
        stamp = pd.Timestamp(self.created_at)
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != pd.Timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        _validate_id(self.job_id)
        if not self.job_id.startswith(f"{self.dataset}-{self.start}-{self.end}-"):
            raise ValueError("job_id does not match dataset and range")
        if not isinstance(self.census_root, str) or not self.census_root:
            raise ValueError("census_root must be a non-empty string")
        for label, value, floor in (("breadth", self.breadth, 1), ("preroll_days", self.preroll_days, 0)):
            if value is not None and (type(value) is not int or value < floor):
                raise ValueError(f"invalid {label}")


def _validate_id(job_id: str) -> None:
    if (
        not isinstance(job_id, str)
        or re.fullmatch(r"(?:metrics|bookDepth|aggTrades)-\d{4}-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{8}T\d{6}Z", job_id)
        is None
    ):
        raise ValueError("invalid job_id")


def _dirs(root: Path) -> tuple[Path, Path, Path]:
    base = Path(root) / "_jobs"
    return base / "pending", base / "done", base / "cancelled"


def _job_to_dict(job: BackfillJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "start": job.start.isoformat(),
        "end": job.end.isoformat(),
        "census_root": job.census_root,
        "breadth": job.breadth,
        "preroll_days": job.preroll_days,
        "created_at": pd.Timestamp(job.created_at).tz_convert("UTC").isoformat(),
    }


def _job_from_dict(obj: object) -> BackfillJob:
    try:
        if not isinstance(obj, dict):
            raise ValueError("job must be an object")
        required = {"job_id", "dataset", "start", "end", "census_root", "breadth", "preroll_days", "created_at"}
        if set(obj) - required - {"summary", "finished_at"} != set() or not required.issubset(set(obj)):
            raise ValueError("invalid job schema")
        if any(
            not isinstance(obj[key], str) for key in ("job_id", "dataset", "start", "end", "census_root", "created_at")
        ):
            raise ValueError("invalid job field type")
        breadth = obj["breadth"]
        preroll = obj["preroll_days"]
        if breadth is not None and (isinstance(breadth, bool) or not isinstance(breadth, int)):
            raise ValueError("invalid breadth")
        if preroll is not None and (isinstance(preroll, bool) or not isinstance(preroll, int)):
            raise ValueError("invalid preroll_days")
        return BackfillJob(
            job_id=str(obj["job_id"]),
            dataset=str(obj["dataset"]),
            start=date.fromisoformat(str(obj["start"])),
            end=date.fromisoformat(str(obj["end"])),
            census_root=str(obj["census_root"]),
            breadth=None if breadth is None else int(breadth),
            preroll_days=None if preroll is None else int(preroll),
            created_at=pd.Timestamp(str(obj["created_at"])),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed backfill job: {obj!r}") from exc


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".job-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def enqueue_backfill(root: Path, job: BackfillJob) -> Path:
    """Persist a backfill job atomically under <root>/_jobs/pending; returns its path."""
    pending, done, cancelled = _dirs(Path(root))
    for directory in (pending, done, cancelled):
        if (directory / f"{job.job_id}.json").exists():
            raise FileExistsError(f"backfill job already exists: {job.job_id}")
    for existing in pending_jobs(Path(root)):
        if existing.dataset != job.dataset:
            continue
        if existing.start <= job.end and job.start <= existing.end:
            raise ValueError(f"overlapping backfill job: {existing.job_id}")
    path = pending / f"{job.job_id}.json"
    _write_atomic(path, _job_to_dict(job))
    return path


def pending_jobs(root: Path) -> list[BackfillJob]:
    """Pending jobs ordered by created_at; malformed files raise DataIntegrityError."""
    return _list_dir(Path(root) / "_jobs" / "pending")


def _list_dir(directory: Path) -> list[BackfillJob]:
    if not directory.is_dir():
        return []
    jobs: list[BackfillJob] = []
    for path in sorted(directory.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise DataIntegrityError(f"malformed backfill job file: {path}") from exc
        job = _job_from_dict(obj)
        if path.name != f"{job.job_id}.json":
            raise DataIntegrityError(f"backfill job id mismatch: {path}")
        jobs.append(job)
    jobs.sort(key=lambda j: (pd.Timestamp(j.created_at).tz_convert("UTC").value, j.job_id))
    return jobs


def done_jobs(root: Path) -> list[BackfillJob]:
    """Finished jobs ordered by created_at."""
    return _list_dir(Path(root) / "_jobs" / "done")


def cancelled_jobs(root: Path) -> list[BackfillJob]:
    """Cancelled jobs ordered by created_at."""
    return _list_dir(Path(root) / "_jobs" / "cancelled")


def _finish_move(root: Path, job_id: str, target: str, summary: Mapping[str, int | str]) -> Path:
    if not job_id or Path(job_id).name != job_id or job_id in (".", ".."):
        raise ValueError("invalid job_id")
    pending, done, cancelled = _dirs(Path(root))
    dest_dir = {"done": done, "cancelled": cancelled}[target]
    src = pending / f"{job_id}.json"
    try:
        obj = json.loads(src.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"malformed backfill job file: {src}") from exc
    job = _job_from_dict(obj)
    if job.job_id != job_id:
        raise DataIntegrityError(f"backfill job id mismatch: {src}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{job_id}.json"
    if dest.exists():
        raise FileExistsError(f"backfill job already exists: {job_id}")
    src.rename(dest)
    payload = _job_to_dict(job)
    payload["summary"] = dict(summary)
    payload["finished_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_atomic(dest, payload)
    return dest


def finish_job(root: Path, job_id: str, *, summary: Mapping[str, int | str]) -> Path:
    """Move a pending job to <root>/_jobs/done with its final summary; returns the new path."""
    return _finish_move(Path(root), job_id, "done", summary)


def cancel_job(root: Path, job_id: str) -> Path:
    """Move a pending job to <root>/_jobs/cancelled with its final summary; returns the new path."""
    return _finish_move(Path(root), job_id, "cancelled", {})
