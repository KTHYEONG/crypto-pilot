"""Append-only, rotation-capped MHS run-history ledger.

One JSON-Lines shard per active run set plus immutable timestamped archives,
with ``latest.json`` holding the most recent run snapshot. No hardcoded
absolute paths: callers derive the history directory dynamically.

Each record is one JSON line in ``active.jsonl``; when appending exceeds
``RUN_HISTORY_SHARD_MAX_BYTES``, the shard rotates to an immutable archive.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUN_HISTORY_SHARD_MAX_BYTES: int = 262144
RUN_HISTORY_MAX_SHARDS: int = 12

_ACTIVE_FILE_NAME = "active.jsonl"
_LATEST_FILE_NAME = "latest.json"
_ARCHIVE_PREFIX = "mhs_run_history_"
_ARCHIVE_SUFFIX = ".jsonl"


def mhs_run_history_dir(target: Path) -> Path:
    """History directory derived from a persisted report target.

    Always ``target.parent / 'mhs_run_history'`` so test fixtures under
    ``tmp_path`` isolate their run history from the repository tree.
    """
    return target.parent / "mhs_run_history"


def _archive_path(history_dir: Path, utc_millis: int) -> Path:
    return history_dir / f"{_ARCHIVE_PREFIX}{utc_millis}{_ARCHIVE_SUFFIX}"


def _unique_archive_path(history_dir: Path) -> Path:
    """Rotated archive name that stays unique even for same-millisecond rotations."""
    utc_millis = int(time.time() * 1000)
    archive = _archive_path(history_dir, utc_millis)
    while archive.exists():
        utc_millis += 1
        archive = _archive_path(history_dir, utc_millis)
    return archive


def _serialize_record(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), ensure_ascii=False, sort_keys=True)


def _prune_archives(history_dir: Path) -> None:
    archives = sorted(history_dir.glob(f"{_ARCHIVE_PREFIX}*{_ARCHIVE_SUFFIX}"))
    excess = len(archives) - RUN_HISTORY_MAX_SHARDS
    for stale in archives[:excess]:
        stale.unlink()


def append_run_history_record(record: Mapping[str, Any], history_dir: Path) -> Path:
    """Append one run record to the ledger, rotating/pruning as needed.

    Returns the active shard path. Rotation and pruning happen only when the
    append would push the active shard past the byte budget, so the steady
    state is a single small append with no directory scan per run.
    """
    history_dir.mkdir(parents=True, exist_ok=True)
    active = history_dir / _ACTIVE_FILE_NAME
    line = _serialize_record(record) + "\n"

    if active.exists() and active.stat().st_size + len(line.encode("utf-8")) > RUN_HISTORY_SHARD_MAX_BYTES:
        active.rename(_unique_archive_path(history_dir))
        _prune_archives(history_dir)

    with active.open("a", encoding="utf-8") as fh:
        fh.write(line)

    latest = history_dir / _LATEST_FILE_NAME
    latest.write_text(line, encoding="utf-8")
    return active
