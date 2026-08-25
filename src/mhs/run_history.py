"""Append-only, rotation-capped MHS run-history ledger.

One JSON-Lines shard per active run set plus immutable timestamped archives,
with ``latest.json`` holding the most recent run snapshot. No hardcoded
absolute paths: callers derive the history directory dynamically.

Each record is one JSON line in ``active.jsonl``; when appending exceeds
``RUN_HISTORY_SHARD_MAX_BYTES``, the shard rotates to an immutable archive.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.mhs.params import SEARCH_TRIALS_ATTEMPTED

RUN_HISTORY_SHARD_MAX_BYTES: int = 262144
RUN_HISTORY_MAX_SHARDS: int = 12

# Repository-canonical history directory used when a caller passes no explicit
# directory; mirrors the persist-time ``<target.parent>/mhs_run_history`` layout.
_DEFAULT_HISTORY_DIR = Path("docs") / "results" / "mhs_run_history"

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


def derive_trials_attempted(history_dir: Path | str | None = None) -> tuple[int, str]:
    """Audit-trials denominator for the DSR from the run history itself.

    Counts the distinct flag configurations (each record's canonical ``flags``
    payload) across every JSONL shard and returns
    ``(SEARCH_TRIALS_ATTEMPTED + counted, 'constant_plus_history')``: the
    registered constant is the floor for the search performed before any
    history existed, and every newly recorded distinct configuration adds on
    top of it, so the denominator is monotone -- never capped -- in
    exploration. ``source`` is ``'constant_plus_history'`` when at least one
    readable record exists, or ``'constant_fallback'`` when the history is
    unreadable (missing directory or malformed records). O(history_lines).
    """
    directory = Path(history_dir) if history_dir is not None else _DEFAULT_HISTORY_DIR
    try:
        seen: set[str] = set()
        observed_records = 0
        for shard in sorted(directory.glob("*.jsonl")):
            with shard.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    flags = record.get("flags") if isinstance(record, dict) else None
                    seen.add(json.dumps(flags, ensure_ascii=False, sort_keys=True))
                    observed_records += 1
    except (OSError, json.JSONDecodeError):
        return SEARCH_TRIALS_ATTEMPTED, "constant_fallback"
    if observed_records == 0:
        # No readable history at all: the denominator's provenance must say so.
        return SEARCH_TRIALS_ATTEMPTED, "constant_fallback"
    return SEARCH_TRIALS_ATTEMPTED + len(seen), "constant_plus_history"


def window_trial_sharpes(
    window: tuple[str, str], history_dir: Path | str | None = None
) -> tuple[float, ...]:
    """Annualized blend Sharpe outcomes recorded for exactly one report window.

    Single pass over every JSONL shard, mirroring ``derive_trials_attempted``:
    a record qualifies when its ``(start, resolved_end)`` equals ``window`` and
    its ``blend.primary_naive_sharpe`` is finite. Records sharing an identical
    canonical ``flags`` payload collapse to one entry (a re-run of the same
    configuration is the same trial, and duplicate outcomes must not shrink
    the DSR trial variance). Returns the outcomes ascending; an unreadable or
    missing history yields ``()``.
    """
    directory = Path(history_dir) if history_dir is not None else _DEFAULT_HISTORY_DIR
    wanted = (str(window[0]), str(window[1]))
    try:
        seen_flags: set[str] = set()
        outcomes: list[float] = []
        for shard in sorted(directory.glob("*.jsonl")):
            with shard.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    if not isinstance(record, dict):
                        continue
                    if (str(record.get("start")), str(record.get("resolved_end"))) != wanted:
                        continue
                    blend = record.get("blend")
                    sharpe = (
                        blend.get("primary_naive_sharpe")
                        if isinstance(blend, dict)
                        else None
                    )
                    if not isinstance(sharpe, (int, float)) or isinstance(sharpe, bool):
                        continue
                    sharpe = float(sharpe)
                    if not math.isfinite(sharpe):
                        continue
                    flags_key = json.dumps(
                        record.get("flags"), ensure_ascii=False, sort_keys=True
                    )
                    if flags_key in seen_flags:
                        continue
                    seen_flags.add(flags_key)
                    outcomes.append(sharpe)
        return tuple(sorted(outcomes))
    except (OSError, json.JSONDecodeError):
        return ()
