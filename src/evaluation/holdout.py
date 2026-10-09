"""One-look holdout journal: a holdout window is evidence only once."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError


def _require_utc(ts: pd.Timestamp, label: str) -> pd.Timestamp:
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
        raise DataIntegrityError(f"{label} must be a valid timestamp")
    if ts.tzinfo is None or ts.utcoffset() is None or ts.utcoffset().total_seconds() != 0:
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return ts.tz_convert("UTC")


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"holdout journal corrupt: {path}") from exc
        if not isinstance(row, dict):
            raise DataIntegrityError(f"holdout journal corrupt: {path}")
        rows.append(row)
    return rows


def _overlaps(start: pd.Timestamp, end: pd.Timestamp, row: dict[str, object]) -> bool:
    try:
        row_start = pd.Timestamp(str(row["window_start"])).tz_convert("UTC")
        row_end = pd.Timestamp(str(row["window_end"])).tz_convert("UTC")
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError("holdout journal corrupt") from exc
    return bool(start < row_end and row_start < end)


def consume_holdout_look(
    strategy_id: str,
    spec_digest: str,
    window: tuple[pd.Timestamp, pd.Timestamp],
    *,
    path: Path,
) -> None:
    """A holdout is evidence only the first time it is looked at. A second evaluation of the same window for the same strategy family raises DataIntegrityError; new data
    appended after the last look forms a new window."""
    if not isinstance(strategy_id, str) or not strategy_id:
        raise DataIntegrityError("strategy_id must be a non-empty string")
    if not isinstance(spec_digest, str) or not spec_digest:
        raise DataIntegrityError("spec_digest must be a non-empty string")
    if not isinstance(window, tuple) or len(window) != 2:
        raise DataIntegrityError("window must be a (start, end) tuple")
    start = _require_utc(window[0], "window_start")
    end = _require_utc(window[1], "window_end")
    if end <= start:
        raise DataIntegrityError("window_end must be after window_start")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_id": strategy_id,
        "spec_digest": spec_digest,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    }
    with target.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for row in _read_rows(target):
            same_family = str(row.get("strategy_id", "")).startswith("flow_mom") and strategy_id.startswith("flow_mom")
            if row.get("strategy_id") != strategy_id and not same_family:
                continue
            if _overlaps(start, end, row):
                raise DataIntegrityError(
                    f"holdout window already consumed for {strategy_id}: {row.get('window_start')}..{row.get('window_end')}"
                )
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def holdout_overlaps(strategy_id: str, window: tuple[pd.Timestamp, pd.Timestamp], *, path: Path) -> bool:
    """True when a window overlaps an already-consumed holdout look."""
    if not isinstance(window, tuple) or len(window) != 2:
        raise DataIntegrityError("window must be a (start, end) tuple")
    start = _require_utc(window[0], "window_start")
    end = _require_utc(window[1], "window_end")
    target = Path(path)
    for row in _read_rows(target):
        same_family = str(row.get("strategy_id", "")).startswith("flow_mom") and strategy_id.startswith("flow_mom")
        if row.get("strategy_id") != strategy_id and not same_family:
            continue
        if _overlaps(start, end, row):
            return True
    return False
