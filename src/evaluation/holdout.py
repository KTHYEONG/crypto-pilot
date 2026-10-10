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


def _family_journal_path(strategy_id: str, root: Path | None = None) -> Path:
    from src.strategy.release import releases_dir

    family = "flow_mom" if strategy_id.startswith("flow_mom") else strategy_id
    return releases_dir(root) / f"{family}.holdout.jsonl"


def _same_family(strategy_id: str, row: dict[str, object]) -> bool:
    return bool(row.get("strategy_id") == strategy_id or (
        strategy_id.startswith("flow_mom") and str(row.get("strategy_id", "")).startswith("flow_mom")
    ))


def read_holdout_looks(strategy_id: str, *, root: Path | None = None) -> tuple[dict[str, object], ...]:
    """Committed holdout looks for the strategy family in file order."""
    if not isinstance(strategy_id, str) or not strategy_id:
        raise DataIntegrityError("strategy_id must be a non-empty string")
    rows = _read_rows(_family_journal_path(strategy_id, root))
    for row in rows:
        _validate_holdout_row(row)
    return tuple(rows)


def _validate_holdout_row(row: dict[str, object]) -> None:
    if any(not isinstance(row.get(key), str) or not row[key] for key in (
        "strategy_id", "spec_digest", "window_start", "window_end",
    )):
        raise DataIntegrityError("holdout journal corrupt")
    try:
        start = _require_utc(pd.Timestamp(str(row["window_start"])), "window_start")
        end = _require_utc(pd.Timestamp(str(row["window_end"])), "window_end")
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError("holdout journal corrupt") from exc
    if end <= start:
        raise DataIntegrityError("holdout journal corrupt")


def holdout_look_recorded(
    strategy_id: str,
    signal_digest: str,
    window: tuple[pd.Timestamp, pd.Timestamp],
    *,
    root: Path | None = None,
) -> bool:
    """Whether this exact window was looked at once for this signal digest."""
    if not isinstance(strategy_id, str) or not strategy_id:
        raise DataIntegrityError("strategy_id must be a non-empty string")
    if not isinstance(signal_digest, str) or not signal_digest:
        raise DataIntegrityError("signal_digest must be a non-empty string")
    if not isinstance(window, tuple) or len(window) != 2:
        raise DataIntegrityError("window must be a (start, end) tuple")
    start = _require_utc(window[0], "window_start")
    end = _require_utc(window[1], "window_end")
    if end <= start:
        raise DataIntegrityError("window_end must be after window_start")
    return any(
        _same_family(strategy_id, row) and row.get("signal_digest") == signal_digest
        and pd.Timestamp(str(row["window_start"])) == start
        and pd.Timestamp(str(row["window_end"])) == end
        for row in read_holdout_looks(strategy_id, root=root)
    )


def holdout_covered(
    strategy_id: str,
    signal_digest: str,
    window: tuple[pd.Timestamp, pd.Timestamp],
    *,
    root: Path | None = None,
) -> bool:
    """Whether a journaled look for this signal digest contains the whole window."""
    if not isinstance(strategy_id, str) or not strategy_id:
        raise DataIntegrityError("strategy_id must be a non-empty string")
    if not isinstance(signal_digest, str) or not signal_digest:
        raise DataIntegrityError("signal_digest must be a non-empty string")
    if not isinstance(window, tuple) or len(window) != 2:
        raise DataIntegrityError("window must be a (start, end) tuple")
    start = _require_utc(window[0], "window_start")
    end = _require_utc(window[1], "window_end")
    if end <= start:
        raise DataIntegrityError("window_end must be after window_start")
    for row in read_holdout_looks(strategy_id, root=root):
        if not _same_family(strategy_id, row) or row.get("signal_digest") != signal_digest:
            continue
        row_start = pd.Timestamp(str(row["window_start"])).tz_convert("UTC")
        row_end = pd.Timestamp(str(row["window_end"])).tz_convert("UTC")
        if row_start <= start and end <= row_end:
            return True
    return False


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
    signal_digest: str,
) -> bool:
    """Journal the one look; returns False when the identical (family, window, signal_digest) look already exists."""
    if not isinstance(strategy_id, str) or not strategy_id:
        raise DataIntegrityError("strategy_id must be a non-empty string")
    if not isinstance(spec_digest, str) or not spec_digest:
        raise DataIntegrityError("spec_digest must be a non-empty string")
    if not isinstance(signal_digest, str) or not signal_digest:
        raise DataIntegrityError("signal_digest must be a non-empty string")
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
        "signal_digest": signal_digest,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    }
    with target.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        rows = _read_rows(target)
        for row in rows:
            _validate_holdout_row(row)
        identical = False
        for row in rows:
            if not _same_family(strategy_id, row):
                continue
            if _overlaps(start, end, row):
                if (
                    row.get("signal_digest") == signal_digest
                    and pd.Timestamp(str(row["window_start"])) == start
                    and pd.Timestamp(str(row["window_end"])) == end
                ):
                    identical = True
                    continue
                raise DataIntegrityError(
                    f"holdout window already consumed for {strategy_id}: {row.get('window_start')}..{row.get('window_end')}"
                )
        if identical:
            return False
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return True


def holdout_overlaps(strategy_id: str, window: tuple[pd.Timestamp, pd.Timestamp], *, path: Path) -> bool:
    """True when a window overlaps an already-consumed holdout look."""
    if not isinstance(window, tuple) or len(window) != 2:
        raise DataIntegrityError("window must be a (start, end) tuple")
    start = _require_utc(window[0], "window_start")
    end = _require_utc(window[1], "window_end")
    target = Path(path)
    for row in _read_rows(target):
        if not _same_family(strategy_id, row):
            continue
        if _overlaps(start, end, row):
            return True
    return False
