"""Attested wall-clock coverage for event streams with no archival backfill."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

COVERAGE_DIRNAME: str = "coverage"

_logger = logging.getLogger(__name__)


def _require_aware(ts: pd.Timestamp, label: str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        raise ValueError(f"{label} requires tz-aware timestamp")
    return out.tz_convert("UTC")


class CoverageTracker:
    """Attests the wall-clock intervals during which an event stream was demonstrably connected.

    For event streams (liquidations) an empty period is ambiguous: no events vs. collector down.
    An interval is attested only between successful stream returns with no transport error in
    between; everything else is an explicit gap. Records are append-only JSONL under
    ``<root>/coverage/<stream>/<YYYYMMDD>.jsonl`` with fields ``stream``, ``start``, ``end`` (ISO UTC).
    """

    def __init__(self, stream: str, root: Path) -> None:
        self._stream = str(stream)
        self._root = Path(root)
        self._open_start: pd.Timestamp | None = None
        self._open_last: pd.Timestamp | None = None
        self._cursor: pd.Timestamp | None = None
        self._closed: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def mark_ok(self, ts: pd.Timestamp) -> None:
        """Record a successful stream return at ``ts`` (opens a segment if none is open)."""
        out = _require_aware(ts, "mark_ok")
        if self._open_start is None:
            self._open_start = out
            self._open_last = out
            if self._cursor is None:
                self._cursor = out
        else:
            if out < self._open_last:
                raise ValueError("mark_ok timestamps must be non-decreasing")
            self._open_last = out

    def mark_error(self, ts: pd.Timestamp) -> None:
        """Close the open segment at the last successful return; ``ts`` itself is not attested."""
        _require_aware(ts, "mark_error")
        if self._open_start is None:
            return
        assert self._open_last is not None
        assert self._cursor is not None
        if self._open_last > self._cursor:
            self._closed.append((self._cursor, self._open_last))
        self._open_start = None
        self._open_last = None
        self._cursor = None

    def _pending(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if self._open_start is None or self._open_last is None or self._cursor is None:
            return None
        if self._open_last <= self._cursor:
            return None
        return (self._cursor, self._open_last)

    def discard_unflushed(self, ts: pd.Timestamp) -> None:
        """Withdraw every attested interval not yet flushed and close the open segment.

        Called when buffered events covering those intervals are discarded without being persisted:
        attesting a period whose events were dropped would certify a gap as complete data. The next
        ``mark_ok`` opens a fresh segment, so the discarded span becomes an explicit coverage gap.

        Raises:
            ValueError: ``ts`` is tz-naive.
        """
        _require_aware(ts, "discard_unflushed")
        self._closed = []
        self._open_start = None
        self._open_last = None
        self._cursor = None

    def flush(self) -> list[Path]:
        """Persist the attested interval accumulated since the previous flush; the open segment continues."""
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = list(self._closed)
        pending = self._pending()
        if pending is not None:
            intervals.append(pending)
        if not intervals:
            return []
        touched: set[Path] = set()
        for start, end in intervals:
            written = _append_intervals(self._root, self._stream, start, end)
            touched.update(written)
        self._closed = []
        if pending is not None and self._open_last is not None:
            self._cursor = self._open_last
        if self._open_start is None:
            self._cursor = None
        return sorted(touched)


def _split_at_midnight(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    pieces: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        day_end = (cursor.normalize() + pd.Timedelta(days=1)).tz_convert("UTC")
        piece_end = day_end if day_end < end else end
        pieces.append((cursor, piece_end))
        cursor = piece_end
    return pieces


def _append_intervals(root: Path, stream: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    directory = Path(root) / COVERAGE_DIRNAME / stream
    directory.mkdir(parents=True, exist_ok=True)
    touched: list[Path] = []
    for piece_start, piece_end in _split_at_midnight(start, end):
        day = piece_start.strftime("%Y%m%d")
        path = directory / f"{day}.jsonl"
        record = {"stream": stream, "start": piece_start.isoformat(), "end": piece_end.isoformat()}
        line = json.dumps(record, sort_keys=True) + "\n"
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        if path.exists():
            with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
                dst.write(src.read())
                dst.write(line)
        else:
            tmp.write_text(line, encoding="utf-8")
        os.replace(tmp, path)
        touched.append(path)
    return touched


def load_coverage(root: Path, stream: str, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Attested intervals clipped to ``[start, end)``, merged where they touch or overlap.

    Returns:
        Columns ``start``, ``end`` (datetime64[ns, UTC]) sorted by ``start``; empty when nothing attested.
    """
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    directory = Path(root) / COVERAGE_DIRNAME / stream
    if end_ts <= start_ts or not directory.exists():
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns, UTC]"), "end": pd.Series(dtype="datetime64[ns, UTC]")})
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            _logger.debug("coverage skip unreadable %s: %s", path, exc)
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                s = pd.Timestamp(rec["start"]).tz_convert("UTC")
                e = pd.Timestamp(rec["end"]).tz_convert("UTC")
            except Exception as exc:
                _logger.debug("coverage skip malformed record: %s", exc)
                continue
            cs = s if s > start_ts else start_ts
            ce = e if e < end_ts else end_ts
            if ce > cs:
                intervals.append((cs, ce))
    if not intervals:
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns, UTC]"), "end": pd.Series(dtype="datetime64[ns, UTC]")})
    intervals.sort()
    merged: list[list[pd.Timestamp]] = [[intervals[0][0], intervals[0][1]]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return pd.DataFrame({"start": [s for s, _ in merged], "end": [e for _, e in merged]})
