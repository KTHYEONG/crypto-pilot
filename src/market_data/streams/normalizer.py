"""Raw-to-derived normalizer: hot journal into the existing parquet layout, checkpointed.

Every cycle derives newly completed hot records (both capture slots, receipt order) into hourly
snapshot partitions, liquidation partitions and coverage intervals. The checkpoint advances only
after every derived write succeeded, so a crash replays at most one cycle and replays are absorbed
by earliest-receipt merges. Cross-slot duplicates are dropped in memory (work saver only);
correctness under restart relies on the merge keys.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.capture.journal import iter_complete_records, last_complete_offset
from src.common.errors import DataIntegrityError
from src.live.lifecycle import ShutdownFlag
from src.market_data.streams.compaction import compact_day, due_compactions
from src.market_data.streams.coverage import CoverageTracker, CoverageTrackerState
from src.market_data.streams.dedupe_window import DedupeWindow
from src.market_data.streams.heartbeat_v3 import (
    CAPTURE_SLOTS,
    build_heartbeat_payload,
    disk_usage_bytes,
    fold_cycle_report,
    local_footprint_bytes,
    new_stream_states,
    refresh_window_states,
    write_heartbeat_atomic,
)
from src.market_data.streams.liquidations import (
    LiquidationEvent,
    append_liquidation_events,
    parse_liquidation,
)
from src.market_data.streams.retention import prune_backed_up, read_backup_status, sweep_partials
from src.market_data.streams.snapshots import (
    parse_book_ticker_payload,
    parse_premium_index_payload,
    write_hourly_partition,
)

_logger = logging.getLogger(__name__)

HOT_STREAMS: tuple[str, str, str] = ("book_ticker", "premium_index", "force_order")
REST_STREAMS: tuple[str, str] = ("book_ticker", "premium_index")
CHECKPOINT_NAME: str = "normalizer_checkpoint.json"


class NormalizerConfig(BaseModel):
    """Cadences, windows and retention bounds of the raw-to-derived normalizer (frozen, validated)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    normalize_interval_s: float = 30.0
    segment_final_grace_s: float = 180.0
    ws_dedupe_window_s: float = 7200.0
    rest_dedupe_window_s: float = 7200.0
    max_bytes_per_cycle: int = 67_108_864
    book_ticker_interval_s: int = 60
    premium_index_interval_s: int = 300
    snapshot_max_rejected_fraction: float = 0.05
    grid_health_window_s: float = 3600.0
    reference_capture_after_utc: str = "00:05"
    compaction_grace_s: float = 1800.0
    archive_lzma_preset: int = 6
    raw_archive_local_retention_days: int = 30
    parquet_local_retention_days: int = 180
    backup_status_max_age_h: float = 72.0
    partial_sweep_age_s: float = 3600.0
    retention_interval_s: float = 3600.0
    heartbeat_interval_s: float = 30.0
    derive_lag_allowance_s: float = 60.0

    @field_validator("normalize_interval_s")
    @classmethod
    def _positive_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("normalize_interval_s must be positive")
        return value

    @field_validator("segment_final_grace_s")
    @classmethod
    def _final_grace(cls, value: float) -> float:
        if value < 60:
            raise ValueError("segment_final_grace_s must be >= 60")
        return value

    @field_validator("ws_dedupe_window_s", "rest_dedupe_window_s")
    @classmethod
    def _dedupe_window(cls, value: float, info: Any) -> float:
        if value < 600:
            raise ValueError(f"{info.field_name} must be >= 600")
        return value

    @field_validator("max_bytes_per_cycle")
    @classmethod
    def _byte_budget(cls, value: int) -> int:
        if value < 1_048_576:
            raise ValueError("max_bytes_per_cycle must be >= 1 MiB")
        return value

    @field_validator("book_ticker_interval_s", "premium_index_interval_s")
    @classmethod
    def _grid_interval(cls, value: int, info: Any) -> int:
        if value <= 0 or 86400 % value != 0:
            raise ValueError(f"{info.field_name} must be a positive divisor of 86400")
        return value

    @field_validator("snapshot_max_rejected_fraction")
    @classmethod
    def _rejected_fraction(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("snapshot_max_rejected_fraction must be in (0, 1)")
        return value

    @field_validator("grid_health_window_s")
    @classmethod
    def _health_window(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("grid_health_window_s must be positive")
        return value

    @field_validator("reference_capture_after_utc")
    @classmethod
    def _cutoff(cls, value: str) -> str:
        if not re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", value):
            raise ValueError("reference_capture_after_utc must be HH:MM UTC")
        return value

    @field_validator("compaction_grace_s")
    @classmethod
    def _compaction_grace(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("compaction_grace_s must be positive")
        return value

    @field_validator("archive_lzma_preset")
    @classmethod
    def _lzma_preset(cls, value: int) -> int:
        if not 0 <= value <= 9:
            raise ValueError("archive_lzma_preset must be in 0..9")
        return value

    @field_validator("raw_archive_local_retention_days")
    @classmethod
    def _archive_retention(cls, value: int) -> int:
        if value < 2:
            raise ValueError("raw_archive_local_retention_days must be >= 2")
        return value

    @field_validator(
        "backup_status_max_age_h",
        "partial_sweep_age_s",
        "retention_interval_s",
        "heartbeat_interval_s",
        "derive_lag_allowance_s",
    )
    @classmethod
    def _positive_float(cls, value: float, info: Any) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("parquet_local_retention_days")
    @classmethod
    def _parquet_retention(cls, value: int) -> int:
        if value < 2:
            raise ValueError("parquet_local_retention_days must be >= 2")
        return value

    @model_validator(mode="after")
    def _cross_field(self) -> NormalizerConfig:
        if self.compaction_grace_s < self.segment_final_grace_s:
            raise ValueError("compaction_grace_s must be >= segment_final_grace_s")
        if self.parquet_local_retention_days < self.raw_archive_local_retention_days:
            raise ValueError("parquet_local_retention_days must be >= raw_archive_local_retention_days")
        if self.retention_interval_s < self.normalize_interval_s:
            raise ValueError("retention_interval_s must be >= normalize_interval_s")
        if self.grid_health_window_s < 2 * max(self.book_ticker_interval_s, self.premium_index_interval_s):
            raise ValueError("grid_health_window_s must be >= max grid interval x 2")
        if self.partial_sweep_age_s < 600:
            raise ValueError("partial_sweep_age_s must be >= 600")
        if self.derive_lag_allowance_s < self.normalize_interval_s:
            raise ValueError("derive_lag_allowance_s must be >= normalize_interval_s")
        return self


@dataclass(frozen=True, slots=True)
class FileCursor:
    """Checkpoint of one hot segment.

    Attributes:
        offset: Byte offset just past the last complete gzip member whose records are durably derived.
        final: Whether the segment was FINAL (spec 01) when ``offset`` was recorded.
    """

    offset: int
    final: bool


@dataclass(frozen=True, slots=True)
class NormalizerCheckpoint:
    """Durable progress of the normalizer; the only mutable state it keeps on disk.

    Attributes:
        files: Cursor per hot segment path relative to ``<capture_root>/raw/hot``.
        coverage: Per-slot restorable force_order coverage state.
        retention_blocked_since: ISO UTC instant the backup-gated prune first became blocked, or
            ``None`` while it is not blocked. Persisted so a restart (every deploy) does not reset the
            timer that decides when a persistent block alerts.
    """

    files: Mapping[str, FileCursor]
    coverage: Mapping[str, CoverageTrackerState]
    retention_blocked_since: str | None = None


def _parse_ts(value: Any) -> pd.Timestamp | None:
    try:
        out = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out) or out.tzinfo is None:
        return None
    return out.tz_convert("UTC")


def _state_from_json(raw: Any) -> CoverageTrackerState:
    if not isinstance(raw, Mapping):
        raise DataIntegrityError("checkpoint coverage entry must be an object")
    open_last = _parse_ts(raw.get("open_last")) if raw.get("open_last") is not None else None
    cursor = _parse_ts(raw.get("cursor")) if raw.get("cursor") is not None else None
    if raw.get("open_last") is not None and open_last is None:
        raise DataIntegrityError("checkpoint coverage open_last is not ISO UTC")
    if raw.get("cursor") is not None and cursor is None:
        raise DataIntegrityError("checkpoint coverage cursor is not ISO UTC")
    return CoverageTrackerState(open_last=open_last, cursor=cursor)


def load_checkpoint(path: Path) -> NormalizerCheckpoint:
    """Read ``raw/normalizer_checkpoint.json``; a missing file is an empty checkpoint.

    Raises:
        DataIntegrityError: The file exists but is undecodable or violates the schema. The normalizer
            must not silently restart from zero, which would re-derive everything; it fails closed and
            the watchdog reports ``normalizer_failing``.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return NormalizerCheckpoint(files={}, coverage={})
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(f"checkpoint undecodable: {exc}") from exc
    if not isinstance(raw, dict):
        raise DataIntegrityError("checkpoint must be a JSON object")
    files: dict[str, FileCursor] = {}
    entries = raw.get("files", {})
    if not isinstance(entries, dict):
        raise DataIntegrityError("checkpoint files must be an object")
    for rel, cursor in entries.items():
        if not isinstance(cursor, dict):
            raise DataIntegrityError(f"checkpoint cursor for {rel!r} must be an object")
        offset = cursor.get("offset")
        final = cursor.get("final", False)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise DataIntegrityError(f"checkpoint offset for {rel!r} must be a non-negative int")
        files[str(rel)] = FileCursor(offset=offset, final=bool(final))
    coverage: dict[str, CoverageTrackerState] = {}
    states = raw.get("coverage", {})
    if not isinstance(states, dict):
        raise DataIntegrityError("checkpoint coverage must be an object")
    for slot, state in states.items():
        coverage[str(slot)] = _state_from_json(state)
    blocked_since = raw.get("retention_blocked_since")
    if blocked_since is not None and (not isinstance(blocked_since, str) or _parse_ts(blocked_since) is None):
        raise DataIntegrityError("checkpoint retention_blocked_since is not ISO UTC")
    return NormalizerCheckpoint(files=files, coverage=coverage, retention_blocked_since=blocked_since)


def save_checkpoint(path: Path, checkpoint: NormalizerCheckpoint) -> None:
    """Atomically replace the checkpoint (same-dir ``.partial`` + fsync + os.replace + dir fsync)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "files": {rel: {"offset": cursor.offset, "final": cursor.final} for rel, cursor in checkpoint.files.items()},
        "coverage": {
            slot: {
                "open_last": state.open_last.isoformat() if state.open_last is not None else None,
                "cursor": state.cursor.isoformat() if state.cursor is not None else None,
            }
            for slot, state in checkpoint.coverage.items()
        },
    }
    if checkpoint.retention_blocked_since is not None:
        payload["retention_blocked_since"] = checkpoint.retention_blocked_since
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    partial = dest.with_name(dest.name + ".partial")
    try:
        with open(partial, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, dest)
        try:
            fd = os.open(dest.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
    except OSError:
        with contextlib.suppress(OSError):
            partial.unlink()
        raise


@dataclass(frozen=True, slots=True)
class CycleReport:
    """Outcome of one derivation cycle, fed into heartbeat v3."""

    records_read: int
    bytes_read: int
    rows_written: Mapping[str, int]
    duplicates_dropped: Mapping[str, int]
    parse_failures: int
    lag_s: float | None
    more_pending: bool


@dataclass(slots=True)
class _Member:
    records: list[dict[str, Any]]
    start: int
    end: int


@dataclass(slots=True)
class _Gathered:
    record: dict[str, Any]
    rel: str
    path: Path
    member_end: int


def _iter_members(path: Path, start: int) -> Any:
    """Yield ``_Member`` groups of complete records past ``start`` in file order."""
    pending: list[dict[str, Any]] = []
    member_start = start
    member_end = start
    for record, end in iter_complete_records(path, start):
        if pending and end != member_end:
            yield _Member(records=pending, start=member_start, end=member_end)
            pending = []
            member_start = member_end
        pending.append(record)
        member_end = end
    if pending:
        yield _Member(records=pending, start=member_start, end=member_end)


def _hot_files(hot_root: Path) -> list[tuple[str, str, str, Path]]:
    """List ``(stream, day, rel, path)`` hot segments in sorted order."""
    found: list[tuple[str, str, str, Path]] = []
    for stream in HOT_STREAMS:
        stream_dir = hot_root / stream
        if not stream_dir.is_dir():
            continue
        for day_dir in sorted(stream_dir.iterdir()):
            if not day_dir.is_dir() or len(day_dir.name) != 8 or not day_dir.name.isdigit():
                continue
            for path in sorted(day_dir.glob("*.jsonl.gz")):
                rel = f"{stream}/{day_dir.name}/{path.name}"
                found.append((stream, day_dir.name, rel, path))
    return found


def _segment_final(path: Path, *, hour: int, day: str, now_ns: int, grace_s: float) -> bool:
    """Whether the hourly segment ended long enough ago and stayed untouched (spec 01 FINAL)."""
    try:
        moment = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}T{hour:02d}:00:00Z")
    except (TypeError, ValueError):
        return False
    hour_end_ns = int(moment.value) + 3_600_000_000_000
    if now_ns < hour_end_ns + int(grace_s * 1_000_000_000):
        return False
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return False
    return mtime_ns + int(grace_s * 1_000_000_000) <= now_ns


def _recv_ns_of(record: Mapping[str, Any]) -> int:
    recv_ns = record.get("recv_ns")
    if isinstance(recv_ns, bool) or not isinstance(recv_ns, int) or recv_ns < 0:
        return 0
    return recv_ns


def _gather_new_records(
    hot_root: Path,
    checkpoint: NormalizerCheckpoint,
    config: NormalizerConfig,
) -> tuple[list[_Gathered], dict[str, int], int, bool, int | None]:
    """Read new complete members up to the byte budget; return items, ends, bytes, and lag state.

    Returns ``(gathered, consumed, consumed_bytes, more_pending, oldest_pending_recv_ns)`` where
    ``consumed`` maps each touched file rel to the member end the checkpoint may advance to.
    Corrupt members raise ``ValueError`` so the cycle fails closed instead of skipping evidence.
    """
    gathered: list[_Gathered] = []
    consumed: dict[str, int] = {}
    consumed_bytes = 0
    budget = config.max_bytes_per_cycle
    more_pending = False
    oldest_pending: int | None = None

    def note_pending(recv_ns: int) -> None:
        nonlocal more_pending, oldest_pending
        more_pending = True
        oldest_pending = recv_ns if oldest_pending is None else min(oldest_pending, recv_ns)

    for _stream, _day, rel, path in _hot_files(hot_root):
        cursor = checkpoint.files.get(rel)
        start = cursor.offset if cursor is not None else 0
        try:
            if path.stat().st_size < start:
                start = 0
        except OSError:
            continue
        if consumed_bytes >= budget:
            first = _peek_first_recv(path, start)
            if first is not None:
                note_pending(first)
            continue
        drained = True
        for member in _iter_members(path, start):
            if consumed_bytes > 0 and consumed_bytes + (member.end - member.start) > budget:
                if member.records:
                    note_pending(_recv_ns_of(member.records[0]))
                drained = False
                break
            gathered.extend(
                _Gathered(record=record, rel=rel, path=path, member_end=member.end) for record in member.records
            )
            consumed[rel] = member.end
            consumed_bytes += member.end - member.start
        if drained:
            continue
    return gathered, consumed, consumed_bytes, more_pending, oldest_pending


def _peek_first_recv(path: Path, start: int) -> int | None:
    """Return the recv instant of the first new complete record, if any."""
    for record, _end in iter_complete_records(path, start):
        return _recv_ns_of(record)
    return None


def _grid_ns(grid: str) -> int | None:
    try:
        moment = pd.Timestamp(grid)
    except (TypeError, ValueError):
        return None
    if pd.isna(moment) or moment.tzinfo is None:
        return None
    return int(moment.tz_convert("UTC").value)


def _parse_rest_body(body: Any) -> Any:
    if not isinstance(body, str):
        return None
    try:
        return json.loads(body)
    except (ValueError, TypeError):
        return None


@dataclass(slots=True)
class _RestOutcome:
    frames: dict[str, list[pd.DataFrame]]
    duplicates: dict[str, int]
    parse_failures: int


def _derive_rest(
    gathered: list[_Gathered],
    config: NormalizerConfig,
    now_ns: int,
    dedupe: DedupeWindow,
) -> _RestOutcome:
    """Parse new REST records into per-stream frames in receipt order."""
    frames: dict[str, list[pd.DataFrame]] = {}
    for stream in REST_STREAMS:
        frames[stream] = []
    duplicates = dict.fromkeys(REST_STREAMS, 0)
    parse_failures = 0
    cycle_done: set[tuple[str, str]] = set()
    for item in gathered:
        record = item.record
        stream = str(record.get("stream"))
        if stream not in REST_STREAMS:
            continue
        if record.get("slot") not in CAPTURE_SLOTS:
            parse_failures += 1
            continue
        grid = record.get("grid")
        if not isinstance(grid, str):
            parse_failures += 1
            continue
        recv_ns = _recv_ns_of(record)
        key = (str(stream), grid)
        if key in cycle_done or dedupe.check_rest(str(stream), grid, recv_ns, now_ns=now_ns):
            duplicates[str(stream)] += 1
            continue
        earlier_twin = dedupe.rest_seen(str(stream), grid)
        grid_ns = _grid_ns(grid)
        if grid_ns is None:
            parse_failures += 1
            continue
        parsed = _parse_rest_record(record, str(stream), grid_ns, recv_ns, config)
        if parsed is None:
            if not earlier_twin:
                dedupe.record_rest_outcome(grid_ns, str(stream), False, 0, 0, 0.0)
            continue
        cycle_done.add(key)
        dedupe.mark_rest_success(str(stream), grid, recv_ns)
        frame, rejected_rows, rejected_fraction = parsed
        if not earlier_twin:
            # 이미 집계된 격자의 더 이른 수신본은 병합 키가 교체만 하므로 통계는 한 번만 센다.
            dedupe.record_rest_outcome(grid_ns, str(stream), True, len(frame), rejected_rows, rejected_fraction)
        frames[str(stream)].append(frame)
    return _RestOutcome(frames=frames, duplicates=duplicates, parse_failures=parse_failures)


def _parse_rest_record(
    record: Mapping[str, Any], stream: str, grid_ns: int, recv_ns: int, config: NormalizerConfig
) -> tuple[pd.DataFrame, int, float] | None:
    """Parse one successful REST sample; ``None`` for failed attempts (never raises)."""
    if record.get("kind") != "rest" or record.get("status") != 200:
        return None
    payload = _parse_rest_body(record.get("body"))
    if payload is None:
        return None
    grid_ts = pd.Timestamp(grid_ns, unit="ns", tz="UTC")
    recv_ts = pd.Timestamp(recv_ns, unit="ns", tz="UTC")
    try:
        if stream == "book_ticker":
            parsed = parse_book_ticker_payload(
                payload, captured_at=grid_ts, fetched_at=recv_ts,
                max_rejected_fraction=config.snapshot_max_rejected_fraction,
            )
        else:
            parsed = parse_premium_index_payload(
                payload, captured_at=grid_ts, fetched_at=recv_ts,
                max_rejected_fraction=config.snapshot_max_rejected_fraction,
            )
    except DataIntegrityError:
        return None
    total = parsed.total_rows
    rejected = parsed.rejected_rows
    fraction = (rejected / total) if total > 0 else 0.0
    return parsed.frame, rejected, fraction


@dataclass(slots=True)
class _WsOutcome:
    events: list[LiquidationEvent]
    duplicates: int
    parse_failures: int


def _derive_ws(
    gathered: list[_Gathered],
    trackers: dict[str, CoverageTracker],
    now_ns: int,
    dedupe: DedupeWindow,
) -> _WsOutcome:
    """Journal forceOrder frames into liquidation events and slot attestation."""
    events: list[LiquidationEvent] = []
    duplicates = 0
    parse_failures = 0
    for item in gathered:
        record = item.record
        if record.get("stream") != "force_order":
            continue
        slot = record.get("slot")
        if slot not in CAPTURE_SLOTS:
            parse_failures += 1
            continue
        tracker = trackers[str(slot)]
        kind = record.get("kind")
        recv_ns = _recv_ns_of(record)
        recv_ts = pd.Timestamp(recv_ns, unit="ns", tz="UTC")
        if kind in ("ws_open", "ws_close"):
            tracker.mark_error(recv_ts)
        elif kind == "frame":
            text = record.get("frame")
            if not isinstance(text, str):
                parse_failures += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if dedupe.check_frame(digest, recv_ns, now_ns=now_ns):
                duplicates += 1
                tracker.mark_ok(recv_ts)
                continue
            event = _parse_ws_frame(text, recv_ts)
            if event is None:
                parse_failures += 1
                continue
            events.append(event)
            tracker.mark_ok(recv_ts)
        else:
            parse_failures += 1
    return _WsOutcome(events=events, duplicates=duplicates, parse_failures=parse_failures)


def _parse_ws_frame(text: str, recv_ts: pd.Timestamp) -> LiquidationEvent | None:
    """Decode one frame into a liquidation event; ``None`` when undecodable."""
    try:
        decoded = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(decoded, Mapping):
        return None
    return parse_liquidation(decoded, ingested_at=recv_ts)


def _restore_trackers(capture_root: Path, checkpoint: NormalizerCheckpoint) -> dict[str, CoverageTracker]:
    """Restore per-slot forceOrder trackers from checkpointed coverage state."""
    trackers: dict[str, CoverageTracker] = {}
    empty = CoverageTrackerState(open_last=None, cursor=None)
    for slot in CAPTURE_SLOTS:
        trackers[slot] = CoverageTracker.restore("liquidations", capture_root, checkpoint.coverage.get(slot, empty))
    return trackers


def _file_final_flags(
    hot_root: Path, files: Mapping[str, FileCursor], now_ns: int, grace_s: float
) -> dict[str, FileCursor]:
    """Refresh FINAL flags for every known hot file; drop entries whose file is gone."""
    refreshed: dict[str, FileCursor] = {}
    for rel, cursor in files.items():
        path = hot_root / rel
        if not path.is_file():
            continue
        try:
            hour = int(path.name.split(".")[0])
            day = path.parent.name
        except (ValueError, IndexError):
            refreshed[rel] = FileCursor(offset=cursor.offset, final=False)
            continue
        refreshed[rel] = FileCursor(
            offset=cursor.offset, final=_segment_final(path, hour=hour, day=day, now_ns=now_ns, grace_s=grace_s)
        )
    return refreshed


def normalize_once(
    capture_root: Path,
    liquidations_dir: Path,
    config: NormalizerConfig,
    *,
    checkpoint: NormalizerCheckpoint,
    now: pd.Timestamp,
    dedupe: DedupeWindow,
) -> tuple[NormalizerCheckpoint, CycleReport]:
    """Derive every newly completed hot record into the existing parquet and coverage layout once.

    Reads only complete gzip members past each file's checkpoint offset, across both capture slots,
    in receipt order, up to ``config.max_bytes_per_cycle``. Duplicates across slots are dropped
    (REST by ``(stream, grid)``, WS by exact frame text). Rows are parsed with the existing
    row-isolating parsers and merged into the hourly partitions. Only after every derived write of
    the cycle succeeded is the returned checkpoint allowed to advance. The caller persists it. A crash
    at any point therefore replays at most the last cycle, and replays are absorbed by the
    earliest-receipt merge keys.

    Args:
        capture_root: ``LIVE_CAPTURE_DIR`` (contains ``raw/``, ``book_ticker/``, ``coverage/`` ...).
        liquidations_dir: Liquidation partition directory.
        config: Normalizer configuration.
        checkpoint: Progress to resume from.
        now: Wall-clock UTC, used to judge FINAL segments and lag.
        dedupe: In-memory cross-slot dedupe window (performance only; correctness does not depend
            on it).

    Returns:
        The advanced checkpoint and the cycle report.

    Raises:
        Exception: A derived write failed. No checkpoint advance is returned, and the caller counts
            a failed cycle.
    """
    capture_root = Path(capture_root)
    hot_root = capture_root / "raw" / "hot"
    now_utc = pd.Timestamp(now, tz="UTC") if pd.Timestamp(now).tzinfo is None else pd.Timestamp(now).tz_convert("UTC")
    now_ns = int(now_utc.value)
    gathered, consumed, consumed_bytes, more_pending, oldest_pending = _gather_new_records(
        hot_root, checkpoint, config
    )
    gathered.sort(key=lambda item: (_recv_ns_of(item.record), str(item.record.get("slot", "")), item.rel))
    trackers = _restore_trackers(capture_root, checkpoint)
    dedupe.discard()
    rows_written: dict[str, int] = {}
    coverage_states: dict[str, CoverageTrackerState] = {}
    try:
        rest = _derive_rest(gathered, config, now_ns, dedupe)
        ws = _derive_ws(gathered, trackers, now_ns, dedupe)
        for stream in REST_STREAMS:
            if rest.frames[stream]:
                combined = pd.concat(rest.frames[stream], ignore_index=True)
                write_hourly_partition(combined, capture_root, stream)
                rows_written[stream] = int(sum(len(frame) for frame in rest.frames[stream]))
        if ws.events:
            append_liquidation_events(ws.events, Path(liquidations_dir))
            rows_written["force_order"] = len(ws.events)
        for slot, tracker in trackers.items():
            tracker.flush()
            coverage_states[slot] = tracker.snapshot_state()
    except BaseException:
        dedupe.discard()
        raise
    dedupe.commit()
    merged_files = dict(checkpoint.files)
    for rel, end in consumed.items():
        previous = merged_files.get(rel, FileCursor(offset=0, final=False))
        merged_files[rel] = FileCursor(offset=end, final=previous.final)
    merged_files = _file_final_flags(hot_root, merged_files, now_ns, config.segment_final_grace_s)
    new_checkpoint = NormalizerCheckpoint(
        files=merged_files,
        coverage=coverage_states,
        retention_blocked_since=checkpoint.retention_blocked_since,
    )
    duplicates = {**rest.duplicates, "force_order": ws.duplicates}
    parse_failures = rest.parse_failures + ws.parse_failures
    lag_s = float((now_ns - oldest_pending) / 1_000_000_000) if more_pending and oldest_pending is not None else 0.0
    report = CycleReport(
        records_read=len(gathered),
        bytes_read=consumed_bytes,
        rows_written=rows_written,
        duplicates_dropped=duplicates,
        parse_failures=parse_failures,
        lag_s=lag_s,
        more_pending=more_pending,
    )
    return new_checkpoint, report


def _pending_complete_bytes(hot_root: Path, checkpoint: NormalizerCheckpoint) -> int:
    """Compressed backlog past checkpoints (complete members only)."""
    total = 0
    for _stream, _day, rel, path in _hot_files(hot_root):
        cursor = checkpoint.files.get(rel)
        start = cursor.offset if cursor is not None else 0
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= start:
            continue
        try:
            total += max(0, last_complete_offset(path) - start)
        except ValueError:
            total += max(0, size - start)
    return total


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def run_normalizer(
    capture_root: Path,
    liquidations_dir: Path,
    config: NormalizerConfig,
    *,
    backup_status_path: Path,
    shutdown: ShutdownFlag,
    now_fn: Callable[[], pd.Timestamp] = _utc_now,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Run derivation, compaction, retention and heartbeat publication until shutdown.

    Each cycle derives new records (repeating immediately while ``more_pending``), then at most once
    per ``retention_interval_s`` runs the ``.partial`` sweep, due compactions and backup-gated
    pruning, and publishes heartbeat v3 every ``heartbeat_interval_s``. Every stage failure is logged
    with its traceback, counted in the heartbeat and retried on the next cycle. The loop itself never
    exits on a stage error, because the capture keeps running regardless and the normalizer can
    always catch up from the raw journal.
    """
    checkpoint_path = capture_root / "raw" / CHECKPOINT_NAME
    started_at = now_fn()
    checkpoint: NormalizerCheckpoint | None = None
    dedupe = DedupeWindow(rest_window_s=config.rest_dedupe_window_s, ws_window_s=config.ws_dedupe_window_s)
    states = new_stream_states()
    consecutive_failures = 0
    last_success_at: pd.Timestamp | None = None
    last_error: str | None = None
    last_report: CycleReport | None = None
    compaction_state: dict[str, Any] = {"last_day": None, "last_result": None, "last_error": None, "last_run_at": None}
    retention_state: dict[str, Any] = {
        "prune_blocked": False, "blocked_reason": None, "blocked_since": None, "backup_started_at": None,
        "footprint_bytes": 0, "last_run_at": None, "pruned_files_total": 0,
    }
    last_retention_run: pd.Timestamp | None = None
    last_heartbeat_run: pd.Timestamp | None = None
    intervals = {"book_ticker": config.book_ticker_interval_s, "premium_index": config.premium_index_interval_s}
    while not shutdown.requested:
        now = now_fn()
        if checkpoint is None:
            try:
                checkpoint = load_checkpoint(checkpoint_path)
            except DataIntegrityError as exc:
                consecutive_failures += 1
                last_error = f"checkpoint: {exc}"
                _logger.exception("[DATA] stage=normalizer status=CYCLE_FAILED error=%s", last_error)
                last_heartbeat_run = _maybe_publish_heartbeat(
                    capture_root, config, states, intervals, dedupe, checkpoint,
                    consecutive_failures, last_success_at, last_error, last_report,
                    compaction_state, retention_state, started_at, now, last_heartbeat_run,
                )
                _sleep_interval(sleep_fn, shutdown, config.normalize_interval_s, now_fn)
                continue
        assert checkpoint is not None
        if retention_state["blocked_since"] is None:
            retention_state["blocked_since"] = checkpoint.retention_blocked_since
        try:
            while True:
                checkpoint, last_report = normalize_once(
                    capture_root, liquidations_dir, config,
                    checkpoint=checkpoint, now=now, dedupe=dedupe,
                )
                fresh = dedupe.drain_fresh_outcomes()
                fold_cycle_report(
                    states,
                    {
                        "rows_written": dict(last_report.rows_written),
                        "duplicates_dropped": dict(last_report.duplicates_dropped),
                        "parse_failures": last_report.parse_failures,
                    },
                    fresh,
                    dedupe.ws_last_recv_ns,
                    now.isoformat(),
                )
                save_checkpoint(checkpoint_path, checkpoint)
                consecutive_failures = 0
                last_success_at = now
                last_error = None
                if not last_report.more_pending or shutdown.requested:
                    break
                now = now_fn()
        except Exception as exc:
            consecutive_failures += 1
            last_error = f"normalize: {type(exc).__name__}: {exc}"
            _logger.exception("[DATA] stage=normalizer status=CYCLE_FAILED error=%s", last_error)
        if shutdown.requested:
            break
        now = now_fn()
        if _retention_due(last_retention_run, retention_state, config, backup_status_path, now):
            checkpoint, compaction_state, retention_state = _run_retention_pass(
                capture_root, liquidations_dir, config, checkpoint,
                backup_status_path, compaction_state, retention_state, now, checkpoint_path,
            )
            last_retention_run = now
        last_heartbeat_run = _maybe_publish_heartbeat(
            capture_root, config, states, intervals, dedupe, checkpoint,
            consecutive_failures, last_success_at, last_error, last_report,
            compaction_state, retention_state, started_at, now, last_heartbeat_run,
        )
        _sleep_interval(sleep_fn, shutdown, config.normalize_interval_s, now_fn)
    _logger.info("[SYS] stage=normalizer status=STOPPED")


def _sleep_interval(
    sleep_fn: Callable[[float], None],
    shutdown: ShutdownFlag,
    interval_s: float,
    now_fn: Callable[[], pd.Timestamp],
) -> None:
    """Sleep the cadence in 1 s slices against the injected clock (fake clocks must advance in ``sleep_fn``)."""
    deadline = now_fn() + pd.Timedelta(seconds=interval_s)
    while not shutdown.requested:
        remaining = (deadline - now_fn()).total_seconds()
        if remaining <= 0:
            return
        sleep_fn(min(1.0, remaining))


def _retention_due(
    last_run: pd.Timestamp | None,
    retention_state: Mapping[str, Any],
    config: NormalizerConfig,
    backup_status_path: Path,
    now: pd.Timestamp,
) -> bool:
    """Whether the sweep/compaction/prune pass should run now.

    It runs on its cadence, and additionally every cycle while the prune is blocked and a fresh
    successful backup status has appeared, so a block clears within one cycle instead of waiting for
    the next cadence tick (an hour). Checking only needs one small status file read.
    """
    if last_run is None or (now - last_run).total_seconds() >= config.retention_interval_s:
        return True
    if retention_state["prune_blocked"] is not True:
        return False
    status = read_backup_status(backup_status_path)
    if status is None:
        return False
    return bool((now - status.finished_at).total_seconds() <= config.backup_status_max_age_h * 3600.0)


def _run_retention_pass(
    capture_root: Path,
    liquidations_dir: Path,
    config: NormalizerConfig,
    checkpoint: NormalizerCheckpoint,
    backup_status_path: Path,
    compaction_state: dict[str, Any],
    retention_state: dict[str, Any],
    now: pd.Timestamp,
    checkpoint_path: Path,
) -> tuple[NormalizerCheckpoint, dict[str, Any], dict[str, Any]]:
    """Sweep temps, compact due days and prune backed-up units; each stage guarded separately."""
    try:
        swept = sweep_partials(
            capture_root, liquidations_dir, now=now, max_age_s=config.partial_sweep_age_s
        )
        if swept:
            _logger.info("[DATA] stage=retention status=SWEEP swept_partials=%d", swept)
    except Exception as exc:
        _logger.exception("[DATA] stage=retention status=SWEEP_FAILED error=%s", exc)
    try:
        for stream, day in due_compactions(capture_root, config=config, now=now):
            try:
                compact_day(capture_root, stream, day, checkpoint=checkpoint, config=config, now=now)
            except DataIntegrityError as exc:
                compaction_state = {
                    "last_day": day, "last_result": "error",
                    "last_error": str(exc), "last_run_at": now.isoformat(),
                }
                _logger.exception(
                    "[DATA] stage=compaction stream=%s day=%s status=FAILED error=%s", stream, day, exc
                )
                continue
            except Exception as exc:
                compaction_state = {
                    "last_day": day, "last_result": "error",
                    "last_error": f"{type(exc).__name__}: {exc}", "last_run_at": now.isoformat(),
                }
                _logger.exception(
                    "[DATA] stage=compaction stream=%s day=%s status=FAILED error=%s", stream, day, exc
                )
                continue
            compaction_state = {
                "last_day": day, "last_result": "ok", "last_error": None, "last_run_at": now.isoformat()
            }
            hot_prefix = f"{stream}/{day}/"
            checkpoint = replace(
                checkpoint,
                files={rel: cursor for rel, cursor in checkpoint.files.items() if not rel.startswith(hot_prefix)},
            )
            save_checkpoint(checkpoint_path, checkpoint)
    except Exception as exc:
        _logger.exception("[DATA] stage=compaction status=LIST_FAILED error=%s", exc)
    try:
        status = read_backup_status(backup_status_path)
        if status is not None:
            retention_state["backup_started_at"] = status.started_at.isoformat()
        report = prune_backed_up(
            capture_root, liquidations_dir, status=status, config=config, now=now,
            backup_status_path=backup_status_path,
        )
        retention_state["prune_blocked"] = report.prune_blocked
        retention_state["blocked_reason"] = report.blocked_reason
        blocked_since = (checkpoint.retention_blocked_since or now.isoformat()) if report.prune_blocked else None
        if blocked_since != checkpoint.retention_blocked_since:
            checkpoint = replace(checkpoint, retention_blocked_since=blocked_since)
            save_checkpoint(checkpoint_path, checkpoint)
        retention_state["blocked_since"] = blocked_since
        retention_state["footprint_bytes"] = local_footprint_bytes(capture_root, liquidations_dir)
        retention_state["last_run_at"] = now.isoformat()
        retention_state["pruned_files_total"] = int(retention_state["pruned_files_total"]) + report.pruned_files
        _logger.info(
            "[DATA] stage=retention status=OK pruned=%d swept=%d blocked=%s",
            report.pruned_files, report.swept_partials, report.prune_blocked,
        )
    except Exception as exc:
        _logger.exception("[DATA] stage=retention status=PRUNE_FAILED error=%s", exc)
    return checkpoint, compaction_state, retention_state


def _maybe_publish_heartbeat(
    capture_root: Path,
    config: NormalizerConfig,
    states: dict[str, dict[str, Any]],
    intervals: Mapping[str, int],
    dedupe: DedupeWindow,
    checkpoint: NormalizerCheckpoint | None,
    consecutive_failures: int,
    last_success_at: pd.Timestamp | None,
    last_error: str | None,
    last_report: CycleReport | None,
    compaction_state: Mapping[str, Any],
    retention_state: Mapping[str, Any],
    started_at: pd.Timestamp,
    now: pd.Timestamp,
    last_run: pd.Timestamp | None,
) -> pd.Timestamp | None:
    """Publish heartbeat v3 when due; return the last publish time."""
    if last_run is not None and (now - last_run).total_seconds() < config.heartbeat_interval_s:
        return last_run
    now_ns = int(now.value)
    refresh_window_states(
        states, dedupe.rest_outcomes, intervals, config.grid_health_window_s, now_ns,
        derive_lag_s=config.derive_lag_allowance_s,
        observed_since_ns=int(started_at.value),
    )
    lag_s: float | None = last_report.lag_s if last_report is not None else None
    hot_bytes, archive_bytes = disk_usage_bytes(capture_root)
    retention = dict(retention_state)
    retention["raw_hot_bytes"] = hot_bytes
    retention["raw_archive_bytes"] = archive_bytes
    payload = build_heartbeat_payload(
        capture_root=capture_root,
        now_iso=now.isoformat(),
        started_at_iso=started_at.isoformat(),
        normalizer={
            "last_run_at": now.isoformat(),
            "last_success_at": last_success_at.isoformat() if last_success_at is not None else None,
            "consecutive_failures": consecutive_failures,
            "lag_s": lag_s,
            "pending_complete_bytes": _pending_complete_bytes(
                capture_root / "raw" / "hot", checkpoint
            ) if checkpoint is not None else 0,
            "last_error": last_error,
        },
        states=states,
        compaction={
            **dict(compaction_state),
            "archived_days_pending": [
                {"stream": stream, "day": day}
                for stream, day in due_compactions(capture_root, config=config, now=now)
            ],
        },
        retention=retention,
        cutoff_utc=config.reference_capture_after_utc,
        now=now,
    )
    try:
        write_heartbeat_atomic(capture_root, payload)
    except OSError as exc:
        _logger.warning("[SYS] stage=normalizer status=HEARTBEAT_FAILED error=%s", exc)
    return now


