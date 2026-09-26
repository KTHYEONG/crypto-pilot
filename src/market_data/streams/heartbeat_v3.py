"""Heartbeat v3 published by the raw-first normalizer for the daemon watchdog."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

HEARTBEAT_NAME: str = "recorder_heartbeat.json"
HEARTBEAT_SCHEMA_VERSION: int = 3
CAPTURE_SLOTS: tuple[str, str] = ("blue", "green")

_logger = logging.getLogger(__name__)


def capture_heartbeat_path(capture_root: Path, slot: str) -> Path:
    """``<capture_root>/raw/capture_<slot>.json`` — written by spec 01's capture process."""
    return Path(capture_root) / "raw" / f"capture_{slot}.json"


def read_capture_heartbeats(capture_root: Path) -> dict[str, Mapping[str, Any] | None]:
    """Decode both slot heartbeats; a missing or undecodable file maps to ``None`` (never raises)."""
    heartbeats: dict[str, Mapping[str, Any] | None] = {}
    for slot in CAPTURE_SLOTS:
        try:
            raw = json.loads(capture_heartbeat_path(capture_root, slot).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _logger.debug("[DATA] stage=heartbeat_v3 status=CAPTURE_UNREADABLE slot=%s error=%s", slot, exc)
            heartbeats[slot] = None
            continue
        heartbeats[slot] = raw if isinstance(raw, dict) else None
    return heartbeats


def write_heartbeat_atomic(capture_root: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace ``<capture_root>/recorder_heartbeat.json`` (same-dir ``.partial`` + fsync + os.replace)."""
    dest = Path(capture_root) / HEARTBEAT_NAME
    partial = dest.with_name(dest.name + ".partial")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        with open(partial, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, dest)
        try:
            fd = os.open(dest.parent, os.O_RDONLY)
        except OSError:
            return dest
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
    return dest


def new_stream_states() -> dict[str, dict[str, Any]]:
    """Fresh per-dataset heartbeat accumulators for the normalizer loop."""
    return {
        "book_ticker": {
            "interval_s": None,
            "last_grid": None, "last_persisted_at": None, "rows_last_write": 0,
            "rejected_rows_last_sample": 0, "rejected_fraction_last_sample": 0.0,
            "rejected_rows_total": 0, "consecutive_rejecting_points": 0,
            "window_expected_points": 0, "window_captured_points": 0,
            "duplicates_dropped_total": 0,
        },
        "premium_index": {
            "interval_s": None,
            "last_grid": None, "last_persisted_at": None, "rows_last_write": 0,
            "rejected_rows_last_sample": 0, "rejected_fraction_last_sample": 0.0,
            "rejected_rows_total": 0, "consecutive_rejecting_points": 0,
            "window_expected_points": 0, "window_captured_points": 0,
            "duplicates_dropped_total": 0,
        },
        "force_order": {
            "last_frame_recv_at": None, "last_persisted_at": None, "frames_total": 0,
            "duplicates_dropped_total": 0, "parse_failures_total": 0,
        },
    }


def fold_cycle_report(
    states: dict[str, dict[str, Any]],
    report: Mapping[str, Any],
    outcomes: list[tuple[int, str, bool, int, int, float]],
    ws_last_recv_ns: int | None,
    now_iso: str,
) -> None:
    """Fold one cycle's report and fresh grid outcomes into the heartbeat accumulators."""
    rows = report.get("rows_written", {})
    dups = report.get("duplicates_dropped", {})
    for stream in ("book_ticker", "premium_index"):
        state = states[stream]
        state["rows_last_write"] = int(rows.get(stream, 0))
        state["duplicates_dropped_total"] = int(state["duplicates_dropped_total"]) + int(dups.get(stream, 0))
        if int(rows.get(stream, 0)) > 0:
            state["last_persisted_at"] = now_iso
        for _grid_ns, outcome_stream, ok, _rows, rejected, _fraction in outcomes:
            if outcome_stream != stream or not ok:
                continue
            state["rejected_rows_total"] = int(state["rejected_rows_total"]) + int(rejected)
    force = states["force_order"]
    force["frames_total"] = int(force["frames_total"]) + int(rows.get("force_order", 0))
    force["duplicates_dropped_total"] = int(force["duplicates_dropped_total"]) + int(dups.get("force_order", 0))
    force["parse_failures_total"] = int(force["parse_failures_total"]) + int(report.get("parse_failures", 0))
    if int(rows.get("force_order", 0)) > 0:
        force["last_persisted_at"] = now_iso
    if ws_last_recv_ns is not None:
        force["last_frame_recv_at"] = _ns_to_iso(ws_last_recv_ns)


def refresh_window_states(
    states: dict[str, dict[str, Any]],
    outcomes: Any,
    intervals: Mapping[str, int],
    health_window_s: float,
    now_ns: int,
    *,
    derive_lag_s: float,
    observed_since_ns: int | None = None,
) -> None:
    """Recompute window capture ratios, last samples and rejecting streaks from outcome history.

    Capture health compares grid points that *should already be derived* with those that were: the
    window covers grid points ``g`` with ``lo <= g <= now - derive_lag_s``, where ``lo`` is
    ``now - health_window_s`` but never earlier than ``observed_since_ns``. Outcomes live only in
    memory, so right after a normalizer restart the history covers just the time since
    ``observed_since_ns``; counting the full window would report a degraded ratio for most of an hour
    after every deploy although nothing was lost. Points newer than ``derive_lag_s`` are excluded from
    both counts, because the capture may have written them while the normalizer has not derived them
    yet; without this a healthy point could be counted as captured but not expected. Both counts use
    the same grid-aligned range, so ``captured <= expected`` always holds.

    Args:
        derive_lag_s: Worst-case delay between a grid point being sampled and being derived
            (normalizer cadence plus sampling slack).
        observed_since_ns: Normalizer start; grid points before it were never observed by this
            process.
    """
    window_ns = int(health_window_s * 1_000_000_000)
    lo_ns = now_ns - window_ns
    if observed_since_ns is not None:
        lo_ns = max(lo_ns, observed_since_ns)
    hi_ns = now_ns - int(derive_lag_s * 1_000_000_000)
    cutoff = lo_ns
    for stream in ("book_ticker", "premium_index"):
        state = states[stream]
        interval_s = intervals[stream]
        interval_ns = interval_s * 1_000_000_000
        state["interval_s"] = interval_s
        first_grid = -(-lo_ns // interval_ns)
        last_grid = hi_ns // interval_ns
        state["window_expected_points"] = max(0, last_grid - first_grid + 1)
        captured: set[int] = set()
        last: tuple[int, str, bool, int, int, float] | None = None
        for outcome in outcomes:
            grid_ns, outcome_stream, ok, _rows, _rejected, _fraction = outcome
            if outcome_stream != stream or grid_ns < cutoff:
                continue
            if ok:
                if grid_ns <= hi_ns:
                    captured.add(grid_ns)
                if last is None or grid_ns >= last[0]:
                    last = outcome
        state["window_captured_points"] = len(captured)
        if last is not None and last[2]:
            state["last_grid"] = _ns_to_iso(last[0])
            state["rejected_rows_last_sample"] = last[4]
            state["rejected_fraction_last_sample"] = last[5]
        streak = 0
        for outcome in reversed(list(outcomes)):
            grid_ns, outcome_stream, ok, _rows, rejected, _ = outcome
            if outcome_stream != stream or grid_ns < cutoff:
                continue
            if not ok:
                continue
            if rejected > 0:
                streak += 1
            else:
                break
        state["consecutive_rejecting_points"] = streak


def _ns_to_iso(value_ns: int | None) -> str | None:
    if value_ns is None:
        return None
    try:
        return str(pd.Timestamp(value_ns, unit="ns", tz="UTC").isoformat())
    except (TypeError, ValueError):
        return None


def reference_section(capture_root: Path, cutoff_utc: str, now: Any) -> dict[str, Any]:
    """Reference completeness for today and yesterday from capture-written files."""
    today_ts = pd.Timestamp(now).tz_convert("UTC").normalize()
    yesterday_ts = today_ts - pd.Timedelta(days=1)
    today = today_ts.strftime("%Y%m%d")
    endpoints = {}
    for name in ("exchange_info", "funding_info", "asset_index"):
        endpoints[name] = {
            "captured": (Path(capture_root) / "reference" / name / f"{today}.json.gz").is_file()
        }
    previous_complete = all(
        (Path(capture_root) / "reference" / name / f"{yesterday_ts.strftime('%Y%m%d')}.json.gz").is_file()
        for name in ("exchange_info", "funding_info", "asset_index")
    )
    return {
        "day": today,
        "cutoff_utc": cutoff_utc,
        "endpoints": endpoints,
        "previous_day": yesterday_ts.strftime("%Y%m%d"),
        "previous_day_complete": previous_complete,
    }


def disk_usage_bytes(capture_root: Path) -> tuple[int, int]:
    """``(raw_hot_bytes, raw_archive_bytes)`` walking the trees while skipping ``.partial`` files."""
    hot = 0
    archive = 0
    for base, bucket in ((Path(capture_root) / "raw" / "hot", 0), (Path(capture_root) / "raw" / "archive", 1)):
        total = 0
        if base.is_dir():
            for path in base.rglob("*"):
                try:
                    if path.is_file() and not path.name.endswith(".partial"):
                        total += path.stat().st_size
                except OSError:
                    continue
        if bucket == 0:
            hot = total
        else:
            archive = total
    return hot, archive


def local_footprint_bytes(capture_root: Path, liquidations_dir: Path) -> int:
    """Total bytes of raw journals plus derived parquet trees, skipping ``.partial`` and temp files.

    Used against the local disk budget while the backup-gated prune is blocked, so a long block that
    would exhaust the disk alerts even before the duration threshold.
    """
    total = 0
    trees = (
        Path(capture_root) / "raw",
        Path(capture_root) / "book_ticker",
        Path(capture_root) / "premium_index",
        Path(liquidations_dir),
    )
    for base in trees:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            try:
                if path.is_file() and not path.name.endswith((".partial", ".tmp")):
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def build_heartbeat_payload(
    *,
    capture_root: Path,
    now_iso: str,
    started_at_iso: str,
    normalizer: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
    compaction: Mapping[str, Any],
    retention: Mapping[str, Any],
    cutoff_utc: str,
    now: Any,
) -> dict[str, Any]:
    """Assemble the v3 heartbeat document; ``capture.*`` is a verbatim passthrough."""
    return {
        "schema_version": HEARTBEAT_SCHEMA_VERSION,
        "ts": now_iso,
        "started_at": started_at_iso,
        "normalizer": dict(normalizer),
        "streams": {name: dict(state) for name, state in states.items()},
        "reference": reference_section(capture_root, cutoff_utc, now),
        "compaction": dict(compaction),
        "retention": dict(retention),
        "capture": dict(read_capture_heartbeats(capture_root)),
    }
