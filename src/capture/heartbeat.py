"""Capture heartbeat document and its atomic file replacement."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .journal import PARTIAL_SUFFIX
from .rest import RestStatus
from .ws import WsStatus


def _ns_to_iso(value_ns: int | None) -> str | None:
    if value_ns is None:
        return None
    return datetime.fromtimestamp(value_ns / 1_000_000_000, tz=UTC).isoformat().replace("+00:00", "Z")


def build_capture_heartbeat(
    *,
    slot: str,
    pid: int,
    fingerprint: str | None,
    started_at_ns: int,
    stopped_at_ns: int | None,
    rest: Mapping[str, RestStatus],
    ws: WsStatus,
    last_flush_at_ns: int | None,
    flush_failures: int,
    now_ns: int,
) -> dict[str, Any]:
    """Return the contract heartbeat document: all times ISO UTC strings or null, ``ready`` included.

    ``ready`` is True iff ``rest["book_ticker"].first_ok_at >= started_at`` and
    ``ws.first_frame_at >= started_at`` and ``flush_failures == 0``.
    """
    book = rest.get("book_ticker")
    ready = (
        flush_failures == 0
        and book is not None
        and book.first_ok_at_ns is not None
        and book.first_ok_at_ns >= started_at_ns
        and ws.first_frame_at_ns is not None
        and ws.first_frame_at_ns >= started_at_ns
    )
    return {
        "slot": slot,
        "pid": pid,
        "fingerprint": fingerprint,
        "started_at": _ns_to_iso(started_at_ns),
        "stopped_at": _ns_to_iso(stopped_at_ns),
        "rest": {
            name: {
                "first_ok_at": _ns_to_iso(entry.first_ok_at_ns),
                "last_ok_at": _ns_to_iso(entry.last_ok_at_ns),
                "consecutive_failures": entry.consecutive_failures,
                "dropped_records": entry.dropped_records,
            }
            for name, entry in rest.items()
        },
        "ws": {
            "connected_at": _ns_to_iso(ws.connected_at_ns),
            "first_frame_at": _ns_to_iso(ws.first_frame_at_ns),
            "last_frame_at": _ns_to_iso(ws.last_frame_at_ns),
            "reconnects": ws.reconnects,
            "pending_dropped": ws.pending_dropped,
        },
        "last_flush_at": _ns_to_iso(last_flush_at_ns),
        "flush_failures": flush_failures,
        "ts": _ns_to_iso(now_ns),
        "ready": ready,
    }


def write_capture_heartbeat(capture_root: Path, slot: str, payload: Mapping[str, Any]) -> Path:
    """Atomically replace ``raw/capture_<slot>.json`` (same-dir ``.partial`` + fsync + ``os.replace``).

    The deploy handover gate and the daemon watchdog read this file from the host, so it must
    never be observed half-written.
    """
    raw_dir = capture_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"capture_{slot}.json"
    partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        with open(partial, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, dest)
        try:
            fd = os.open(raw_dir, os.O_RDONLY)
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
