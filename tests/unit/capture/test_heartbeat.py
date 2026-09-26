"""Invariant guards for the capture heartbeat document and its atomic write."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.capture.heartbeat import build_capture_heartbeat, write_capture_heartbeat
from src.capture.rest import RestStatus
from src.capture.ws import WsStatus


def _ns(hour: int, minute: int = 0) -> int:
    moment = datetime(2026, 9, 26, hour, minute, tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000_000)


def _payload(**overrides: Any) -> dict[str, Any]:
    started = _ns(10, 0)
    rest = {"book_ticker": RestStatus(first_ok_at_ns=_ns(10, 1), last_ok_at_ns=_ns(10, 1))}
    ws = WsStatus(connected_at_ns=_ns(10, 0), first_frame_at_ns=_ns(10, 1), last_frame_at_ns=_ns(10, 1))
    values: dict[str, Any] = {
        "slot": "blue",
        "pid": 7,
        "fingerprint": "abc",
        "started_at_ns": started,
        "stopped_at_ns": None,
        "rest": rest,
        "ws": ws,
        "last_flush_at_ns": _ns(10, 1),
        "flush_failures": 0,
        "now_ns": _ns(10, 2),
    }
    values.update(overrides)
    return build_capture_heartbeat(**values)  # type: ignore[arg-type]


def test_ready_requires_all_three() -> None:
    """Spec 01: ready is True only with fresh first-ok, fresh first-frame and zero failures."""
    assert _payload()["ready"] is True
    assert _payload(flush_failures=1)["ready"] is False
    assert _payload(ws=WsStatus())["ready"] is False
    assert _payload(rest={"book_ticker": RestStatus()})["ready"] is False
    stale_rest = {"book_ticker": RestStatus(first_ok_at_ns=_ns(9, 59), last_ok_at_ns=_ns(9, 59))}
    assert _payload(rest=stale_rest)["ready"] is False
    stale_ws = WsStatus(first_frame_at_ns=_ns(9, 59))
    assert _payload(ws=stale_ws)["ready"] is False


def test_atomic_replace_leaves_no_partial(tmp_path: Path) -> None:
    """Spec 01: consecutive heartbeat writes parse as the latest payload with no residue."""
    first = _payload(now_ns=_ns(10, 2))
    second = _payload(now_ns=_ns(10, 3))
    write_capture_heartbeat(tmp_path, "blue", first)
    dest = write_capture_heartbeat(tmp_path, "blue", second)
    assert dest == tmp_path / "raw" / "capture_blue.json"
    assert json.loads(dest.read_text(encoding="utf-8"))["ts"] == second["ts"]
    assert list((tmp_path / "raw").glob("*.partial")) == []


def test_stopped_marker(tmp_path: Path) -> None:
    """Spec 01: stopped_at is an ISO string when set; ready is still computed."""
    payload = _payload(stopped_at_ns=_ns(11, 0))
    assert payload["stopped_at"] == "2026-09-26T11:00:00Z"
    assert payload["ready"] is True
    write_capture_heartbeat(tmp_path, "green", payload)
    assert (tmp_path / "raw" / "capture_green.json").exists()


def test_heartbeat_schema_keys() -> None:
    """Contract keys match the spec exactly."""
    payload = _payload()
    assert set(payload) == {
        "slot", "pid", "fingerprint", "started_at", "stopped_at", "rest", "ws",
        "last_flush_at", "flush_failures", "ts", "ready",
    }
    assert set(payload["rest"]["book_ticker"]) == {"first_ok_at", "last_ok_at", "consecutive_failures", "dropped_records"}
    assert set(payload["ws"]) == {"connected_at", "first_frame_at", "last_frame_at", "reconnects", "pending_dropped"}


def test_write_tolerates_directory_fsync_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unopenable or unsyncable heartbeat directories still leave a valid file."""
    import os as _os

    payload = _payload()
    monkeypatch.setattr(_os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    dest = write_capture_heartbeat(tmp_path, "blue", payload)
    assert dest.exists()
    monkeypatch.undo()
    calls: list[int] = []
    real_fsync = _os.fsync

    def selective_fsync(fd: int) -> None:
        calls.append(fd)
        if len(calls) > 1:
            raise OSError("dir")
        real_fsync(fd)

    monkeypatch.setattr(_os, "fsync", selective_fsync)
    write_capture_heartbeat(tmp_path, "blue", payload)
    assert dest.exists()
    assert len(calls) >= 2


def test_write_failure_cleans_partial_and_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed atomic replace raises and leaves no partial behind."""
    import os as _os

    real_replace = _os.replace
    monkeypatch.setattr(_os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        write_capture_heartbeat(tmp_path, "blue", _payload())
    assert list((tmp_path / "raw").glob("*.partial")) == []
    _ = real_replace
