"""Invariant guards for heartbeat v3 assembly and atomic publication."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.market_data.streams.heartbeat_v3 import (
    CAPTURE_SLOTS,
    HEARTBEAT_NAME,
    HEARTBEAT_SCHEMA_VERSION,
    build_heartbeat_payload,
    disk_usage_bytes,
    fold_cycle_report,
    new_stream_states,
    read_capture_heartbeats,
    reference_section,
    refresh_window_states,
    write_heartbeat_atomic,
)


def _now() -> pd.Timestamp:
    return pd.Timestamp("2026-09-26T12:00:00Z")


def test_capture_heartbeat_paths() -> None:
    """Both slots resolve under raw/ with the spec 01 filename."""
    from src.market_data.streams.heartbeat_v3 import capture_heartbeat_path

    assert capture_heartbeat_path("/cap", "blue").name == "capture_blue.json"
    assert set(CAPTURE_SLOTS) == {"blue", "green"}
    assert HEARTBEAT_NAME == "recorder_heartbeat.json"
    assert HEARTBEAT_SCHEMA_VERSION == 3


def test_read_capture_heartbeats_never_raises(tmp_path: Path) -> None:
    """Missing, corrupt and scalar slot files all map to None."""
    assert read_capture_heartbeats(tmp_path) == {"blue": None, "green": None}
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "capture_blue.json").write_text("{oops")
    (raw / "capture_green.json").write_text("[1]")
    assert read_capture_heartbeats(tmp_path) == {"blue": None, "green": None}
    (raw / "capture_blue.json").write_text(json.dumps({"ready": True}))
    assert read_capture_heartbeats(tmp_path) == {"blue": {"ready": True}, "green": None}


def test_write_heartbeat_atomic_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unfsyncable directories still publish; failed replaces clean up and raise."""
    import os as _os

    payload = {"schema_version": 3}
    dest = write_heartbeat_atomic(tmp_path, payload)
    assert json.loads(dest.read_text()) == payload
    monkeypatch.setattr(_os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    write_heartbeat_atomic(tmp_path, payload)
    monkeypatch.undo()
    calls: list[int] = []
    real_fsync = _os.fsync

    def selective_fsync(fd: int) -> None:
        calls.append(fd)
        if len(calls) > 1:
            raise OSError("no")
        real_fsync(fd)

    monkeypatch.setattr(_os, "fsync", selective_fsync)
    write_heartbeat_atomic(tmp_path, payload)
    assert len(calls) >= 2
    monkeypatch.undo()
    monkeypatch.setattr(_os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        write_heartbeat_atomic(tmp_path, payload)
    assert list(tmp_path.glob("*.partial")) == []


def test_fold_and_refresh_window_states() -> None:
    """Cycle reports accumulate; windows derive ratios, samples and streaks."""
    states = new_stream_states()
    now = _now()
    outcomes = [
        (int((now - pd.Timedelta(minutes=30)).value), "book_ticker", True, 10, 0, 0.0),
        (int((now - pd.Timedelta(minutes=10)).value), "book_ticker", True, 10, 2, 0.2),
        (int((now - pd.Timedelta(minutes=5)).value), "book_ticker", False, 0, 0, 0.0),
    ]
    fold_cycle_report(
        states,
        {"rows_written": {"book_ticker": 10, "force_order": 3},
         "duplicates_dropped": {"book_ticker": 1, "force_order": 2}, "parse_failures": 4},
        outcomes,
        int(now.value),
        now.isoformat(),
    )
    assert states["book_ticker"]["rows_last_write"] == 10
    assert states["book_ticker"]["duplicates_dropped_total"] == 1
    assert states["book_ticker"]["rejected_rows_total"] == 2
    assert states["book_ticker"]["last_persisted_at"] == now.isoformat()
    assert states["force_order"]["frames_total"] == 3
    assert states["force_order"]["parse_failures_total"] == 4
    assert states["force_order"]["last_frame_recv_at"] is not None
    refresh_window_states(
        states, outcomes, {"book_ticker": 60, "premium_index": 300}, 3600.0, int(now.value), derive_lag_s=45.0
    )
    assert states["book_ticker"]["window_expected_points"] == 60
    assert states["book_ticker"]["window_captured_points"] == 2
    assert states["book_ticker"]["rejected_rows_last_sample"] == 2
    assert states["book_ticker"]["consecutive_rejecting_points"] == 1


def test_reference_and_disk_sections(tmp_path: Path) -> None:
    """Reference completeness reads capture files; disk walks skip partials."""
    ref = tmp_path / "reference" / "exchange_info"
    ref.mkdir(parents=True)
    (ref / "20260926.json.gz").write_bytes(b"x")
    section = reference_section(tmp_path, "00:05", _now())
    assert section["day"] == "20260926"
    assert section["endpoints"]["exchange_info"]["captured"] is True
    assert section["endpoints"]["funding_info"]["captured"] is False
    assert section["previous_day"] == "20260925"
    assert section["previous_day_complete"] is False
    hot = tmp_path / "raw" / "hot"
    hot.mkdir(parents=True)
    (hot / "a.gz").write_bytes(b"12345")
    (hot / "b.partial").write_bytes(b"1234567890")
    assert disk_usage_bytes(tmp_path) == (5, 0)


def test_build_payload_passes_capture_through(tmp_path: Path) -> None:
    """The assembled heartbeat carries verbatim capture slots and the v3 schema."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "capture_blue.json").write_text(json.dumps({"ready": True, "ts": "x"}))
    payload = build_heartbeat_payload(
        capture_root=tmp_path, now_iso=_now().isoformat(), started_at_iso=_now().isoformat(),
        normalizer={"consecutive_failures": 0}, states=new_stream_states(),
        compaction={}, retention={}, cutoff_utc="00:05", now=_now(),
    )
    assert payload["schema_version"] == 3
    assert payload["capture"] == {"blue": {"ready": True, "ts": "x"}, "green": None}
    assert set(payload["streams"]) == {"book_ticker", "premium_index", "force_order"}


def test_ns_to_iso_corners() -> None:
    """None stays None; unparsable values stay None instead of raising."""
    from src.market_data.streams.heartbeat_v3 import _ns_to_iso

    assert _ns_to_iso(None) is None
    assert _ns_to_iso(object()) is None  # type: ignore[arg-type]
    assert _ns_to_iso(0) is not None


def test_disk_usage_skips_vanished_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file vanishing mid-walk is skipped safely."""
    from pathlib import Path as _Path

    hot = tmp_path / "raw" / "hot"
    hot.mkdir(parents=True)
    target = hot / "a.gz"
    target.write_bytes(b"12345")
    real_stat = _Path.stat

    def _vanishing(self: Path, *args: object, **kwargs: object) -> object:
        if self == target:
            raise OSError("gone")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "stat", _vanishing)
    assert disk_usage_bytes(tmp_path) == (0, 0)


def _grid(text: str) -> int:
    return int(pd.Timestamp(text).value)


def test_window_excludes_points_not_yet_derived() -> None:
    """A grid point captured seconds ago but still inside the derive lag is neither expected nor captured.

    Regression: right after a restart the window floor undercounted expected points while the freshly
    derived point was already counted as captured, which read as a malformed (captured > expected)
    heartbeat and alerted a healthy sampler.
    """
    states = new_stream_states()
    now = pd.Timestamp("2026-09-26T15:25:14Z")
    outcomes = [
        (_grid("2026-09-26T15:20:00Z"), "premium_index", True, 910, 0, 0.0),
        (_grid("2026-09-26T15:25:00Z"), "premium_index", True, 910, 0, 0.0),
    ]
    refresh_window_states(
        states,
        outcomes,
        {"book_ticker": 60, "premium_index": 300},
        3600.0,
        int(now.value),
        derive_lag_s=60.0,
        observed_since_ns=_grid("2026-09-26T15:16:43Z"),
    )
    state = states["premium_index"]
    assert state["window_expected_points"] == 1
    assert state["window_captured_points"] == 1
    assert state["window_captured_points"] <= state["window_expected_points"]
    assert state["last_grid"] == "2026-09-26T15:25:00+00:00"
    assert state["interval_s"] == 300


def test_window_is_empty_before_first_grid_point_after_start() -> None:
    """Within one interval of start there is nothing to expect yet, so no ratio can be degraded."""
    states = new_stream_states()
    now = pd.Timestamp("2026-09-26T15:18:15Z")
    refresh_window_states(
        states,
        [],
        {"book_ticker": 60, "premium_index": 300},
        3600.0,
        int(now.value),
        derive_lag_s=60.0,
        observed_since_ns=_grid("2026-09-26T15:16:43Z"),
    )
    assert states["premium_index"]["window_expected_points"] == 0
    assert states["premium_index"]["window_captured_points"] == 0
    assert states["book_ticker"]["window_expected_points"] == 1


def test_local_footprint_counts_finished_files_and_survives_vanishing_ones(tmp_path: Path, monkeypatch) -> None:
    """Journals and parquet count; temp files are skipped; a file removed mid-walk never raises."""
    from src.market_data.streams.heartbeat_v3 import local_footprint_bytes

    (tmp_path / "raw" / "hot").mkdir(parents=True)
    (tmp_path / "raw" / "hot" / "a.gz").write_bytes(b"12345")
    (tmp_path / "raw" / "hot" / "b.partial").write_bytes(b"xxxxxxxxxx")
    (tmp_path / "book_ticker").mkdir()
    (tmp_path / "book_ticker" / "1.parquet").write_bytes(b"123")
    liq = tmp_path / "liq"
    liq.mkdir()
    (liq / "l.parquet").write_bytes(b"1234567")
    (liq / ".l.tmp").write_bytes(b"zz")
    assert local_footprint_bytes(tmp_path, liq) == 5 + 3 + 7

    real_stat = Path.stat

    def _vanishing(self: Path, *args, **kwargs):
        if self.name == "a.gz":
            raise FileNotFoundError(self.name)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _vanishing)
    assert local_footprint_bytes(tmp_path, liq) == 3 + 7
