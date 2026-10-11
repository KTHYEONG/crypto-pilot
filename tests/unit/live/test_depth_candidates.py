"""Invariant guards for the exec-depth candidate sidecar."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from src.live.depth_candidates import (
    DEPTH_CANDIDATES_NAME,
    read_depth_candidates,
    write_depth_candidates,
)

_DAY = pd.Timestamp("2026-09-22", tz="UTC")


def test_roundtrip_same_day(tmp_path: Path) -> None:
    """Written symbols read back in order under the read limit."""
    path = tmp_path / DEPTH_CANDIDATES_NAME
    write_depth_candidates(path, _DAY, ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"])
    assert read_depth_candidates(path, _DAY, limit=3) == ("AUSDT", "BUSDT", "CUSDT")


def test_stale_day_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A sidecar from the previous day reads as empty with a stale log."""
    path = tmp_path / DEPTH_CANDIDATES_NAME
    write_depth_candidates(path, _DAY - pd.Timedelta(days=1), ["AUSDT"])
    with caplog.at_level(logging.INFO, logger="src.live.depth_candidates"):
        assert read_depth_candidates(path, _DAY, limit=10) == ()
    assert any("status=stale" in record.message for record in caplog.records)


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing sidecar reads as empty without raising."""
    assert read_depth_candidates(tmp_path / DEPTH_CANDIDATES_NAME, _DAY, limit=10) == ()


def test_malformed_file_returns_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Invalid JSON reads as empty with a malformed log and no exception."""
    path = tmp_path / DEPTH_CANDIDATES_NAME
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="src.live.depth_candidates"):
        assert read_depth_candidates(path, _DAY, limit=10) == ()
    assert any("status=malformed" in record.message for record in caplog.records)


def test_zero_limit_disables(tmp_path: Path) -> None:
    """A non-positive limit always reads as empty."""
    path = tmp_path / DEPTH_CANDIDATES_NAME
    write_depth_candidates(path, _DAY, ["AUSDT"])
    assert read_depth_candidates(path, _DAY, limit=0) == ()


@pytest.mark.parametrize("stamp", [None, 0, "NaT", "not-a-date"])
def test_invalid_decision_day_is_malformed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, stamp: object,
) -> None:
    path = tmp_path / DEPTH_CANDIDATES_NAME
    path.write_text(json.dumps({"decision_day": stamp, "symbols": ["AUSDT"]}), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="src.live.depth_candidates"):
        assert read_depth_candidates(path, _DAY, limit=10) == ()
    assert len(caplog.records) == 1
    assert "status=malformed" in caplog.records[0].message


def test_utc_date_matches_across_timezones(tmp_path: Path) -> None:
    path = tmp_path / DEPTH_CANDIDATES_NAME
    write_depth_candidates(path, pd.Timestamp("2026-09-23T01:00:00+09:00"), ["AUSDT"])
    assert read_depth_candidates(path, _DAY + pd.Timedelta(hours=23), limit=1) == ("AUSDT",)


def test_failed_replacement_preserves_previous_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / DEPTH_CANDIDATES_NAME
    write_depth_candidates(path, _DAY, ["AUSDT"])
    before = path.read_bytes()

    def fail_replace(*args: object) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr("src.common.durable_io.os.replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        write_depth_candidates(path, _DAY, ["BUSDT"])
    assert path.read_bytes() == before
    assert read_depth_candidates(path, _DAY, limit=10) == ("AUSDT",)
    assert list(tmp_path.glob("*.tmp")) == []


def test_empty_list_writes_valid_sidecar(tmp_path: Path) -> None:
    """An empty ranking still writes a readable sidecar."""
    path = tmp_path / DEPTH_CANDIDATES_NAME
    write_depth_candidates(path, _DAY, [])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["symbols"] == []
    assert read_depth_candidates(path, _DAY, limit=10) == ()


def test_non_object_payload_returns_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A valid-JSON but non-object sidecar reads as malformed."""
    path = tmp_path / DEPTH_CANDIDATES_NAME
    path.write_text(json.dumps(["AUSDT"]), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="src.live.depth_candidates"):
        assert read_depth_candidates(path, _DAY, limit=10) == ()
    assert any("status=malformed" in record.message for record in caplog.records)


def test_non_list_symbols_returns_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A sidecar whose symbols are not strings reads as malformed."""
    path = tmp_path / DEPTH_CANDIDATES_NAME
    path.write_text(
        json.dumps({"decision_day": _DAY.isoformat(), "symbols": "AUSDT"}), encoding="utf-8",
    )
    with caplog.at_level(logging.INFO, logger="src.live.depth_candidates"):
        assert read_depth_candidates(path, _DAY, limit=10) == ()
    assert any("status=malformed" in record.message for record in caplog.records)
