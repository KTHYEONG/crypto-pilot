"""Invariant guards for pure recorder-heartbeat evaluation."""

from __future__ import annotations

import pytest

from src.market_data.streams.recorder_health import (
    RecorderWatchThresholds,
    evaluate_recorder_heartbeat,
    read_recorder_heartbeat,
)

_DEFAULTS = RecorderWatchThresholds(
    heartbeat_stale_s=600.0,
    liquidation_silence_s=900.0,
    liquidation_max_failed_connections=5,
    sampler_stale_s=1800.0,
)

_NOW = "2026-09-24T06:35:00Z"


def _payload(**overrides):
    now = _NOW
    base = {
        "ts": now,
        "started_at": "2026-09-24T05:00:00Z",
        "book_ticker": {"last_success_at": now, "rows_last_flush": 1, "consecutive_failures": 0},
        "premium_index": {"last_success_at": now, "rows_last_flush": 1, "consecutive_failures": 0},
        "reference": {"last_success_at": now, "rows_last_flush": 3, "consecutive_failures": 0},
        "liquidations": {
            "last_event_at": now,
            "last_connected_at": now,
            "consecutive_failed_connections": 0,
            "last_disconnect_reason": None,
        },
    }
    base.update(overrides)
    return base


def test_healthy_heartbeat_yields_no_findings() -> None:
    """A fresh heartbeat with recent events and no failures is quiet."""
    now = _NOW
    payload = _payload(
        ts="2026-09-24T06:34:30Z",
        liquidations={
            "last_event_at": "2026-09-24T06:34:40Z",
            "last_connected_at": now,
            "consecutive_failed_connections": 0,
            "last_disconnect_reason": None,
        },
        book_ticker={"last_success_at": "2026-09-24T06:34:00Z"},
        premium_index={"last_success_at": "2026-09-24T06:34:00Z"},
    )
    assert (
        evaluate_recorder_heartbeat(
            payload,
            now=now,
            watch_started_at="2026-09-24T05:00:00Z",
            thresholds=_DEFAULTS,
        )
        == ()
    )


def test_sixty_eight_minute_outage_is_caught() -> None:
    """The 2026-09-24 zero-event outage fires a single silence finding with its age."""
    (finding,) = evaluate_recorder_heartbeat(
        _payload(liquidations={
            "last_event_at": "2026-09-24T06:19:00Z",
            "last_connected_at": "2026-09-24T06:19:00Z",
            "consecutive_failed_connections": 1,
            "last_disconnect_reason": "EVENT_STALL: silent",
        }),
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    assert finding.key == "liquidation_silent:liquidations"
    assert "age_s=960" in finding.detail


def test_silence_boundary_is_strict() -> None:
    """Exactly at the silence threshold nothing fires; one second later it does."""
    thresholds = _DEFAULTS
    at_threshold = evaluate_recorder_heartbeat(
        _payload(liquidations={
            "last_event_at": "2026-09-24T06:20:00Z",
            "last_connected_at": "2026-09-24T06:20:00Z",
            "consecutive_failed_connections": 0,
            "last_disconnect_reason": None,
        }),
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=thresholds,
    )
    assert all(f.check != "liquidation_silent" for f in at_threshold)
    (finding,) = [
        f
        for f in evaluate_recorder_heartbeat(
            _payload(liquidations={
                "last_event_at": "2026-09-24T06:19:59Z",
                "last_connected_at": "2026-09-24T06:19:59Z",
                "consecutive_failed_connections": 0,
                "last_disconnect_reason": None,
            }),
            now=_NOW,
            watch_started_at="2026-09-24T05:00:00Z",
            thresholds=thresholds,
        )
        if f.check == "liquidation_silent"
    ]
    assert finding.key == "liquidation_silent:liquidations"


def test_never_seen_event_ages_from_process_start() -> None:
    """A null last event falls back to the recorder start instant."""
    (finding,) = [
        f
        for f in evaluate_recorder_heartbeat(
            _payload(
                started_at="2026-09-24T06:15:00Z",
                liquidations={
                    "last_event_at": None,
                    "last_connected_at": "2026-09-24T06:15:00Z",
                    "consecutive_failed_connections": 0,
                    "last_disconnect_reason": None,
                },
            ),
            now=_NOW,
            watch_started_at="2026-09-24T05:00:00Z",
            thresholds=_DEFAULTS,
        )
        if f.check == "liquidation_silent"
    ]
    assert finding.subject == "liquidations"


def test_failed_connections_threshold() -> None:
    """The failing check fires at the count threshold, not one below."""
    healthy_liq = {
        "last_event_at": _NOW,
        "last_connected_at": _NOW,
        "consecutive_failed_connections": 5,
        "last_disconnect_reason": "DISCONNECTED: close",
    }
    (finding,) = evaluate_recorder_heartbeat(
        _payload(liquidations=healthy_liq),
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    assert finding.key == "liquidation_failing:liquidations"
    below = dict(healthy_liq, consecutive_failed_connections=4)
    assert (
        evaluate_recorder_heartbeat(
            _payload(liquidations=below),
            now=_NOW,
            watch_started_at="2026-09-24T05:00:00Z",
            thresholds=_DEFAULTS,
        )
        == ()
    )


def test_stale_heartbeat_suppresses_derived_checks() -> None:
    """Only the staleness itself is reported when the heartbeat describes the past."""
    (finding,) = evaluate_recorder_heartbeat(
        _payload(
            ts="2026-09-24T06:24:00Z",
            liquidations={
                "last_event_at": "2026-09-24T05:24:00Z",
                "last_connected_at": "2026-09-24T05:24:00Z",
                "consecutive_failed_connections": 9,
                "last_disconnect_reason": "DISCONNECTED: close",
            },
        ),
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    assert finding.key == "heartbeat_stale:recorder"


def test_missing_heartbeat_tolerated_during_startup() -> None:
    """An absent heartbeat is tolerated inside the grace window, then reported."""
    assert (
        evaluate_recorder_heartbeat(
            None,
            now=_NOW,
            watch_started_at="2026-09-24T06:30:00Z",
            thresholds=_DEFAULTS,
        )
        == ()
    )
    (finding,) = evaluate_recorder_heartbeat(
        None,
        now=_NOW,
        watch_started_at="2026-09-24T06:24:00Z",
        thresholds=_DEFAULTS,
    )
    assert finding.key == "heartbeat_missing:recorder"


def test_legacy_heartbeat_without_liquidation_entry_fails_closed() -> None:
    """A payload predating the liquidation entry counts as silent."""
    payload = {
        "ts": _NOW,
        "started_at": "2026-09-24T05:00:00Z",
        "book_ticker": {"last_success_at": _NOW},
        "premium_index": {"last_success_at": _NOW},
    }
    (finding,) = evaluate_recorder_heartbeat(
        payload,
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    assert finding.key == "liquidation_silent:liquidations"
    assert "entry=missing" in finding.detail


def test_stale_sampler_detected_per_dataset() -> None:
    """Only the stale dataset is reported when its sibling is fresh."""
    (finding,) = evaluate_recorder_heartbeat(
        _payload(premium_index={"last_success_at": "2026-09-24T06:04:00Z"}),
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    assert finding.key == "sampler_stale:premium_index"


def test_unreadable_file_reads_as_none(tmp_path) -> None:
    """Invalid JSON and non-object JSON both read as unavailable."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert read_recorder_heartbeat(bad) is None
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    assert read_recorder_heartbeat(listed) is None
    assert read_recorder_heartbeat(tmp_path / "absent.json") is None


def test_naive_clock_rejected() -> None:
    """Tz-naive evaluation instants fail closed with ValueError."""
    with pytest.raises(ValueError, match="now"):
        evaluate_recorder_heartbeat(
            _payload(),
            now="2026-09-24 06:35:00",
            watch_started_at="2026-09-24T05:00:00Z",
            thresholds=_DEFAULTS,
        )
    with pytest.raises(ValueError, match="watch_started_at"):
        evaluate_recorder_heartbeat(
            _payload(),
            now=_NOW,
            watch_started_at="2026-09-24 05:00:00",
            thresholds=_DEFAULTS,
        )


def test_timestamp_shapes_and_garbage() -> None:
    """Datetime objects parse like strings; bools, numbers, and garbage do not."""
    import datetime

    import pandas as pd

    object_payload = _payload(
        ts=pd.Timestamp(_NOW),
        started_at=datetime.datetime(2026, 9, 24, 5, 0, tzinfo=datetime.UTC),
        liquidations={
            "last_event_at": datetime.datetime(2026, 9, 24, 6, 34, 40, tzinfo=datetime.UTC),
            "last_connected_at": pd.Timestamp(_NOW),
            "consecutive_failed_connections": 0,
            "last_disconnect_reason": None,
        },
        book_ticker={"last_success_at": pd.Timestamp(_NOW)},
        premium_index={"last_success_at": pd.Timestamp(_NOW)},
    )
    assert (
        evaluate_recorder_heartbeat(
            object_payload,
            now=pd.Timestamp(_NOW),
            watch_started_at=pd.Timestamp("2026-09-24T05:00:00Z"),
            thresholds=_DEFAULTS,
        )
        == ()
    )
    for bad_ts in (True, 12345, "not-a-time"):
        (finding,) = evaluate_recorder_heartbeat(
            _payload(ts=bad_ts),
            now=_NOW,
            watch_started_at="2026-09-24T05:00:00Z",
            thresholds=_DEFAULTS,
        )
        assert finding.key == "heartbeat_stale:recorder"
        assert "ts=missing" in finding.detail


def test_malformed_liquidation_entry_fires_both_checks() -> None:
    """An entry with unparseable event time and counter fails both of its checks."""
    payload = _payload()
    del payload["started_at"]
    payload["liquidations"] = {
        "last_event_at": "tomorrow",
        "last_connected_at": _NOW,
        "consecutive_failed_connections": "lots",
        "last_disconnect_reason": None,
    }
    findings = evaluate_recorder_heartbeat(
        payload,
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    assert [f.key for f in findings] == [
        "liquidation_failing:liquidations",
        "liquidation_silent:liquidations",
    ]
    assert "entry=malformed" in findings[0].detail
    assert "entry=malformed" in findings[1].detail


def test_malformed_sampler_entry_and_bool_counter() -> None:
    """Naive sampler timestamps and boolean counters count as malformed."""
    payload = _payload(
        book_ticker={"last_success_at": "2026-09-22 10:00:00"},
        liquidations={
            "last_event_at": _NOW,
            "last_connected_at": _NOW,
            "consecutive_failed_connections": True,
            "last_disconnect_reason": None,
        },
    )
    del payload["started_at"]
    findings = evaluate_recorder_heartbeat(
        payload,
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    assert [f.key for f in findings] == [
        "liquidation_failing:liquidations",
        "sampler_stale:book_ticker",
    ]
    assert "entry=malformed" in findings[1].detail


def test_missing_sampler_entry_and_started_at_fallback() -> None:
    """An absent dataset entry fails closed; a null success ages from process start."""
    payload = _payload(started_at="2026-09-24T06:00:00Z")
    del payload["premium_index"]
    payload["book_ticker"] = {"last_success_at": None}
    findings = evaluate_recorder_heartbeat(
        payload,
        now=_NOW,
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    by_key = {f.key: f for f in findings}
    assert by_key["sampler_stale:premium_index"].detail == "entry=missing"
    assert "age_s=2100" in by_key["sampler_stale:book_ticker"].detail
    assert "last_success_at=none" in by_key["sampler_stale:book_ticker"].detail
