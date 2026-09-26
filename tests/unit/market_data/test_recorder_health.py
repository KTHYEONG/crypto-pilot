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
    sampler_max_consecutive_failures=5,
    min_capture_ratio=0.9,
    capture_ratio_min_points=10,
    persist_stale_s=1200.0,
    max_consecutive_flush_failures=3,
    reference_grace_s=3600.0,
    rejected_fraction_alert=0.01,
    rejected_max_consecutive_points=60,
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
            watch_started_at="2026-09-24T06:30:00Z",
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
        watch_started_at="2026-09-24T06:30:00Z",
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
        watch_started_at="2026-09-24T06:30:00Z",
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
            watch_started_at="2026-09-24T06:30:00Z",
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
            watch_started_at="2026-09-24T06:30:00Z",
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
        watch_started_at="2026-09-24T06:30:00Z",
        thresholds=_DEFAULTS,
    )
    assert finding.key == "liquidation_failing:liquidations"
    below = dict(healthy_liq, consecutive_failed_connections=4)
    assert (
        evaluate_recorder_heartbeat(
            _payload(liquidations=below),
            now=_NOW,
            watch_started_at="2026-09-24T06:30:00Z",
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
        watch_started_at="2026-09-24T06:30:00Z",
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
        watch_started_at="2026-09-24T06:30:00Z",
        thresholds=_DEFAULTS,
    )
    assert finding.key == "liquidation_silent:liquidations"
    assert "entry=missing" in finding.detail


def test_stale_sampler_detected_per_dataset() -> None:
    """Only the stale dataset is reported when its sibling is fresh."""
    (finding,) = evaluate_recorder_heartbeat(
        _payload(premium_index={"last_success_at": "2026-09-24T06:04:00Z"}),
        now=_NOW,
        watch_started_at="2026-09-24T06:30:00Z",
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
            watch_started_at="2026-09-24T06:30:00Z",
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
            watch_started_at=pd.Timestamp("2026-09-24T06:30:00Z"),
            thresholds=_DEFAULTS,
        )
        == ()
    )
    for bad_ts in (True, 12345, "not-a-time"):
        (finding,) = evaluate_recorder_heartbeat(
            _payload(ts=bad_ts),
            now=_NOW,
            watch_started_at="2026-09-24T06:30:00Z",
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
        watch_started_at="2026-09-24T06:30:00Z",
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
        watch_started_at="2026-09-24T06:30:00Z",
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
        watch_started_at="2026-09-24T06:30:00Z",
        thresholds=_DEFAULTS,
    )
    by_key = {f.key: f for f in findings}
    assert by_key["sampler_stale:premium_index"].detail == "entry=missing"
    assert "age_s=2100" in by_key["sampler_stale:book_ticker"].detail
    assert "last_success_at=none" in by_key["sampler_stale:book_ticker"].detail


def _v2_sampler(success_at: str, **overrides):
    base = {
        "last_success_at": success_at,
        "consecutive_failures": 0,
        "skipped_grid_points": 0,
        "rows_last_flush": 100,
        "last_persisted_at": success_at,
        "consecutive_flush_failures": 0,
        "pending_rows": 0,
        "dropped_rows_total": 0,
        "rejected_rows_last_sample": 0,
        "rejected_rows_total": 0,
        "rejected_fraction_last_sample": 0.0,
        "consecutive_rejecting_points": 0,
        "window_expected_points": 60,
        "window_captured_points": 60,
    }
    base.update(overrides)
    return base


def _v2_reference(
    day: str,
    captured: bool = True,
    cutoff: str = "00:05",
    last_error: str | None = None,
    previous_day: str | None = "20260923",
    previous_complete: bool | None = True,
):
    def _endpoint(name: str, ok: bool):
        return {
            "captured": ok,
            "consecutive_failures": 0 if ok else 2,
            "last_attempt_at": None if ok else "2026-09-24T01:06:00Z",
            "last_error": None if ok else last_error,
        }

    return {
        "day": day,
        "cutoff_utc": cutoff,
        "endpoints": {
            "exchange_info": _endpoint("exchange_info", True),
            "funding_info": _endpoint("funding_info", captured),
            "asset_index": _endpoint("asset_index", True),
        },
        "last_success_at": "2026-09-24T01:06:00Z" if captured else None,
        "previous_day": previous_day,
        "previous_day_complete": previous_complete,
    }


def _v2_liquidations(event_at: str, **overrides):
    base = {
        "last_event_at": event_at,
        "last_connected_at": event_at,
        "consecutive_failed_connections": 0,
        "last_disconnect_reason": None,
        "last_persisted_at": event_at,
        "consecutive_flush_failures": 0,
        "pending_events": 0,
        "dropped_events_total": 0,
    }
    base.update(overrides)
    return base


def _v2_payload(now: str, **overrides):
    base = {
        "schema_version": 2,
        "ts": now,
        "started_at": "2026-09-24T00:00:00Z",
        "book_ticker": _v2_sampler(now),
        "premium_index": _v2_sampler(now),
        "reference": _v2_reference("20260924"),
        "liquidations": _v2_liquidations(now),
    }
    base.update(overrides)
    return base


def _evaluate(payload, now: str):
    return evaluate_recorder_heartbeat(
        payload,
        now=now,
        watch_started_at="2026-09-24T00:00:00Z",
        thresholds=_DEFAULTS,
    )


def test_sampler_failing_alerts_within_minutes() -> None:
    """Five consecutive failures alert while freshness stays quiet."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T06:34:00Z", consecutive_failures=5),
    )
    keys = {finding.key for finding in _evaluate(payload, now)}
    assert "sampler_failing:book_ticker" in keys
    assert "sampler_stale:book_ticker" not in keys


def test_sampler_four_consecutive_failures_stay_quiet() -> None:
    """Below-threshold failures do not page."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T06:34:00Z", consecutive_failures=4),
    )
    assert "sampler_failing:book_ticker" not in {finding.key for finding in _evaluate(payload, now)}


def test_sampler_degraded_by_capture_ratio() -> None:
    """Intermittent misses are caught by the trailing capture ratio."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler(
            "2026-09-24T06:34:00Z", window_expected_points=60, window_captured_points=50
        ),
    )
    (finding,) = [
        finding for finding in _evaluate(payload, now) if finding.key == "sampler_degraded:book_ticker"
    ]
    assert "ratio=0.833" in finding.detail


def test_sampler_capture_ratio_ignored_at_startup() -> None:
    """Too few expected points means no ratio verdict yet."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler(
            "2026-09-24T06:34:00Z", window_expected_points=5, window_captured_points=0
        ),
    )
    assert "sampler_degraded:book_ticker" not in {finding.key for finding in _evaluate(payload, now)}


def test_sampler_flush_failures_detected_while_fetches_succeed() -> None:
    """Fresh fetches with failing writes page persistence, not freshness."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T06:34:00Z", consecutive_flush_failures=3),
    )
    assert "sampler_unpersisted:book_ticker" in {finding.key for finding in _evaluate(payload, now)}


def test_sampler_stale_persistence_with_pending_rows_detected() -> None:
    """Buffered rows with no recent write page persistence."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler(
            "2026-09-24T06:34:00Z", pending_rows=4000, last_persisted_at="2026-09-24T06:13:20Z"
        ),
    )
    assert "sampler_unpersisted:book_ticker" in {finding.key for finding in _evaluate(payload, now)}


def test_sampler_idle_buffer_is_not_unpersisted() -> None:
    """Nothing pending means nothing unpersisted, however old the last write."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler(
            "2026-09-24T06:34:00Z", pending_rows=0, last_persisted_at="2026-09-24T05:11:40Z"
        ),
    )
    assert "sampler_unpersisted:book_ticker" not in {finding.key for finding in _evaluate(payload, now)}


def test_sampler_single_row_rejection_stays_quiet() -> None:
    """Expected venue noise stays in logs and counters, not in alerts."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler(
            "2026-09-24T06:34:00Z",
            rejected_rows_last_sample=2,
            rejected_fraction_last_sample=0.0026,
            consecutive_rejecting_points=3,
            rejected_rows_total=6,
        ),
    )
    assert "sampler_rows_rejected:book_ticker" not in {
        finding.key for finding in _evaluate(payload, now)
    }


def test_sampler_high_rejected_fraction_alerts() -> None:
    """A schema-scale rejection fraction pages immediately."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler(
            "2026-09-24T06:34:00Z",
            rejected_rows_last_sample=15,
            rejected_fraction_last_sample=0.02,
            consecutive_rejecting_points=1,
            rejected_rows_total=15,
        ),
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "sampler_rows_rejected:book_ticker"
    ]
    assert "fraction=0.0200" in finding.detail


def test_sampler_persistent_rejection_alerts() -> None:
    """A long-lived malformed row surfaces once it persists long enough."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler(
            "2026-09-24T06:34:00Z",
            rejected_rows_last_sample=1,
            rejected_fraction_last_sample=0.0013,
            consecutive_rejecting_points=60,
            rejected_rows_total=60,
        ),
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "sampler_rows_rejected:book_ticker"
    ]
    assert "consecutive_points=60" in finding.detail


def test_liquidation_flush_failure_detected() -> None:
    """Failing liquidation writes page even with a fresh event stream."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(now, liquidations=_v2_liquidations(now, consecutive_flush_failures=3))
    keys = {finding.key for finding in _evaluate(payload, now)}
    assert "liquidation_unpersisted:liquidations" in keys
    assert "liquidation_silent:liquidations" not in keys


def test_reference_incomplete_after_deadline_flagged() -> None:
    """A missing endpoint past the deadline names the endpoint and error."""
    now = "2026-09-24T01:10:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T01:09:00Z"),
        premium_index=_v2_sampler("2026-09-24T01:09:00Z"),
        liquidations=_v2_liquidations("2026-09-24T01:09:00Z"),
        reference=_v2_reference("20260924", captured=False, last_error="ClientResponseError: 403"),
    )
    (finding,) = [
        finding for finding in _evaluate(payload, now) if finding.key == "reference_missing:reference"
    ]
    assert "funding_info" in finding.detail
    assert "403" in finding.detail


def test_reference_quiet_before_deadline() -> None:
    """Today's endpoints may still arrive before the deadline."""
    now = "2026-09-24T00:30:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T00:29:00Z"),
        premium_index=_v2_sampler("2026-09-24T00:29:00Z"),
        liquidations=_v2_liquidations("2026-09-24T00:29:00Z"),
        reference=_v2_reference("20260924", captured=False),
    )
    assert "reference_missing:reference" not in {finding.key for finding in _evaluate(payload, now)}


def test_reference_previous_incomplete_day_stays_flagged() -> None:
    """An incomplete yesterday does not falsely recover at midnight."""
    now = "2026-09-24T00:30:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T00:29:00Z"),
        premium_index=_v2_sampler("2026-09-24T00:29:00Z"),
        liquidations=_v2_liquidations("2026-09-24T00:29:00Z"),
        reference=_v2_reference(
            "20260924", captured=False, previous_day="20260923", previous_complete=False
        ),
    )
    assert "reference_missing:reference" in {finding.key for finding in _evaluate(payload, now)}


def test_v1_heartbeat_uses_legacy_checks_only() -> None:
    """A young v1 payload evaluates exactly like before, without a schema finding."""
    payload = _payload(ts="2026-09-24T06:34:30Z")
    findings = evaluate_recorder_heartbeat(
        payload,
        now="2026-09-24T06:35:00Z",
        watch_started_at="2026-09-24T06:30:00Z",
        thresholds=_DEFAULTS,
    )
    assert findings == ()
    assert not any(finding.check == "heartbeat_schema" for finding in findings)


def test_v1_outdated_recorder_image_flagged() -> None:
    """A v1 file outliving one stale window means an outdated image."""
    payload = _payload(ts="2026-09-24T06:34:30Z")
    findings = evaluate_recorder_heartbeat(
        payload,
        now="2026-09-24T06:35:00Z",
        watch_started_at="2026-09-24T05:00:00Z",
        thresholds=_DEFAULTS,
    )
    assert "heartbeat_schema:recorder" in {finding.key for finding in findings}


def test_malformed_v2_counter_fails_closed() -> None:
    """A boolean where a counter belongs fails that check closed."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T06:34:00Z", consecutive_flush_failures=True),
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "sampler_unpersisted:book_ticker"
    ]
    assert finding.detail == "entry=malformed"


def test_sampler_bool_failure_counter_fails_closed() -> None:
    """A boolean failure count fails that check closed."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now, book_ticker=_v2_sampler("2026-09-24T06:34:00Z", consecutive_failures=True)
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "sampler_failing:book_ticker"
    ]
    assert finding.detail == "entry=malformed"


def test_sampler_bool_window_fails_closed() -> None:
    """A boolean window count fails the ratio check closed."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now, book_ticker=_v2_sampler("2026-09-24T06:34:00Z", window_expected_points=True)
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "sampler_degraded:book_ticker"
    ]
    assert finding.detail == "entry=malformed"


def test_sampler_bad_rejected_fraction_fails_closed() -> None:
    """Non-numeric rejection fractions fail that check closed."""
    now = "2026-09-24T06:35:00Z"
    for bad in (True, "high"):
        payload = _v2_payload(
            now,
            book_ticker=_v2_sampler(
                "2026-09-24T06:34:00Z", rejected_fraction_last_sample=bad
            ),
        )
        (finding,) = [
            finding
            for finding in _evaluate(payload, now)
            if finding.key == "sampler_rows_rejected:book_ticker"
        ]
        assert finding.detail == "entry=malformed"


def test_sampler_garbage_persisted_timestamp_fails_closed() -> None:
    """An unparseable persistence timestamp fails that check closed."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler(
            "2026-09-24T06:34:00Z", pending_rows=10, last_persisted_at="not-a-time"
        ),
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "sampler_unpersisted:book_ticker"
    ]
    assert finding.detail == "entry=malformed"


def test_sampler_pending_without_time_reference_fails_closed() -> None:
    """Pending rows with no time anchor fail that check closed."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(now)
    del payload["started_at"]
    payload["book_ticker"] = _v2_sampler(
        "2026-09-24T06:34:00Z", pending_rows=10, last_persisted_at=None
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "sampler_unpersisted:book_ticker"
    ]
    assert finding.detail == "entry=malformed"


def test_sampler_missing_entry_skips_v2_checks() -> None:
    """A missing sampler entry keeps its stale finding and gains no v2 ones."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(now, premium_index=None)
    keys = {finding.key for finding in _evaluate(payload, now)}
    assert "sampler_stale:premium_index" in keys
    assert "sampler_failing:premium_index" not in keys
    assert "sampler_degraded:premium_index" not in keys


def test_reference_malformed_cutoff_flagged() -> None:
    """An unreadable cutoff cannot compute a deadline and fails closed."""
    now = "2026-09-24T06:35:00Z"
    for bad_cutoff in (True, "25:99"):
        payload = _v2_payload(
            now, reference=_v2_reference("20260924", cutoff=bad_cutoff)  # type: ignore[arg-type]
        )
        (finding,) = [
            finding
            for finding in _evaluate(payload, now)
            if finding.key == "reference_missing:reference"
        ]
        assert finding.detail == "entry=malformed"


def test_reference_non_mapping_entry_fails_closed() -> None:
    """A non-mapping reference entry fails closed."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(now, reference="bad")
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "reference_missing:reference"
    ]
    assert finding.detail == "entry=malformed"


def test_reference_malformed_day_endpoints_after_deadline() -> None:
    """Unreadable day fields fail closed once the deadline passes."""
    now = "2026-09-24T01:10:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T01:09:00Z"),
        premium_index=_v2_sampler("2026-09-24T01:09:00Z"),
        liquidations=_v2_liquidations("2026-09-24T01:09:00Z"),
        reference={
            "day": "20260924",
            "cutoff_utc": "00:05",
            "endpoints": "bad",
            "last_success_at": None,
            "previous_day": "20260923",
            "previous_day_complete": True,
        },
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "reference_missing:reference"
    ]
    assert finding.detail == "entry=malformed"


def test_reference_malformed_day_endpoints_quiet_before_deadline() -> None:
    """Unreadable day fields stay quiet while endpoints may still arrive."""
    now = "2026-09-24T00:30:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T00:29:00Z"),
        premium_index=_v2_sampler("2026-09-24T00:29:00Z"),
        liquidations=_v2_liquidations("2026-09-24T00:29:00Z"),
        reference={
            "day": "20260924",
            "cutoff_utc": "00:05",
            "endpoints": "bad",
            "last_success_at": None,
            "previous_day": "20260923",
            "previous_day_complete": True,
        },
    )
    assert "reference_missing:reference" not in {finding.key for finding in _evaluate(payload, now)}


def test_reference_stale_day_after_deadline() -> None:
    """An entry still pointing at yesterday past the deadline is stale."""
    now = "2026-09-24T01:10:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T01:09:00Z"),
        premium_index=_v2_sampler("2026-09-24T01:09:00Z"),
        liquidations=_v2_liquidations("2026-09-24T01:09:00Z"),
        reference=_v2_reference("20260923"),
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "reference_missing:reference"
    ]
    assert "entry=stale" in finding.detail


def test_reference_yesterday_incomplete_before_deadline() -> None:
    """Yesterday's incomplete endpoints stay flagged until today completes."""
    now = "2026-09-24T00:30:00Z"
    payload = _v2_payload(
        now,
        book_ticker=_v2_sampler("2026-09-24T00:29:00Z"),
        premium_index=_v2_sampler("2026-09-24T00:29:00Z"),
        liquidations=_v2_liquidations("2026-09-24T00:29:00Z"),
        reference=_v2_reference(
            "20260923", captured=False, previous_day=None, previous_complete=None
        ),
    )
    (finding,) = [
        finding
        for finding in _evaluate(payload, now)
        if finding.key == "reference_missing:reference"
    ]
    assert "funding_info" in finding.detail


def test_reference_unpublished_cutoff_quiet_right_after_start() -> None:
    """기동 직후 첫 publish 전(cutoff 빈 값)에는 기한을 알 수 없으므로 reference_missing을 내지 않는다."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(
        now,
        started_at="2026-09-24T06:34:00Z",
        reference=_v2_reference("20260924", cutoff=""),
    )
    keys = {finding.key for finding in _evaluate(payload, now)}
    assert "reference_missing:reference" not in keys


def test_reference_unpublished_cutoff_flagged_after_grace() -> None:
    """grace가 지나도 cutoff가 비어 있으면 malformed로 fail-closed 한다."""
    now = "2026-09-24T06:35:00Z"
    payload = _v2_payload(now, reference=_v2_reference("20260924", cutoff=""))
    (finding,) = [f for f in _evaluate(payload, now) if f.key == "reference_missing:reference"]
    assert finding.detail == "entry=malformed"
