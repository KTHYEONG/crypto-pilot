"""Invariant guards for heartbeat v3 evaluation."""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import pytest

from src.market_data.streams.recorder_health import (
    RecorderFinding,
    RecorderWatchThresholds,
    evaluate_recorder_heartbeat,
)


def _now() -> pd.Timestamp:
    return pd.Timestamp("2026-09-26T12:00:00Z")


def _thresholds(**overrides: Any) -> RecorderWatchThresholds:
    values: dict[str, Any] = {
        "heartbeat_stale_s": 600.0,
        "liquidation_silence_s": 900.0,
        "sampler_stale_s": 1800.0,
        "sampler_max_consecutive_failures": 5,
        "min_capture_ratio": 0.9,
        "capture_ratio_min_points": 10,
        "persist_stale_s": 1200.0,
        "max_consecutive_flush_failures": 3,
        "reference_grace_s": 3600.0,
        "rejected_fraction_alert": 0.01,
        "rejected_max_consecutive_points": 60,
        "capture_stale_s": 120.0,
        "capture_ready_grace_s": 900.0,
        "capture_dual_active_max_s": 1200.0,
        "normalizer_max_lag_s": 600.0,
        "normalizer_max_consecutive_failures": 5,
        "compaction_max_delay_s": 10800.0,
    }
    values.update(overrides)
    return RecorderWatchThresholds(**values)


def _slot(started_ago_s: float = 100.0, ready: bool = True, **overrides: Any) -> dict[str, Any]:
    now = _now()
    started = (now - pd.Timedelta(seconds=started_ago_s)).isoformat()
    first_ok = started if ready else None
    first_frame = started if ready else None
    payload: dict[str, Any] = {
        "slot": "blue",
        "pid": 7,
        "fingerprint": "fp",
        "started_at": started,
        "stopped_at": None,
        "rest": {
            "book_ticker": {"first_ok_at": first_ok, "last_ok_at": first_ok, "consecutive_failures": 0},
            "premium_index": {"first_ok_at": first_ok, "last_ok_at": first_ok, "consecutive_failures": 0},
        },
        "ws": {"connected_at": started, "first_frame_at": first_frame,
               "last_frame_at": now.isoformat(), "reconnects": 0, "pending_dropped": 0},
        "last_flush_at": now.isoformat(),
        "flush_failures": 0,
        "ts": now.isoformat(),
        "ready": ready,
    }
    payload.update(overrides)
    return payload


def _payload(**overrides: Any) -> dict[str, Any]:
    now = _now()
    payload: dict[str, Any] = {
        "schema_version": 3,
        "ts": now.isoformat(),
        "started_at": (now - pd.Timedelta(hours=2)).isoformat(),
        "normalizer": {"last_run_at": now.isoformat(), "last_success_at": now.isoformat(),
                       "consecutive_failures": 0, "lag_s": 0.0,
                       "pending_complete_bytes": 0, "last_error": None},
        "streams": {
            "book_ticker": {"last_grid": "2026-09-26T11:55:00+00:00",
                            "last_persisted_at": now.isoformat(), "rows_last_write": 10,
                            "rejected_rows_last_sample": 0, "rejected_fraction_last_sample": 0.0,
                            "rejected_rows_total": 0, "consecutive_rejecting_points": 0,
                            "window_expected_points": 60, "window_captured_points": 60,
                            "duplicates_dropped_total": 0},
            "premium_index": {"last_grid": "2026-09-26T11:55:00+00:00",
                              "last_persisted_at": now.isoformat(), "rows_last_write": 5,
                              "rejected_rows_last_sample": 0, "rejected_fraction_last_sample": 0.0,
                              "rejected_rows_total": 0, "consecutive_rejecting_points": 0,
                              "window_expected_points": 12, "window_captured_points": 12,
                              "duplicates_dropped_total": 0},
            "force_order": {"last_frame_recv_at": now.isoformat(),
                            "last_persisted_at": now.isoformat(), "frames_total": 100,
                            "duplicates_dropped_total": 0, "parse_failures_total": 0},
        },
        "reference": {"day": "20260926", "cutoff_utc": "00:05",
                      "endpoints": {"exchange_info": {"captured": True},
                                    "funding_info": {"captured": True},
                                    "asset_index": {"captured": True}},
                      "previous_day": "20260925", "previous_day_complete": True},
        "compaction": {"last_day": "20260925", "last_result": "ok", "last_error": None,
                       "last_run_at": now.isoformat(), "archived_days_pending": []},
        "retention": {"prune_blocked": False, "blocked_reason": None, "backup_started_at": None,
                      "last_run_at": now.isoformat(), "pruned_files_total": 0,
                      "raw_hot_bytes": 10, "raw_archive_bytes": 20},
        "capture": {"blue": _slot(), "green": None},
    }
    payload.update(overrides)
    return payload


def _keys(findings: tuple[RecorderFinding, ...]) -> set[str]:
    return {finding.key for finding in findings}


def test_healthy_snapshot_yields_no_findings() -> None:
    """One fresh READY slot, lagless normalizer, ok compaction and open retention prune clean."""
    assert evaluate_recorder_heartbeat(_payload(), now=_now(),
                                       watch_started_at=_now() - pd.Timedelta(hours=3),
                                       thresholds=_thresholds()) == ()


def test_schema_other_than_3_yields_only_schema() -> None:
    """A v2 payload fails only the schema check."""
    payload = _payload(schema_version=2)
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert _keys(findings) == {"heartbeat_schema:recorder"}


def test_dead_capture_live_normalizer_is_missing() -> None:
    """No fresh slot heartbeat surfaces as capture_missing."""
    payload = _payload(capture={"blue": None, "green": None})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert "capture_missing:capture" in _keys(findings)


def test_unfinished_handover_is_dual_active() -> None:
    """Two fresh slots older than the dual-active window raise; a young pair does not."""
    now = _now()
    payload = _payload(capture={"blue": _slot(started_ago_s=1300), "green": _slot(started_ago_s=1300)})
    findings = evaluate_recorder_heartbeat(payload, now=now,
                                           watch_started_at=now - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert "capture_dual_active:capture" in _keys(findings)
    young = _payload(capture={"blue": _slot(started_ago_s=600), "green": _slot(started_ago_s=600)})
    young_findings = evaluate_recorder_heartbeat(young, now=now,
                                                 watch_started_at=now - pd.Timedelta(hours=3),
                                                 thresholds=_thresholds())
    assert "capture_dual_active:capture" not in _keys(young_findings)


def test_slot_never_ready() -> None:
    """A fresh slot past the ready grace without a first frame raises not_ready."""
    payload = _payload(capture={"blue": _slot(started_ago_s=1000, ready=False), "green": None})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert "capture_not_ready:blue" in _keys(findings)


def test_healthy_slot_masks_overlap_rest_failures() -> None:
    """Green failing book_ticker 6 times is masked while blue stays healthy."""
    failing = _slot()
    failing["rest"]["book_ticker"]["consecutive_failures"] = 6
    payload = _payload(capture={"blue": _slot(), "green": failing})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert "capture_rest_failing:book_ticker" not in _keys(findings)


def test_normalizer_lag_and_failures() -> None:
    """Lag and consecutive failures each raise their own finding."""
    payload = _payload(normalizer={"last_run_at": _now().isoformat(), "last_success_at": None,
                                   "consecutive_failures": 5, "lag_s": 700.0,
                                   "pending_complete_bytes": 9, "last_error": "boom"})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert {"normalizer_lagging:normalizer", "normalizer_failing:normalizer"} <= _keys(findings)


def test_compaction_and_retention_problems() -> None:
    """Error result, overdue pending day and blocked prune all surface."""
    payload = _payload(
        compaction={"last_day": "20260920", "last_result": "error", "last_error": "x",
                    "last_run_at": _now().isoformat(),
                    "archived_days_pending": [{"stream": "book_ticker", "day": "20260920"}]},
        retention={"prune_blocked": True, "blocked_reason": "status_missing",
                   "backup_started_at": None, "last_run_at": _now().isoformat(),
                   "pruned_files_total": 0, "raw_hot_bytes": 1, "raw_archive_bytes": 2},
    )
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert {"compaction_failed:compaction", "compaction_overdue:compaction",
            "prune_blocked:retention"} <= _keys(findings)


def test_stale_heartbeat_masks_everything() -> None:
    """A stale ts yields only heartbeat_stale."""
    payload = _payload(ts=(_now() - pd.Timedelta(seconds=3600)).isoformat(), capture={"blue": None})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert _keys(findings) == {"heartbeat_stale:recorder"}


def test_missing_heartbeat_grace_and_alert() -> None:
    """A missing heartbeat is tolerated while the watch is young, then reported."""
    young = evaluate_recorder_heartbeat(None, now=_now(), watch_started_at=_now(),
                                        thresholds=_thresholds())
    assert young == ()
    old = evaluate_recorder_heartbeat(None, now=_now(),
                                      watch_started_at=_now() - pd.Timedelta(hours=3),
                                      thresholds=_thresholds())
    assert _keys(old) == {"heartbeat_missing:recorder"}


def test_heartbeat_io_variants(tmp_path) -> None:
    """Missing, corrupt and non-object heartbeat files read as None; naive clocks rejected."""
    import pytest

    from src.market_data.streams.recorder_health import read_recorder_heartbeat

    assert read_recorder_heartbeat(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    assert read_recorder_heartbeat(bad) is None
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[1,2]", encoding="utf-8")
    assert read_recorder_heartbeat(scalar) is None
    with pytest.raises(ValueError, match="tz-aware"):
        evaluate_recorder_heartbeat(_payload(), now=pd.Timestamp("2026-09-26T12:00:00"),
                                    watch_started_at=_now() - pd.Timedelta(hours=3),
                                    thresholds=_thresholds())
    stale_ts = evaluate_recorder_heartbeat(_payload(ts=True), now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert {finding.key for finding in stale_ts} == {"heartbeat_stale:recorder"}


def test_malformed_sections_fail_closed() -> None:
    """Missing sections fail their own checks without masking the rest."""
    payload = _payload(
        streams={},
        normalizer=None,
        compaction=None,
        retention={},
        capture={"blue": {**_slot(), "flush_failures": "bad"}},
    )
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    keys = {finding.key for finding in findings}
    assert "sampler_stale:book_ticker" in keys
    assert "normalizer_lagging:normalizer" in keys
    assert "normalizer_failing:normalizer" in keys
    assert "compaction_failed:compaction" in keys
    assert "prune_blocked:retention" in keys
    assert "capture_flush_failing:blue" in keys
    assert "liquidation_unpersisted:force_order" in keys


def test_all_slots_rest_failing_and_silent() -> None:
    """Every fresh slot failing a stream, and silent WS, raise their findings."""
    bad = _slot()
    bad["rest"]["book_ticker"]["consecutive_failures"] = 9
    bad["rest"]["premium_index"]["consecutive_failures"] = 9
    bad["ws"]["last_frame_at"] = "2026-09-26T10:00:00+00:00"
    payload = _payload(capture={"blue": bad, "green": None})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    keys = {finding.key for finding in findings}
    assert "capture_rest_failing:book_ticker" in keys
    assert "capture_ws_silent:force_order" in keys


def test_sampler_degraded_rejecting_and_unpersisted_fire() -> None:
    """Thin ratios, hot rejects and a stale liquidation persist raise findings."""
    payload = _payload()
    payload["streams"]["book_ticker"]["window_captured_points"] = 5
    payload["streams"]["premium_index"]["rejected_fraction_last_sample"] = 0.5
    payload["streams"]["force_order"]["last_persisted_at"] = "2026-09-26T10:00:00+00:00"
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    keys = {finding.key for finding in findings}
    assert "sampler_degraded:book_ticker" in keys
    assert "sampler_rejecting:premium_index" in keys
    assert "liquidation_unpersisted:force_order" in keys


def test_reference_variants() -> None:
    """Cutoff, staleness and completeness corners of the reference rule."""
    base = _payload()

    def _evaluate(reference: object) -> set[str]:
        payload = _payload(reference=reference)
        return {finding.key for finding in evaluate_recorder_heartbeat(
            payload, now=_now(), watch_started_at=_now() - pd.Timedelta(hours=3),
            thresholds=_thresholds())}

    assert "reference_missing:reference" in _evaluate({"cutoff_utc": None})
    recent_start = (_now() - pd.Timedelta(seconds=10)).isoformat()
    young_payload = _payload(reference={"cutoff_utc": None}, started_at=recent_start)
    young_keys = {finding.key for finding in evaluate_recorder_heartbeat(
        young_payload, now=_now(), watch_started_at=_now() - pd.Timedelta(seconds=10),
        thresholds=_thresholds())}
    assert "reference_missing:reference" not in young_keys
    incomplete_today = dict(base["reference"])
    incomplete_today["endpoints"] = {"exchange_info": {"captured": False},
                                     "funding_info": {"captured": True},
                                     "asset_index": {"captured": True}}
    assert "reference_missing:reference" in _evaluate(incomplete_today)
    assert "reference_missing:reference" in _evaluate("yesterday-stale")


def test_compaction_overdue_variants() -> None:
    """Pending lists fail closed on shape errors and fire past the deadline."""
    base_pending = [{"stream": "book_ticker", "day": "20990101"}]

    def _evaluate(compaction: object) -> set[str]:
        payload = _payload(compaction=compaction)
        return {finding.key for finding in evaluate_recorder_heartbeat(
            payload, now=_now(), watch_started_at=_now() - pd.Timedelta(hours=3),
            thresholds=_thresholds())}

    assert _evaluate({"last_day": None, "last_result": None, "last_error": None,
                       "last_run_at": None, "archived_days_pending": base_pending}) == set()
    assert "compaction_overdue:compaction" in _evaluate(
        {"last_day": None, "last_result": None, "last_error": None,
         "last_run_at": None, "archived_days_pending": "oops"})
    assert "compaction_overdue:compaction" in _evaluate(
        {"last_day": None, "last_result": None, "last_error": None,
         "last_run_at": None, "archived_days_pending": [{"stream": "book_ticker"}]})
    assert "compaction_overdue:compaction" in _evaluate(
        {"last_day": None, "last_result": None, "last_error": None,
         "last_run_at": None, "archived_days_pending": [{"stream": "book_ticker", "day": "20260920"}]})


def test_dual_active_and_not_ready_missing_started() -> None:
    """Missing started_at fails the handover checks closed."""
    no_start = _slot()
    del no_start["started_at"]
    payload = _payload(capture={"blue": no_start, "green": _slot()})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    keys = {finding.key for finding in findings}
    assert "capture_dual_active:capture" in keys
    assert "capture_not_ready:blue" in keys


def test_parse_helpers_cover_input_shapes() -> None:
    """Scalar parsers accept, coerce and reject each documented shape."""
    from datetime import datetime as _datetime

    from src.market_data.streams.recorder_health import (
        _parse_count,
        _parse_cutoff,
        _parse_ratio,
        _parse_ts,
    )

    assert _parse_ts(True) is None
    assert _parse_ts(pd.Timestamp("2026-09-26T12:00:00Z")) is not None
    assert _parse_ts("2026-09-26T12:00:00Z") is not None
    assert _parse_ts("yesterday") is None
    assert _parse_ts(_datetime(2026, 9, 26, 12, 0)) is None
    assert _parse_ts(_datetime(2026, 9, 26, 12, 0, tzinfo=datetime.UTC)) is not None
    assert _parse_ts(None) is None
    assert _parse_ts(123) is None
    assert _parse_count(True) is None
    assert _parse_count(3) == 3
    assert _parse_count(3.5) is None
    assert _parse_ratio(True) is None
    assert _parse_ratio(2) == 2.0
    assert _parse_ratio("x") is None
    assert _parse_cutoff(True) is None
    assert _parse_cutoff("25:00") is None
    assert _parse_cutoff("00:05") == (0, 5)


def test_malformed_mappings_fail_each_check() -> None:
    """Mapping-shaped but key-missing entries fail their own checks."""
    payload = _payload(
        streams={"book_ticker": {}, "premium_index": {}, "force_order": {}},
        normalizer={"lag_s": None, "consecutive_failures": None},
    )
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    keys = {finding.key for finding in findings}
    assert "sampler_stale:book_ticker" in keys
    assert "sampler_degraded:book_ticker" in keys
    assert "sampler_rejecting:book_ticker" in keys
    assert "liquidation_unpersisted:force_order" in keys
    assert "normalizer_lagging:normalizer" in keys
    assert "normalizer_failing:normalizer" in keys


def test_ws_null_frame_within_grace_is_quiet() -> None:
    """A fresh slot with no frame yet inside the ready grace is not silent."""
    slot = _slot()
    slot["ws"]["last_frame_at"] = None
    slot["ws"]["first_frame_at"] = None
    slot["started_at"] = (_now() - pd.Timedelta(seconds=100)).isoformat()
    slot["rest"]["book_ticker"]["first_ok_at"] = slot["started_at"]
    payload = _payload(capture={"blue": slot, "green": None})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert "capture_ws_silent:force_order" not in {finding.key for finding in findings}


def test_capture_non_mapping_is_missing() -> None:
    """A non-object capture section reads as no fresh slot."""
    payload = _payload(capture="oops")
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert {finding.key for finding in findings} == {"capture_missing:capture"}


def test_slot_without_book_ticker_never_ready() -> None:
    """A slot heartbeat missing book_ticker cannot be READY."""
    slot = _slot(started_ago_s=1000.0)
    slot["rest"] = {"premium_index": slot["rest"]["premium_index"]}
    payload = _payload(capture={"blue": slot, "green": None})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert "capture_not_ready:blue" in {finding.key for finding in findings}


def test_slot_flush_failures_counted() -> None:
    """A non-zero flush failure count on a fresh slot raises its finding."""
    slot = _slot()
    slot["flush_failures"] = 2
    payload = _payload(capture={"blue": slot, "green": None})
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert "capture_flush_failing:blue" in {finding.key for finding in findings}


@pytest.mark.parametrize("pending", [None, "absent"])
def test_compaction_pending_missing_is_malformed(pending: object) -> None:
    """A missing or null pending list cannot prove nothing is overdue, so it fails closed."""
    compaction: dict[str, object] = {"last_day": None, "last_result": None, "last_error": None, "last_run_at": None}
    if pending is None:
        compaction["archived_days_pending"] = None
    payload = _payload(compaction=compaction)
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    overdue = [finding for finding in findings if finding.key == "compaction_overdue:compaction"]
    assert overdue
    assert overdue[0].detail == "entry=malformed"


def test_retention_null_is_missing() -> None:
    """A null retention section reads as a missing prune report."""
    payload = _payload(retention=None)
    findings = evaluate_recorder_heartbeat(payload, now=_now(),
                                           watch_started_at=_now() - pd.Timedelta(hours=3),
                                           thresholds=_thresholds())
    assert "prune_blocked:retention" in {finding.key for finding in findings}


def test_reference_stale_and_incomplete_days() -> None:
    """Post-deadline shape errors, stale days and incomplete history all raise."""
    def _keys(reference: object) -> set[str]:
        payload = _payload(reference=reference)
        return {finding.key for finding in evaluate_recorder_heartbeat(
            payload, now=_now(), watch_started_at=_now() - pd.Timedelta(hours=3),
            thresholds=_thresholds())}

    assert "reference_missing:reference" in _keys({"day": "20260926", "cutoff_utc": "00:05"})
    assert "reference_missing:reference" in _keys({
        "day": "20260920", "cutoff_utc": "00:05",
        "endpoints": {"exchange_info": {"captured": True}, "funding_info": {"captured": True},
                      "asset_index": {"captured": True}},
        "previous_day": "20260925", "previous_day_complete": True})
    assert "reference_missing:reference" in _keys({
        "day": "20260926", "cutoff_utc": "00:05",
        "endpoints": {"exchange_info": {"captured": True}, "funding_info": {"captured": True},
                      "asset_index": {"captured": True}},
        "previous_day": "20260925", "previous_day_complete": False})
    assert "reference_missing:reference" in _keys({
        "day": "20260925", "cutoff_utc": "00:05",
        "endpoints": {"exchange_info": {"captured": False}, "funding_info": {"captured": True},
                      "asset_index": {"captured": True}},
        "previous_day": "20260924", "previous_day_complete": True})
    early = pd.Timestamp("2026-09-26T00:30:00Z")
    payload = _payload(reference={
        "day": "20260925", "cutoff_utc": "00:05",
        "endpoints": {"exchange_info": {"captured": False}, "funding_info": {"captured": True},
                      "asset_index": {"captured": True}},
        "previous_day": "20260924", "previous_day_complete": True})
    early_findings = evaluate_recorder_heartbeat(
        payload, now=early, watch_started_at=early - pd.Timedelta(hours=3),
        thresholds=_thresholds())
    assert "reference_missing:reference" in {finding.key for finding in early_findings}
