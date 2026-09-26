"""Invariant guards for the recorder watchdog episode alerting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.recorder_watch as watch_mod
from src.live.recorder_watch import RecorderWatchdog, build_recorder_watchdog
from src.live.settings import LiveSettings
from src.market_data.streams.recorder_health import RecorderWatchThresholds

_NOW = "2026-09-24T06:35:00Z"


def _thresholds() -> RecorderWatchThresholds:
    return RecorderWatchThresholds(
        heartbeat_stale_s=600.0,
        liquidation_silence_s=900.0,
        sampler_stale_s=1800.0,
        sampler_max_consecutive_failures=5,
        min_capture_ratio=0.9,
        capture_ratio_min_points=10,
        persist_stale_s=1200.0,
        max_consecutive_flush_failures=3,
        reference_grace_s=3600.0,
        rejected_fraction_alert=0.01,
        rejected_max_consecutive_points=60,
        capture_stale_s=120.0,
        capture_ready_grace_s=900.0,
        capture_dual_active_max_s=1200.0,
        normalizer_max_lag_s=600.0,
        normalizer_max_consecutive_failures=5,
        compaction_max_delay_s=10800.0,
        startup_grace_s=120.0,
        prune_blocked_alert_after_s=21600.0,
        local_disk_budget_bytes=8 * 1024**3,
    )


class _ManualNow:
    def __init__(self, start: str) -> None:
        self.t = start

    def __call__(self) -> pd.Timestamp:
        return pd.Timestamp(self.t)


def _fresh_slot(ts: str) -> dict[str, Any]:
    started = "2026-09-24T05:00:00Z"
    return {
        "slot": "blue",
        "pid": 7,
        "fingerprint": "fp",
        "started_at": started,
        "stopped_at": None,
        "rest": {
            "book_ticker": {"first_ok_at": started, "last_ok_at": ts, "consecutive_failures": 0},
            "premium_index": {"first_ok_at": started, "last_ok_at": ts, "consecutive_failures": 0},
        },
        "ws": {"connected_at": started, "first_frame_at": started, "last_frame_at": ts,
               "reconnects": 0, "pending_dropped": 0},
        "last_flush_at": ts,
        "flush_failures": 0,
        "ts": ts,
        "ready": True,
    }


def _heartbeat(
    *,
    ts: str = _NOW,
    dead_capture: bool = False,
    premium_at: str | None = None,
    schema_version: int = 3,
    consecutive_failures: int = 0,
) -> dict[str, Any]:
    day_ts = pd.Timestamp(ts).tz_convert("UTC")
    day = day_ts.strftime("%Y%m%d")
    previous = (day_ts - pd.Timedelta(days=1)).strftime("%Y%m%d")
    premium_persisted = premium_at if premium_at is not None else ts
    return {
        "schema_version": schema_version,
        "ts": ts,
        "started_at": "2026-09-24T05:00:00Z",
        "normalizer": {"last_run_at": ts, "last_success_at": ts,
                       "consecutive_failures": consecutive_failures, "lag_s": 0.0,
                       "pending_complete_bytes": 0, "last_error": None},
        "streams": {
            "book_ticker": {"last_grid": "2026-09-24T06:30:00+00:00", "last_persisted_at": ts,
                            "rows_last_write": 10, "rejected_rows_last_sample": 0,
                            "rejected_fraction_last_sample": 0.0, "rejected_rows_total": 0,
                            "consecutive_rejecting_points": 0, "window_expected_points": 60,
                            "window_captured_points": 60, "duplicates_dropped_total": 0},
            "premium_index": {"last_grid": "2026-09-24T06:30:00+00:00",
                              "last_persisted_at": premium_persisted, "rows_last_write": 5,
                              "rejected_rows_last_sample": 0, "rejected_fraction_last_sample": 0.0,
                              "rejected_rows_total": 0, "consecutive_rejecting_points": 0,
                              "window_expected_points": 12, "window_captured_points": 12,
                              "duplicates_dropped_total": 0},
            "force_order": {"last_frame_recv_at": ts, "last_persisted_at": ts, "frames_total": 10,
                            "duplicates_dropped_total": 0, "parse_failures_total": 0},
        },
        "reference": {"day": day, "cutoff_utc": "00:05",
                      "endpoints": {"exchange_info": {"captured": True},
                                    "funding_info": {"captured": True},
                                    "asset_index": {"captured": True}},
                      "previous_day": previous, "previous_day_complete": True},
        "compaction": {"last_day": previous, "last_result": "ok", "last_error": None,
                       "last_run_at": ts, "archived_days_pending": []},
        "retention": {"prune_blocked": False, "blocked_reason": None, "backup_started_at": None,
                      "last_run_at": ts, "pruned_files_total": 0,
                      "raw_hot_bytes": 1, "raw_archive_bytes": 2},
        "capture": {"blue": None if dead_capture else _fresh_slot(ts), "green": None},
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _watch(
    path: Path, calls: list[tuple[str, str]], results: list[bool] | None = None
) -> RecorderWatchdog:
    results = [] if results is None else results

    def _alert(event: str, detail: str) -> bool:
        calls.append((event, detail))
        return results.pop(0) if results else True

    return RecorderWatchdog(
        heartbeat_path=path,
        thresholds=_thresholds(),
        interval_s=60.0,
        alert=_alert,
        now_fn=_ManualNow(_NOW),
    )


def test_one_alert_per_episode(tmp_path: Path) -> None:
    """Repeated unhealthy checks alert only once until the finding set changes."""
    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat(dead_capture=True))
    calls: list[tuple[str, str]] = []
    watch = _watch(path, calls)
    for _ in range(3):
        findings = watch.check_once()
    assert len(findings) == 1
    assert [event for event, _ in calls] == ["recorder_unhealthy"]
    assert "capture_missing:capture" in calls[0][1]


def test_new_failure_during_episode_realerts(tmp_path: Path) -> None:
    """A second finding joins the episode with one alert listing both keys."""
    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat(dead_capture=True))
    calls: list[tuple[str, str]] = []
    watch = _watch(path, calls)
    watch.check_once()
    _write(
        path,
        _heartbeat(dead_capture=True, premium_at="2026-09-24T06:04:00Z"),
    )
    watch.check_once()
    assert [event for event, _ in calls] == ["recorder_unhealthy", "recorder_unhealthy"]
    assert "capture_missing:capture" in calls[1][1]
    assert "sampler_stale:premium_index" in calls[1][1]


def test_recovery_announced_once_and_relapse_realerts(tmp_path: Path) -> None:
    """A cleared episode announces recovery once; a later relapse opens a new episode."""
    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat(dead_capture=True))
    calls: list[tuple[str, str]] = []
    watch = _watch(path, calls)
    watch.check_once()
    _write(path, _heartbeat())
    watch.check_once()
    watch.check_once()
    assert [event for event, _ in calls] == ["recorder_unhealthy", "recorder_recovered"]
    _write(path, _heartbeat(dead_capture=True))
    watch.check_once()
    assert [event for event, _ in calls] == [
        "recorder_unhealthy",
        "recorder_recovered",
        "recorder_unhealthy",
    ]


def test_undelivered_alert_retried(tmp_path: Path) -> None:
    """A failed delivery is retried on the next check without duplicating after success."""
    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat(dead_capture=True))
    calls: list[tuple[str, str]] = []
    watch = _watch(path, calls, results=[False, True])
    watch.check_once()
    watch.check_once()
    watch.check_once()
    assert [event for event, _ in calls] == ["recorder_unhealthy", "recorder_unhealthy"]


def test_healthy_without_prior_alert_sends_nothing(tmp_path: Path) -> None:
    """A healthy heartbeat never touches the alert channel."""
    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat())
    calls: list[tuple[str, str]] = []
    watch = _watch(path, calls)
    for _ in range(3):
        assert watch.check_once() == ()
    assert calls == []


def test_check_failures_never_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An exploding read is swallowed with an ERROR record instead of propagating."""
    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat())

    def _boom(path: Path) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(watch_mod, "read_recorder_heartbeat", _boom)
    calls: list[tuple[str, str]] = []
    watch = _watch(path, calls)
    with caplog.at_level(logging.ERROR, logger="src.live.recorder_watch"):
        assert watch.check_once() == ()
    assert calls == []
    assert any(
        "stage=recorder_watch" in record.message for record in caplog.records
    )


def test_thread_stops_promptly(tmp_path: Path) -> None:
    """A far-future interval still stops immediately with no check executed."""
    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat())
    calls: list[tuple[str, str]] = []
    watch = RecorderWatchdog(
        heartbeat_path=path,
        thresholds=_thresholds(),
        interval_s=3600.0,
        alert=lambda event, detail: calls.append((event, detail)) or True,
        now_fn=_ManualNow(_NOW),
    )
    watch.start()
    thread = watch._thread
    assert thread is not None
    assert thread.is_alive()
    assert thread.daemon
    assert thread.name == "recorder-watch"
    watch.stop(timeout_s=1.0)
    assert not thread.is_alive()
    assert calls == []


def test_disabled_setting_builds_nothing() -> None:
    """An opted-out daemon gets no watchdog thread."""
    assert (
        build_recorder_watchdog(
            LiveSettings(recorder_watch_enabled=False), alert=lambda event, detail: True
        )
        is None
    )


def test_enabled_setting_builds_production_watchdog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Thresholds, interval, and path come from the recorder_* settings."""
    import src.live.recorder_watch as mod

    monkeypatch.setattr(mod, "LIVE_CAPTURE_DIR", tmp_path)
    settings = LiveSettings(
        recorder_watch_interval_s=30.0,
        recorder_heartbeat_stale_s=60.0,
        recorder_liquidation_silence_s=120.0,
        recorder_sampler_stale_s=180.0,
        recorder_sampler_max_consecutive_failures=4,
        recorder_min_capture_ratio=0.8,
        recorder_capture_ratio_min_points=5,
        recorder_persist_stale_s=900.0,
        recorder_max_consecutive_flush_failures=2,
        recorder_reference_grace_s=1800.0,
        recorder_rejected_fraction_alert=0.02,
        recorder_rejected_max_consecutive_points=30,
        recorder_capture_stale_s=45.0,
        recorder_capture_ready_grace_s=400.0,
        recorder_capture_dual_active_max_s=800.0,
        recorder_normalizer_max_lag_s=300.0,
        recorder_normalizer_max_consecutive_failures=4,
        recorder_compaction_max_delay_s=7200.0,
    )
    watch = build_recorder_watchdog(settings, alert=lambda event, detail: True)
    assert watch is not None
    assert watch._heartbeat_path == tmp_path / "recorder_heartbeat.json"
    assert watch._interval_s == 30.0
    assert watch._thresholds == RecorderWatchThresholds(
        heartbeat_stale_s=60.0,
        liquidation_silence_s=120.0,
        sampler_stale_s=180.0,
        sampler_max_consecutive_failures=4,
        min_capture_ratio=0.8,
        capture_ratio_min_points=5,
        persist_stale_s=900.0,
        max_consecutive_flush_failures=2,
        reference_grace_s=1800.0,
        rejected_fraction_alert=0.02,
        rejected_max_consecutive_points=30,
        capture_stale_s=45.0,
        capture_ready_grace_s=400.0,
        capture_dual_active_max_s=800.0,
        normalizer_max_lag_s=300.0,
        normalizer_max_consecutive_failures=4,
        compaction_max_delay_s=7200.0,
        startup_grace_s=120.0,
        prune_blocked_alert_after_s=21600.0,
        local_disk_budget_bytes=8 * 1024**3,
    )


def test_raising_alert_never_escapes_unhealthy_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An exploding alert callable is swallowed without advancing episode state."""
    import logging

    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat(dead_capture=True))

    def _boom(event: str, detail: str) -> bool:
        raise RuntimeError("webhook down")

    watch = RecorderWatchdog(
        heartbeat_path=path,
        thresholds=_thresholds(),
        interval_s=60.0,
        alert=_boom,
        now_fn=_ManualNow(_NOW),
    )
    with caplog.at_level(logging.ERROR, logger="src.live.recorder_watch"):
        findings = watch.check_once()
    assert [f.key for f in findings] == ["capture_missing:capture"]
    assert any("ALERT_FAILED" in record.message for record in caplog.records)


def test_raising_alert_never_escapes_recovery_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A recovery that cannot be delivered is retried on the next check."""
    import logging

    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat(dead_capture=True))
    calls: list[tuple[str, str]] = []
    watch = _watch(path, calls)
    watch.check_once()
    _write(path, _heartbeat())

    def _boom(event: str, detail: str) -> bool:
        calls.append((event, detail))
        raise RuntimeError("webhook down")

    watch._alert = _boom  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger="src.live.recorder_watch"):
        watch.check_once()
    assert [event for event, _ in calls] == ["recorder_unhealthy", "recorder_recovered"]
    assert any("ALERT_FAILED" in record.message for record in caplog.records)


def test_start_is_idempotent_and_loop_checks(tmp_path: Path) -> None:
    """A second start reuses the thread; the loop evaluates on its cadence."""
    import time

    path = tmp_path / "recorder_heartbeat.json"
    _write(path, _heartbeat(dead_capture=True))
    calls: list[tuple[str, str]] = []
    watch = RecorderWatchdog(
        heartbeat_path=path,
        thresholds=_thresholds(),
        interval_s=0.01,
        alert=lambda event, detail: calls.append((event, detail)) or True,
        now_fn=_ManualNow(_NOW),
    )
    watch.start()
    first = watch._thread
    watch.start()
    assert watch._thread is first
    deadline = time.monotonic() + 5.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    watch.stop(timeout_s=1.0)
    assert calls
    assert calls[0][0] == "recorder_unhealthy"






def test_production_thresholds_wired_from_settings() -> None:
    """Non-default watchdog settings reach the thresholds verbatim."""
    from src.live.recorder_watch import build_recorder_watchdog as _build

    settings = LiveSettings(
        recorder_sampler_max_consecutive_failures=4,
        recorder_min_capture_ratio=0.8,
        recorder_capture_ratio_min_points=5,
        recorder_persist_stale_s=900.0,
        recorder_max_consecutive_flush_failures=2,
        recorder_reference_grace_s=1800.0,
        recorder_rejected_fraction_alert=0.02,
        recorder_rejected_max_consecutive_points=30,
        recorder_capture_stale_s=90.0,
        recorder_capture_ready_grace_s=800.0,
        recorder_capture_dual_active_max_s=1000.0,
        recorder_normalizer_max_lag_s=500.0,
        recorder_normalizer_max_consecutive_failures=4,
        recorder_compaction_max_delay_s=9000.0,
        recorder_startup_grace_s=90.0,
        recorder_prune_blocked_alert_after_s=7200.0,
        recorder_local_disk_budget_bytes=4 * 1024**3,
    )
    watch = _build(settings, alert=lambda event, detail: True)
    assert watch is not None
    assert watch._thresholds.sampler_max_consecutive_failures == 4
    assert watch._thresholds.min_capture_ratio == 0.8
    assert watch._thresholds.capture_ratio_min_points == 5
    assert watch._thresholds.persist_stale_s == 900.0
    assert watch._thresholds.max_consecutive_flush_failures == 2
    assert watch._thresholds.reference_grace_s == 1800.0
    assert watch._thresholds.rejected_fraction_alert == 0.02
    assert watch._thresholds.rejected_max_consecutive_points == 30
    assert watch._thresholds.capture_stale_s == 90.0
    assert watch._thresholds.capture_ready_grace_s == 800.0
    assert watch._thresholds.capture_dual_active_max_s == 1000.0
    assert watch._thresholds.normalizer_max_lag_s == 500.0
    assert watch._thresholds.normalizer_max_consecutive_failures == 4
    assert watch._thresholds.compaction_max_delay_s == 9000.0
    assert watch._thresholds.startup_grace_s == 90.0
    assert watch._thresholds.prune_blocked_alert_after_s == 7200.0
    assert watch._thresholds.local_disk_budget_bytes == 4 * 1024**3
    assert not hasattr(watch._thresholds, "liquidation_max_failed_connections")


def test_outage_replay_alerts_in_minutes(tmp_path: Path) -> None:
    """A normalizer outage pages once, within minutes of the fifth failed cycle."""
    from src.live.recorder_watch import RecorderWatchdog as _Watchdog

    start = pd.Timestamp("2026-09-25T08:02:00Z")
    now = _ManualNow(start.isoformat())
    calls: list[tuple[str, str]] = []

    def _alert(event: str, detail: str) -> bool:
        calls.append((event, detail))
        return True

    watch = _Watchdog(
        heartbeat_path=tmp_path / "recorder_heartbeat.json",
        thresholds=_thresholds(),
        interval_s=60.0,
        alert=_alert,
        now_fn=now,
    )
    for minute in range(7):
        stamp = (start + pd.Timedelta(minutes=minute)).isoformat()
        now.t = stamp
        payload = _heartbeat(ts=stamp, consecutive_failures=minute)
        _write(watch._heartbeat_path, payload)
        watch.check_once()
    assert [event for event, _ in calls] == ["recorder_unhealthy"]
    assert "normalizer_failing:normalizer" in calls[0][1]


def test_settings_reject_inconsistent_capture_window() -> None:
    """A capture staleness at or beyond heartbeat staleness fails validation."""
    import pytest

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="recorder_capture_stale_s"):
        LiveSettings(recorder_capture_stale_s=600.0, recorder_heartbeat_stale_s=600.0)
    with pytest.raises(ValidationError, match="recorder_capture_dual_active_max_s"):
        LiveSettings(recorder_capture_dual_active_max_s=100.0, recorder_capture_ready_grace_s=900.0)
