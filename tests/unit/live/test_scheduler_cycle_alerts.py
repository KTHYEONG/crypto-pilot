"""SCENARIO_LIVE_DAEMON_*: cycle-outcome alerting and execute-stage binding contracts of the daemon."""

from __future__ import annotations

import json

import pandas as pd

from src.live.scheduler import DECISION_RELEASE_OFFSET
from src.live.settings import LiveSettings

DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")
READY_NOW = DECISION_TIME + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)


def _daemon_clock(start: pd.Timestamp):
    cur = [start]

    def now_fn() -> pd.Timestamp:
        return cur[0]

    return cur, now_fn


def test_freeze_only_degraded_cycle_does_not_page_per_cycle(tmp_path, monkeypatch) -> None:
    """Freeze-only DEGRADED writes heartbeat and advances state without a cycle_degraded alert."""

    import src.live.scheduler as sched
    from src.live.runner import CycleReport

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb_freeze.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w_freeze.parquet"
    artifact.touch()
    state_path = tmp_path / "state_freeze.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    beats: list[tuple[str, str]] = []

    def _record(path, *, decision_time, status, attempts, consecutive_halts, now, stage="idle", detail="", expected_by=None, **_k):
        beats.append((status, stage))

    monkeypatch.setattr(sched, "write_heartbeat", _record)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda *a, **k: CycleReport(status="DEGRADED", reason="unresolved_own_orders", decision_time=target, intent_count=1),
    )

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=lambda s: None, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )

    assert not [event for event, _ in alerts if event == "cycle_degraded"]
    assert ("DEGRADED", "idle") in beats
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(saved["last_processed_decision_time"]) == target


def test_derisk_degraded_cycle_still_alerts(tmp_path, monkeypatch) -> None:
    """A DEGRADED cycle that also carries de-risk reasons still pages once."""
    import src.live.scheduler as sched
    from src.live.runner import CycleReport

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb_mix.json")
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w_mix.parquet"
    artifact.touch()
    state_path = tmp_path / "state_mix.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    monkeypatch.setattr(sched, "write_heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda *a, **k: CycleReport(status="DEGRADED", reason="reconciliation_breach,unresolved_own_orders", decision_time=target, intent_count=1),
    )

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=lambda s: None, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )

    assert len([event for event, _ in alerts if event == "cycle_degraded"]) == 1


def test_execute_stage_binds_iteration_decision_time(tmp_path, monkeypatch) -> None:
    """The execute stage captures each iteration's decision time with the clock read at stage start."""
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb_bind.json")
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: True)
    calls: list[tuple] = []

    def _fake_cycle(settings, decision_time, artifact, now=None, **k):
        calls.append((pd.Timestamp(decision_time), now))
        return CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(decision_time), intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _fake_cycle)
    cur, now_fn = _daemon_clock(READY_NOW)

    def _sleep(seconds: float) -> None:
        cur[0] += pd.Timedelta(seconds=seconds)

    sched.run_daemon(
        LiveSettings(alert_outbox_path=str(tmp_path / "outbox_bind.json")), tmp_path / "w_bind.parquet", tmp_path / "state_bind.json",
        sleep_fn=_sleep, now_fn=now_fn, max_iterations=2,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None, venue_fn=lambda t: None,
    )

    assert len(calls) == 2
    first, second = calls
    assert second[0] - first[0] == pd.Timedelta(days=1)
    for decision_time, now in calls:
        assert now >= decision_time + sched.DECISION_RELEASE_OFFSET


def test_execute_stage_omits_shutdown_kwarg_when_absent(tmp_path, monkeypatch) -> None:
    """Without a shutdown flag the cycle double without a shutdown parameter still works."""
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb_noshu.json")
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: True)
    calls: list = []

    def _fake_cycle(settings, decision_time, artifact, *, now):
        calls.append(pd.Timestamp(decision_time))
        return CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(decision_time), intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _fake_cycle)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    sched.run_daemon(
        LiveSettings(alert_outbox_path=str(tmp_path / "outbox_noshu.json")), tmp_path / "w_noshu.parquet", tmp_path / "state_noshu.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None, venue_fn=lambda t: None,
    )

    assert calls == [target]


def test_awaiting_data_backoff_wait_failure_logged_and_continues(tmp_path, monkeypatch, caplog) -> None:
    """A failing backoff wait in AWAITING_DATA is logged with traceback and the loop continues."""
    import logging
    from types import SimpleNamespace

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb_wait.json")
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: True)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    stale = SimpleNamespace(ok=False, staleness_hours=999.0, failed=1, total=1)
    waits = {"n": 0}

    def _sleep(seconds: float) -> None:
        waits["n"] += 1
        if waits["n"] == 1:
            raise RuntimeError("sleep down")

    with caplog.at_level(logging.INFO, logger="LiveScheduler"):
        sched.run_daemon(
            LiveSettings(alert_outbox_path=str(tmp_path / "outbox_wait.json")), tmp_path / "w_wait.parquet", tmp_path / "state_wait.json",
            sleep_fn=_sleep, now_fn=lambda: ready, max_iterations=1,
            refresh_fn=lambda *a, **k: stale, signal_step_fn=lambda t: None, prune_fn=lambda: None, venue_fn=lambda t: None,
        )

    assert waits["n"] == 1
    assert any("[SYS] awaiting-data backoff wait failed" in r.message for r in caplog.records)


def test_sizing_note_falls_back_on_missing_attributes() -> None:
    """A frozen report lacking unit_observations falls back to empty; a full one formats the suffix."""
    import src.live.scheduler as sched

    class _Partial:
        exposure = 1.5
        equity_usdt = 2000.0

    assert sched._sizing_note(_Partial()) == ""

    class _Full:
        exposure = 1.23456
        equity_usdt = 123.456
        unit_observations = 7

    assert sched._sizing_note(_Full()) == " exposure=1.2346 equity_usdt=123.46 unit_observations=7"
