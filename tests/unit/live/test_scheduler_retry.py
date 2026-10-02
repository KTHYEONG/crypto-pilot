"""Retry-budget invariants for the live daemon scheduler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.scheduler as scheduler
from src.common.daemon_stages import BUSY_STAGES
from src.live.lifecycle import ShutdownFlag
from src.live.runner import CycleReport
from src.live.settings import LiveSettings

TARGET = pd.Timestamp("2026-08-24 00:00Z")
READY = TARGET + scheduler.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)


def _run_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    attempts_cap: int = 5,
    post_cycle: bool = False,
    shutdown_on_wait: bool = False,
) -> tuple[list[float], list[tuple[str, str]], dict[str, Any], int]:
    monkeypatch.setattr(scheduler, "_strategy_params_present", lambda _: True, raising=False)
    monkeypatch.setattr(scheduler, "prune_old_audit_logs", lambda *_: 0)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        scheduler,
        "_daemon_alert",
        lambda _settings, *, event, detail, **_kwargs: alerts.append((event, detail)) or True,
    )
    calls: list[int] = []

    def fail_cycle(*_args: Any, **_kwargs: Any) -> CycleReport:
        calls.append(1)
        return CycleReport(status="HALT", reason="test failure", decision_time=TARGET, intent_count=0)

    if post_cycle:
        monkeypatch.setattr(scheduler, "run_shadow_cycle", fail_cycle)
    waits: list[float] = []
    shutdown = ShutdownFlag()

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        if shutdown_on_wait:
            shutdown.request("SIGTERM")

    def fail_signal(_target: pd.Timestamp) -> None:
        calls.append(1)
        raise RuntimeError("halt")

    settings = LiveSettings(daemon_max_attempts_per_day=attempts_cap, alert_halt_streak=100)
    artifact = tmp_path / "weights.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    scheduler.run_daemon(
        settings,
        artifact,
        state_path,
        sleep_fn=sleep,
        now_fn=lambda: READY,
        max_iterations=attempts_cap,
        shutdown=shutdown if shutdown_on_wait else None,
        refresh_fn=lambda *_a, **_k: None,
        signal_step_fn=(lambda _target: None) if post_cycle else fail_signal,
        prune_fn=lambda: None,
        venue_fn=lambda _target: None,
    )
    return waits, alerts, json.loads(state_path.read_text(encoding="utf-8")), len(calls)


@pytest.mark.parametrize("cap", [5, 7])
def test_signal_halt_obeys_configured_cap_and_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cap: int
) -> None:
    waits, alerts, state, attempts = _run_failure(monkeypatch, tmp_path, attempts_cap=cap)

    assert [event for event, _ in alerts].count("day_skipped") == 1
    assert attempts == cap
    assert waits == [300.0] * sum((min(300 * 2 ** i, 2400) // 300) for i in range(cap - 1))
    assert state["pending_decision_time"] is None
    assert state["attempts"] == 0
    assert pd.Timestamp(state["last_processed_decision_time"]) == TARGET


def test_post_cycle_failure_uses_same_retry_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    waits, alerts, state, attempts = _run_failure(monkeypatch, tmp_path, post_cycle=True)

    assert attempts == 5
    assert waits == [300.0] * 15
    assert [event for event, _ in alerts].count("day_skipped") == 1
    assert state["attempts"] == 0
    assert pd.Timestamp(state["last_processed_decision_time"]) == TARGET


@pytest.mark.parametrize("post_cycle", [False, True])
def test_shutdown_during_backoff_keeps_attempt_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, post_cycle: bool
) -> None:
    waits, alerts, state, attempts = _run_failure(
        monkeypatch, tmp_path, shutdown_on_wait=True, post_cycle=post_cycle
    )

    assert attempts == 1
    assert waits == [300.0]
    assert all(event != "day_skipped" for event, _ in alerts)
    assert pd.Timestamp(state["pending_decision_time"]) == TARGET
    assert state["attempts"] == 1


def test_shutdown_already_requested_aborts_before_first_wait(tmp_path: Path) -> None:
    shutdown = ShutdownFlag(requested=True)
    state_path = tmp_path / "state.json"
    assert scheduler._retry_or_skip(
        LiveSettings(),
        state_path=state_path,
        last_processed=None,
        target=TARGET,
        attempts=0,
        failure_cause="test failure",
        now_fn=lambda: READY,
        wait=lambda *_args: pytest.fail("shutdown must be checked before waiting"),
        shutdown=shutdown,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["attempts"] == 1
    assert pd.Timestamp(state["pending_decision_time"]) == TARGET


def test_busy_stage_members_are_the_only_interruptible_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        scheduler,
        "_daemon_alert",
        lambda _settings, *, event, **_kwargs: events.append(event) or True,
    )
    heartbeat_path = tmp_path / "heartbeat.json"
    for stage in (*BUSY_STAGES, "idle"):
        heartbeat_path.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "decision_time": TARGET.isoformat(),
                    "ts": READY.isoformat(),
                }
            ),
            encoding="utf-8",
        )
        scheduler._handle_interrupted_stage(LiveSettings(), heartbeat_path, READY)

    assert len(events) == len(BUSY_STAGES)
    assert set(events) == {"cycle_interrupted"}


def test_attempt_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="daemon_max_attempts_per_day"):
        LiveSettings(daemon_max_attempts_per_day=0)
