# ruff: noqa
"""Execution parity part 3: the daemon waits until release plus the catch-up buffer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import src.live.scheduler as sched
from src.live.runner import CycleReport
from src.live.scheduler import DECISION_RELEASE_OFFSET, run_daemon
from src.live.settings import LiveSettings

TARGET = pd.Timestamp("2026-08-24 00:00Z")


def _report(decision_time: pd.Timestamp = TARGET) -> CycleReport:
    return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)


def _install_daemon_stubs(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001, ANN202
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, d, a, now=None, **k: _report(d))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)


def _run_with_clock(monkeypatch, tmp_path: Path, settings: LiveSettings, start: pd.Timestamp, fns: dict) -> tuple[list, Any]:  # noqa: ANN001, ANN202
    _install_daemon_stubs(monkeypatch, tmp_path)
    cur = [start]

    def now_fn() -> pd.Timestamp:
        return cur[0]

    def sleep_fn(seconds: float) -> None:
        cur[0] += pd.Timedelta(seconds=seconds)

    (tmp_path / "w.parquet").touch()
    run_daemon(
        settings,
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=1,
        refresh_fn=fns.get("refresh", lambda *a, **k: None),
        signal_step_fn=fns.get("signal", lambda *a, **k: None),
        venue_fn=fns.get("venue", lambda *a, **k: None),
        prefetch_fn=fns.get("prefetch", lambda *a, **k: None),
        prune_fn=lambda: None,
    )
    return cur, now_fn


def test_wait_ends_at_release_plus_buffer(tmp_path, monkeypatch) -> None:
    settings = LiveSettings(funding_prefetch_enabled=False, daemon_catchup_buffer_minutes=0.25)
    wait_until = TARGET + DECISION_RELEASE_OFFSET + pd.Timedelta(seconds=15)
    refresh_at: list[pd.Timestamp] = []

    def fake_refresh(t: pd.Timestamp) -> None:
        refresh_at.append(holder_now[0])
        return None

    _install_daemon_stubs(monkeypatch, tmp_path)
    holder_now = [wait_until - pd.Timedelta(seconds=60)]

    def now_fn() -> pd.Timestamp:
        return holder_now[0]

    def sleep_fn(seconds: float) -> None:
        holder_now[0] += pd.Timedelta(seconds=seconds)

    (tmp_path / "w.parquet").touch()
    run_daemon(
        settings,
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=1,
        refresh_fn=fake_refresh,
        signal_step_fn=lambda *a, **k: None,
        venue_fn=lambda *a, **k: None,
        prefetch_fn=lambda *a, **k: None,
        prune_fn=lambda: None,
    )
    assert len(refresh_at) == 1
    assert wait_until <= refresh_at[0] <= wait_until + pd.Timedelta(seconds=sched.DAEMON_POLL_INTERVAL_SECONDS)


