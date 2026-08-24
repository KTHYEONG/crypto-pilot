"""SCENARIO_LIVE_DAEMON_*: 무인 데몬 스케줄러 계약 검증(실시간 대기 없음)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.scheduler as scheduler_mod
from src.live.audit import AUDIT_LOG_ROOT
from src.live.runner import CycleReport
from src.live.scheduler import (
    DAEMON_CATCHUP_BUFFER,
    next_decision_time,
    run_daemon,
)
from src.live.settings import LiveSettings
from src.live.signal import _SIGNAL_LAG

DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")
#: 데몬이 사이클을 즉시 실행할 수 있는 하한 시각(target + SIGNAL_LAG + 버퍼).
READY_NOW = DECISION_TIME + _SIGNAL_LAG + DAEMON_CATCHUP_BUFFER


class _StopWaitingError(Exception):
    """대기 루프를 테스트가 강제 종료하기 위한 센티널."""


def _report(decision_time: pd.Timestamp = DECISION_TIME) -> CycleReport:
    return CycleReport(
        status="COMPLETE",
        reason=None,
        decision_time=decision_time,
        intent_count=0,
    )


def test_SCENARIO_LIVE_DAEMON_01_next_decision_time_sequential() -> None:
    assert next_decision_time(None, pd.Timestamp("2026-08-24 15:30Z")) == pd.Timestamp(
        "2026-08-24 00:00Z"
    )
    # 밀린 날짜도 now와 무관하게 마지막 처리일 다음날로 순차 진행한다.
    assert next_decision_time(
        pd.Timestamp("2026-08-20 00:00Z"), pd.Timestamp("2026-08-24 15:30Z")
    ) == pd.Timestamp("2026-08-21 00:00Z")
    with pytest.raises(ValueError, match="tz-aware"):
        next_decision_time(None, pd.Timestamp("2026-08-24 15:30"))
    with pytest.raises(ValueError, match="tz-aware"):
        next_decision_time(pd.Timestamp("2026-08-20 00:00Z"), pd.Timestamp("2026-08-24 15:30"))


def test_SCENARIO_LIVE_DAEMON_04_one_cycle_per_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_path = tmp_path / "deployed_target_weights.parquet"
    artifact_path.touch()
    state_path = tmp_path / "state" / "live_daemon_last_run.json"
    cycle_calls: list[pd.Timestamp] = []
    prune_calls: list[tuple[Path, pd.Timestamp]] = []

    def fake_cycle(
        settings: LiveSettings,
        decision_time: pd.Timestamp,
        artifact: Path,
        *,
        now: pd.Timestamp,
    ) -> CycleReport:
        cycle_calls.append(decision_time)
        return _report(decision_time)

    def fake_prune(root: Path, reference_date: pd.Timestamp, **_: int) -> int:
        prune_calls.append((root, reference_date))
        return 0

    monkeypatch.setattr(scheduler_mod, "run_shadow_cycle", fake_cycle)
    monkeypatch.setattr(scheduler_mod, "prune_old_audit_logs", fake_prune)

    run_daemon(
        LiveSettings(),
        artifact_path,
        state_path,
        sleep_fn=lambda seconds: pytest.fail("window already passed; must not sleep"),
        now_fn=lambda: READY_NOW,
        max_iterations=1,
    )

    assert cycle_calls == [DECISION_TIME]
    assert prune_calls == [(AUDIT_LOG_ROOT / "live", DECISION_TIME)]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(saved["last_processed_decision_time"]) == DECISION_TIME


def test_SCENARIO_LIVE_DAEMON_05_idempotent_skip_on_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_path = tmp_path / "deployed_target_weights.parquet"
    state_path = tmp_path / "state" / "live_daemon_last_run.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"last_processed_decision_time": DECISION_TIME.isoformat()}),
        encoding="utf-8",
    )

    cycle_calls: list[Any] = []

    def fake_cycle(*args: Any, **kwargs: Any) -> CycleReport:
        cycle_calls.append(args)
        return _report()

    monkeypatch.setattr(scheduler_mod, "run_shadow_cycle", fake_cycle)
    monkeypatch.setattr(scheduler_mod, "prune_old_audit_logs", lambda *_a: 0)

    sleeps: list[float] = []

    def limited_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise _StopWaitingError

    with pytest.raises(_StopWaitingError):
        run_daemon(
            LiveSettings(),
            artifact_path,
            state_path,
            sleep_fn=limited_sleep,
            now_fn=lambda: DECISION_TIME,  # 오늘자는 이미 처리됨 -> 내일자 윈도우 대기 상태
            max_iterations=1,
        )

    assert len(sleeps) >= 1
    assert cycle_calls == []  # 동일 날짜 재실행 없음


def test_SCENARIO_LIVE_DAEMON_06_crash_does_not_kill_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_path = tmp_path / "deployed_target_weights.parquet"
    state_path = tmp_path / "state" / "live_daemon_last_run.json"

    def boom(*args: Any, **kwargs: Any) -> CycleReport:
        raise RuntimeError("boom")

    monkeypatch.setattr(scheduler_mod, "run_shadow_cycle", boom)
    monkeypatch.setattr(scheduler_mod, "prune_old_audit_logs", lambda *_a: 0)

    run_daemon(
        LiveSettings(),
        artifact_path,
        state_path,
        sleep_fn=lambda seconds: None,
        now_fn=lambda: READY_NOW,
        max_iterations=1,
    )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(saved["last_processed_decision_time"]) == DECISION_TIME


def test_SCENARIO_LIVE_DAEMON_07_catchup_no_extra_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_path = tmp_path / "deployed_target_weights.parquet"
    state_path = tmp_path / "state" / "live_daemon_last_run.json"
    cycle_calls: list[pd.Timestamp] = []

    def fake_cycle(
        settings: LiveSettings,
        decision_time: pd.Timestamp,
        artifact: Path,
        *,
        now: pd.Timestamp,
    ) -> CycleReport:
        cycle_calls.append(decision_time)
        return _report(decision_time)

    monkeypatch.setattr(scheduler_mod, "run_shadow_cycle", fake_cycle)
    monkeypatch.setattr(scheduler_mod, "prune_old_audit_logs", lambda *_a: 0)

    run_daemon(
        LiveSettings(),
        artifact_path,
        state_path,
        sleep_fn=lambda seconds: pytest.fail("catch-up must run without extra wait"),
        now_fn=lambda: DECISION_TIME + pd.Timedelta(hours=3),
        max_iterations=1,
    )

    assert cycle_calls == [DECISION_TIME]


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_01_NEXT_DECISION_TIME_SEQUENTIAL",
    "SCENARIO_LIVE_DAEMON_04_RUN_DAEMON_PROCESSES_ONE_CYCLE_PER_ITERATION",
    "SCENARIO_LIVE_DAEMON_05_IDEMPOTENT_SKIP_ON_RESTART",
    "SCENARIO_LIVE_DAEMON_06_CRASH_DOES_NOT_KILL_LOOP",
    "SCENARIO_LIVE_DAEMON_07_CATCHUP_NO_EXTRA_WAIT",
)
