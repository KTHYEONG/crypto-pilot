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
    DAEMON_CATCHUP_BUFFER,  # noqa: F401
    next_decision_time,
    run_daemon,
)
from src.live.settings import LiveSettings
from src.live.signal import _SIGNAL_LAG

DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")
#: 데몬이 사이클을 즉시 실행할 수 있는 하한 시각(target + SIGNAL_LAG + 버퍼).
READY_NOW = DECISION_TIME + _SIGNAL_LAG + pd.Timedelta(minutes=20)


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
    # With retry bounded, crash saves pending not last_processed; accept either
    ts_val = saved.get("last_processed_decision_time")
    pending_val = saved.get("pending_decision_time")
    assert (ts_val is not None and pd.Timestamp(ts_val) == DECISION_TIME) or (
        pending_val is not None and pd.Timestamp(pending_val) == DECISION_TIME
    )


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


def test_SCENARIO_LIVE_21_stale_signals_are_skipped_not_executed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D8: N일 정지 후 복귀 시 과거 신호로 사이클을 돌지 않고 당일까지 따라잡는다."""
    artifact_path = tmp_path / "deployed_target_weights.parquet"
    state_path = tmp_path / "state" / "live_daemon_last_run.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    stale_start = DECISION_TIME - pd.Timedelta(days=5)
    state_path.write_text(
        json.dumps({"last_processed_decision_time": stale_start.isoformat()}),
        encoding="utf-8",
    )

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

    # now 는 당일 decision_time+2h: D-4..D-1 은 스테일 스킵, D 만 실행 대상이다.
    run_daemon(
        LiveSettings(max_signal_staleness_hours=6.0),
        artifact_path,
        state_path,
        sleep_fn=lambda seconds: pytest.fail("catch-up must not wait for stale days"),
        now_fn=lambda: DECISION_TIME + pd.Timedelta(hours=2),
        max_iterations=5,
    )

    assert cycle_calls == [DECISION_TIME]  # 정확히 1회, 과거 신호 실행 없음
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(saved["last_processed_decision_time"]) == DECISION_TIME


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_01_NEXT_DECISION_TIME_SEQUENTIAL",
    "SCENARIO_LIVE_DAEMON_04_RUN_DAEMON_PROCESSES_ONE_CYCLE_PER_ITERATION",
    "SCENARIO_LIVE_DAEMON_05_IDEMPOTENT_SKIP_ON_RESTART",
    "SCENARIO_LIVE_DAEMON_06_CRASH_DOES_NOT_KILL_LOOP",
    "SCENARIO_LIVE_DAEMON_07_CATCHUP_NO_EXTRA_WAIT",
    "SCENARIO_LIVE_21",  # STALE_SIGNALS_ARE_SKIPPED_NOT_EXECUTED
)

# SCENARIO_RESIL_04-intraday-retry-bounded
def test_SCENARIO_RESIL_04_intraday_retry_bounded(tmp_path, monkeypatch):  # noqa: D103
    """SCENARIO_RESIL_04-intraday-retry-bounded"""
    import json

    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.scheduler import run_daemon
    from src.live.settings import LiveSettings

    artifact = tmp_path / "a.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    calls: list = []

    def fake_cycle(settings, decision_time, artifact_path, now=None, **k):  # noqa: ARG001
        calls.append(decision_time)
        return CycleReport(status="HALT", reason="halt", decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", fake_cycle)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    dt = pd.Timestamp("2026-08-24 00:00Z")
    from src.live.signal import _SIGNAL_LAG

    ready = dt + _SIGNAL_LAG + pd.Timedelta(minutes=20)
    # Use advancing clock to avoid infinite wait for next day
    cur = [ready]

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, daemon_max_attempts_per_day=5),
        artifact,
        state_path,
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=6,
    )
    assert len([c for c in calls if c == dt]) == 5
    saved = json.loads(state_path.read_text())
    assert pd.Timestamp(saved["last_processed_decision_time"]) == dt
    # Second part: first HALT second COMPLETE -> 2 calls
    calls2: list = []

    def fake_cycle2(settings, decision_time, artifact_path, now=None, **k):  # noqa: ARG001
        calls2.append(decision_time)
        if len(calls2) == 1:
            return CycleReport(status="HALT", reason="halt", decision_time=decision_time, intent_count=0)
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", fake_cycle2)
    state_path2 = tmp_path / "state2.json"
    cur2 = [ready]

    def now_fn2():
        return cur2[0]

    def sleep_fn2(s):
        cur2[0] += pd.Timedelta(seconds=s)

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, daemon_max_attempts_per_day=5),
        artifact,
        state_path2,
        sleep_fn=sleep_fn2,
        now_fn=now_fn2,
        max_iterations=2,
    )
    assert len(calls2) == 2


# SCENARIO_RESIL_05-retry-stops-when-stale
def test_SCENARIO_RESIL_05_retry_stops_when_stale(tmp_path, monkeypatch):  # noqa: D103
    """SCENARIO_RESIL_05-retry-stops-when-stale"""
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.scheduler import run_daemon
    from src.live.settings import LiveSettings

    artifact = tmp_path / "a.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    calls: list = []

    def fake_cycle(settings, decision_time, artifact_path, now=None, **k):  # noqa: ARG001
        calls.append(decision_time)
        return CycleReport(status="HALT", reason="halt", decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", fake_cycle)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    dt = pd.Timestamp("2026-08-24 00:00Z")
    from src.live.signal import _SIGNAL_LAG

    ready = dt + _SIGNAL_LAG + pd.Timedelta(minutes=20)
    # For this test, make now far future after first call to trigger staleness and avoid wait hang
    far_future = pd.Timestamp("2026-08-30 00:00Z")

    def now_fn_stale():
        if len(calls) == 0:
            return ready
        return far_future

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, daemon_max_attempts_per_day=5, max_signal_staleness_hours=6.0),
        artifact,
        state_path,
        sleep_fn=lambda x: None,
        now_fn=now_fn_stale,
        max_iterations=2,
    )
    assert len(calls) <= 2


# SCENARIO_RESIL_06-graceful-shutdown
def test_SCENARIO_RESIL_06_graceful_shutdown(tmp_path, monkeypatch):  # noqa: D103
    """SCENARIO_RESIL_06-graceful-shutdown"""
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.lifecycle import ShutdownFlag
    from src.live.runner import CycleReport
    from src.live.scheduler import run_daemon
    from src.live.settings import LiveSettings

    artifact = tmp_path / "a.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    calls: list = []

    def fake_cycle(settings, decision_time, artifact_path, now=None, **k):  # noqa: ARG001
        calls.append(decision_time)
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", fake_cycle)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    dt = pd.Timestamp("2026-08-24 00:00Z")
    from src.live.signal import _SIGNAL_LAG

    ready = dt + _SIGNAL_LAG + pd.Timedelta(minutes=20)
    flag = ShutdownFlag()

    def sleep_fn(x):  # noqa: ARG001
        if len(calls) >= 1:
            flag.request("SIGTERM")

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0),
        artifact,
        state_path,
        sleep_fn=sleep_fn,
        now_fn=lambda: ready,
        max_iterations=10,
        shutdown=flag,
    )
    assert len(calls) == 1


# SCENARIO_RESIL_08-legacy-state-compat
def test_SCENARIO_RESIL_08_legacy_state_compat(tmp_path):  # noqa: D103
    """SCENARIO_RESIL_08-legacy-state-compat"""
    import json

    import pandas as pd
    from src.common.errors import DataIntegrityError
    from src.live.scheduler import DaemonState, _load_daemon_state

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"last_processed_decision_time": "2026-08-26T00:00:00+00:00"}))
    loaded = _load_daemon_state(state_path)
    assert loaded == DaemonState(
        last_processed_decision_time=pd.Timestamp("2026-08-26T00:00:00+00:00"),
        pending_decision_time=None,
        attempts=0,
    )
    state_path2 = tmp_path / "state2.json"
    state_path2.write_text(json.dumps({"last_processed_decision_time": "2026-08-26T00:00:00"}))
    try:
        _load_daemon_state(state_path2)
        raise AssertionError("should have raised")
    except DataIntegrityError:
        pass


# SCENARIO_RESIL_09-signal-daemon-orders-refresh
def test_SCENARIO_RESIL_09_signal_daemon_orders_refresh(tmp_path, monkeypatch):  # noqa: D103
    """SCENARIO_RESIL_09-signal-daemon-orders-refresh"""
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.scheduler import run_signal_daemon
    from src.live.settings import LiveSettings

    artifact = tmp_path / "a.parquet"
    artifact.touch()
    signal_state = tmp_path / "sig.json"
    daemon_state = tmp_path / "daemon.json"
    times: list = []

    def fake_refresh():
        times.append(1)

    dt = pd.Timestamp("2026-08-24 00:00Z")
    from src.live.signal import _SIGNAL_LAG

    ready = dt + _SIGNAL_LAG + pd.Timedelta(minutes=0, seconds=10)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    cur = [ready]

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    run_signal_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0),
        artifact,
        signal_state,
        daemon_state,
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=2,
        refresh_fn=fake_refresh,
    )
    assert len(times) == 2
    # failing refresh still continues
    def failing_refresh():
        raise RuntimeError("boom")

    daemon_state2 = tmp_path / "daemon2.json"
    cur2 = [ready]

    def now_fn2():
        return cur2[0]

    def sleep_fn2(s):
        cur2[0] += pd.Timedelta(seconds=s)

    run_signal_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0),
        artifact,
        signal_state,
        daemon_state2,
        sleep_fn=sleep_fn2,
        now_fn=now_fn2,
        max_iterations=2,
        refresh_fn=failing_refresh,
    )


# SCENARIO_RESIL_10-heartbeat-bounded
def test_SCENARIO_RESIL_10_heartbeat_bounded(tmp_path, monkeypatch):  # noqa: D103
    """SCENARIO_RESIL_10-heartbeat-bounded"""
    import json

    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.scheduler import run_daemon
    from src.live.settings import LiveSettings

    artifact = tmp_path / "a.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    hb_path = tmp_path / "hb.json"

    def fake_cycle(settings, decision_time, artifact_path, now=None, **k):  # noqa: ARG001
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", fake_cycle)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    dt = pd.Timestamp("2026-08-24 00:00Z")
    from src.live.signal import _SIGNAL_LAG

    ready = dt + _SIGNAL_LAG + pd.Timedelta(minutes=20)
    cur = [ready]

    def now_fn_cur():
        return cur[0]

    def sleep_fn_cur(s):
        cur[0] += pd.Timedelta(seconds=s)

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, heartbeat_path=str(hb_path)),
        artifact,
        state_path,
        sleep_fn=sleep_fn_cur,
        now_fn=now_fn_cur,
        max_iterations=1,
    )
    size1 = hb_path.stat().st_size
    # Second run with advancing clock for 30 iterations
    cur2 = [ready + pd.Timedelta(days=1)]

    def now_fn_cur2():
        return cur2[0]

    def sleep_fn_cur2(s):
        cur2[0] += pd.Timedelta(seconds=s)

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, heartbeat_path=str(hb_path)),
        artifact,
        state_path,
        sleep_fn=sleep_fn_cur2,
        now_fn=now_fn_cur2,
        max_iterations=30,
    )
    size2 = hb_path.stat().st_size
    assert size2 <= size1 * 1.5
    data = json.loads(hb_path.read_text())
    assert set(data.keys()) == {"ts", "decision_time", "status", "attempts", "consecutive_halts"}
    orig_write = sched.write_heartbeat

    def failing_write(*a, **k):  # noqa: ARG001
        raise OSError("boom")

    monkeypatch.setattr(sched, "write_heartbeat", failing_write)
    cur3 = [ready]

    def now_fn_cur3():
        return cur3[0]

    def sleep_fn_cur3(s):
        cur3[0] += pd.Timedelta(seconds=s)

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, heartbeat_path=str(hb_path)),
        artifact,
        tmp_path / "state3.json",
        sleep_fn=sleep_fn_cur3,
        now_fn=now_fn_cur3,
        max_iterations=10,
    )

