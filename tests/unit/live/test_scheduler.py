# ruff: noqa
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
from src.live.scheduler import DECISION_RELEASE_OFFSET

DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")
#: 데몬이 사이클을 즉시 실행할 수 있는 하한 시각(target + SIGNAL_LAG + 버퍼).
READY_NOW = DECISION_TIME + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)


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
    monkeypatch.setattr(scheduler_mod, "_strategy_params_present", lambda settings: True, raising=False)
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )

    assert cycle_calls == [DECISION_TIME]
    assert prune_calls == [(AUDIT_LOG_ROOT / "live", DECISION_TIME)]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(saved["last_processed_decision_time"]) == DECISION_TIME


def test_SCENARIO_LIVE_DAEMON_05_idempotent_skip_on_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scheduler_mod, "_strategy_params_present", lambda settings: True, raising=False)
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )

    assert len(sleeps) >= 1
    assert cycle_calls == []  # 동일 날짜 재실행 없음


def test_SCENARIO_LIVE_DAEMON_06_crash_does_not_kill_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scheduler_mod, "_strategy_params_present", lambda settings: True, raising=False)
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
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
    monkeypatch.setattr(scheduler_mod, "_strategy_params_present", lambda settings: True, raising=False)
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
        now_fn=lambda: DECISION_TIME + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20),
        max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
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

# SCENARIO_RESIL_04-intraday-retry-bounded
def test_SCENARIO_RESIL_04_intraday_retry_bounded(tmp_path, monkeypatch):  # noqa: D103
    """SCENARIO_RESIL_04-intraday-retry-bounded"""
    import pandas as pd
    import src.live.scheduler as sched
    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
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
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    ready = dt + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )
    assert len(calls2) == 2


# SCENARIO_RESIL_06-graceful-shutdown
def test_SCENARIO_RESIL_06_graceful_shutdown(tmp_path, monkeypatch):  # noqa: D103
    """SCENARIO_RESIL_06-graceful-shutdown"""
    import pandas as pd
    import src.live.scheduler as sched
    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
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
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    ready = dt + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
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


# SCENARIO_RESIL_10-heartbeat-bounded
def test_SCENARIO_RESIL_10_heartbeat_bounded(tmp_path, monkeypatch):  # noqa: D103
    """SCENARIO_RESIL_10-heartbeat-bounded"""
    import pandas as pd
    import src.live.scheduler as sched
    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
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
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    ready = dt + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )
    size2 = hb_path.stat().st_size
    assert size2 <= size1 * 1.5
    data = json.loads(hb_path.read_text())
    assert set(data.keys()) == {"ts", "decision_time", "status", "attempts", "consecutive_halts", "stage", "detail", "expected_by"}
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
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )

def test_run_daemon_has_no_params_gate(monkeypatch, tmp_path) -> None:
    import json

    import pandas as pd

    import src.live.scheduler as sched

    order: list[str] = []
    from src.live.runner import CycleReport

    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda s, t, w, now=None: order.append("cycle") or CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=0),
    )

    hb = tmp_path / "hb.json"
    settings = sched.LiveSettings(heartbeat_path=str(hb))
    sched.run_daemon(settings, tmp_path / "w.parquet", tmp_path / "state.json",
                     sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-08-25 23:20:00", tz="UTC"),
                     max_iterations=1, signal_step_fn=lambda target: order.append("signal"),
                     refresh_fn=lambda *a, **k: order.append("refresh"), prune_fn=lambda: order.append("prune"),
                     venue_fn=lambda target: order.append("venue"))
    assert order == ["venue", "refresh", "signal", "cycle", "prune"]
    assert json.loads(hb.read_text())["status"] == "COMPLETE"


def test_run_daemon_runs_signal_then_cycle(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.live.scheduler as sched

    order = []
    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, t, w, now=None: order.append(("cycle", pd.Timestamp(t))) or sched.CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=0))

    def _sig(target):
        order.append(("signal", pd.Timestamp(target)))

    settings = sched.LiveSettings(heartbeat_path=str(tmp_path / "hb.json"))
    sched.run_daemon(settings, tmp_path / "w.parquet", tmp_path / "state.json",
                     sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-08-25 23:20:00", tz="UTC"),
                     max_iterations=1, signal_step_fn=_sig, refresh_fn=lambda *a, **k: None, prune_fn=lambda: None)
    assert [k for k, _ in order] == ["signal", "cycle"]
    assert order[0][1] == order[1][1]



# --- auto appended from contract ---
def test_run_daemon_awaiting_data_when_refresh_fails(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    # 디스크 데이터도 낡았을 때만 AWAITING_DATA -- staleness 판정을 무한대로 고정.
    monkeypatch.setattr("src.live.data_refresh.market_data_staleness_hours", lambda *a, **k: float("inf"))
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    step_calls: list[object] = []

    def _bad_refresh(*a: object, **k: object) -> None:
        raise RuntimeError("cold")

    sched.run_daemon(
        LiveSettings(),
        artifact,
        tmp_path / "state.json",
        sleep_fn=lambda _s: None,
        now_fn=lambda: sched_now(),
        max_iterations=1,
        refresh_fn=_bad_refresh,
        signal_step_fn=lambda *a, **k: step_calls.append(a),
        prune_fn=lambda: None,
    )

    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb["status"] == "AWAITING_DATA"
    assert step_calls == []


def sched_now() -> "pd.Timestamp":
    import pandas as pd
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    return pd.Timestamp("2026-08-24 00:00Z") + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)




def test_daemon_runs_prune_after_execute(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: __import__("src.live.runner", fromlist=["CycleReport"]).CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp("2026-08-24 00:00Z"), intent_count=0))
    order: list[str] = []
    artifact = tmp_path / "w.parquet"
    artifact.touch()

    sched.run_daemon(
        LiveSettings(),
        artifact,
        tmp_path / "state.json",
        sleep_fn=lambda _s: None,
        now_fn=lambda: pd.Timestamp("2026-08-24 00:00Z") + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20),
        max_iterations=1,
        refresh_fn=lambda *a, **k: order.append("refresh"),
        signal_step_fn=lambda *a, **k: order.append("signal"),
        prune_fn=lambda: order.append("prune"),
    )

    assert order == ["refresh", "signal", "prune"]


def test_daemon_prune_failure_is_non_fatal(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: __import__("src.live.runner", fromlist=["CycleReport"]).CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp("2026-08-24 00:00Z"), intent_count=0))
    ran: list[str] = []
    artifact = tmp_path / "w.parquet"
    artifact.touch()

    def _boom() -> None:
        raise RuntimeError("disk busy")

    sched.run_daemon(
        LiveSettings(),
        artifact,
        tmp_path / "state.json",
        sleep_fn=lambda _s: None,
        now_fn=lambda: pd.Timestamp("2026-08-24 00:00Z") + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20),
        max_iterations=1,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=lambda *a, **k: ran.append("signal"),
        prune_fn=_boom,
    )

    assert ran == ["signal"]


def test_default_data_refresh_splits_crypto_and_seeds_long_window(monkeypatch, tmp_path) -> None:
    import json

    import src.live.scheduler as sched
    from src.live.data_refresh import RefreshReport
    from src.mhs.params import LIVE_FROZEN_WARMUP_DAYS

    payload = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "COIN"},
            {"symbol": "AAPLUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "EQUITY"},
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto_symbols.json")
    monkeypatch.setattr(sched, "DATA_DIR", tmp_path / "data")

    captured: dict = {}

    def _fake_refresh(*a, **k):
        captured.update(k)
        return RefreshReport(total=1, fresh=0, refreshed=1, failed=0, deadline_skipped=0, elapsed_s=0.1, deadline_hit=False, staleness_hours=1.0, ok=True)

    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", _fake_refresh)
    import pandas as pd

    from src.live.settings import LiveSettings

    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"))
    (tmp_path / "ledger.json").write_text(
        json.dumps({"positions": {"GHOSTUSDT": "1"}, "cash_usdt": "1000"}), encoding="utf-8"
    )
    decision_time = pd.Timestamp("2026-09-01 00:00Z")
    rep = sched._default_data_refresh(settings, tmp_path / "w.parquet", decision_time)
    assert rep.ok is True
    assert captured["symbols"] == ["BTCUSDT"]
    assert captured["seed_lookback_days"] == LIVE_FROZEN_WARMUP_DAYS + 30
    saved = json.loads((tmp_path / "non_crypto_symbols.json").read_text(encoding="utf-8"))
    assert saved["symbols"] == ["AAPLUSDT"]
    assert (tmp_path / "data" / "state" / "venue_listing" / "20260901.json.gz").exists()




def test_run_daemon_emails_alert_on_halt_streak(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr("src.live.alerting.post_alert", lambda *a, **k: True)

    emails: list[str] = []

    def _fake_email(*, gmail_user, gmail_app_password, event, detail, decision_time, now):
        emails.append(event)
        return True

    monkeypatch.setattr("src.live.alerting.send_email_alert", _fake_email)

    def _halt_cycle(settings, decision_time, artifact, *, now):
        from src.live.runner import CycleReport
        return CycleReport(status="HALT", reason="x", decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _halt_cycle)
    artifact = tmp_path / "w.parquet"
    artifact.touch()

    base = pd.Timestamp("2026-08-24 00:00Z")
    cur = [base + pd.Timedelta(days=5) + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)]

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    sched.run_daemon(
        LiveSettings(
            alert_halt_streak=2,
            daemon_max_attempts_per_day=1,
            alert_gmail_user="bot@gmail.com",
            alert_gmail_app_password="pw",
            alert_outbox_path=str(tmp_path / "outbox.json"),
        ),
        artifact,
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=3,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=lambda *a, **k: None,
        prune_fn=lambda: None,
    )

    assert "halt_streak" in emails
    assert emails.count("halt_streak") == 2


def test_run_daemon_proceeds_degraded_when_cached_data_fresh_enough(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import scheduler as sched
    from src.live.data_refresh import RefreshReport
    from src.live.settings import ExecutionMode, LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda s: True, raising=False)
    alerts: list[str] = []
    monkeypatch.setattr(sched, "_daemon_alert", lambda s, *, event, detail, decision_time, now, **_k: alerts.append(event))

    class _Report:
        pass

    rep = RefreshReport(total=500, fresh=0, refreshed=400, failed=100, deadline_skipped=0,
                        elapsed_s=12.0, deadline_hit=False, staleness_hours=20.0, ok=False)

    steps: list[str] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda settings, target, wp, now=None: type("R", (), {"status": "COMPLETE", "reason": None})())

    settings = LiveSettings(mode=ExecutionMode.PAPER, heartbeat_path=str(tmp_path / "hb.json"), max_market_data_staleness_hours=30.0)
    sched.run_daemon(
        settings, tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-09-01T23:20:00Z"),
        max_iterations=1,
        refresh_fn=lambda *a, **k: rep,
        signal_step_fn=lambda target: steps.append("signal"),
        prune_fn=lambda: steps.append("prune"),
    )

    assert steps == ["signal", "prune"]
    assert "data_degraded" in alerts
    import json
    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb["status"] == "COMPLETE"


def test_run_daemon_awaiting_data_when_staleness_beyond_limit(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    from src.live import scheduler as sched
    from src.live.data_refresh import RefreshReport
    from src.live.settings import ExecutionMode, LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda s: True, raising=False)
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: None)

    rep = RefreshReport(total=500, fresh=0, refreshed=0, failed=500, deadline_skipped=0,
                        elapsed_s=5.0, deadline_hit=False, staleness_hours=200.0, ok=False)
    called: list[str] = []

    settings = LiveSettings(mode=ExecutionMode.PAPER, heartbeat_path=str(tmp_path / "hb.json"), max_market_data_staleness_hours=30.0)
    sched.run_daemon(
        settings, tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-09-01T23:20:00Z"),
        max_iterations=1,
        refresh_fn=lambda *a, **k: rep,
        signal_step_fn=lambda target: called.append("signal"),
        prune_fn=lambda: called.append("prune"),
    )

    assert called == []
    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb["status"] == "AWAITING_DATA"


def test_run_daemon_legacy_none_refresh_still_proceeds(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    from src.live import scheduler as sched
    from src.live.settings import ExecutionMode, LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda s: True, raising=False)
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: None)
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda settings, target, wp, now=None: type("R", (), {"status": "COMPLETE", "reason": None})())

    steps: list[str] = []
    settings = LiveSettings(mode=ExecutionMode.PAPER, heartbeat_path=str(tmp_path / "hb.json"))
    sched.run_daemon(
        settings, tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-09-01T23:20:00Z"),
        max_iterations=1,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=lambda target: steps.append("signal"),
        prune_fn=lambda: steps.append("prune"),
    )

    assert steps == ["signal", "prune"]
    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb["status"] == "COMPLETE"


def test_run_daemon_alerts_day_skipped_when_cycle_halts_on_last_attempt(tmp_path, monkeypatch) -> None:
    import subprocess
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched,
        "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    import json

    def _halt_cycle(settings, decision_time, artifact_path, *, now):
        return CycleReport(status="HALT", reason="reconcile_breach", decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _halt_cycle)

    sched.run_daemon(
        LiveSettings(daemon_max_attempts_per_day=1, alert_halt_streak=5),
        artifact,
        state_path,
        sleep_fn=lambda s: None,
        now_fn=lambda: ready,
        max_iterations=1,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=lambda *a, **k: None,
        prune_fn=lambda: None,
    )

    skipped = [detail for event, detail in alerts if event == "day_skipped"]
    assert skipped == ["attempts=1 cause=cycle status=HALT reason=reconcile_breach"]
    saved = json.loads(state_path.read_text())
    assert pd.Timestamp(saved["last_processed_decision_time"]) == target


def test_run_daemon_alerts_day_skipped_when_signal_step_fails_on_last_attempt(tmp_path, monkeypatch) -> None:
    import subprocess
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched,
        "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    cycle_calls: list[pd.Timestamp] = []
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now: cycle_calls.append(decision_time),
    )

    def _failing_signal_step(t):
        raise subprocess.CalledProcessError(1, ["python", "-m", "src.cli.main", "live", "frozen-step"])

    sched.run_daemon(
        LiveSettings(daemon_max_attempts_per_day=1, alert_halt_streak=1),
        artifact,
        state_path,
        sleep_fn=lambda s: None,
        now_fn=lambda: ready,
        max_iterations=1,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=_failing_signal_step,
        prune_fn=lambda: None,
    )

    assert cycle_calls == []
    assert any(event == "halt_streak" and detail.startswith("consecutive_halts=1 cause=frozen_step CalledProcessError:") for event, detail in alerts)
    assert any(event == "day_skipped" and detail.startswith("attempts=1 cause=frozen_step CalledProcessError:") for event, detail in alerts)


def test_run_daemon_halt_streak_detail_names_crashed_cycle_exception(tmp_path, monkeypatch) -> None:
    import subprocess
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched,
        "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    def _crash(settings, decision_time, artifact_path, *, now):
        raise RuntimeError("boom")

    monkeypatch.setattr(sched, "run_shadow_cycle", _crash)

    sched.run_daemon(
        LiveSettings(daemon_max_attempts_per_day=5, alert_halt_streak=1),
        artifact,
        state_path,
        sleep_fn=lambda s: None,
        now_fn=lambda: ready,
        max_iterations=1,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=lambda *a, **k: None,
        prune_fn=lambda: None,
    )

    assert [a for a in alerts if a[0] != "venue_capture_failed"] == [("halt_streak", "consecutive_halts=1 cause=cycle crashed RuntimeError")]


def test_run_daemon_alerts_halt_streak_once_per_decision_day(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    events: list[tuple[str, pd.Timestamp]] = []
    monkeypatch.setattr(
        "src.live.alerting.post_alert",
        lambda url, *, event, detail, decision_time, now, **_k: events.append((event, decision_time)) or True,
    )

    def _halt_cycle(settings, decision_time, artifact, *, now):
        return CycleReport(status="HALT", reason="x", decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _halt_cycle)
    artifact = tmp_path / "w.parquet"
    artifact.touch()

    base = pd.Timestamp("2026-08-24 00:00Z")
    day1 = base + pd.Timedelta(days=5)
    cur = [day1 + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)]

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    sched.run_daemon(
        LiveSettings(alert_webhook_url="https://h.example", alert_halt_streak=2, daemon_max_attempts_per_day=1, alert_outbox_path=str(tmp_path / "outbox.json")),
        artifact,
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=3,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=lambda *a, **k: None,
        prune_fn=lambda: None,
    )

    day2 = day1 + pd.Timedelta(days=1)
    day3 = day1 + pd.Timedelta(days=2)
    assert [dt for ev, dt in events if ev == "halt_streak"] == [day2, day3]
    assert [dt for ev, dt in events if ev == "day_skipped"] == [day1, day2, day3]


def test_run_daemon_degraded_alert_detail_includes_refresh_failure_count(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import scheduler as sched
    from src.live.data_refresh import RefreshReport
    from src.live.settings import ExecutionMode, LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda s: True, raising=False)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda s, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    rep = RefreshReport(total=500, fresh=0, refreshed=21, failed=479, deadline_skipped=0,
                        elapsed_s=12.0, deadline_hit=False, staleness_hours=0.1, ok=False)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, target, wp, now=None: type("R", (), {"status": "COMPLETE", "reason": None})(),
    )

    settings = LiveSettings(mode=ExecutionMode.PAPER, heartbeat_path=str(tmp_path / "hb.json"), max_market_data_staleness_hours=30.0)
    sched.run_daemon(
        settings, tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-09-01T23:20:00Z"),
        max_iterations=1,
        refresh_fn=lambda *a, **k: rep,
        signal_step_fn=lambda target: None,
        prune_fn=lambda: None,
    )

    assert any(ev == "data_degraded" and detail.startswith("staleness_h=0.1 failed=479/500 err=None") for ev, detail in alerts)


def test_run_daemon_refresh_failed_alert_detail_includes_refresh_failure_count(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import scheduler as sched
    from src.live.data_refresh import RefreshReport
    from src.live.settings import ExecutionMode, LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda s: True, raising=False)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda s, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    rep = RefreshReport(total=500, fresh=0, refreshed=0, failed=500, deadline_skipped=0,
                        elapsed_s=5.0, deadline_hit=False, staleness_hours=200.0, ok=False)

    settings = LiveSettings(mode=ExecutionMode.PAPER, heartbeat_path=str(tmp_path / "hb.json"), max_market_data_staleness_hours=30.0)
    sched.run_daemon(
        settings, tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-09-01T23:20:00Z"),
        max_iterations=1,
        refresh_fn=lambda *a, **k: rep,
        signal_step_fn=lambda target: None,
        prune_fn=lambda: None,
    )

    assert any(ev == "data_refresh_failed" and detail.startswith("staleness_h=200.0 failed=500/500 err=None") for ev, detail in alerts)


def test_run_daemon_halt_streak_detail_names_generic_signal_step_exception(tmp_path, monkeypatch) -> None:
    import subprocess
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched,
        "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    cycle_calls: list[pd.Timestamp] = []
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now: cycle_calls.append(decision_time),
    )

    def _crashing_signal_step(t):
        raise RuntimeError("worker died")

    sched.run_daemon(
        LiveSettings(daemon_max_attempts_per_day=1, alert_halt_streak=1),
        artifact,
        state_path,
        sleep_fn=lambda s: None,
        now_fn=lambda: ready,
        max_iterations=1,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=_crashing_signal_step,
        prune_fn=lambda: None,
    )

    assert cycle_calls == []
    assert ("halt_streak", "consecutive_halts=1 cause=frozen_step RuntimeError: worker died") in alerts
    assert ("day_skipped", "attempts=1 cause=frozen_step RuntimeError: worker died") in alerts


def test_save_daemon_state_is_atomic_when_replace_fails(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.scheduler import DaemonState

    state_path = tmp_path / "state.json"
    decision = pd.Timestamp("2026-08-24 00:00Z")
    sched._save_daemon_state(state_path, DaemonState(last_processed_decision_time=decision, pending_decision_time=None, attempts=0))
    original = state_path.read_text(encoding="utf-8")

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(sched.os, "replace", _boom)
    with pytest.raises(OSError):
        sched._save_daemon_state(
            state_path,
            DaemonState(last_processed_decision_time=decision, pending_decision_time=decision + pd.Timedelta(days=1), attempts=2),
        )

    assert state_path.read_text(encoding="utf-8") == original
    assert json.loads(original)["last_processed_decision_time"] == "2026-08-24T00:00:00+00:00"


def test_write_heartbeat_is_atomic_and_carries_stage(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched

    hb = tmp_path / "hb.json"
    decision = pd.Timestamp("2026-08-24 00:00Z")
    sched.write_heartbeat(hb, decision_time=decision, status="RUNNING", attempts=1, consecutive_halts=0, now=decision, stage="signal")
    running = json.loads(hb.read_text(encoding="utf-8"))
    sched.write_heartbeat(hb, decision_time=decision, status="COMPLETE", attempts=0, consecutive_halts=0, now=decision)
    complete_text = hb.read_text(encoding="utf-8")

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(sched.os, "replace", _boom)
    with pytest.raises(OSError):
        sched.write_heartbeat(hb, decision_time=decision, status="HALT", attempts=2, consecutive_halts=1, now=decision)

    assert running["stage"] == "signal"
    assert set(running) == {"ts", "decision_time", "status", "attempts", "consecutive_halts", "stage", "detail", "expected_by"}
    assert json.loads(complete_text)["stage"] == "idle"
    assert hb.read_text(encoding="utf-8") == complete_text


def test_run_daemon_writes_full_state_once_and_legacy_writer_removed(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    writes: list[object] = []
    original_save = sched._save_daemon_state

    def _recording_save(path, state):
        writes.append(state)
        original_save(path, state)

    monkeypatch.setattr(sched, "_save_daemon_state", _recording_save)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now: CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0),
    )
    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=lambda s: None, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )
    complete_keys = set(json.loads(state_path.read_text(encoding="utf-8")))
    complete_writes = len(writes)

    halted_state = tmp_path / "halted.json"
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now: CycleReport(status="HALT", reason="x", decision_time=decision_time, intent_count=0),
    )
    sched.run_daemon(
        LiveSettings(daemon_max_attempts_per_day=1, alert_halt_streak=5), artifact, halted_state,
        sleep_fn=lambda s: None, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )

    assert not hasattr(sched, "_save_last_processed")
    assert not hasattr(sched, "_load_last_processed")
    assert complete_writes == 1
    assert complete_keys == {"last_processed_decision_time", "pending_decision_time", "attempts"}
    assert set(json.loads(halted_state.read_text(encoding="utf-8"))) == {"last_processed_decision_time", "pending_decision_time", "attempts"}
    assert len(writes) == 2


def test_run_daemon_corrupt_state_alerts_and_idles_without_reset(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    state_path.write_text("{", encoding="utf-8")
    sleeps: list[float] = []
    cycles: list[object] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: cycles.append(a))

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=sleeps.append, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: cycles.append(t), prune_fn=lambda: None,
    )

    heartbeat = json.loads(hb_path.read_text(encoding="utf-8"))
    assert alerts == [("state_corrupt", "path=state.json error=DataIntegrityError")]
    assert heartbeat["status"] == "STATE_CORRUPT"
    assert heartbeat["stage"] == "idle"
    assert heartbeat["decision_time"] == "2026-08-24T00:00:00+00:00"
    assert sleeps == [sched.DAEMON_POLL_INTERVAL_SECONDS]
    assert state_path.read_text(encoding="utf-8") == "{"
    assert cycles == []


def test_run_daemon_heartbeat_stage_transitions_for_complete_cycle(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    beats: list[tuple[str, str]] = []

    def _record(path, *, decision_time, status, attempts, consecutive_halts, now, stage="idle", detail="", expected_by=None, **_k):
        beats.append((status, stage))

    monkeypatch.setattr(sched, "write_heartbeat", _record)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now: CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0),
    )

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=lambda s: None, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )

    assert beats == [("RUNNING", "refresh"), ("RUNNING", "signal"), ("RUNNING", "execute"), ("COMPLETE", "idle")]


@pytest.mark.timeout(20)
def test_run_daemon_default_wait_is_interrupted_by_shutdown(tmp_path, monkeypatch) -> None:
    import json
    import threading
    import time
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.lifecycle import ShutdownFlag
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    cycles: list[object] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: cycles.append(a))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"last_processed_decision_time": "2026-08-23T00:00:00+00:00"}), encoding="utf-8")
    flag = ShutdownFlag()
    timer = threading.Timer(0.2, flag.request, args=("SIGTERM",))
    timer.daemon = True
    started = time.monotonic()
    timer.start()

    sched.run_daemon(
        LiveSettings(), tmp_path / "w.parquet", state_path,
        now_fn=lambda: pd.Timestamp("2026-08-23 12:00Z"), max_iterations=3, shutdown=flag,
        refresh_fn=lambda *a, **k: cycles.append("refresh"), signal_step_fn=lambda t: cycles.append(t), prune_fn=lambda: None,
    )

    assert time.monotonic() - started < 5.0
    assert cycles == []


def test_run_daemon_honors_shutdown_at_stage_boundaries(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    from src.live.lifecycle import ShutdownFlag

    calls: list[str] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: calls.append("cycle"))

    def _run(flag, *, venue_hook, refresh_hook, signal_hook, state_file):
        sched.run_daemon(
            LiveSettings(), artifact, tmp_path / state_file, sleep_fn=lambda s: None, now_fn=lambda: ready,
            max_iterations=1, shutdown=flag, refresh_fn=refresh_hook, signal_step_fn=signal_hook, prune_fn=lambda: None,
            venue_fn=venue_hook,
        )

    before_venue = ShutdownFlag()
    _run(
        before_venue,
        venue_hook=lambda target: (before_venue.request("SIGTERM"), calls.append("venue-1")),
        refresh_hook=lambda *a, **k: calls.append("refresh-1"),
        signal_hook=lambda t: calls.append("signal-1"),
        state_file="s1.json",
    )
    before_refresh = ShutdownFlag()
    _run(
        before_refresh,
        venue_hook=lambda target: None,
        refresh_hook=lambda *a, **k: (before_refresh.request("SIGTERM"), calls.append("refresh-2")),
        signal_hook=lambda t: calls.append("signal-2"),
        state_file="s2.json",
    )
    before_signal = ShutdownFlag()
    _run(
        before_signal,
        venue_hook=lambda target: None,
        refresh_hook=lambda *a, **k: None,
        signal_hook=lambda t: (before_signal.request("SIGTERM"), calls.append("signal-3")),
        state_file="s3.json",
    )

    assert calls == ["venue-1", "refresh-2", "signal-3"]
    assert not (tmp_path / "s1.json").exists()
    assert not (tmp_path / "s2.json").exists()
    assert not (tmp_path / "s3.json").exists()


def test_run_daemon_passes_shutdown_to_cycle(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    from src.live.lifecycle import ShutdownFlag

    seen: list[object] = []

    def _cycle(settings, decision_time, artifact_path, *, now, shutdown=None):
        seen.append(shutdown)
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _cycle)
    flag = ShutdownFlag()

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=lambda s: None, now_fn=lambda: ready,
        max_iterations=1, shutdown=flag, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )

    assert seen == [flag]


def test_run_daemon_catchup_skips_stale_days_to_earliest_fresh_decision(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    class _Stop(Exception):
        pass

    def _stop_on_wait(seconds):
        raise _Stop

    cycles: list[object] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: cycles.append(a))
    state_path.write_text(json.dumps({"last_processed_decision_time": "2026-08-20T00:00:00+00:00"}), encoding="utf-8")

    with pytest.raises(_Stop):
        sched.run_daemon(
            LiveSettings(max_signal_staleness_hours=6.0), artifact, state_path, sleep_fn=_stop_on_wait,
            now_fn=lambda: pd.Timestamp("2026-08-24 10:00Z"), max_iterations=1,
            refresh_fn=lambda *a, **k: cycles.append("refresh"), signal_step_fn=lambda t: cycles.append(t), prune_fn=lambda: None,
        )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert alerts == [("day_skipped", "catchup skipped=2026-08-21..2026-08-24")]
    assert saved == {"last_processed_decision_time": "2026-08-24T00:00:00+00:00", "pending_decision_time": None, "attempts": 0}
    assert cycles == []


def test_run_daemon_catchup_skips_stale_pending_retry(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    class _Stop(Exception):
        pass

    def _stop_on_wait(seconds):
        raise _Stop

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: pytest.fail("stale pending must not run"))
    state_path.write_text(
        json.dumps({"last_processed_decision_time": "2026-08-20T00:00:00+00:00", "pending_decision_time": "2026-08-21T00:00:00+00:00", "attempts": 3}),
        encoding="utf-8",
    )

    with pytest.raises(_Stop):
        sched.run_daemon(
            LiveSettings(max_signal_staleness_hours=6.0), artifact, state_path, sleep_fn=_stop_on_wait,
            now_fn=lambda: pd.Timestamp("2026-08-24 10:00Z"), max_iterations=1,
            refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: pytest.fail("stale pending must not run"), prune_fn=lambda: None,
        )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert alerts == [("day_skipped", "catchup skipped=2026-08-21..2026-08-24")]
    assert saved["attempts"] == 0
    assert saved["pending_decision_time"] is None
    assert saved["last_processed_decision_time"] == "2026-08-24T00:00:00+00:00"


def test_run_daemon_no_catchup_within_freshness_window(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    ran: list[pd.Timestamp] = []

    def _cycle(settings, decision_time, artifact_path, *, now):
        ran.append(decision_time)
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _cycle)
    state_path.write_text(json.dumps({"last_processed_decision_time": "2026-08-23T00:00:00+00:00"}), encoding="utf-8")

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=lambda s: None,
        now_fn=lambda: pd.Timestamp("2026-08-24 23:20Z"), max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )

    assert ran == [target]
    assert [event for event, _ in alerts if event == "day_skipped"] == []


def test_run_daemon_restores_consecutive_halts_from_halted_heartbeat(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    hb_path.write_text(
        json.dumps({"ts": "2026-08-23T02:00:00+00:00", "decision_time": "2026-08-23T00:00:00+00:00", "status": "HALT", "attempts": 4, "consecutive_halts": 3}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now: CycleReport(status="HALT", reason="x", decision_time=decision_time, intent_count=0),
    )

    sched.run_daemon(
        LiveSettings(alert_halt_streak=4, daemon_max_attempts_per_day=5), artifact, state_path,
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )

    assert ("halt_streak", "consecutive_halts=4 cause=cycle status=HALT reason=x") in alerts


def test_run_daemon_does_not_restore_halts_from_complete_or_unreadable_heartbeat(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now: CycleReport(status="HALT", reason="x", decision_time=decision_time, intent_count=0),
    )
    heartbeats = [
        json.dumps({"status": "COMPLETE", "consecutive_halts": 3}),
        "{",
        json.dumps({"status": "HALT", "consecutive_halts": "many"}),
    ]
    for index, content in enumerate(heartbeats):
        hb_path.write_text(content, encoding="utf-8")
        sched.run_daemon(
            LiveSettings(alert_halt_streak=2, daemon_max_attempts_per_day=5), artifact, tmp_path / f"state{index}.json",
            sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
            refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        )

    assert [event for event, _ in alerts if event == "halt_streak"] == []


def test_run_daemon_corrupt_state_survives_heartbeat_write_failure(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    state_path.write_text("{", encoding="utf-8")
    waits: list[float] = []
    cycles: list[object] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: cycles.append(a))

    def _broken_heartbeat(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sched, "write_heartbeat", _broken_heartbeat)

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=waits.append, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: cycles.append(t), prune_fn=lambda: None,
    )

    assert alerts == [("state_corrupt", "path=state.json error=DataIntegrityError")]
    assert waits == [sched.DAEMON_POLL_INTERVAL_SECONDS]
    assert state_path.read_text(encoding="utf-8") == "{"
    assert not hb_path.exists()
    assert cycles == []


def test_run_daemon_signal_step_halt_waits_backoff_and_keeps_pending(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    import subprocess

    waits: list[float] = []
    cycles: list[object] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: cycles.append(a))

    def _failing_signal_step(t):
        raise subprocess.CalledProcessError(1, ["python", "-m", "src.cli.main", "live", "frozen-step"])

    sched.run_daemon(
        LiveSettings(daemon_max_attempts_per_day=5, alert_halt_streak=5), artifact, state_path,
        sleep_fn=waits.append, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=_failing_signal_step, prune_fn=lambda: None,
        venue_fn=lambda t: "captured",
    )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["pending_decision_time"] == "2026-08-24T00:00:00+00:00"
    assert saved["attempts"] == 1
    assert waits == [sched.DAEMON_RETRY_BACKOFF_SECONDS[0]]
    assert cycles == []
    assert alerts == []



def test_run_daemon_logs_elapsed_per_stage(tmp_path, monkeypatch, caplog) -> None:
    import logging
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: None)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    import re

    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now: CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0),
    )

    with caplog.at_level(logging.INFO, logger="LiveScheduler"):
        sched.run_daemon(
            LiveSettings(), artifact, state_path, sleep_fn=lambda s: None, now_fn=lambda: ready,
            max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        )

    pattern = re.compile(r"^\[SYS\] stage=(refresh|signal|execute) decision_time=2026-08-24T00:00:00\+00:00 elapsed_s=\d+\.\d$")
    stages = [m.group(1) for m in (pattern.match(r.getMessage()) for r in caplog.records) if m]
    assert stages == ["refresh", "signal", "execute"]


def test_run_daemon_logs_stage_elapsed_even_when_stage_fails(tmp_path, monkeypatch, caplog) -> None:
    import logging
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: None)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    import src.live.data_refresh as data_refresh

    cycles: list[object] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: cycles.append(a))
    monkeypatch.setattr(data_refresh, "market_data_staleness_hours", lambda root, *, now, partition="dev": 0.0)

    def _refresh_boom(*a: object, **k: object):
        raise RuntimeError("refresh down")

    def _signal_boom(t):
        raise RuntimeError("signal down")

    with caplog.at_level(logging.INFO, logger="LiveScheduler"):
        sched.run_daemon(
            LiveSettings(daemon_max_attempts_per_day=1), artifact, state_path,
            sleep_fn=lambda s: None, now_fn=lambda: ready,
            max_iterations=1, refresh_fn=_refresh_boom, signal_step_fn=_signal_boom, prune_fn=lambda: None,
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("[SYS] stage=refresh decision_time=") for m in messages)
    assert any(m.startswith("[SYS] stage=signal decision_time=") for m in messages)
    assert not any(m.startswith("[SYS] stage=execute") for m in messages)
    assert cycles == []

















# --- halt_reason_persistence contract: new scenarios ---

def test_write_heartbeat_default_detail_is_empty_string(tmp_path) -> None:
    import json
    import pandas as pd
    import src.live.scheduler as sched

    # Given: no detail kwarg passed
    hb = tmp_path / "hb.json"
    decision = pd.Timestamp("2026-08-24 00:00Z")

    # When
    sched.write_heartbeat(hb, decision_time=decision, status="COMPLETE", attempts=0, consecutive_halts=0, now=decision)

    # Then
    payload = json.loads(hb.read_text(encoding="utf-8"))
    assert payload["detail"] == ""

def test_write_heartbeat_persists_explicit_detail(tmp_path) -> None:
    import json
    import pandas as pd
    import src.live.scheduler as sched

    # Given
    hb = tmp_path / "hb.json"
    decision = pd.Timestamp("2026-08-24 00:00Z")

    # When
    sched.write_heartbeat(
        hb, decision_time=decision, status="HALT", attempts=2, consecutive_halts=1, now=decision,
        stage="idle", detail="signal_step ValueError",
    )

    # Then
    payload = json.loads(hb.read_text(encoding="utf-8"))
    assert payload["detail"] == "signal_step ValueError"

def test_run_daemon_persists_signal_halt_cause_to_heartbeat(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    # Given: strategy params present, refresh/prune no-ops, signal step raises ValueError
    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    artifact = tmp_path / "a.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)

    dt = pd.Timestamp("2026-08-24 00:00Z")
    ready = dt + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    cur = [ready]

    def now_fn(): return cur[0]
    def sleep_fn(s): cur[0] += pd.Timedelta(seconds=s)

    def boom_signal_step(target, **k):
        raise ValueError("bad panel")

    # When
    sched.run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, heartbeat_path=str(hb_path)),
        artifact, state_path,
        sleep_fn=sleep_fn, now_fn=now_fn, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=boom_signal_step, prune_fn=lambda: None,
    )

    # Then: heartbeat detail carries the same redacted cause the alert would have used
    payload = json.loads(hb_path.read_text(encoding="utf-8"))
    assert payload["status"] == "HALT"
    assert payload["detail"] == "frozen_step ValueError: bad panel"

def test_run_daemon_persists_execute_halt_cause_to_heartbeat(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    # Given: signal step succeeds, run_shadow_cycle reports a HALT with a reason code
    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)

    def fake_cycle(settings, decision_time, artifact_path, now=None, **k):  # noqa: ARG001
        return CycleReport(status="HALT", reason="STALE_MARK", decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", fake_cycle)
    artifact = tmp_path / "a.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)

    dt = pd.Timestamp("2026-08-24 00:00Z")
    ready = dt + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    cur = [ready]

    def now_fn(): return cur[0]
    def sleep_fn(s): cur[0] += pd.Timedelta(seconds=s)

    # When
    sched.run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, heartbeat_path=str(hb_path)),
        artifact, state_path,
        sleep_fn=sleep_fn, now_fn=now_fn, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )

    # Then
    payload = json.loads(hb_path.read_text(encoding="utf-8"))
    assert payload["status"] == "HALT"
    assert payload["detail"] == "cycle status=HALT reason=STALE_MARK"

def test_run_daemon_persists_awaiting_data_detail_to_heartbeat(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    # Given: strategy params present, refresh_fn raises and cached-panel staleness exceeds the hard ceiling
    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    import src.live.data_refresh as data_refresh_mod
    monkeypatch.setattr(data_refresh_mod, "market_data_staleness_hours", lambda *a, **k: 999.0)

    def boom_refresh(*a: object, **k: object):
        raise RuntimeError("network down")

    artifact = tmp_path / "a.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)

    dt = pd.Timestamp("2026-08-24 00:00Z")
    ready = dt + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    cur = [ready]

    def now_fn(): return cur[0]
    def sleep_fn(s): cur[0] += pd.Timedelta(seconds=s)

    # When
    sched.run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, heartbeat_path=str(hb_path), max_market_data_staleness_hours=6.0),
        artifact, state_path,
        sleep_fn=sleep_fn, now_fn=now_fn, max_iterations=1,
        refresh_fn=boom_refresh, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )

    # Then: heartbeat detail carries the same staleness/err summary the data_refresh_failed alert used
    payload = json.loads(hb_path.read_text(encoding="utf-8"))
    assert payload["status"] == "AWAITING_DATA"
    assert "staleness_h=999.0" in payload["detail"]
    assert "network down" in payload["detail"]


def test_run_daemon_persists_state_corrupt_detail_to_heartbeat(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    # Given: daemon state file contains invalid JSON
    artifact = tmp_path / "a.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    state_path.write_text("{not valid json")
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)

    now = pd.Timestamp("2026-08-24 00:00Z")

    # When
    sched.run_daemon(
        LiveSettings(heartbeat_path=str(hb_path)),
        artifact, state_path,
        sleep_fn=lambda s: None, now_fn=lambda: now, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )

    # Then: heartbeat detail carries the same path/error-type summary the state_corrupt alert used
    payload = json.loads(hb_path.read_text(encoding="utf-8"))
    assert payload["status"] == "STATE_CORRUPT"
    assert payload["detail"] == "path=state.json error=DataIntegrityError"


# --- auto appended from contract: live_alert_gaps ---




def test_run_daemon_alerts_interrupted_stage_once_and_logs_banner(tmp_path, monkeypatch, caplog) -> None:

    import json
    import subprocess
    from types import SimpleNamespace

    import pandas as pd
    import pytest

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET
    from src.live.signal_step_result import SignalStepResult, signal_step_result_path, write_signal_step_result

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    result_path = signal_step_result_path(artifact)

    import logging

    hb_path.write_text(
        json.dumps({
            "ts": "2026-08-24T01:10:00+00:00", "decision_time": "2026-08-24T00:00:00+00:00", "status": "RUNNING",
            "attempts": 1, "consecutive_halts": 2, "stage": "signal", "detail": "",
        }),
        encoding="utf-8",
    )

    class _Stop(Exception):
        pass

    def _stop(seconds):
        raise _Stop

    early = target + pd.Timedelta(minutes=30)
    caplog.set_level(logging.INFO, logger="LiveScheduler")

    # When: 재시작 직후(아직 실행 창 전이라 대기에서 멈춤)
    with pytest.raises(_Stop):
        sched.run_daemon(
            LiveSettings(), artifact, state_path, sleep_fn=_stop, now_fn=lambda: early, max_iterations=1,
            refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        )

    # Then
    assert alerts == [("cycle_interrupted", "stage=signal decision_time=2026-08-24T00:00:00+00:00 heartbeat_ts=2026-08-24T01:10:00+00:00")]
    heartbeat = json.loads(hb_path.read_text(encoding="utf-8"))
    assert heartbeat["stage"] == "idle"
    assert heartbeat["status"] == "INTERRUPTED"
    assert heartbeat["detail"] == "interrupted stage=signal"
    assert heartbeat["decision_time"] == "2026-08-24T00:00:00+00:00"
    assert (heartbeat["attempts"], heartbeat["consecutive_halts"]) == (1, 2)
    assert any("daemon start" in record.getMessage() for record in caplog.records)

    # When: 다시 재시작 -> idle 이므로 재알림 없음
    alerts.clear()
    with pytest.raises(_Stop):
        sched.run_daemon(
            LiveSettings(), artifact, state_path, sleep_fn=_stop, now_fn=lambda: early, max_iterations=1,
            refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        )
    assert alerts == []


def test_run_daemon_touches_heartbeat_ts_while_waiting_without_creating_it(tmp_path, monkeypatch) -> None:

    import json
    import subprocess
    from types import SimpleNamespace

    import pandas as pd
    import pytest

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET
    from src.live.signal_step_result import SignalStepResult, signal_step_result_path, write_signal_step_result

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    result_path = signal_step_result_path(artifact)

    base_payload = {
        "decision_time": "2026-08-23T00:00:00+00:00", "status": "HALT", "attempts": 2,
        "consecutive_halts": 3, "stage": "idle", "detail": "cycle status=HALT reason=x",
    }
    hb_path.write_text(json.dumps({"ts": "2026-08-23T02:00:00+00:00", **base_payload}), encoding="utf-8")
    start = target + pd.Timedelta(minutes=10)
    cur = [start]
    calls: list[float] = []

    class _Stop(Exception):
        pass

    def _sleep(seconds):
        calls.append(seconds)
        if len(calls) >= 2:
            raise _Stop
        cur[0] += pd.Timedelta(seconds=seconds)

    # When: 실행 창 전 대기 1회 후 중단
    with pytest.raises(_Stop):
        sched.run_daemon(
            LiveSettings(), artifact, state_path, sleep_fn=_sleep, now_fn=lambda: cur[0], max_iterations=1,
            refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        )

    # Then: ts 와 expected_by(다음 행동 시각 + grace)만 갱신, 상태 필드 보존
    heartbeat = json.loads(hb_path.read_text(encoding="utf-8"))
    tick = start + pd.Timedelta(seconds=calls[0])
    assert heartbeat["ts"] == tick.isoformat()
    assert pd.Timestamp(heartbeat["expected_by"]) > tick
    assert {k: v for k, v in heartbeat.items() if k not in ("ts", "expected_by")} == base_payload

    # Given: heartbeat 부재 -> 생존 틱이 파일을 만들지 않는다
    hb_path.unlink()
    calls.clear()
    cur[0] = start
    with pytest.raises(_Stop):
        sched.run_daemon(
            LiveSettings(), artifact, state_path, sleep_fn=_sleep, now_fn=lambda: cur[0], max_iterations=1,
            refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        )
    assert not hb_path.exists()


def test_daemon_alert_dispatches_through_outbox(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    settings = LiveSettings(
        alert_outbox_path=str(tmp_path / "outbox.json"),
        alert_gmail_user="bot@gmail.com",
        alert_gmail_app_password="pw",
    )
    now = pd.Timestamp("2026-08-24 01:10Z")

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(sched, "dispatch_alert", lambda s, *, event, detail, decision_time, dedupe_key, now: calls.append((event, dedupe_key)) or True)

    assert sched._daemon_alert(settings, event="halt_streak", detail="d", decision_time=None, now=now) is True
    assert calls == [("halt_streak", "halt_streak:none")]
    assert sched._daemon_alert(settings, event="halt_streak", detail="d", decision_time=None, now=now, dedupe_key="custom") is True
    assert calls[-1] == ("halt_streak", "custom")


def test_scheduler_alert_gap_helpers_tolerate_bad_inputs(tmp_path, monkeypatch) -> None:
    import json
    from types import SimpleNamespace

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    now = pd.Timestamp("2026-08-24 01:30Z")
    hb = tmp_path / "hb.json"

    # _read_heartbeat: 부재/깨짐/비-dict -> None
    assert sched._read_heartbeat(hb) is None
    hb.write_text("{", encoding="utf-8")
    assert sched._read_heartbeat(hb) is None
    hb.write_text("[1]", encoding="utf-8")
    assert sched._read_heartbeat(hb) is None

    # _touch_heartbeat: 비-dict 는 건드리지 않고, 쓰기 실패는 삼킨다
    sched._touch_heartbeat(hb, now)
    assert hb.read_text(encoding="utf-8") == "[1]"
    hb.write_text(json.dumps({"ts": "old", "stage": "idle"}), encoding="utf-8")

    def _disk_full(path, text):
        raise OSError("disk full")

    original_write = sched._atomic_write_text
    monkeypatch.setattr(sched, "_atomic_write_text", _disk_full)
    sched._touch_heartbeat(hb, now)
    assert json.loads(hb.read_text(encoding="utf-8"))["ts"] == "old"
    monkeypatch.setattr(sched, "_atomic_write_text", original_write)

    # _refresh_note
    assert sched._refresh_note(None, RuntimeError("x")) == "refresh=error:RuntimeError"
    assert sched._refresh_note(None, None) == "refresh=n/a"
    assert sched._refresh_note(SimpleNamespace(ok=True), None) == "refresh=n/a"

    # _handle_interrupted_stage: naive decision_time/비정수 카운터는 안전한 기본값으로 기록
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(sched, "_daemon_alert", lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)))
    hb.write_text(json.dumps({"stage": "execute", "decision_time": "2026-08-24T00:00:00", "attempts": "many", "ts": "t0"}), encoding="utf-8")
    sched._handle_interrupted_stage(LiveSettings(), hb, now)
    rewritten = json.loads(hb.read_text(encoding="utf-8"))
    assert alerts == [("cycle_interrupted", "stage=execute decision_time=2026-08-24T00:00:00 heartbeat_ts=t0")]
    assert (rewritten["decision_time"], rewritten["attempts"], rewritten["consecutive_halts"]) == ("2026-08-24T00:00:00+00:00", 0, 0)
    assert (rewritten["status"], rewritten["stage"]) == ("INTERRUPTED", "idle")

    # heartbeat 재기록 실패도 시작을 막지 않는다
    hb.write_text(json.dumps({"stage": "refresh", "decision_time": "2026-08-24T00:00:00+00:00", "ts": "t1"}), encoding="utf-8")

    def _broken_heartbeat(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sched, "write_heartbeat", _broken_heartbeat)
    sched._handle_interrupted_stage(LiveSettings(), hb, now)
    assert alerts[-1] == ("cycle_interrupted", "stage=refresh decision_time=2026-08-24T00:00:00+00:00 heartbeat_ts=t1")




def test_default_data_refresh_propagates_exchange_info_failure(monkeypatch) -> None:
    import urllib.error

    import pytest

    import src.live.scheduler as sched

    def _blocked(*a, **k):
        raise urllib.error.URLError("venue down")

    monkeypatch.setattr("urllib.request.urlopen", _blocked)
    from pathlib import Path

    import pandas as pd

    from src.live.settings import LiveSettings

    with pytest.raises(urllib.error.URLError):
        sched._default_data_refresh(
            LiveSettings(), Path("/tmp/w.parquet"),
            pd.Timestamp("2026-09-01 00:00Z"),
        )







def test_run_daemon_degraded_alert_detail_flags_funding_block(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import scheduler as sched
    from src.live.data_refresh import RefreshReport
    from src.live.settings import ExecutionMode, LiveSettings

    # Given
    monkeypatch.setattr(sched, "_strategy_params_present", lambda s: True, raising=False)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda s, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    rep = RefreshReport(total=500, fresh=0, refreshed=21, failed=479, deadline_skipped=0,
                        elapsed_s=12.0, deadline_hit=False, staleness_hours=0.1, ok=False, funding_blocked=True)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, target, wp, now=None: type("R", (), {"status": "COMPLETE", "reason": None})(),
    )
    settings = LiveSettings(mode=ExecutionMode.PAPER, heartbeat_path=str(tmp_path / "hb.json"), max_market_data_staleness_hours=30.0)

    # When
    sched.run_daemon(
        settings, tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-09-01T23:20:00Z"),
        max_iterations=1,
        refresh_fn=lambda *a, **k: rep,
        signal_step_fn=lambda target: None,
        prune_fn=lambda: None,
    )

    # Then
    assert any(ev == "data_degraded" and detail.startswith("staleness_h=0.1 failed=479/500 err=None") and "funding_blocked=True" in detail for ev, detail in alerts)







def test_frozen_delivery_boundary_stays_under_deploy_mhs() -> None:
    from src.common.paths import DATA_DIR, DEPLOY_MHS_DIR
    from src.live.settings import LiveSettings

    settings = LiveSettings()
    assert DATA_DIR not in DEPLOY_MHS_DIR.parents
    assert str(DEPLOY_MHS_DIR) in settings.unit_bootstrap_path
    assert str(DEPLOY_MHS_DIR) in settings.venue_fallback_path


def test_daemon_waits_until_release_plus_buffer(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, t, w, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=0))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    day = pd.Timestamp("2026-08-24 00:00Z")
    cur = [pd.Timestamp("2026-08-24 22:00Z")]
    first_stage_at: list[pd.Timestamp] = []

    def _venue(target) -> None:
        first_stage_at.append(cur[0])

    sched.run_daemon(
        LiveSettings(), tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: cur.__setitem__(0, cur[0] + pd.Timedelta(seconds=s)),
        now_fn=lambda: cur[0], max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        venue_fn=_venue,
    )

    assert first_stage_at and first_stage_at[0] >= pd.Timestamp("2026-08-24 23:03Z")


def test_daemon_frozen_step_integrity_error_halts_with_cause(tmp_path, monkeypatch) -> None:
    import json

    import pandas as pd

    import src.live.scheduler as sched
    from src.common.errors import DataIntegrityError
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    def _gap(target):
        raise DataIntegrityError("unit history gap")

    sched.run_daemon(
        LiveSettings(daemon_max_attempts_per_day=5), tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=_gap, prune_fn=lambda: None, venue_fn=lambda target: None,
    )

    hb = json.loads((tmp_path / "hb.json").read_text(encoding="utf-8"))
    assert hb["status"] == "HALT"
    assert hb["detail"] == "frozen_step DataIntegrityError: unit history gap"
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["pending_decision_time"] == "2026-08-24T00:00:00+00:00"


def test_daemon_venue_capture_failure_never_stops_cycle(tmp_path, monkeypatch) -> None:
    import json

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, t, w, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=0))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    def _boom(target) -> None:
        raise RuntimeError("bracket endpoint down")

    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json")), tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None, venue_fn=_boom,
    )

    assert json.loads((tmp_path / "hb.json").read_text(encoding="utf-8"))["status"] == "COMPLETE"


def test_same_day_row_stays_fresh_at_submission(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.signal import assert_signal_fresh

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, t, w, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=0))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    day = pd.Timestamp("2026-08-24 00:00Z")
    settings = LiveSettings()
    assert_signal_fresh(day, day + pd.Timedelta(hours=23, minutes=3), pd.Timedelta(hours=settings.max_signal_staleness_hours))

    ran: list[pd.Timestamp] = []
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, t, w, now=None, **k: ran.append(pd.Timestamp(t)) or CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=0))
    (tmp_path / "state.json").write_text(json.dumps({"last_processed_decision_time": "2026-08-23T00:00:00+00:00"}), encoding="utf-8")

    sched.run_daemon(
        settings, tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: day + pd.Timedelta(hours=23, minutes=20), max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None, venue_fn=lambda target: None,
    )

    assert ran == [day]


def test_digest_carries_sizing_fields(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from types import SimpleNamespace

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, t, w, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=3))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    details: list[str] = []
    monkeypatch.setattr(sched, "_daemon_alert", lambda s, *, event, detail, decision_time, now, **_k: details.append(detail) if event == "cycle_complete" else None)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    frozen = SimpleNamespace(exposure=2.5, equity_usdt=2100.0, unit_observations=300)
    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json")), tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: frozen, prune_fn=lambda: None, venue_fn=lambda target: None,
    )

    assert len(details) == 1
    assert "exposure=2.5000" in details[0]
    assert "equity_usdt=2100.00" in details[0]


def test_default_venue_capture_stores_snapshot_and_tolerates_failure(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    seen: dict = {}
    target = pd.Timestamp("2026-09-21T00:00:00Z")
    monkeypatch.setattr(sched, "VENUE_RULES_DIR", tmp_path)

    def _fake_fetch(*, api_key=None, api_secret=None, max_rejected_fraction=0.0):
        seen.update(api_key=api_key, api_secret=api_secret)
        return "snapshot"

    written: dict = {}

    def _fake_write(snapshot, root, *, slot_day=None):
        written.update(snapshot=snapshot, root=root, slot_day=slot_day)
        return tmp_path / "20260921.json.gz"

    monkeypatch.setattr("src.market_data.binance.venue_rules.fetch_venue_rules", _fake_fetch)
    monkeypatch.setattr("src.market_data.binance.venue_rules.write_venue_rule_snapshot", _fake_write)
    assert sched._default_venue_capture(LiveSettings(), target) == "captured"
    assert seen == {"api_key": None, "api_secret": None}
    assert written["snapshot"] == "snapshot"
    assert written["root"] == tmp_path
    assert pd.Timestamp(written["slot_day"]) == target

    def _boom(*, api_key=None, api_secret=None, max_rejected_fraction=0.0):
        raise RuntimeError("bracket endpoint down")

    monkeypatch.setattr("src.market_data.binance.venue_rules.fetch_venue_rules", _boom)
    assert sched._default_venue_capture(LiveSettings(), target) == "failed"


def test_default_venue_capture_fills_decision_day_slot(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings
    from src.market_data.binance.venue_rules import parse_venue_rules

    brackets = [
        {
            "symbol": "BTCUSDT",
            "brackets": [
                {
                    "bracket": 1,
                    "initialLeverage": 125,
                    "notionalCap": 50000,
                    "notionalFloor": 0,
                    "maintMarginRatio": 0.004,
                    "cum": 0,
                },
            ],
        }
    ]
    info = {"symbols": [{"symbol": "BTCUSDT", "filters": []}]}
    snapshot = parse_venue_rules(
        brackets, info, captured_at=pd.Timestamp("2026-09-23T00:10:00Z")
    )
    monkeypatch.setattr(
        "src.market_data.binance.venue_rules.fetch_venue_rules", lambda **kwargs: snapshot
    )
    monkeypatch.setattr(sched, "VENUE_RULES_DIR", tmp_path)

    assert sched._default_venue_capture(LiveSettings(), pd.Timestamp("2026-09-22T00:00:00Z")) == "captured"
    assert (tmp_path / "20260922.json.gz").exists()
    assert not (tmp_path / "20260923.json.gz").exists()


def test_default_venue_capture_skips_signed_fetch_when_slot_exists(monkeypatch, tmp_path, caplog) -> None:
    import logging

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    (tmp_path / "20260922.json.gz").write_bytes(b"sentinel")
    monkeypatch.setattr(sched, "VENUE_RULES_DIR", tmp_path)

    def _must_not_fetch(**kwargs):
        raise AssertionError("fetch must not run when the slot already exists")

    monkeypatch.setattr("src.market_data.binance.venue_rules.fetch_venue_rules", _must_not_fetch)
    with caplog.at_level(logging.INFO, logger="LiveScheduler"):
        assert sched._default_venue_capture(LiveSettings(), pd.Timestamp("2026-09-22T00:00:00Z")) == "already_captured"
    assert any("ALREADY_CAPTURED" in rec.message and "slot=20260922" in rec.message for rec in caplog.records)


def test_daemon_passes_cycle_target_to_venue_fn(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, t, w, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=0))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    received: list = []
    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json")), tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        venue_fn=lambda decision_time: received.append(pd.Timestamp(decision_time)),
    )
    assert received == [target]


def test_default_venue_capture_failure_stays_non_fatal(monkeypatch, tmp_path, caplog) -> None:
    import logging

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "VENUE_RULES_DIR", tmp_path)

    def _boom(**kwargs):
        raise RuntimeError("bracket endpoint down")

    monkeypatch.setattr("src.market_data.binance.venue_rules.fetch_venue_rules", _boom)
    with caplog.at_level(logging.WARNING, logger="LiveScheduler"):
        assert sched._default_venue_capture(LiveSettings(), pd.Timestamp("2026-09-21T00:00:00Z")) == "failed"
    assert any("status=FAILED" in rec.message for rec in caplog.records)


def test_default_frozen_step_wires_run_args_and_fails_without_non_crypto_list(monkeypatch, tmp_path) -> None:
    import pandas as pd
    import pytest

    import src.live.scheduler as sched
    from src.common.errors import DataIntegrityError
    from src.live.settings import LiveSettings

    (tmp_path / "non_crypto.json").write_text('{"captured_at": "2026-09-21T00:00:00+00:00", "symbols": ["AAPLUSDT"]}', encoding="utf-8")
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    seen: dict = {}

    def _fake_run(decision_day, **kwargs):
        seen.update(decision_day=decision_day, **kwargs)
        return "frozen-report"

    monkeypatch.setattr("src.live.frozen_signal.run_frozen_signal_step", _fake_run)
    weights = tmp_path / "state" / "deployed_target_weights.parquet"
    settings = LiveSettings()
    target = pd.Timestamp("2026-08-24 00:00Z")

    assert sched._default_frozen_step(target, settings, weights) == "frozen-report"
    assert seen["non_crypto"] == frozenset({"AAPLUSDT"})
    assert seen["seed_equity_usdt"] == settings.notional_equity_usdt
    assert seen["unit_forward_path"] == weights.parent / "frozen_unit_forward.parquet"

    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "missing.json")
    with pytest.raises(DataIntegrityError):
        sched._default_frozen_step(target, settings, weights)


def test_default_frozen_step_rejects_malformed_non_crypto_list(monkeypatch, tmp_path) -> None:
    import pandas as pd
    import pytest

    import src.live.scheduler as sched
    from src.common.errors import DataIntegrityError
    from src.live.settings import LiveSettings

    target = pd.Timestamp("2026-08-24 00:00Z")
    settings = LiveSettings()
    weights = tmp_path / "w.parquet"
    bad = tmp_path / "bad.json"
    bad.write_text('{"captured_at": "2026-09-21T00:00:00+00:00", "symbols": "BTCUSDT"}', encoding="utf-8")
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", bad)
    with pytest.raises(DataIntegrityError):
        sched._default_frozen_step(target, settings, weights)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", corrupt)
    with pytest.raises(DataIntegrityError):
        sched._default_frozen_step(target, settings, weights)


def test_sizing_note_ignores_missing_fields() -> None:
    import src.live.scheduler as sched

    assert sched._sizing_note(None) == ""
    assert sched._sizing_note(object()) == ""


def test_daemon_defaults_wire_frozen_step_and_venue(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, t, w, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp(t), intent_count=0))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    frozen_calls: list = []
    venue_calls: list = []
    monkeypatch.setattr(sched, "_default_frozen_step", lambda t, settings=None, weights_path=None: frozen_calls.append(t) or None)
    monkeypatch.setattr(sched, "_default_venue_capture", lambda settings=None, decision_time=None: venue_calls.append((settings, decision_time)) or None)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json")), tmp_path / "w.parquet", tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1, refresh_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )

    assert frozen_calls == [target]
    assert len(venue_calls) == 1
    assert venue_calls[0][1] == target


def test_prune_runs_after_execution(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    order: list[str] = []
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    def _fake_cycle(settings, decision_time, artifact_path, *, now=None, **k):
        order.append("execute")
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _fake_cycle)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    sched.run_daemon(
        LiveSettings(),
        artifact,
        tmp_path / "state.json",
        sleep_fn=lambda s: None,
        now_fn=lambda: ready,
        max_iterations=1,
        refresh_fn=lambda *a, **k: order.append("refresh"),
        signal_step_fn=lambda t: order.append("signal"),
        prune_fn=lambda: order.append("prune"),
        venue_fn=lambda target: None,
    )
    assert order == ["refresh", "signal", "execute", "prune"]


def test_prune_failure_is_isolated(tmp_path, monkeypatch) -> None:
    import json
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda *a, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp("2026-08-24 00:00Z"), intent_count=0),
    )
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"

    def _boom() -> None:
        raise RuntimeError("disk busy")

    sched.run_daemon(
        LiveSettings(),
        artifact,
        state_path,
        sleep_fn=lambda s: None,
        now_fn=lambda: ready,
        max_iterations=1,
        refresh_fn=lambda *a, **k: None,
        signal_step_fn=lambda *a, **k: None,
        prune_fn=_boom,
        venue_fn=lambda target: None,
    )
    hb = json.loads((tmp_path / "hb.json").read_text(encoding="utf-8"))
    assert hb["status"] == "COMPLETE"
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(saved["last_processed_decision_time"]) == target


def test_live_mode_fetches_account_equity(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import ExecutionMode, LiveSettings

    captured: dict[str, object] = {}
    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    (tmp_path / "non_crypto.json").write_text('{"symbols": []}', encoding="utf-8")
    def _fake_fetch(settings, now):
        captured["called"] = True
        return 3000.0

    monkeypatch.setattr("src.live.runner.fetch_live_account_equity", _fake_fetch)

    def _fake_step(target, **kwargs):
        captured.update(kwargs)
        return type("R", (), {"status": "COMPLETE"})()

    monkeypatch.setattr("src.live.frozen_signal.run_frozen_signal_step", _fake_step)
    settings = LiveSettings(mode=ExecutionMode.LIVE_TESTNET, order_api_key="k", order_api_secret="s", heartbeat_path=str(tmp_path / "hb.json"))
    sched._default_frozen_step(pd.Timestamp("2026-08-24 00:00Z"), settings, tmp_path / "w.parquet")
    assert captured.get("account_equity_usdt") == 3000.0
    assert captured.get("called") is True


def test_paper_mode_never_fetches_account_equity(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    (tmp_path / "non_crypto.json").write_text('{"symbols": []}', encoding="utf-8")
    called: list[bool] = []
    monkeypatch.setattr("src.live.runner.fetch_live_account_equity", lambda *a, **k: called.append(True) or 3000.0)
    seen: dict[str, object] = {}

    def _fake_step(target, **kwargs):
        seen.update(kwargs)
        return type("R", (), {"status": "COMPLETE"})()

    monkeypatch.setattr("src.live.frozen_signal.run_frozen_signal_step", _fake_step)
    sched._default_frozen_step(pd.Timestamp("2026-08-24 00:00Z"), LiveSettings(), tmp_path / "w.parquet")
    assert called == []
    assert seen.get("account_equity_usdt") is None


def test_fetch_failure_halts_step(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    import src.live.scheduler as sched
    from src.common.errors import DataIntegrityError
    from src.live.settings import ExecutionMode, LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    (tmp_path / "non_crypto.json").write_text('{"symbols": []}', encoding="utf-8")

    def _boom(settings, now):
        raise DataIntegrityError("venue down")

    monkeypatch.setattr("src.live.runner.fetch_live_account_equity", _boom)
    monkeypatch.setattr("src.live.frozen_signal.run_frozen_signal_step", lambda *a, **k: pytest.fail("must not reach signal step"))
    settings = LiveSettings(mode=ExecutionMode.LIVE_TESTNET, order_api_key="k", order_api_secret="s", heartbeat_path=str(tmp_path / "hb.json"))
    with pytest.raises(DataIntegrityError):
        sched._default_frozen_step(pd.Timestamp("2026-08-24 00:00Z"), settings, tmp_path / "w.parquet")


def _fast_pulse(monkeypatch, interval_s: float = 0.02) -> None:
    import src.live.scheduler as sched

    _orig = sched._run_heartbeat_pulse

    def _wrapper(stop, **kwargs):
        return _orig(stop, interval_s=interval_s, **kwargs)

    monkeypatch.setattr(sched, "_run_heartbeat_pulse", _wrapper)


def _daemon_ready(target):
    import pandas as pd

    import src.live.scheduler as sched

    return target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)


def test_heartbeat_pulse_writes_fresh_during_slow_execute(tmp_path, monkeypatch) -> None:
    import json
    import time

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    _fast_pulse(monkeypatch, interval_s=0.02)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = _daemon_ready(target)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    records: list[dict] = []
    _orig_write = sched.write_heartbeat

    def _recording(path, *, decision_time, status, attempts, consecutive_halts, now, stage="idle", detail="", expected_by=None, **_k):
        records.append({"status": status, "stage": stage, "ts": str(now)})
        return _orig_write(path, decision_time=decision_time, status=status, attempts=attempts, consecutive_halts=consecutive_halts, now=now, stage=stage, detail=detail)

    monkeypatch.setattr(sched, "write_heartbeat", _recording)

    def _slow_cycle(settings, decision_time, artifact_path, *, now=None, **k):
        time.sleep(0.09)
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _slow_cycle)
    sched.run_daemon(
        LiveSettings(), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )
    pulses = [r for r in records if r["status"] == "RUNNING" and r["stage"] == "execute"]
    assert len(pulses) >= 2
    assert records[-1]["status"] == "COMPLETE"
    payload = json.loads(hb_path.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"


def test_heartbeat_pulse_never_overwrites_final(tmp_path, monkeypatch) -> None:
    import json
    import time

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    _fast_pulse(monkeypatch, interval_s=0.02)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = _daemon_ready(target)
    artifact = tmp_path / "w.parquet"
    artifact.touch()

    def _slow_cycle(settings, decision_time, artifact_path, *, now=None, **k):
        time.sleep(0.07)
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _slow_cycle)
    sched.run_daemon(
        LiveSettings(), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )
    first = hb_path.read_text(encoding="utf-8")
    time.sleep(0.08)
    second = hb_path.read_text(encoding="utf-8")
    third = hb_path.read_text(encoding="utf-8")
    assert first == second == third
    assert json.loads(first)["status"] == "COMPLETE"


def test_heartbeat_pulse_survives_crashing_cycle(tmp_path, monkeypatch) -> None:
    import json
    import threading
    import time

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    _fast_pulse(monkeypatch, interval_s=0.02)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = _daemon_ready(target)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    records: list[tuple[str, str]] = []
    _orig_write = sched.write_heartbeat

    def _recording(path, *, decision_time, status, attempts, consecutive_halts, now, stage="idle", detail="", expected_by=None, **_k):
        records.append((status, stage))
        return _orig_write(path, decision_time=decision_time, status=status, attempts=attempts, consecutive_halts=consecutive_halts, now=now, stage=stage, detail=detail)

    monkeypatch.setattr(sched, "write_heartbeat", _recording)

    def _crash(settings, decision_time, artifact_path, *, now=None, **k):
        time.sleep(0.06)
        raise RuntimeError("boom")

    monkeypatch.setattr(sched, "run_shadow_cycle", _crash)
    sched.run_daemon(
        LiveSettings(daemon_max_attempts_per_day=1), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )
    assert ("RUNNING", "execute") in records
    assert json.loads(hb_path.read_text(encoding="utf-8"))["status"] == "HALT"
    assert not [t for t in threading.enumerate() if t.name == "live-heartbeat-pulse" and t.is_alive()]


def test_heartbeat_pulse_write_failure_does_not_affect_cycle(tmp_path, monkeypatch) -> None:
    import json
    import time

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    _fast_pulse(monkeypatch, interval_s=0.02)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = _daemon_ready(target)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    _orig_write = sched.write_heartbeat
    calls = {"n": 0}

    def _flaky(path, *, decision_time, status, attempts, consecutive_halts, now, stage="idle", detail="", expected_by=None, **_k):
        import threading as _th
        if status == "RUNNING" and stage == "execute" and _th.current_thread().name == "live-heartbeat-pulse" and calls["n"] == 0:
            calls["n"] += 1
            raise OSError("disk full")
        return _orig_write(path, decision_time=decision_time, status=status, attempts=attempts, consecutive_halts=consecutive_halts, now=now, stage=stage, detail=detail)

    monkeypatch.setattr(sched, "write_heartbeat", _flaky)

    def _slow_cycle(settings, decision_time, artifact_path, *, now=None, **k):
        time.sleep(0.07)
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _slow_cycle)
    sched.run_daemon(
        LiveSettings(), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )
    assert json.loads(hb_path.read_text(encoding="utf-8"))["status"] == "COMPLETE"


def test_heartbeat_pulse_no_lingering_thread_after_fast_cycle(tmp_path, monkeypatch) -> None:
    import threading

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = _daemon_ready(target)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0),
    )
    sched.run_daemon(
        LiveSettings(), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )
    assert not [t for t in threading.enumerate() if t.name == "live-heartbeat-pulse" and t.is_alive()]


def test_heartbeat_pulse_uses_real_wall_clock(tmp_path, monkeypatch) -> None:
    import threading
    import time

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    _fast_pulse(monkeypatch, interval_s=0.02)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = _daemon_ready(target)
    main_thread = threading.get_ident()

    def _main_only_now():
        assert threading.get_ident() == main_thread, "now_fn called off main thread"
        return ready

    artifact = tmp_path / "w.parquet"
    artifact.touch()

    def _slow_cycle(settings, decision_time, artifact_path, *, now=None, **k):
        time.sleep(0.07)
        return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)

    monkeypatch.setattr(sched, "run_shadow_cycle", _slow_cycle)
    sched.run_daemon(
        LiveSettings(), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=_main_only_now, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )


def test_heartbeat_pulse_join_timeout_warns(tmp_path, monkeypatch, caplog) -> None:
    import logging
    import time

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    monkeypatch.setattr(sched, "_HEARTBEAT_PULSE_JOIN_TIMEOUT_S", 0.01)

    def _stuck_pulse(stop, **kwargs):
        time.sleep(0.2)

    monkeypatch.setattr(sched, "_run_heartbeat_pulse", _stuck_pulse)
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = _daemon_ready(target)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact_path, *, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0),
    )
    with caplog.at_level(logging.WARNING, logger="LiveScheduler"):
        sched.run_daemon(
            LiveSettings(), artifact, tmp_path / "state.json",
            sleep_fn=lambda s: None, now_fn=lambda: ready, max_iterations=1,
            refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
        )
    assert any("heartbeat pulse thread still alive" in r.getMessage() for r in caplog.records)


def test_day_skipped_survives_delivery_outage(tmp_path, monkeypatch) -> None:
    """전송 장애 시 day_skipped는 유실되지 않고 채널 복구 후 1회 배달된다."""
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    delivered: list[str] = []
    state = {"up": False}
    monkeypatch.setattr(
        "src.live.alerting.post_alert",
        lambda url, *, event, detail, decision_time, now, **_k: delivered.append(event) or state["up"],
    )
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda settings, target, wp, now=None, **k: CycleReport(status="HALT", reason="x", decision_time=target, intent_count=0))
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    base = pd.Timestamp("2026-08-24 00:00Z")
    cur = [base + pd.Timedelta(days=5) + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)]

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    settings = LiveSettings(alert_webhook_url="https://h.example", daemon_max_attempts_per_day=1, alert_outbox_path=str(tmp_path / "outbox.json"))
    sched.run_daemon(settings, artifact, tmp_path / "state.json", sleep_fn=sleep_fn, now_fn=now_fn, max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None)
    assert "day_skipped" in delivered
    state["up"] = True
    from src.live.alerting import drain_alerts

    report = drain_alerts(settings, now=cur[0] + pd.Timedelta(seconds=3600), blocking=True)
    assert report.pending == 0
    assert delivered.count("day_skipped") == 2


def test_halt_streak_alerts_once_per_streak_length_across_restart(tmp_path, monkeypatch) -> None:
    """연속 HALT 3회는 streak 2·3 알림만 남기고 재시작 후에도 중복하지 않는다."""
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    delivered: list[str] = []
    monkeypatch.setattr(
        "src.live.alerting.post_alert",
        lambda url, *, event, detail, decision_time, now, **_k: delivered.append(detail) or True,
    )
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda settings, target, wp, now=None, **k: CycleReport(status="HALT", reason="x", decision_time=target, intent_count=0))
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    base = pd.Timestamp("2026-08-24 00:00Z")
    cur = [base + pd.Timedelta(days=5) + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)]

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    settings = LiveSettings(alert_webhook_url="https://h.example", alert_halt_streak=2, daemon_max_attempts_per_day=5, alert_outbox_path=str(tmp_path / "outbox.json"))

    def _run_once(max_iterations):
        sched.run_daemon(settings, artifact, tmp_path / "state.json", sleep_fn=sleep_fn, now_fn=now_fn, max_iterations=max_iterations, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None)

    _run_once(2)
    _run_once(1)
    streaks = sorted({d.split("consecutive_halts=")[1].split()[0].rstrip(",") for d in delivered if d.startswith("consecutive_halts=")})
    assert streaks == ["2", "3"]


def test_heartbeat_carries_expected_by_for_every_stage(tmp_path, monkeypatch) -> None:
    """refresh·signal·execute·idle 하트비트는 모두 ts보다 늦은 expected_by를 가진다."""
    import json
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: True)
    writes: list[dict] = []
    original_write = sched.write_heartbeat

    def _capture(path, **kwargs):
        original_write(path, **kwargs)
        writes.append({"stage": kwargs.get("stage"), "ts": kwargs.get("now"), "expected_by": kwargs.get("expected_by")})

    monkeypatch.setattr(sched, "write_heartbeat", _capture)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    base = pd.Timestamp("2026-08-24 00:00Z")
    cur = [base + pd.Timedelta(days=5) + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)]

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda settings, target, wp, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=target, intent_count=0))
    sched.run_daemon(
        LiveSettings(alert_outbox_path=str(tmp_path / "outbox.json")), artifact, tmp_path / "state.json",
        sleep_fn=sleep_fn, now_fn=now_fn, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )
    stages = {w["stage"] for w in writes}
    assert {"refresh", "signal", "execute", "idle"} <= stages
    for w in writes:
        assert w["expected_by"] is not None
        assert pd.Timestamp(w["expected_by"]) > pd.Timestamp(w["ts"])


def test_pulse_covers_long_refresh(tmp_path, monkeypatch) -> None:
    """차단형 refresh 동안 펄스가 ts를 갱신하고 stage를 refresh로 유지한다."""
    import json
    import threading
    import time
    import pandas as pd
    import src.live.scheduler as sched

    hb_path = tmp_path / "hb.json"
    decision = pd.Timestamp("2026-08-24 00:00Z")
    sched.write_heartbeat(hb_path, decision_time=decision, status="RUNNING", attempts=0, consecutive_halts=0, now=decision, stage="refresh")
    before = json.loads(hb_path.read_text(encoding="utf-8"))["ts"]
    stop = threading.Event()
    thread = threading.Thread(
        target=sched._run_heartbeat_pulse,
        kwargs={"stop": stop, "heartbeat_path": hb_path, "decision_time": decision, "attempts": 0, "consecutive_halts": 0, "interval_s": 0.02, "stage": "refresh"},
        daemon=True,
    )
    thread.start()
    time.sleep(0.08)
    stop.set()
    thread.join(timeout=5)
    after = json.loads(hb_path.read_text(encoding="utf-8"))
    assert after["stage"] == "refresh"
    assert pd.Timestamp(after["ts"]) > pd.Timestamp(before)


def test_daemon_ping_and_drain_failures_never_stop_cycle(tmp_path, monkeypatch) -> None:
    """deadman 핑·아웃박스 drain 예외는 사이클을 멈추지 않는다."""
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: True)

    def _boom_ping(self, *, now, failing):
        raise RuntimeError("deadman down")

    monkeypatch.setattr(sched.DeadmanPinger, "maybe_ping", _boom_ping)

    def _boom_drain(*a, **k):
        raise RuntimeError("outbox lock down")

    monkeypatch.setattr(sched, "drain_alerts", _boom_drain)
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda settings, target, wp, now=None, **k: CycleReport(status="COMPLETE", reason=None, decision_time=target, intent_count=0))
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    base = pd.Timestamp("2026-08-24 00:00Z")
    cur = [base + pd.Timedelta(days=5) + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)]

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    sched.run_daemon(
        LiveSettings(alert_outbox_path=str(tmp_path / "outbox.json"), deadman_ping_url="https://hc.example.com/ping"),
        artifact, tmp_path / "state.json", sleep_fn=sleep_fn, now_fn=now_fn, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )
    assert (tmp_path / "hb.json").exists()


def test_state_corrupt_dedupe_key_uses_mtime_or_date_fallback(tmp_path) -> None:
    """state_corrupt 키는 mtime 기반이며 stat 실패 시 날짜 폴백을 사용한다."""
    import pandas as pd
    import src.live.scheduler as sched

    state_path = tmp_path / "state.json"
    state_path.write_text("{bad", encoding="utf-8")
    now = pd.Timestamp("2026-08-24 05:00Z")
    assert sched._state_corrupt_dedupe_key(state_path, now) == f"state_corrupt:state.json:{state_path.stat().st_mtime_ns}"
    assert sched._state_corrupt_dedupe_key(tmp_path / "missing.json", now) == "state_corrupt:missing.json:2026-08-24"


def test_wait_drain_failure_never_blocks_release_wait(tmp_path, monkeypatch) -> None:
    """대기 중 drain 예외는 릴리스 대기를 막지 않는다."""
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings
    from src.live.scheduler import DECISION_RELEASE_OFFSET

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    monkeypatch.setattr(sched, "_daemon_alert", lambda *a, **k: True)

    def _boom_drain(*a, **k):
        raise RuntimeError("drain down")

    monkeypatch.setattr(sched, "drain_alerts", _boom_drain)
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: None)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    base = pd.Timestamp("2026-08-24 00:00Z")
    cur = [base + pd.Timedelta(minutes=10)]

    waited: list[float] = []

    class _StopLoop(Exception):
        pass

    def sleep_fn(s):
        waited.append(s)
        cur[0] += pd.Timedelta(seconds=s)
        if len(waited) >= 2:
            raise _StopLoop

    try:
        sched.run_daemon(
            LiveSettings(alert_outbox_path=str(tmp_path / "outbox.json")), artifact, tmp_path / "state.json",
            sleep_fn=sleep_fn, now_fn=lambda: cur[0], max_iterations=1,
            refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
        )
    except _StopLoop:
        pass
    assert len(waited) >= 2


def test_interrupted_report_keeps_day_pending(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """INTERRUPTED report keeps the day pending: daemon state has pending == target with unchanged attempts, heartbeat stage execute/INTERRUPTED, no cycle_complete alert, consecutive halts unchanged."""
    from src.live.scheduler import DaemonState, _load_daemon_state, _read_heartbeat

    heartbeat_path = tmp_path / "heartbeat.json"
    state_path = tmp_path / "state.json"
    artifact_path = tmp_path / "w.parquet"
    artifact_path.touch()
    settings = LiveSettings(heartbeat_path=str(heartbeat_path))
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        scheduler_mod, "_daemon_alert",
        lambda s, *, event, detail, decision_time, now, dedupe_key=None: alerts.append((event, detail)) or False,
    )
    monkeypatch.setattr(
        scheduler_mod, "run_shadow_cycle",
        lambda *a, **k: CycleReport(status="INTERRUPTED", reason="shutdown_requested", decision_time=DECISION_TIME, intent_count=1),
    )
    monkeypatch.setattr(scheduler_mod, "prune_old_audit_logs", lambda *_a: 0)

    run_daemon(
        settings, artifact_path, state_path,
        sleep_fn=lambda s: pytest.fail("must not sleep"),
        now_fn=lambda: READY_NOW, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None,
        prune_fn=lambda: None, venue_fn=lambda t: None,
    )

    saved = _load_daemon_state(state_path)
    assert saved.pending_decision_time == DECISION_TIME
    assert saved.attempts == 0
    assert saved.last_processed_decision_time is None
    beat = _read_heartbeat(heartbeat_path)
    assert beat is not None
    assert beat["stage"] == "execute"
    assert beat["status"] == "INTERRUPTED"
    assert beat["attempts"] == 0
    assert beat["consecutive_halts"] == 0
    assert all(event != "cycle_complete" for event, _ in alerts)


def test_restart_after_interruption_alerts_and_resumes_same_day(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Restart after interruption alerts cycle_interrupted once and the next cycle targets the pending decision time."""
    from src.live.scheduler import _read_heartbeat, _save_daemon_state, DaemonState, write_heartbeat

    heartbeat_path = tmp_path / "heartbeat.json"
    state_path = tmp_path / "state.json"
    artifact_path = tmp_path / "w.parquet"
    artifact_path.touch()
    settings = LiveSettings(heartbeat_path=str(heartbeat_path))
    _save_daemon_state(state_path, DaemonState(last_processed_decision_time=None, pending_decision_time=DECISION_TIME, attempts=0))
    write_heartbeat(
        heartbeat_path, decision_time=DECISION_TIME, status="RUNNING", attempts=0,
        consecutive_halts=0, now=READY_NOW - pd.Timedelta(minutes=5), stage="execute",
    )
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        scheduler_mod, "_daemon_alert",
        lambda s, *, event, detail, decision_time, now, dedupe_key=None: alerts.append((event, detail)) or False,
    )
    cycles: list[pd.Timestamp] = []
    monkeypatch.setattr(
        scheduler_mod, "run_shadow_cycle",
        lambda *a, **k: cycles.append(k.get("now", None) and a[1] or a[1]) or CycleReport(status="COMPLETE", reason=None, decision_time=a[1], intent_count=0),
    )
    monkeypatch.setattr(scheduler_mod, "prune_old_audit_logs", lambda *_a: 0)

    run_daemon(
        settings, artifact_path, state_path,
        sleep_fn=lambda s: pytest.fail("must not sleep"),
        now_fn=lambda: READY_NOW, max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None,
        prune_fn=lambda: None, venue_fn=lambda t: None,
    )

    interrupted = [a for a in alerts if a[0] == "cycle_interrupted"]
    assert len(interrupted) == 1
    assert cycles and cycles[0] == DECISION_TIME


def test_run_daemon_degraded_processes_day_and_alerts_once(tmp_path, monkeypatch) -> None:
    """DEGRADED processes the day and alerts."""
    import json

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)
    beats: list[tuple[str, str]] = []

    def _record(path, *, decision_time, status, attempts, consecutive_halts, now, stage="idle", detail="", expected_by=None, **_k):
        beats.append((status, stage))

    monkeypatch.setattr(sched, "write_heartbeat", _record)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda *a, **k: CycleReport(status="DEGRADED", reason="reconciliation_breach", decision_time=target, intent_count=1),
    )

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=lambda s: None, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=lambda: None,
    )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(saved["last_processed_decision_time"]) == target
    assert saved["pending_decision_time"] is None
    assert saved["attempts"] == 0
    assert ("DEGRADED", "idle") in beats
    degraded = [detail for event, detail in alerts if event == "cycle_degraded"]
    assert len(degraded) == 1
    assert not [event for event, _ in alerts if event == "cycle_complete"]


def test_run_daemon_degraded_survives_heartbeat_and_prune_failures(tmp_path, monkeypatch) -> None:
    """DEGRADED still processes the day when heartbeat and prune writes fail."""
    import json

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    hb_path = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb_path)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda settings, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)),
    )
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    state_path = tmp_path / "state.json"
    target = pd.Timestamp("2026-08-24 00:00Z")
    ready = target + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)

    def _boom_write(*a, **k):
        raise OSError("disk full")

    def _boom_prune():
        raise RuntimeError("prune down")

    monkeypatch.setattr(sched, "write_heartbeat", _boom_write)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda *a, **k: CycleReport(status="DEGRADED", reason="free_margin_floor", decision_time=target, intent_count=1),
    )

    sched.run_daemon(
        LiveSettings(), artifact, state_path, sleep_fn=lambda s: None, now_fn=lambda: ready,
        max_iterations=1, refresh_fn=lambda *a, **k: None, signal_step_fn=lambda t: None, prune_fn=_boom_prune,
        venue_fn=lambda t: "captured",
    )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(saved["last_processed_decision_time"]) == target
    assert [event for event, _ in alerts] == ["cycle_degraded"]


def test_required_symbols_unions_ledger_and_weights(tmp_path) -> None:
    import json

    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps({"positions": {"HELDUSDT": "1.5", "FLATUSDT": "0"}, "cash_usdt": "1000"}),
        encoding="utf-8",
    )
    weights_path = tmp_path / "deployed_target_weights.parquet"
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-24 00:00Z"), pd.Timestamp("2026-08-25 00:00Z")])
    pd.DataFrame({"BOOKUSDT": [0.0, 0.02], "ZEROUSDT": [0.0, 0.0]}, index=idx).to_parquet(weights_path, index=True)
    settings = LiveSettings(ledger_path=str(ledger_path))

    assert sched._required_symbols(settings, weights_path) == {"HELDUSDT", "BOOKUSDT"}


def test_required_symbols_missing_files_contribute_nothing(tmp_path) -> None:
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    settings = LiveSettings(ledger_path=str(tmp_path / "no-ledger.json"))

    assert sched._required_symbols(settings, tmp_path / "no-weights.parquet") == set()


def test_required_symbols_corrupt_ledger_fails_closed(tmp_path) -> None:
    import pytest

    import src.live.scheduler as sched
    from src.common.errors import DataIntegrityError
    from src.live.settings import LiveSettings

    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text("{not json", encoding="utf-8")
    settings = LiveSettings(ledger_path=str(ledger_path))

    with pytest.raises(DataIntegrityError):
        sched._required_symbols(settings, tmp_path / "no-weights.parquet")


def test_required_symbols_skips_non_numeric_weight_columns(tmp_path) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    weights_path = tmp_path / "deployed_target_weights.parquet"
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-25 00:00Z")])
    frame = pd.DataFrame({"BOOKUSDT": [0.02], "NOTE": ["held"]}, index=idx)
    frame.to_parquet(weights_path, index=True)
    settings = LiveSettings(ledger_path=str(tmp_path / "no-ledger.json"))

    assert sched._required_symbols(settings, weights_path) == {"BOOKUSDT"}


def test_run_daemon_binds_default_wiring_without_network(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.live.scheduler as sched
    from src.live.lifecycle import ShutdownFlag
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    flag = ShutdownFlag()
    flag.request("SIGTERM")
    sched.run_daemon(
        LiveSettings(),
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=lambda s: None,
        now_fn=lambda: pd.Timestamp("2026-08-24 01:00Z"),
        max_iterations=1,
        shutdown=flag,
        refresh_fn=None,
        signal_step_fn=None,
        prune_fn=lambda: None,
    )


def test_prefetch_runs_once_per_target_before_refresh(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    target = pd.Timestamp("2026-08-24 00:00Z")
    prefetch_at = target + pd.Timedelta(hours=20.25)
    release = target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=3)
    assert prefetch_at < release
    cur = [prefetch_at - pd.Timedelta(minutes=10)]
    order: list[str] = []

    def now_fn():
        return cur[0]

    def sleep_fn(s):
        cur[0] += pd.Timedelta(seconds=s)

    def prefetch_fn(t):
        order.append(f"prefetch:{pd.Timestamp(t).isoformat()}")
        return None

    def refresh_fn(t):
        order.append("refresh")
        return None

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: CycleReport(status="COMPLETE", reason=None, decision_time=target, intent_count=0))
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json")), artifact, tmp_path / "state.json",
        sleep_fn=sleep_fn, now_fn=now_fn, max_iterations=1,
        refresh_fn=refresh_fn, signal_step_fn=lambda t: order.append("signal"),
        prune_fn=lambda: None, venue_fn=lambda t: "captured", prefetch_fn=prefetch_fn,
    )
    assert order.count(f"prefetch:{target.isoformat()}") == 1
    assert order.index(f"prefetch:{target.isoformat()}") < order.index("refresh")


def test_prefetch_failure_never_blocks_cycle(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    target = pd.Timestamp("2026-08-24 00:00Z")
    cur = [target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)]
    ran: list[str] = []

    def boom(t):
        raise RuntimeError("prefetch down")

    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: CycleReport(status="COMPLETE", reason=None, decision_time=target, intent_count=0))
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json")), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: cur[0], max_iterations=1,
        refresh_fn=lambda t: ran.append("refresh"),
        signal_step_fn=lambda t: ran.append("signal"),
        prune_fn=lambda: None, venue_fn=lambda t: "captured", prefetch_fn=boom,
    )
    assert ran == ["refresh", "signal"]


def test_required_incomplete_alerts_with_symbols(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.data_refresh import RefreshReport
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(sched, "_daemon_alert", lambda s, *, event, detail, decision_time, now, **_k: alerts.append((event, detail)))
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp("2026-08-24 00:00Z"), intent_count=0))
    rep = RefreshReport(total=200, fresh=199, refreshed=0, failed=0, deadline_skipped=0, incomplete=1, required_incomplete=("XUSDT",), not_current_sample=("XUSDT",), elapsed_s=1.0, deadline_hit=False, staleness_hours=1.0, ok=False)
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json")), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-08-24 00:00Z") + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20),
        max_iterations=1, refresh_fn=lambda t: rep,
        signal_step_fn=lambda t: None, prune_fn=lambda: None, venue_fn=lambda t: "captured",
        prefetch_fn=lambda t: None,
    )
    assert any(ev == "refresh_incomplete" and "XUSDT" in detail for ev, detail in alerts)


def test_venue_capture_failure_alerts_once(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    alerts: list[str] = []
    monkeypatch.setattr(sched, "_daemon_alert", lambda s, *, event, detail, decision_time, now, **_k: alerts.append(event))
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp("2026-08-24 00:00Z"), intent_count=0))
    target = pd.Timestamp("2026-08-24 00:00Z")
    cur = [target + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20)]
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json"), daemon_max_attempts_per_day=1), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: cur[0], max_iterations=1,
        refresh_fn=lambda t: None, signal_step_fn=lambda t: (_ for _ in ()).throw(RuntimeError("halt")),
        prune_fn=lambda: None, venue_fn=lambda t: "failed", prefetch_fn=lambda t: None,
    )
    assert alerts.count("venue_capture_failed") == 1


def test_stale_ladder_warns_before_halting(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.runner import CycleReport
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "_strategy_params_present", lambda settings: True, raising=False)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    alerts: list[str] = []
    monkeypatch.setattr(sched, "_daemon_alert", lambda s, *, event, detail, decision_time, now, **_k: alerts.append(event))
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda *a, **k: CycleReport(status="COMPLETE", reason=None, decision_time=pd.Timestamp("2026-08-24 00:00Z"), intent_count=0))
    frozen = type("R", (), {"venue_snapshot_age_days": 3.0, "venue_snapshot": "20260921.json.gz", "decision_bar_missing": 0, "venue_gap_excluded": (), "exposure": 0.0, "equity_usdt": 1.0, "unit_observations": 1})()
    artifact = tmp_path / "w.parquet"
    artifact.touch()
    sched.run_daemon(
        LiveSettings(heartbeat_path=str(tmp_path / "hb.json"), venue_rules_warn_age_days=2.0), artifact, tmp_path / "state.json",
        sleep_fn=lambda s: None, now_fn=lambda: pd.Timestamp("2026-08-24 00:00Z") + sched.DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=20),
        max_iterations=1, refresh_fn=lambda t: None,
        signal_step_fn=lambda t: frozen, prune_fn=lambda: None, venue_fn=lambda t: "captured",
        prefetch_fn=lambda t: None,
    )
    assert "venue_rules_stale" in alerts


def test_default_funding_prefetch_returns_none_without_listing(monkeypatch, tmp_path) -> None:
    import src.live.scheduler as sched
    from src.live.settings import LiveSettings

    monkeypatch.setattr(sched, "DATA_DIR", tmp_path / "data")
    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"))
    assert sched._default_funding_prefetch(settings, tmp_path / "w.parquet", sched.DECISION_RELEASE_OFFSET) is None


def test_default_funding_prefetch_uses_universe_and_reports(monkeypatch, tmp_path) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.data_refresh import FundingPrefetchReport
    from src.live.settings import LiveSettings
    from src.live.venue_listing import VenueListingEntry, VenueListingSnapshot

    listing_root = tmp_path / "data" / "state" / "venue_listing"
    listing_root.mkdir(parents=True)
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    entries = {
        "BTCUSDT": VenueListingEntry(symbol="BTCUSDT", status="TRADING", contract_type="PERPETUAL", underlying_type="COIN", quote_asset="USDT", delivery_time=None, announced_delisting=False, delisting_first_seen_at=None),
    }
    from src.live.venue_listing import write_venue_listing_snapshot

    write_venue_listing_snapshot(VenueListingSnapshot(captured_at=now, entries=entries), listing_root, slot_day=now)
    monkeypatch.setattr(sched, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    (tmp_path / "non_crypto.json").write_text('{"captured_at": "2026-09-01T00:00:00+00:00", "symbols": []}', encoding="utf-8")
    (tmp_path / "ledger.json").write_text('{"positions": {}, "cash_usdt": "1000"}', encoding="utf-8")
    seen: dict = {}

    def _fake_prefetch(*a, **k):
        seen.update(k)
        return FundingPrefetchReport(total=1, fetched=1, fresh=0, failed=0, deadline_skipped=0, funding_blocked=False, elapsed_s=0.5)

    monkeypatch.setattr("src.live.data_refresh.refresh_funding_tails", _fake_prefetch)
    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"))
    report = sched._default_funding_prefetch(settings, tmp_path / "w.parquet", pd.Timestamp("2026-09-01T00:00:00Z"))
    assert report.fetched == 1
    assert seen["symbols"] == ["BTCUSDT"]


def test_default_funding_prefetch_tolerates_corrupt_non_crypto_file(monkeypatch, tmp_path) -> None:
    import pandas as pd
    import src.live.scheduler as sched
    from src.live.data_refresh import FundingPrefetchReport
    from src.live.settings import LiveSettings
    from src.live.venue_listing import VenueListingEntry, VenueListingSnapshot, write_venue_listing_snapshot

    listing_root = tmp_path / "data" / "state" / "venue_listing"
    listing_root.mkdir(parents=True)
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    write_venue_listing_snapshot(
        VenueListingSnapshot(
            captured_at=now,
            entries={"BTCUSDT": VenueListingEntry(symbol="BTCUSDT", status="TRADING", contract_type="PERPETUAL", underlying_type="COIN", quote_asset="USDT", delivery_time=None, announced_delisting=False, delisting_first_seen_at=None)},
        ),
        listing_root, slot_day=now,
    )
    monkeypatch.setattr(sched, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    (tmp_path / "non_crypto.json").write_bytes(b"{corrupt")
    (tmp_path / "ledger.json").write_text('{"positions": {}, "cash_usdt": "1000"}', encoding="utf-8")
    monkeypatch.setattr(
        "src.live.data_refresh.refresh_funding_tails",
        lambda *a, **k: FundingPrefetchReport(total=1, fetched=0, fresh=1, failed=0, deadline_skipped=0, funding_blocked=False, elapsed_s=0.1),
    )
    report = sched._default_funding_prefetch(LiveSettings(ledger_path=str(tmp_path / "ledger.json")), tmp_path / "w.parquet", now)
    assert report.fresh == 1


def _daemon_clock(start: pd.Timestamp):
    cur = [start]

    def now_fn() -> pd.Timestamp:
        return cur[0]

    return cur, now_fn


def _expected_by(hb_path: Path) -> pd.Timestamp:
    return pd.Timestamp(json.loads(hb_path.read_text(encoding="utf-8"))["expected_by"])


def test_idle_wait_keeps_expected_by_in_future(tmp_path, monkeypatch) -> None:
    """사이클 완료 후 하루 대기 동안 heartbeat expected_by는 항상 현재 시각보다 뒤여야 한다."""
    import src.live.scheduler as sched

    hb = tmp_path / "hb.json"
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb)
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda settings, decision_time, artifact, now=None, **k: _report(decision_time))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    cur, now_fn = _daemon_clock(READY_NOW)
    lags: list[float] = []

    def sleep_fn(seconds: float) -> None:
        if hb.exists():
            lags.append((_expected_by(hb) - cur[0]).total_seconds())
        cur[0] += pd.Timedelta(seconds=seconds)

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0),
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=2,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )
    assert len(lags) > 200  # 약 하루 대기
    assert min(lags) > 0


def test_halt_backoff_keeps_expected_by_and_pings_fail(tmp_path, monkeypatch) -> None:
    """HALT 재시도 backoff 동안 expected_by가 만료되지 않고, 대기 중 핑은 /fail(failing=True)이다."""
    import src.live.scheduler as sched

    hb = tmp_path / "hb.json"
    pings: list[bool] = []
    in_wait = [False]

    class _Pinger:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def maybe_ping(self, *, now: pd.Timestamp, failing: bool) -> bool:
            # 대기(_wait) 직후의 핑만 수집: 단계 진입 RUNNING 핑은 정상 상태이므로 제외
            if in_wait[0]:
                pings.append(failing)
                in_wait[0] = False
            return True

    monkeypatch.setattr(sched, "DeadmanPinger", _Pinger)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb)
    monkeypatch.setattr(
        sched, "run_shadow_cycle",
        lambda settings, decision_time, artifact, now=None, **k: CycleReport(status="HALT", reason="halt", decision_time=decision_time, intent_count=0),
    )
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    cur, now_fn = _daemon_clock(READY_NOW)
    lags: list[float] = []

    def sleep_fn(seconds: float) -> None:
        if hb.exists():
            lags.append((_expected_by(hb) - cur[0]).total_seconds())
        cur[0] += pd.Timedelta(seconds=seconds)
        in_wait[0] = True

    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0, daemon_max_attempts_per_day=5),
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=4,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
        venue_fn=lambda t: None,
    )
    assert lags and min(lags) > 0
    assert pings and all(pings)


def test_stage_pulse_receives_deadman_pinger(tmp_path, monkeypatch) -> None:
    """장시간 단계의 heartbeat pulse에는 dead-man pinger가 전달되어야 한다."""
    import src.live.scheduler as sched

    seen: list[Any] = []

    class _Pinger:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def maybe_ping(self, *, now: pd.Timestamp, failing: bool) -> bool:
            return True

    def _spy_pulse(stop, **kwargs: Any) -> None:
        seen.append((kwargs["stage"], kwargs["pinger"]))
        stop.wait()

    monkeypatch.setattr(sched, "DeadmanPinger", _Pinger)
    monkeypatch.setattr(sched, "_run_heartbeat_pulse", _spy_pulse)
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda settings, decision_time, artifact, now=None, **k: _report(decision_time))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)
    run_daemon(
        LiveSettings(daemon_catchup_buffer_minutes=20.0),
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=lambda s: None,
        now_fn=lambda: READY_NOW,
        max_iterations=1,
        refresh_fn=lambda *a, **k: None, signal_step_fn=lambda *a, **k: None, prune_fn=lambda: None,
    )
    stages = {stage for stage, _ in seen}
    assert "execute" in stages
    assert all(isinstance(p, _Pinger) for _, p in seen)
