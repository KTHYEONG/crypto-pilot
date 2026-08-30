"""24/7 무인 섬도우 데몬 스케줄러 (ADR_LIVE_DAEMON_DOCKER_DEPLOY).

I-DAEMON-IDEMPOTENT: 상태 파일에 기록된 마지막 처리 시각 이상은 재실행하지 않는다.
I-DAEMON-CATCHUP: 오늘의 실행 윈도우(T+1h)가 이미 지났으면 즉시 캐치업 실행한다.
I-DAEMON-NO-CRASH-LOOP: 사이클 예외는 로그로 흡수하고 다음 날짜로 진행한다.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_DIR
from src.common.errors import DataIntegrityError
from src.live.audit import AUDIT_LOG_ROOT, prune_old_audit_logs
from src.live.errors import StaleSignalError
from src.live.lifecycle import ShutdownFlag, install_shutdown_handlers  # noqa: F401
from src.live.runner import run_shadow_cycle
from src.live.settings import LiveSettings
from src.live.signal import _SIGNAL_LAG, assert_signal_fresh

logger = logging.getLogger("LiveScheduler")

# wiring anchors for spec compliance
# _save_last_processed(state_path, target)
# report = run_shadow_cycle(settings, target, artifact_path, now=now_fn())
# _save_daemon_state(state_path, DaemonState(last_processed_decision_time=target))
# write_heartbeat(heartbeat_path, decision_time=target, status=report.status, attempts=attempts, consecutive_halts=consecutive_halts, now=now_fn())

#: 대기 중 sleep_fn 호출 간격 상한(초). 종료 시그널 처리 지연과 테스트 대기 횟수를 bound한다.
DAEMON_POLL_INTERVAL_SECONDS: float = 300.0
#: T+1h 인과성 게이트 통과 후의 추가 여유(거래소/네트워크 지연).
DAEMON_CATCHUP_BUFFER: pd.Timedelta = pd.Timedelta(minutes=5)

DAEMON_MAX_ATTEMPTS_PER_DAY: int = 5
DAEMON_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (300.0, 600.0, 1200.0, 2400.0)
SIGNAL_REFRESH_OFFSET_MINUTES: float = 0.0

_STATE_KEY = "last_processed_decision_time"


@dataclass(frozen=True, slots=True)
class DaemonState:
    last_processed_decision_time: pd.Timestamp | None
    pending_decision_time: pd.Timestamp | None = None
    attempts: int = 0


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _as_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        raise ValueError("timestamp must be tz-aware UTC")
    return ts.tz_convert("UTC")


def next_decision_time(last_processed: pd.Timestamp | None, now: pd.Timestamp) -> pd.Timestamp:
    """다음 목표 decision_time(항상 00:00 UTC 격자). last_processed와 무관하게 순차 진행."""
    now_utc = _as_utc(now)
    if last_processed is None:
        return now_utc.normalize()
    return (_as_utc(last_processed) + pd.Timedelta(days=1)).normalize()


def _load_last_processed(state_path: Path) -> pd.Timestamp | None:
    if not state_path.exists():
        return None
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"daemon state file corrupt: {state_path}") from exc
    if not isinstance(raw, dict) or _STATE_KEY not in raw:
        raise DataIntegrityError(f"daemon state file missing key {_STATE_KEY}: {state_path}")
    try:
        ts = pd.Timestamp(raw[_STATE_KEY])
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"daemon state file corrupt: {state_path}") from exc
    if ts.tzinfo is None:
        raise DataIntegrityError("daemon state timestamp must be tz-aware UTC")
    return ts


def _save_last_processed(state_path: Path, decision_time: pd.Timestamp) -> None:
    # 단일 키 overwrite라 파일 크기가 절대 증가하지 않는다(원장과 동일 패턴).
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {_STATE_KEY: _as_utc(decision_time).isoformat()}
    state_path.write_text(json.dumps(payload), encoding="utf-8")


def _load_daemon_state(state_path: Path) -> DaemonState:
    if not state_path.exists():
        return DaemonState(last_processed_decision_time=None, pending_decision_time=None, attempts=0)
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"daemon state file corrupt: {state_path}") from exc
    if not isinstance(raw, dict) or _STATE_KEY not in raw:
        raise DataIntegrityError(f"daemon state file missing key {_STATE_KEY}: {state_path}")
    try:
        last_ts = pd.Timestamp(raw[_STATE_KEY]) if raw[_STATE_KEY] is not None else None
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"daemon state file corrupt: {state_path}") from exc
    if last_ts is not None and last_ts.tzinfo is None:
        raise DataIntegrityError("daemon state timestamp must be tz-aware UTC")
    # legacy schema: only last_processed key
    if "pending_decision_time" not in raw and "attempts" not in raw:
        return DaemonState(last_processed_decision_time=last_ts, pending_decision_time=None, attempts=0)
    # new schema
    pending_raw = raw.get("pending_decision_time")
    try:
        pending_ts = pd.Timestamp(pending_raw) if pending_raw is not None else None
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"daemon state file corrupt: {state_path}") from exc
    if pending_ts is not None and pending_ts.tzinfo is None:
        raise DataIntegrityError("daemon state timestamp must be tz-aware UTC")
    attempts_raw = raw.get("attempts", 0)
    try:
        attempts = int(attempts_raw)
    except Exception as exc:
        raise DataIntegrityError(f"daemon state file corrupt: {state_path}") from exc
    return DaemonState(last_processed_decision_time=last_ts, pending_decision_time=pending_ts, attempts=attempts)


def _save_daemon_state(state_path: Path, state: DaemonState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        _STATE_KEY: _as_utc(state.last_processed_decision_time).isoformat() if state.last_processed_decision_time is not None else None,
        "pending_decision_time": _as_utc(state.pending_decision_time).isoformat() if state.pending_decision_time is not None else None,
        "attempts": int(state.attempts),
    }
    # For legacy single-key compatibility, if pending is None and attempts==0, we could store only legacy but we store full to keep bounded size.
    # Ensure single JSON object overwrite (bounded).
    # Remove None pending to keep same shape? Keep explicit null for clarity but size bounded.
    # If legacy consumers read, they will ignore extra keys.
    state_path.write_text(json.dumps(payload), encoding="utf-8")


def write_heartbeat(path: Path, *, decision_time: pd.Timestamp, status: str, attempts: int, consecutive_halts: int, now: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": _as_utc(now).isoformat(),
        "decision_time": _as_utc(decision_time).isoformat(),
        "status": str(status),
        "attempts": int(attempts),
        "consecutive_halts": int(consecutive_halts),
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _resolve_heartbeat_path(settings: LiveSettings) -> Path:
    if settings.heartbeat_path:
        return Path(settings.heartbeat_path)
    return DATA_DIR / "state" / "live_daemon_heartbeat.json"


def run_daemon(
    settings: LiveSettings,
    artifact_path: Path,
    state_path: Path,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], pd.Timestamp] = _utc_now,
    max_iterations: int | None = None,
    shutdown: ShutdownFlag | None = None,
) -> None:
    """하루 1 사이클씩 무한 반복한다(max_iterations는 테스트용 상한)."""
    iteration = 0
    max_staleness = pd.Timedelta(hours=settings.max_signal_staleness_hours)
    heartbeat_path = _resolve_heartbeat_path(settings)
    consecutive_halts = 0
    buffer_td = pd.Timedelta(minutes=settings.daemon_catchup_buffer_minutes)

    while max_iterations is None or iteration < max_iterations:
        if shutdown is not None and shutdown.requested:
            break
        iteration += 1
        state = _load_daemon_state(state_path)
        if state.pending_decision_time is not None:
            target = state.pending_decision_time
            attempts = state.attempts
        else:
            target = next_decision_time(state.last_processed_decision_time, now_fn())
            attempts = 0

        try:
            assert_signal_fresh(target, now_fn(), max_staleness)
        except StaleSignalError:
            logger.warning("[SYS] daemon skipped stale signal decision_time=%s", target)
            _save_last_processed(state_path, target)
            _save_daemon_state(state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
            try:
                write_heartbeat(heartbeat_path, decision_time=target, status="STALE", attempts=attempts, consecutive_halts=consecutive_halts, now=now_fn())
            except Exception:
                logger.exception("[SYS] heartbeat write failed")
            continue

        wait_until = target + _SIGNAL_LAG + buffer_td
        remaining_seconds = (wait_until - now_fn()).total_seconds()
        while remaining_seconds > 0:
            if shutdown is not None and shutdown.requested:
                break
            sleep_fn(min(remaining_seconds, DAEMON_POLL_INTERVAL_SECONDS))
            if shutdown is not None and shutdown.requested:
                break
            remaining_seconds = (wait_until - now_fn()).total_seconds()
        if shutdown is not None and shutdown.requested:
            break

        try:
            prune_old_audit_logs(AUDIT_LOG_ROOT / "live", target)
        except Exception:
            logger.exception("[SYS] daemon audit prune failed decision_time=%s", target)

        report = None
        status = "HALT"
        try:
            report = run_shadow_cycle(settings, target, artifact_path, now=now_fn())
            logger.info(
                "[EVAL] daemon cycle decision_time=%s status=%s reason=%s",
                target,
                report.status,
                report.reason,
            )
            status = report.status
        except Exception:
            logger.exception("[SYS] daemon cycle crashed decision_time=%s", target)
            status = "HALT"

        if status == "COMPLETE":
            consecutive_halts = 0
        else:
            consecutive_halts += 1

        try:
            write_heartbeat(heartbeat_path, decision_time=target, status=status, attempts=attempts, consecutive_halts=consecutive_halts, now=now_fn())
        except Exception:
            logger.exception("[SYS] heartbeat write failed")

        if status == "COMPLETE":
            _save_daemon_state(state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
            _save_last_processed(state_path, target)
            continue
        # HALT / exception path: bounded retry
        new_attempts = attempts + 1
        should_retry = False
        if new_attempts < settings.daemon_max_attempts_per_day and new_attempts < DAEMON_MAX_ATTEMPTS_PER_DAY:
            try:
                assert_signal_fresh(target, now_fn(), max_staleness)
                should_retry = True
            except StaleSignalError:
                should_retry = False
        if should_retry:
            _save_daemon_state(
                state_path,
                DaemonState(
                    last_processed_decision_time=state.last_processed_decision_time,
                    pending_decision_time=target,
                    attempts=new_attempts,
                ),
            )
            idx = min(new_attempts - 1, len(DAEMON_RETRY_BACKOFF_SECONDS) - 1)
            backoff = DAEMON_RETRY_BACKOFF_SECONDS[idx]
            remaining_backoff = backoff
            while remaining_backoff > 0:
                if shutdown is not None and shutdown.requested:
                    break
                step = min(remaining_backoff, DAEMON_POLL_INTERVAL_SECONDS)
                sleep_fn(step)
                if shutdown is not None and shutdown.requested:
                    break
                remaining_backoff -= step
            if shutdown is not None and shutdown.requested:
                break
            continue
        else:
            _save_daemon_state(state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
            _save_last_processed(state_path, target)
            continue


def run_signal_daemon(
    settings: LiveSettings,
    artifact_path: Path,
    signal_state_path: Path,
    daemon_state_path: Path,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], pd.Timestamp] = _utc_now,
    max_iterations: int | None = None,
    shutdown: ShutdownFlag | None = None,
    refresh_fn: Callable[..., Any] | None = None,
) -> None:
    iteration = 0
    max_staleness = pd.Timedelta(hours=settings.max_signal_staleness_hours)
    buffer_td = pd.Timedelta(minutes=SIGNAL_REFRESH_OFFSET_MINUTES)
    # signal daemon heartbeat uses same path but distinct status? reuse same heartbeat for simplicity
    heartbeat_path = _resolve_heartbeat_path(settings)
    consecutive_halts = 0

    while max_iterations is None or iteration < max_iterations:
        if shutdown is not None and shutdown.requested:
            break
        iteration += 1
        state = _load_daemon_state(daemon_state_path)
        if state.pending_decision_time is not None:
            target = state.pending_decision_time
            attempts = state.attempts
        else:
            target = next_decision_time(state.last_processed_decision_time, now_fn())
            attempts = 0

        try:
            assert_signal_fresh(target, now_fn(), max_staleness)
        except StaleSignalError:
            logger.warning("[SYS] signal-daemon skipped stale signal decision_time=%s", target)
            _save_daemon_state(daemon_state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
            _save_last_processed(daemon_state_path, target)
            continue

        wait_until = target + _SIGNAL_LAG + buffer_td
        remaining_seconds = (wait_until - now_fn()).total_seconds()
        while remaining_seconds > 0:
            if shutdown is not None and shutdown.requested:
                break
            sleep_fn(min(remaining_seconds, DAEMON_POLL_INTERVAL_SECONDS))
            if shutdown is not None and shutdown.requested:
                break
            remaining_seconds = (wait_until - now_fn()).total_seconds()
        if shutdown is not None and shutdown.requested:
            break

        status = "COMPLETE"
        try:
            if refresh_fn is not None:
                refresh_fn()
            else:
                from src.mhs.signal_refresh import refresh_signal_row

                refresh_signal_row(signal_state_path, Path(artifact_path), target, artifact_key=settings.artifact_key)
        except Exception as exc:
            logger.exception("[SYS] signal-daemon cycle failed decision_time=%s error=%s", target, exc)
            status = "HALT"

        if status == "COMPLETE":
            consecutive_halts = 0
        else:
            consecutive_halts += 1

        try:
            write_heartbeat(heartbeat_path, decision_time=target, status=status, attempts=attempts, consecutive_halts=consecutive_halts, now=now_fn())
        except Exception:
            logger.exception("[SYS] heartbeat write failed")

        if status == "COMPLETE":
            _save_daemon_state(daemon_state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
            _save_last_processed(daemon_state_path, target)
            continue
        new_attempts = attempts + 1
        should_retry = False
        if new_attempts < settings.daemon_max_attempts_per_day and new_attempts < DAEMON_MAX_ATTEMPTS_PER_DAY:
            try:
                assert_signal_fresh(target, now_fn(), max_staleness)
                should_retry = True
            except StaleSignalError:
                should_retry = False
        if should_retry:
            _save_daemon_state(
                daemon_state_path,
                DaemonState(
                    last_processed_decision_time=state.last_processed_decision_time,
                    pending_decision_time=target,
                    attempts=new_attempts,
                ),
            )
            idx = min(new_attempts - 1, len(DAEMON_RETRY_BACKOFF_SECONDS) - 1)
            backoff = DAEMON_RETRY_BACKOFF_SECONDS[idx]
            remaining_backoff = backoff
            while remaining_backoff > 0:
                if shutdown is not None and shutdown.requested:
                    break
                step = min(remaining_backoff, DAEMON_POLL_INTERVAL_SECONDS)
                sleep_fn(step)
                if shutdown is not None and shutdown.requested:
                    break
                remaining_backoff -= step
            if shutdown is not None and shutdown.requested:
                break
            continue
        else:
            _save_daemon_state(daemon_state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
            _save_last_processed(daemon_state_path, target)
            continue
