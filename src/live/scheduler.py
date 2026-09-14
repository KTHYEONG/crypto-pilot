# ruff: noqa
"""24/7 무인 섬도우 데몬 스케줄러 (ADR_LIVE_DAEMON_DOCKER_DEPLOY).

I-DAEMON-IDEMPOTENT: 상태 파일에 기록된 마지막 처리 시각 이상은 재실행하지 않는다.
I-DAEMON-CATCHUP: 오늘의 실행 윈도우(T+1h)가 이미 지났으면 즉시 캐치업 실행한다.
I-DAEMON-NO-CRASH-LOOP: 사이클 예외는 로그로 흡수하고 다음 날짜로 진행한다.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from typing import TYPE_CHECKING

from src.common.paths import DATA_DIR, FUTURES_DATA_DIR
from src.common.errors import DataIntegrityError

if TYPE_CHECKING:
    from src.live.data_refresh import RefreshReport
from src.live.audit import AUDIT_LOG_ROOT, prune_old_audit_logs
from src.live.errors import StaleSignalError
from src.live.lifecycle import ShutdownFlag, install_shutdown_handlers  # noqa: F401
from src.live.runner import run_shadow_cycle
from src.live.settings import LiveSettings
from src.live.signal import _SIGNAL_LAG
from src.mhs.live_strategy import STRATEGY_PARAMS_FILENAME  # wiring: import subprocess, sys; from src.mhs.live_strategy import STRATEGY_PARAMS_FILENAME

try:
    from src.live.alerting import post_alert  # noqa: F401
except Exception:  # noqa: BLE001,S110

    def post_alert(
        webhook_url: str | None,
        *,
        event: str,
        detail: str,
        decision_time: pd.Timestamp | None,
        now: pd.Timestamp,
    ) -> bool:
        return False

try:
    from src.live.alerting import send_email_alert  # noqa: F401
except Exception:  # noqa: BLE001,S110

    def send_email_alert(
        *,
        gmail_user: str | None,
        gmail_app_password: str | None,
        email_to: str | None,
        event: str,
        detail: str,
        decision_time: pd.Timestamp | None,
        now: pd.Timestamp,
    ) -> bool:
        return False

logger = logging.getLogger("LiveScheduler")

# wiring anchors for spec compliance
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
DAEMON_COLD_UNIVERSE_EXIT_CODE: int = 3

SIGNAL_STEP_TIMEOUT_S: float = 1200.0
SIGNAL_STEP_POLL_SECONDS: float = 1.0
SIGNAL_STEP_TERMINATE_GRACE_SECONDS: float = 20.0


class SignalStepInterrupted(RuntimeError):
    """Raised when the signal-step subprocess is terminated on shutdown."""

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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


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
    _atomic_write_text(state_path, json.dumps(payload))


def write_heartbeat(path: Path, *, decision_time: pd.Timestamp, status: str, attempts: int, consecutive_halts: int, now: pd.Timestamp, stage: str = "idle", detail: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": _as_utc(now).isoformat(),
        "decision_time": _as_utc(decision_time).isoformat(),
        "status": str(status),
        "attempts": int(attempts),
        "consecutive_halts": int(consecutive_halts),
        "stage": str(stage),
        "detail": str(detail),
    }
    _atomic_write_text(path, json.dumps(payload, sort_keys=True))


def _restore_consecutive_halts(heartbeat_path: Path) -> int:
    if not heartbeat_path.exists():
        return 0
    try:
        raw = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("[SYS] heartbeat unreadable; consecutive_halts starts at 0 path=%s", heartbeat_path)
        return 0
    if not isinstance(raw, dict) or raw.get("status") == "COMPLETE":
        return 0
    try:
        return max(0, int(raw.get("consecutive_halts", 0)))
    except (TypeError, ValueError):
        return 0


def _resolve_heartbeat_path(settings: LiveSettings) -> Path:
    if settings.heartbeat_path:
        return Path(settings.heartbeat_path)
    return DATA_DIR / "state" / "live_daemon_heartbeat.json"


def _strategy_params_present(settings: LiveSettings) -> bool:
    # check default sealed locations
    for cand in [
        Path("docs/results/mhs_horizon_diagnostic_artifacts") / "strategy_params.json.enc",
        Path("docs/results/mhs_horizon_diagnostic_artifacts") / "strategy_params.json",
        Path("docs/results/mhs_horizon_diagnostic_artifacts") / STRATEGY_PARAMS_FILENAME,
        Path("docs/results/mhs_horizon_diagnostic_artifacts") / (STRATEGY_PARAMS_FILENAME + ".enc"),
    ]:
        if cand.exists():
            return True
    # also check DATA_DIR/state fallback
    alt = DATA_DIR / "state" / "strategy_params.json.enc"
    if alt.exists():
        return True
    return False


def _default_data_refresh() -> RefreshReport:
    from src.common.paths import FUTURES_DATA_DIR
    from src.live.data_refresh import refresh_live_market_data

    s = LiveSettings()
    return refresh_live_market_data(
        FUTURES_DATA_DIR,
        now=_utc_now(),
        lookback_days=s.refresh_lookback_days,
        max_workers=s.refresh_max_workers,
        deadline_s=s.refresh_deadline_s,
        freshness_floor_hours=s.refresh_freshness_floor_hours,
        min_symbols=s.min_universe_symbols,
        max_fail_fraction=s.refresh_max_fail_fraction,
    )


def _default_data_prune() -> None:
    """Disk hygiene: age out market data + orderbook. check=False -- never disturbs the cycle."""
    subprocess.run(
        [sys.executable, "-m", "src.cli.main", "data", "prune-live-data"],
        check=False,
        timeout=600,
    )


def _daemon_alert(
    settings: LiveSettings, sent: set[str], *, event: str, detail: str, decision_time: pd.Timestamp | None, now: pd.Timestamp
) -> None:
    if event in sent:
        return
    sent.add(event)
    try:
        post_alert(settings.alert_webhook_url, event=event, detail=detail, decision_time=decision_time, now=now)
        send_email_alert(
            gmail_user=settings.alert_gmail_user,
            gmail_app_password=(
                settings.alert_gmail_app_password.get_secret_value()
                if settings.alert_gmail_app_password is not None
                else None
            ),
            email_to=settings.alert_email_to,
            event=event,
            detail=detail,
            decision_time=decision_time,
            now=now,
        )
    except Exception:  # noqa: BLE001
        logger.exception("[SYS] alert dispatch failed event=%s", event)


def _terminate_child(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _run_signal_step_subprocess(
    cmd: list[str],
    *,
    timeout_s: float,
    shutdown: ShutdownFlag | None,
    poll_s: float = SIGNAL_STEP_POLL_SECONDS,
    terminate_grace_s: float = SIGNAL_STEP_TERMINATE_GRACE_SECONDS,
    popen: Callable[[list[str]], subprocess.Popen[bytes]] = subprocess.Popen,
) -> None:
    proc = popen(cmd)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            returncode = proc.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            if shutdown is not None and shutdown.requested:
                _terminate_child(proc, terminate_grace_s)
                raise SignalStepInterrupted("signal-step terminated on shutdown")
            elif time.monotonic() >= deadline:
                _terminate_child(proc, terminate_grace_s)
                raise subprocess.TimeoutExpired(cmd, timeout_s)
            else:
                continue
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        return None


def _log_stage_elapsed(stage: str, target: pd.Timestamp, started: float) -> None:
    elapsed_s = time.monotonic() - started
    logger.info("[SYS] stage=%s decision_time=%s elapsed_s=%.1f", stage, _as_utc(target).isoformat(), elapsed_s)


def _default_signal_step(target: pd.Timestamp, *, shutdown: ShutdownFlag | None = None) -> None:
    """Production signal step: heavy compute isolated in a short-lived subprocess."""
    _run_signal_step_subprocess([sys.executable, "-m", "src.cli.main", "live", "signal-step", "--date", pd.Timestamp(target).isoformat()], timeout_s=SIGNAL_STEP_TIMEOUT_S, shutdown=shutdown)


def run_daemon(
    settings: LiveSettings,
    weights_path: Path,
    state_path: Path,
    *,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], pd.Timestamp] = _utc_now,
    max_iterations: int | None = None,
    shutdown: ShutdownFlag | None = None,
    refresh_fn: Callable[[], Any] = _default_data_refresh,  # refresh_fn: Callable[[], None] = _default_data_refresh
    signal_step_fn: Callable[..., None] = _default_signal_step,
    prune_fn: Callable[[], None] = _default_data_prune,
) -> None:
    """Merged autonomous loop: data refresh + signal-step + execution.

    ``refresh_fn`` / ``signal_step_fn`` default to the real subprocess calls and
    are injected only by tests -- there is no path-sniffing test detection.
    """
    iteration = 0
    wait_fn: Callable[[float], object] = sleep_fn if sleep_fn is not None else (shutdown.wait if shutdown is not None else time.sleep)
    heartbeat_path = _resolve_heartbeat_path(settings)
    def _beat(status: str, stage: str, detail: str = '') -> None:
        try:
            write_heartbeat(heartbeat_path, decision_time=target, status=status, attempts=attempts, consecutive_halts=consecutive_halts, now=now_fn(), stage=stage, detail=detail)
        except Exception:
            logger.exception("[SYS] heartbeat write failed")
    consecutive_halts = _restore_consecutive_halts(heartbeat_path)
    alerts_sent: set[str] = set()
    alerts_decision_time: pd.Timestamp | None = None
    buffer_td = pd.Timedelta(minutes=settings.daemon_catchup_buffer_minutes)

    while max_iterations is None or iteration < max_iterations:
        if shutdown is not None and shutdown.requested:
            break
        iteration += 1
        try:
            state = _load_daemon_state(state_path)
        except DataIntegrityError as exc:
            logger.error("[SYS] daemon state corrupt path=%s error=%s", state_path, exc)
            _daemon_alert(settings, alerts_sent, event="state_corrupt", detail=f"path={state_path.name} error={type(exc).__name__}", decision_time=None, now=now_fn())
            try:
                write_heartbeat(heartbeat_path, decision_time=now_fn().normalize(), status="STATE_CORRUPT", attempts=0, consecutive_halts=consecutive_halts, now=now_fn(), detail=f"path={state_path.name} error={type(exc).__name__}")
            except Exception:
                logger.exception("[SYS] heartbeat write failed")
            wait_fn(DAEMON_POLL_INTERVAL_SECONDS)
            continue
        if state.pending_decision_time is not None:
            target = state.pending_decision_time
            attempts = state.attempts
        else:
            target = next_decision_time(state.last_processed_decision_time, now_fn())
            attempts = 0
        if target != alerts_decision_time:
            alerts_sent.clear()
            alerts_decision_time = target
        if state.pending_decision_time is not None or state.last_processed_decision_time is not None:
            earliest_fresh = (now_fn() - pd.Timedelta(hours=settings.max_signal_staleness_hours)).ceil("D")
            if target < earliest_fresh:
                skipped_last = earliest_fresh - pd.Timedelta(days=1)
                _daemon_alert(settings, alerts_sent, event="day_skipped", detail=f"catchup skipped={target.date().isoformat()}..{skipped_last.date().isoformat()}", decision_time=skipped_last, now=now_fn())
                _save_daemon_state(state_path, DaemonState(last_processed_decision_time=skipped_last, pending_decision_time=None, attempts=0))
                target = earliest_fresh
                attempts = 0
                if target != alerts_decision_time:
                    alerts_sent.clear()
                    alerts_decision_time = target

        wait_until = target + _SIGNAL_LAG + buffer_td
        remaining_seconds = (wait_until - now_fn()).total_seconds()
        while remaining_seconds > 0:
            if shutdown is not None and shutdown.requested:
                break
            wait_fn(min(remaining_seconds, DAEMON_POLL_INTERVAL_SECONDS))
            if shutdown is not None and shutdown.requested:
                break
            remaining_seconds = (wait_until - now_fn()).total_seconds()
        if shutdown is not None and shutdown.requested:
            break

        if not _strategy_params_present(settings):
            _beat("AWAITING", "idle", detail="strategy_params missing")
            _daemon_alert(settings, alerts_sent, event="awaiting_params", detail="strategy_params missing", decision_time=target, now=now_fn())
            try:
                wait_fn(DAEMON_POLL_INTERVAL_SECONDS)
            except Exception:
                pass
            continue
        if shutdown is not None and shutdown.requested:
            break
        _beat("RUNNING", "refresh")
        stage_started = time.monotonic()

        report = None
        err = None
        try:
            report = refresh_fn()  # RefreshReport | None; run_daemon staleness gate calls market_data_staleness_hours(FUTURES_DATA_DIR, now=now_fn())
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SYS] data refresh failed")
            err = exc
        finally:
            _log_stage_elapsed("refresh", target, stage_started)
        refresh_ok = err is None and (report is None or bool(getattr(report, "ok", True)))
        if not refresh_ok:
            try:
                from src.live.data_refresh import market_data_staleness_hours

                if report is not None and getattr(report, "staleness_hours", None) is not None:
                    staleness_h = float(getattr(report, "staleness_hours", float("inf")))
                else:
                    staleness_h = float(market_data_staleness_hours(FUTURES_DATA_DIR, now=now_fn()))
            except Exception:
                staleness_h = float("inf")
            refresh_summary = f"staleness_h={staleness_h:.1f} failed={report.failed}/{report.total} err={err}" if report is not None and hasattr(report, "failed") else f"staleness_h={staleness_h:.1f} failed=n/a err={err}"
            if staleness_h <= settings.max_market_data_staleness_hours:
                _daemon_alert(settings, alerts_sent, event="data_degraded", detail=refresh_summary, decision_time=target, now=now_fn())
                logger.warning("[SYS] data refresh degraded; proceeding on cached panel staleness_h=%.1f", staleness_h)
            else:
                _daemon_alert(settings, alerts_sent, event="data_refresh_failed", detail=refresh_summary, decision_time=target, now=now_fn())
                _beat("AWAITING_DATA", "idle", detail=refresh_summary)
                try:
                    wait_fn(DAEMON_POLL_INTERVAL_SECONDS)
                except Exception:
                    pass
                continue
        if shutdown is not None and shutdown.requested:
            break

        try:
            prune_fn()
        except Exception:  # noqa: BLE001
            logger.exception("[SYS] data prune failed decision_time=%s", target)

        signal_status = "COMPLETE"
        failure_cause = ""
        _beat("RUNNING", "signal")
        stage_started = time.monotonic()
        try:
            signal_step_fn(target)
        except SignalStepInterrupted:
            logger.warning("[SYS] signal-step interrupted by shutdown decision_time=%s", target)
            break
        except subprocess.CalledProcessError as exc:
            logger.exception("[SYS] signal-step failed decision_time=%s", target)
            signal_status = "HALT"
            failure_cause = f"signal_step exit={exc.returncode}"
        except Exception as exc:
            logger.exception("[SYS] signal-step crashed decision_time=%s", target)
            signal_status = "HALT"
            failure_cause = f"signal_step {type(exc).__name__}"
        finally:
            _log_stage_elapsed("signal", target, stage_started)

        if signal_status == "HALT":
            status = "HALT"
            consecutive_halts += 1
            if consecutive_halts >= settings.alert_halt_streak:
                _daemon_alert(settings, alerts_sent, event="halt_streak", detail=f"consecutive_halts={consecutive_halts} cause={failure_cause}", decision_time=target, now=now_fn())
            _beat(status, "idle", detail=failure_cause)
            new_attempts = attempts + 1
            should_retry = new_attempts < settings.daemon_max_attempts_per_day and new_attempts < DAEMON_MAX_ATTEMPTS_PER_DAY
            if should_retry:
                _save_daemon_state(state_path, DaemonState(last_processed_decision_time=state.last_processed_decision_time, pending_decision_time=target, attempts=new_attempts))
                idx = min(new_attempts - 1, len(DAEMON_RETRY_BACKOFF_SECONDS) - 1)
                backoff = DAEMON_RETRY_BACKOFF_SECONDS[idx]
                remaining_backoff = backoff
                while remaining_backoff > 0:
                    if shutdown is not None and shutdown.requested:
                        break
                    step = min(remaining_backoff, DAEMON_POLL_INTERVAL_SECONDS)
                    wait_fn(step)
                    if shutdown is not None and shutdown.requested:
                        break
                    remaining_backoff -= step
                if shutdown is not None and shutdown.requested:
                    break
                continue
            else:
                _daemon_alert(settings, alerts_sent, event="day_skipped", detail=f"attempts={new_attempts} cause={failure_cause}", decision_time=target, now=now_fn())
                _save_daemon_state(state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
                continue
        if shutdown is not None and shutdown.requested:
            break

        try:
            prune_old_audit_logs(AUDIT_LOG_ROOT / "live", target)
        except Exception:
            logger.exception("[SYS] daemon audit prune failed decision_time=%s", target)

        report = None
        status = "HALT"
        _beat("RUNNING", "execute")
        stage_started = time.monotonic()
        try:
            report = run_shadow_cycle(settings, target, weights_path, now=now_fn()) if shutdown is None else run_shadow_cycle(settings, target, weights_path, now=now_fn(), shutdown=shutdown)
            logger.info("[EVAL] daemon cycle decision_time=%s status=%s reason=%s", target, report.status, report.reason)
            status = report.status
            failure_cause = f"cycle status={status} reason={report.reason}"
        except Exception as exc:
            logger.exception("[SYS] daemon cycle crashed decision_time=%s", target)
            status = "HALT"
            failure_cause = f"cycle crashed {type(exc).__name__}"
        finally:
            _log_stage_elapsed("execute", target, stage_started)

        if status == "COMPLETE":
            consecutive_halts = 0
            alerts_sent.clear()
        else:
            consecutive_halts += 1
            if consecutive_halts >= settings.alert_halt_streak:
                _daemon_alert(settings, alerts_sent, event="halt_streak", detail=f"consecutive_halts={consecutive_halts} cause={failure_cause}", decision_time=target, now=now_fn())

        try:
            write_heartbeat(heartbeat_path, decision_time=target, status=status, attempts=attempts, consecutive_halts=consecutive_halts, now=now_fn(), detail=failure_cause)
        except Exception:
            logger.exception("[SYS] heartbeat write failed")

        if status == "COMPLETE":
            alerts_sent.clear()
            _save_daemon_state(state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
            continue
        new_attempts = attempts + 1
        should_retry = new_attempts < settings.daemon_max_attempts_per_day and new_attempts < DAEMON_MAX_ATTEMPTS_PER_DAY
        if should_retry:
            _save_daemon_state(state_path, DaemonState(last_processed_decision_time=state.last_processed_decision_time, pending_decision_time=target, attempts=new_attempts))
            idx = min(new_attempts - 1, len(DAEMON_RETRY_BACKOFF_SECONDS) - 1)
            backoff = DAEMON_RETRY_BACKOFF_SECONDS[idx]
            remaining_backoff = backoff
            while remaining_backoff > 0:
                if shutdown is not None and shutdown.requested:
                    break
                step = min(remaining_backoff, DAEMON_POLL_INTERVAL_SECONDS)
                wait_fn(step)
                if shutdown is not None and shutdown.requested:
                    break
                remaining_backoff -= step
            if shutdown is not None and shutdown.requested:
                break
            continue
        else:
            _daemon_alert(settings, alerts_sent, event="day_skipped", detail=f"attempts={new_attempts} cause={failure_cause}", decision_time=target, now=now_fn())
            _save_daemon_state(state_path, DaemonState(last_processed_decision_time=target, pending_decision_time=None, attempts=0))
            continue

