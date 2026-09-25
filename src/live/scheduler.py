# ruff: noqa
"""24/7 무인 섬도우 데몬 스케줄러 (ADR_LIVE_DAEMON_DOCKER_DEPLOY).

I-DAEMON-IDEMPOTENT: 상태 파일에 기록된 마지막 처리 시각 이상은 재실행하지 않는다.
I-DAEMON-CATCHUP: 오늘의 실행 윈도우(T+1h)가 이미 지났으면 즉시 캐치업 실행한다.
I-DAEMON-NO-CRASH-LOOP: 사이클 예외는 로그로 흡수하고 다음 날짜로 진행한다.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from typing import TYPE_CHECKING

from src.common.paths import DATA_DIR, FUTURES_DATA_DIR, VENUE_RULES_DIR
from src.common.errors import DataIntegrityError
from src.live.errors import CausalityViolation

if TYPE_CHECKING:
    from src.live.data_refresh import RefreshReport
    from src.live.frozen_signal import FrozenStepReport
from src.live.audit import AUDIT_LOG_ROOT, prune_old_audit_logs
from src.live.errors import StaleSignalError
from src.live.lifecycle import ShutdownFlag, install_shutdown_handlers  # noqa: F401
from src.live.runner import run_shadow_cycle
from src.live.settings import LiveSettings
from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2

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
# stale_after_s(2700초/45분)보다 한참 짧게 잡아, 스케줄러 지연이 겹쳐도 여유가 크다.
DAEMON_HEARTBEAT_PULSE_INTERVAL_SECONDS: float = 120.0
_HEARTBEAT_PULSE_JOIN_TIMEOUT_S: float = 10.0
#: T+1h 인과성 게이트 통과 후의 추가 여유(거래소/네트워크 지연).
DAEMON_CATCHUP_BUFFER: pd.Timedelta = pd.Timedelta(minutes=5)

#: frozen 신호 공개 시각이며, 공식 메이커 원장의 제출봉과 같은 기준이다.
DECISION_RELEASE_OFFSET: pd.Timedelta = pd.Timedelta(hours=FROZEN_MHS_TOP20_V2.release_hour_utc)

DAEMON_MAX_ATTEMPTS_PER_DAY: int = 5
DAEMON_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (300.0, 600.0, 1200.0, 2400.0)
# src/application/ops/daemon_idle_gate.py BUSY_STAGES 와 같은 의미
INTERRUPTIBLE_STAGES: frozenset[str] = frozenset({"refresh", "signal", "execute"})
DAEMON_ALERT_SYMBOL_SAMPLE: int = 10
SIGNAL_REFRESH_OFFSET_MINUTES: float = 0.0
DAEMON_COLD_UNIVERSE_EXIT_CODE: int = 3

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


NON_CRYPTO_SYMBOLS_PATH: Path = DATA_DIR / "state" / "non_crypto_symbols.json"


def _default_data_refresh() -> RefreshReport:
    import urllib.request

    from src.live.data_refresh import EXCHANGE_INFO_TIMEOUT_S, EXCHANGE_INFO_URL, listed_crypto_perpetuals, refresh_live_market_data
    from src.mhs.params import LIVE_FROZEN_WARMUP_DAYS

    with urllib.request.urlopen(EXCHANGE_INFO_URL, timeout=EXCHANGE_INFO_TIMEOUT_S) as resp:  # noqa: S310
        payload = json.loads(resp.read())
    crypto, non_crypto = listed_crypto_perpetuals(payload)
    NON_CRYPTO_SYMBOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = NON_CRYPTO_SYMBOLS_PATH.with_suffix(NON_CRYPTO_SYMBOLS_PATH.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"captured_at": _utc_now().isoformat(), "symbols": sorted(non_crypto)}, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, NON_CRYPTO_SYMBOLS_PATH)
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
        symbols=sorted(crypto),
        seed_lookback_days=LIVE_FROZEN_WARMUP_DAYS + 30,
    )


def _default_venue_capture(settings: LiveSettings, decision_time: pd.Timestamp) -> None:
    """Best-effort venue rule snapshot for ``decision_time``'s slot; failures never stop the cycle.

    The slot is keyed by the decision day so a catch-up cycle after midnight does not consume the next
    decision day's slot. An existing slot (an earlier attempt for the same decision) is reused without
    a signed API call.
    """
    try:
        from src.market_data.binance.venue_rules import (
            fetch_venue_rules,
            venue_rule_snapshot_exists,
            write_venue_rule_snapshot,
        )

        slot = _as_utc(decision_time).strftime("%Y%m%d")
        if venue_rule_snapshot_exists(VENUE_RULES_DIR, slot):
            logger.info("[DATA] stage=venue_capture status=ALREADY_CAPTURED slot=%s", slot)
            return None
        key = settings.api_key.get_secret_value() if settings.api_key is not None else None
        secret = settings.api_secret.get_secret_value() if settings.api_secret is not None else None
        snapshot = fetch_venue_rules(api_key=key, api_secret=secret)
        write_venue_rule_snapshot(snapshot, VENUE_RULES_DIR, slot_day=decision_time)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DATA] stage=venue_capture status=FAILED error=%s", exc)
        return None


def _default_frozen_step(target: pd.Timestamp, settings: LiveSettings, weights_path: Path) -> FrozenStepReport:
    from src.live.frozen_signal import run_frozen_signal_step
    from src.live.ledger import default_ledger_path
    from src.live.runner import fetch_live_account_equity

    try:
        raw = json.loads(NON_CRYPTO_SYMBOLS_PATH.read_text(encoding="utf-8"))
        symbols_raw = raw.get("symbols")
        if not isinstance(raw, dict) or not isinstance(symbols_raw, list):
            raise DataIntegrityError(f"non-crypto symbols file malformed: {NON_CRYPTO_SYMBOLS_PATH}")
        non_crypto = frozenset(str(s) for s in symbols_raw)
    except FileNotFoundError as exc:
        raise DataIntegrityError(f"non-crypto symbols file missing: {NON_CRYPTO_SYMBOLS_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"non-crypto symbols file corrupt: {NON_CRYPTO_SYMBOLS_PATH}") from exc
    return run_frozen_signal_step(
        target,
        now=_utc_now(),
        data_root=FUTURES_DATA_DIR,
        weights_path=Path(weights_path),
        unit_bootstrap_path=Path(settings.unit_bootstrap_path),
        unit_forward_path=Path(weights_path).parent / "frozen_unit_forward.parquet",
        venue_rules_dir=VENUE_RULES_DIR,
        fallback_venue_path=Path(settings.venue_fallback_path),
        ledger_path=Path(settings.ledger_path) if settings.ledger_path else default_ledger_path(),
        seed_equity_usdt=settings.notional_equity_usdt,
        account_equity_usdt=None if settings.mode.suppresses_mutations else fetch_live_account_equity(settings, _utc_now()),
        non_crypto=non_crypto,
        artifact_key=settings.artifact_key,
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
) -> bool:
    if event in sent:
        return False
    try:
        delivered = bool(post_alert(settings.alert_webhook_url, event=event, detail=detail, decision_time=decision_time, now=now))
        emailed = bool(
            send_email_alert(
                gmail_user=settings.alert_gmail_user,
                gmail_app_password=(
                    settings.alert_gmail_app_password.get_secret_value()
                    if settings.alert_gmail_app_password is not None
                    else None
                ),
                event=event,
                detail=detail,
                decision_time=decision_time,
                now=now,
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("[SYS] alert dispatch failed event=%s", event)
        return False
    # 발송 성공 시에만 중복 제거 집합에 기록
    if delivered or emailed:
        sent.add(event)
        return True
    if settings.alert_webhook_url or settings.alert_gmail_user:
        logger.warning("[SYS] alert not delivered event=%s", event)
    return False


def _read_heartbeat(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _touch_heartbeat(path: Path, now: pd.Timestamp) -> None:
    raw = _read_heartbeat(path)
    if raw is None:
        return
    # 대기 중 생존 틱: ts만 갱신하고 상태 전이는 건드리지 않음
    raw["ts"] = _as_utc(now).isoformat()
    try:
        _atomic_write_text(Path(path), json.dumps(raw, sort_keys=True))
    except OSError as exc:
        logger.warning("[SYS] heartbeat touch failed path=%s error=%s", path, exc)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _handle_interrupted_stage(settings: LiveSettings, heartbeat_path: Path, sent: set[str], now: pd.Timestamp) -> None:
    raw = _read_heartbeat(heartbeat_path)
    if raw is None or raw.get("stage") not in INTERRUPTIBLE_STAGES:
        return
    stage = str(raw["stage"])
    detail = f"stage={stage} decision_time={raw.get('decision_time')} heartbeat_ts={raw.get('ts')}"
    logger.warning("[SYS] cycle interrupted %s", detail)
    _daemon_alert(settings, sent, event="cycle_interrupted", detail=detail, decision_time=None, now=now)
    try:
        decision_time = pd.Timestamp(raw["decision_time"])
        if decision_time.tzinfo is None:
            raise ValueError("naive decision_time")
    except (KeyError, ValueError, TypeError):
        # 깨진 시각은 오늘 날짜로 복구
        decision_time = _as_utc(now).normalize()
    attempts = _int_or_zero(raw.get("attempts", 0))
    consecutive_halts = _int_or_zero(raw.get("consecutive_halts", 0))
    try:
        write_heartbeat(heartbeat_path, decision_time=decision_time, status="INTERRUPTED", attempts=attempts, consecutive_halts=consecutive_halts, now=now, stage="idle", detail=f"interrupted stage={stage}")
    except OSError:
        logger.exception("[SYS] heartbeat write failed")


def _refresh_note(report: Any, err: BaseException | None) -> str:
    if err is not None:
        return f"refresh=error:{type(err).__name__}"
    if report is None or not hasattr(report, "failed"):
        return "refresh=n/a"
    return f"refresh fresh={getattr(report, 'fresh', 'n/a')} refreshed={getattr(report, 'refreshed', 'n/a')} failed={report.failed}/{getattr(report, 'total', 'n/a')} staleness_h={float(getattr(report, 'staleness_hours', float('nan'))):.1f}"


def _sizing_note(frozen_report: Any) -> str:
    if frozen_report is None:
        return ""
    try:
        exposure = float(getattr(frozen_report, "exposure"))
        equity = float(getattr(frozen_report, "equity_usdt"))
        unit_obs = int(getattr(frozen_report, "unit_observations"))
    except (TypeError, ValueError, AttributeError):
        return ""
    return f" exposure={exposure:.4f} equity_usdt={equity:.2f} unit_observations={unit_obs}"


def _run_heartbeat_pulse(
    stop: threading.Event,
    *,
    heartbeat_path: Path,
    decision_time: pd.Timestamp,
    attempts: int,
    consecutive_halts: int,
    interval_s: float = DAEMON_HEARTBEAT_PULSE_INTERVAL_SECONDS,
) -> None:
    """Keep the heartbeat fresh while a long blocking stage (execute) runs.

    Runs in a background thread started just before the blocking call and stopped in its
    ``finally``. Writes ``status="RUNNING", stage="execute"`` at ``interval_s`` using the real
    UTC wall clock (never an injected/test clock -- the deploy gate reads this file from a
    separate process, so freshness must reflect actual elapsed time, and a fake clock shared
    across threads would race). One write failure is logged and never stops the loop; the pulse
    is a liveness signal only and must never affect the trading cycle's outcome.
    """
    while True:
        if stop.wait(timeout=interval_s):
            return
        try:
            write_heartbeat(
                heartbeat_path,
                decision_time=decision_time,
                status="RUNNING",
                attempts=attempts,
                consecutive_halts=consecutive_halts,
                now=pd.Timestamp.now(tz="UTC"),
                stage="execute",
            )
        except Exception:  # noqa: BLE001
            logger.exception("[SYS] heartbeat pulse write failed")


def _log_stage_elapsed(stage: str, target: pd.Timestamp, started: float) -> None:
    elapsed_s = time.monotonic() - started
    logger.info("[SYS] stage=%s decision_time=%s elapsed_s=%.1f", stage, _as_utc(target).isoformat(), elapsed_s)


def run_daemon(
    settings: LiveSettings,
    weights_path: Path,
    state_path: Path,
    *,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], pd.Timestamp] = _utc_now,
    max_iterations: int | None = None,
    shutdown: ShutdownFlag | None = None,
    refresh_fn: Callable[[], Any] = _default_data_refresh,
    signal_step_fn: Callable[[pd.Timestamp], Any] | None = None,
    prune_fn: Callable[[], None] = _default_data_prune,
    venue_fn: Callable[[pd.Timestamp], None] | None = None,
) -> None:
    """Merged autonomous loop: venue snapshot + data refresh + frozen step + execution.

    ``refresh_fn`` / ``signal_step_fn`` / ``venue_fn`` default to the live wiring and
    are injected only by tests -- there is no path-sniffing test detection.
    """
    if signal_step_fn is None:
        signal_step_fn = functools.partial(_default_frozen_step, settings=settings, weights_path=weights_path)
    if venue_fn is None:
        venue_fn = functools.partial(_default_venue_capture, settings)
    iteration = 0
    wait_fn: Callable[[float], object] = sleep_fn if sleep_fn is not None else (shutdown.wait if shutdown is not None else time.sleep)
    heartbeat_path = _resolve_heartbeat_path(settings)
    def _wait(seconds: float) -> None:
        wait_fn(seconds)
        _touch_heartbeat(heartbeat_path, now_fn())
    def _beat(status: str, stage: str, detail: str = '') -> None:
        try:
            write_heartbeat(heartbeat_path, decision_time=target, status=status, attempts=attempts, consecutive_halts=consecutive_halts, now=now_fn(), stage=stage, detail=detail)
        except Exception:
            logger.exception("[SYS] heartbeat write failed")
    consecutive_halts = _restore_consecutive_halts(heartbeat_path)
    alerts_sent: set[str] = set()
    alerts_decision_time: pd.Timestamp | None = None
    buffer_td = pd.Timedelta(minutes=settings.daemon_catchup_buffer_minutes)
    startup = _read_heartbeat(heartbeat_path)
    logger.info("[SYS] daemon start mode=%s pid=%d weights=%s state=%s heartbeat_stage=%s heartbeat_status=%s", settings.mode.value, os.getpid(), weights_path, state_path, (startup or {}).get("stage"), (startup or {}).get("status"))
    _handle_interrupted_stage(settings, heartbeat_path, alerts_sent, now_fn())

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
            _wait(DAEMON_POLL_INTERVAL_SECONDS)
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

        wait_until = target + DECISION_RELEASE_OFFSET + buffer_td
        remaining_seconds = (wait_until - now_fn()).total_seconds()
        while remaining_seconds > 0:
            if shutdown is not None and shutdown.requested:
                break
            _wait(min(remaining_seconds, DAEMON_POLL_INTERVAL_SECONDS))
            if shutdown is not None and shutdown.requested:
                break
            remaining_seconds = (wait_until - now_fn()).total_seconds()
        if shutdown is not None and shutdown.requested:
            break

        try:
            venue_fn(target)
        except Exception:  # noqa: BLE001
            logger.exception("[SYS] venue capture failed decision_time=%s", target)
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
        refresh_note = _refresh_note(report, err)
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
            if report is not None and getattr(report, "funding_blocked", False):
                refresh_summary += " funding_blocked=True"
            if staleness_h <= settings.max_market_data_staleness_hours:
                _daemon_alert(settings, alerts_sent, event="data_degraded", detail=refresh_summary, decision_time=target, now=now_fn())
                logger.warning("[SYS] data refresh degraded; proceeding on cached panel staleness_h=%.1f", staleness_h)
            else:
                _daemon_alert(settings, alerts_sent, event="data_refresh_failed", detail=refresh_summary, decision_time=target, now=now_fn())
                _beat("AWAITING_DATA", "idle", detail=refresh_summary)
                try:
                    _wait(DAEMON_POLL_INTERVAL_SECONDS)
                except Exception:
                    pass
                continue
        if shutdown is not None and shutdown.requested:
            break

        signal_status = "COMPLETE"
        failure_cause = ""
        frozen_report = None
        quarantined = 0
        _beat("RUNNING", "signal")
        stage_started = time.monotonic()
        try:
            frozen_report = signal_step_fn(target)
        except (DataIntegrityError, CausalityViolation) as exc:
            logger.exception("[SYS] frozen step halted decision_time=%s", target)
            signal_status = "HALT"
            failure_cause = f"frozen_step {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SYS] frozen step crashed decision_time=%s", target)
            signal_status = "HALT"
            failure_cause = f"frozen_step {type(exc).__name__}: {exc}"
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
                    _wait(step)
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
        pulse_stop = threading.Event()
        pulse_thread = threading.Thread(
            target=_run_heartbeat_pulse,
            kwargs={
                "stop": pulse_stop,
                "heartbeat_path": heartbeat_path,
                "decision_time": target,
                "attempts": attempts,
                "consecutive_halts": consecutive_halts,
            },
            name="live-heartbeat-pulse",
            daemon=True,
        )
        pulse_thread.start()
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
            pulse_stop.set()
            pulse_thread.join(timeout=_HEARTBEAT_PULSE_JOIN_TIMEOUT_S)
            if pulse_thread.is_alive():
                logger.warning("[SYS] heartbeat pulse thread still alive decision_time=%s", target)
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

        try:
            prune_fn()
        except Exception:  # noqa: BLE001
            logger.exception("[SYS] data prune failed decision_time=%s", target)

        if status == "COMPLETE":
            alerts_sent.clear()
            if settings.alert_daily_digest:
                digest_detail = f"intents={getattr(report, 'intent_count', 0)} reason={getattr(report, 'reason', None)} dropped_fraction={float(getattr(report, 'dropped_notional_fraction', 0.0)):.4f} quarantined={quarantined} {refresh_note}{_sizing_note(frozen_report)}"
                _daemon_alert(settings, alerts_sent, event="cycle_complete", detail=digest_detail, decision_time=target, now=now_fn())
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
                _wait(step)
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

