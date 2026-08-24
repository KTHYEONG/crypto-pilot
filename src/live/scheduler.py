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
from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.audit import AUDIT_LOG_ROOT, prune_old_audit_logs
from src.live.runner import run_shadow_cycle
from src.live.settings import LiveSettings
from src.live.signal import _SIGNAL_LAG

logger = logging.getLogger("LiveScheduler")

#: 대기 중 sleep_fn 호출 간격 상한(초). 종료 시그널 처리 지연과 테스트 대기 횟수를 bound한다.
DAEMON_POLL_INTERVAL_SECONDS: float = 300.0
#: T+1h 인과성 게이트 통과 후의 추가 여유(거래소/네트워크 지연).
DAEMON_CATCHUP_BUFFER: pd.Timedelta = pd.Timedelta(minutes=5)

_STATE_KEY = "last_processed_decision_time"


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


def run_daemon(
    settings: LiveSettings,
    artifact_path: Path,
    state_path: Path,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], pd.Timestamp] = _utc_now,
    max_iterations: int | None = None,
) -> None:
    """하루 1 사이클씩 무한 반복한다(max_iterations는 테스트용 상한).

    COMPLETE/HALT/예외 모두 '처리됨'으로 기록해 동일 decision_time 재실행을 막는다.
    반복마다 누적 상태가 없어 O(1) 메모리로 무한 실행 가능하다.
    """
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        last = _load_last_processed(state_path)
        target = next_decision_time(last, now_fn())
        wait_until = target + _SIGNAL_LAG + DAEMON_CATCHUP_BUFFER
        remaining_seconds = (wait_until - now_fn()).total_seconds()
        while remaining_seconds > 0:
            sleep_fn(min(remaining_seconds, DAEMON_POLL_INTERVAL_SECONDS))
            remaining_seconds = (wait_until - now_fn()).total_seconds()

        try:
            prune_old_audit_logs(AUDIT_LOG_ROOT / "live", target)
        except Exception:
            logger.exception("[SYS] daemon audit prune failed decision_time=%s", target)

        try:
            report = run_shadow_cycle(settings, target, artifact_path, now=now_fn())
            logger.info(
                "[EVAL] daemon cycle decision_time=%s status=%s reason=%s",
                target,
                report.status,
                report.reason,
            )
        except Exception:
            logger.exception("[SYS] daemon cycle crashed decision_time=%s", target)

        _save_last_processed(state_path, target)
