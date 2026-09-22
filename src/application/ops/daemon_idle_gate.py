"""CI deploy gate: proceed only when the live daemon is idle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

BUSY_STAGES: frozenset[str] = frozenset({"refresh", "signal", "execute"})

EXIT_PROCEED: int = 0
EXIT_WAIT: int = 10

# 신호 공개(23:00 UTC) 직전부터 막아 공개 직후 제출이 재시작으로 밀리지 않게 한다.
DECISION_WINDOW_LEAD_MINUTES: int = 15
# frozen 신호 공개 시각(UTC). FROZEN_MHS_TOP20_V2.release_hour_utc 와 같아야 한다(의존성 없는 CI 모듈이라 복제).
DECISION_RELEASE_HOUR_UTC: int = 23
# 신호 스테일 상한(시간). LiveSettings.max_signal_staleness_hours 와 같아야 한다.
DECISION_SIGNAL_STALENESS_HOURS: float = 26.0

# 최악 재시도 경로 02:00 + 실행 단계 66분을 덮는다.
DEFAULT_MAX_WAIT_S: float = 3 * 3600.0
DEFAULT_STALE_AFTER_S: float = 2700.0


@dataclass(frozen=True, slots=True)
class GateDecision:
    action: Literal["proceed", "wait", "proceed_stale", "proceed_timeout"]
    reason: str


def _window_decision_day(now: datetime) -> datetime | None:
    now_utc = now.astimezone(UTC)
    base = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    hits: list[datetime] = []
    for delta in (-2, -1, 0, 1):
        day = base + timedelta(days=delta)
        start = day + timedelta(hours=DECISION_RELEASE_HOUR_UTC) - timedelta(minutes=DECISION_WINDOW_LEAD_MINUTES)
        end = day + timedelta(hours=DECISION_SIGNAL_STALENESS_HOURS)
        if start <= now_utc < end:
            hits.append(day)
    if not hits:
        return None
    return max(hits)


def _parse_heartbeat_time(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def decide_deploy(
    heartbeat: Mapping[str, object] | None,
    *,
    now: datetime,
    waited_s: float,
    max_wait_s: float,
    stale_after_s: float,
) -> GateDecision:
    """Decide whether CI may restart the live daemon now.

    Inside the decision window -- from ``DECISION_WINDOW_LEAD_MINUTES`` before the day's
    ``DECISION_RELEASE_HOUR_UTC`` release until the signal staleness limit
    (decision day 00:00 + ``DECISION_SIGNAL_STALENESS_HOURS``) -- a restart is allowed only once
    the heartbeat shows ``status == "COMPLETE"`` for that window's decision day, because a
    restart anywhere in the cycle (including retry back-off and AWAITING_DATA, which report
    stage ``idle``) delays submission and late submission is the dominant live growth loss.
    Outside the window the existing busy-stage rule applies. Stale heartbeats and
    ``max_wait_s`` keep their existing escape semantics.

    Returns:
        ``GateDecision`` with action ``proceed`` | ``wait`` | ``proceed_stale`` | ``proceed_timeout``.
    Raises:
        ValueError: ``now`` is naive.
    """
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")
    window_day = _window_decision_day(now)
    if window_day is not None:
        ts = _parse_heartbeat_time(heartbeat.get("ts") if heartbeat is not None else None)
        if ts is not None:
            age_s = (now.astimezone(UTC) - ts).total_seconds()
            stage = heartbeat.get("stage") if heartbeat is not None else None
            stage_label = str(stage)
            if age_s > stale_after_s:
                return GateDecision("proceed_stale", f"stale:{stage_label} age_s={int(age_s)}")
            if waited_s >= max_wait_s:
                return GateDecision("proceed_timeout", f"max_wait:{stage_label} waited_s={int(waited_s)}")
        elif waited_s >= max_wait_s:
            return GateDecision("proceed_timeout", f"max_wait:unknown waited_s={int(waited_s)}")
        status = heartbeat.get("status") if heartbeat is not None else None
        decision_raw = heartbeat.get("decision_time") if heartbeat is not None else None
        decision_ts = _parse_heartbeat_time(decision_raw)
        if status == "COMPLETE" and decision_ts is not None and decision_ts.replace(hour=0, minute=0, second=0, microsecond=0) == window_day:
            return GateDecision("proceed", "cycle_complete")
        status_label = str(status) if isinstance(status, str) else "unknown"
        return GateDecision("wait", f"decision_window:{status_label}")
    ts_raw = heartbeat.get("ts") if heartbeat is not None else None
    if not isinstance(ts_raw, str):
        return GateDecision("proceed", "no_heartbeat")
    try:
        ts = datetime.fromisoformat(ts_raw)
    except ValueError:
        return GateDecision("proceed", "no_heartbeat")
    if ts.tzinfo is None:
        return GateDecision("proceed", "no_heartbeat")
    stage = heartbeat.get("stage") if heartbeat is not None else None
    if stage not in BUSY_STAGES:
        return GateDecision("proceed", "idle")
    age_s = (now - ts).total_seconds()
    if age_s > stale_after_s:
        return GateDecision("proceed_stale", f"stale:{stage} age_s={int(age_s)}")
    if waited_s >= max_wait_s:
        return GateDecision("proceed_timeout", f"max_wait:{stage} waited_s={int(waited_s)}")
    return GateDecision("wait", f"busy:{stage}")


def main(argv: Sequence[str] | None = None) -> int:
    """Provide the dependency-free deployment gate entry point. Args: existing heartbeat and wait CLI arguments. Returns: 10 for wait, otherwise 0. Raises: existing argument and file-read errors."""
    parser = argparse.ArgumentParser(description="Deploy gate: wait for daemon idle.")
    parser.add_argument("--heartbeat-file", required=True)
    parser.add_argument("--waited-s", type=float, required=True)
    parser.add_argument("--max-wait-s", type=float, default=DEFAULT_MAX_WAIT_S)
    parser.add_argument("--stale-after-s", type=float, default=DEFAULT_STALE_AFTER_S)
    parser.add_argument("--now", default=None)
    args = parser.parse_args(argv)
    raw = Path(args.heartbeat_file).read_text(encoding="utf-8")
    if not raw.strip():
        heartbeat: dict[str, object] | None = None
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            heartbeat = None
        else:
            heartbeat = parsed if isinstance(parsed, dict) else None
    now = datetime.fromisoformat(args.now) if args.now is not None else datetime.now(UTC)
    decision = decide_deploy(
        heartbeat,
        now=now,
        waited_s=args.waited_s,
        max_wait_s=args.max_wait_s,
        stale_after_s=args.stale_after_s,
    )
    print(f"action={decision.action} reason={decision.reason}")  # noqa: T201 - contract-mandated CI output
    return EXIT_WAIT if decision.action == "wait" else EXIT_PROCEED


if __name__ == "__main__":
    raise SystemExit(main())
