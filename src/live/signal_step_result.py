"""Cross-process result sidecar between the signal-step subprocess and the daemon (observability only)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger("LiveSignalStepResult")

SIGNAL_STEP_RESULT_NAME: str = "signal_step_result.json"
SIGNAL_STEP_REASON_MAX_CHARS: int = 500
SIGNAL_STEP_STATUS_OK: str = "OK"
SIGNAL_STEP_STATUS_FAILED: str = "FAILED"


@dataclass(frozen=True, slots=True)
class SignalStepResult:
    decision_time: pd.Timestamp
    status: str
    error_type: str = ""
    reason: str = ""
    quarantine: tuple[tuple[str, str], ...] = ()


def signal_step_result_path(weights_path: Path) -> Path:
    return Path(weights_path).parent / SIGNAL_STEP_RESULT_NAME


def write_signal_step_result(path: Path, result: SignalStepResult) -> None:
    if result.decision_time.tzinfo is None:
        raise ValueError("signal step result decision_time must be tz-aware")
    if result.status not in (SIGNAL_STEP_STATUS_OK, SIGNAL_STEP_STATUS_FAILED):
        raise ValueError(f"signal step result status must be OK or FAILED, got {result.status!r}")
    # 낡은 사이드카 오인 방지용 UTC 정규화
    payload = {
        "decision_time": pd.Timestamp(result.decision_time).tz_convert("UTC").isoformat(),
        "status": result.status,
        "error_type": result.error_type,
        "reason": result.reason[:SIGNAL_STEP_REASON_MAX_CHARS],
        "quarantine": [{"symbol": s, "reason": r} for s, r in result.quarantine],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


def read_signal_step_result(path: Path, decision_time: pd.Timestamp) -> SignalStepResult | None:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        if pd.Timestamp(raw["decision_time"]) != pd.Timestamp(decision_time):
            return None
        # 결정일 일치 시에만 현 시도의 원인으로 사용
        quarantine = tuple((str(r["symbol"]), str(r["reason"])) for r in raw.get("quarantine", []))
        return SignalStepResult(
            decision_time=pd.Timestamp(decision_time),
            status=str(raw["status"]),
            error_type=str(raw.get("error_type", "")),
            reason=str(raw.get("reason", "")),
            quarantine=quarantine,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("[SYS] signal_step_result unreadable path=%s error=%s", target, exc)
        return None


def load_quarantine_records(sidecar_path: Path, decision_time: pd.Timestamp) -> tuple[tuple[str, str], ...]:
    target = Path(sidecar_path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return ()
        if pd.Timestamp(raw["decision_time"]) != pd.Timestamp(decision_time):
            return ()
        # write_quarantine_sidecar 포맷 판독
        return tuple((str(r["symbol"]), str(r["reason"])) for r in raw.get("records", []))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("[SYS] signal_step_result unreadable path=%s error=%s", target, exc)
        return ()
