"""Append-only JSONL audit log. 시크릿/서명은 절대 기록하지 않는다."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from src.common.config import BASE_DIR

#: 감사 로그가 허용되는 유일한 프로젝트 하위 루트(외부 /tmp 금지).
AUDIT_LOG_ROOT = BASE_DIR / "logs"

#: 3개월 섬도우 검증 기간을 커버하는 감사로그 보존 일수.
AUDIT_LOG_RETENTION_DAYS: int = 90

_DATE_PARTITION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")

_SECRET_KEYS = frozenset(
    {"signature", "apiKey", "apiSecret", "secret", "api_key", "api_secret", "artifact_key"}
)


def default_audit_log_path(name: str, for_date: pd.Timestamp | None = None) -> Path:
    """decision_time 날짜로 파티셔닝된 감사 로그 경로를 반환한다(무한 증가 방지)."""
    if for_date is None:
        date_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    else:
        ts = pd.Timestamp(for_date)
        if ts.tzinfo is None:
            raise ValueError("for_date must be tz-aware UTC")
        date_str = ts.strftime("%Y-%m-%d")
    return AUDIT_LOG_ROOT / "live" / name / f"{date_str}.jsonl"


def prune_old_audit_logs(
    root: Path,
    reference_date: pd.Timestamp,
    *,
    retain_days: int = AUDIT_LOG_RETENTION_DAYS,
) -> int:
    """retain_days보다 오래된 {YYYY-MM-DD}.jsonl 만 삭제하고 건수를 반환한다."""
    if not root.exists():
        return 0
    cutoff = pd.Timestamp(reference_date).normalize() - pd.Timedelta(days=retain_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    removed = 0
    for name_dir in sorted(root.iterdir()):
        if not name_dir.is_dir():
            continue
        for candidate in name_dir.iterdir():
            if _DATE_PARTITION_RE.match(candidate.name) is None:
                continue
            # ISO 날짜는 사전식 비교가 시간순과 동일하다.
            if not candidate.is_file() or candidate.name[:10] >= cutoff_str:
                continue
            candidate.unlink()
            removed += 1
    return removed


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _sanitize(v) for k, v in value.items() if k not in _SECRET_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


class AuditLog:
    """모든 레코드에 ts/run_id/mode/event를 강제하는 append-only JSONL 로그.

    핫루프 syscall 폭증을 피하기 위해 파일 핸들을 열어두고 재사용한다(매 record
    open/close 금지). 각 레코드는 flush 되므로 크래시 시에도 직전 레코드까지 가시적이다.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context: dict[str, Any] = {}
        self._handle: TextIO | None = None

    def _ensure_handle(self) -> TextIO:
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()

    def record(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            **self.context,
            "event": event,
            **fields,
        }
        clean = {k: _jsonable(v) for k, v in _sanitize(record).items()}
        line = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
        handle = self._ensure_handle()
        handle.write(line + "\n")
        handle.flush()

    def record_suppressed(self, method: str, path: str, payload: dict[str, Any] | None = None) -> None:
        """SHADOW 모드에서 억제된 변이 요청을 기록한다. 원문 대신 다이제스트만 남긴다."""
        from src.live.errors import payload_digest

        self.record(
            "suppressed",
            method=method,
            path=path,
            payload_sha256_12=payload_digest(repr(payload or {})),
        )
