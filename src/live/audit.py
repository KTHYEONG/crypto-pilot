"""Append-only JSONL audit log. 시크릿/서명은 절대 기록하지 않는다."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.config import BASE_DIR

#: 감사 로그가 허용되는 유일한 프로젝트 하위 루트(외부 /tmp 금지).
AUDIT_LOG_ROOT = BASE_DIR / "logs"

_SECRET_KEYS = frozenset({"signature", "apiKey", "apiSecret", "secret", "api_key", "api_secret"})


def default_audit_log_path(name: str) -> Path:
    """프로젝트 logs/ 하위의 감사 로그 경로를 반환한다."""
    return AUDIT_LOG_ROOT / "live" / f"{name}.jsonl"


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
    """모든 레코드에 ts/run_id/mode/event를 강제하는 append-only JSONL 로그."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context: dict[str, Any] = {}

    def record(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            **self.context,
            "event": event,
            **fields,
        }
        clean = {k: _jsonable(v) for k, v in _sanitize(record).items()}
        line = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def record_suppressed(self, method: str, path: str, payload: dict[str, Any] | None = None) -> None:
        """SHADOW 모드에서 억제된 변이 요청을 기록한다. 원문 대신 다이제스트만 남긴다."""
        from src.live.errors import payload_digest

        self.record(
            "suppressed",
            method=method,
            path=path,
            payload_sha256_12=payload_digest(repr(payload or {})),
        )
