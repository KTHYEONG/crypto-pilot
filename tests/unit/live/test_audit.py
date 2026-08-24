"""SCENARIO_LIVE_11: 시크릿은 로그/예외/repr 어디에도 평문으로 나타나지 않는다."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import SecretStr

from src.live.audit import AUDIT_LOG_ROOT, AuditLog, default_audit_log_path
from src.live.errors import VenueError
from src.live.settings import LiveSettings

SECRET = "SUPERSECRET123"  # noqa: S105 - fixture value, not a real credential


def _settings() -> LiveSettings:
    return LiveSettings(
        api_key=SecretStr("MYKEY"),
        api_secret=SecretStr(SECRET),
    )


def test_SCENARIO_LIVE_11_no_secrets_in_logs(tmp_path: Path) -> None:
    settings = _settings()
    rendered = f"{settings!r} {settings!s} {settings.model_dump_json()}"
    assert SECRET not in rendered
    assert "MYKEY" not in rendered

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)
    payload = {"symbol": "BTCUSDT", "signature": "deadbeef", "note": SECRET}
    audit.record_suppressed("POST", "/fapi/v1/order", payload)

    line = audit_path.read_text(encoding="utf-8")
    assert "signature" not in line
    assert SECRET not in line
    record = json.loads(line)
    assert record["event"] == "suppressed"

    error = VenueError(
        "rejected",
        code=-2010,
        http_status=400,
        path="/fapi/v1/order",
        payload_digest="abc123def456",
    )
    rendered_error = str(error)
    assert "abc123def456" in rendered_error
    assert SECRET not in rendered_error
    assert len(error.payload_digest) == 12

    live_path = default_audit_log_path("shadow_cycle")
    assert live_path.is_relative_to(AUDIT_LOG_ROOT)

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_11_NO_SECRETS_IN_LOGS",
)
