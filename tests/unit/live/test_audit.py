"""SCENARIO_LIVE_11: 시크릿은 로그/예외/repr 어디에도 평문으로 나타나지 않는다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import SecretStr

from src.live.audit import (
    AuditLog,
    default_audit_log_path,
    prune_old_audit_logs,
)
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

    import src.live.audit as audit_module

    live_path = default_audit_log_path("shadow_cycle")
    assert live_path.is_relative_to(audit_module.AUDIT_LOG_ROOT)


def test_SCENARIO_LIVE_DAEMON_02_audit_path_date_partitioned() -> None:
    path = default_audit_log_path("orders", for_date=pd.Timestamp("2026-08-24 00:00Z"))
    assert str(path).endswith("logs/live/orders/2026-08-24.jsonl")

    # for_date 생략 시 오늘(UTC) 날짜로 대체되고 예외 없이 반환한다.
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    fallback = default_audit_log_path("orders")
    assert str(fallback).endswith(f"logs/live/orders/{today}.jsonl")

    with pytest.raises(ValueError, match="for_date"):
        default_audit_log_path("orders", for_date=pd.Timestamp("2026-08-24"))


def test_SCENARIO_LIVE_DAEMON_03_prune_deletes_only_stale_dated_files(tmp_path: Path) -> None:
    root = tmp_path / "live"
    orders = root / "orders"
    orders.mkdir(parents=True)
    names = ("2026-01-01.jsonl", "2026-08-01.jsonl", "2026-08-24.jsonl", "not_a_date.jsonl")
    for name in names:
        (orders / name).write_text("", encoding="utf-8")

    removed = prune_old_audit_logs(root, pd.Timestamp("2026-08-24 00:00Z"), retain_days=90)

    assert removed == 1
    assert not (orders / "2026-01-01.jsonl").exists()
    for kept in names[1:]:
        assert (orders / kept).exists()

    # root가 없으면 0을 반환하고 예외 없다(첫 실행).
    missing = tmp_path / "missing"
    assert prune_old_audit_logs(missing, pd.Timestamp("2026-08-24 00:00Z"), retain_days=90) == 0

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_11_NO_SECRETS_IN_LOGS",
    "SCENARIO_LIVE_DAEMON_02_AUDIT_PATH_DATE_PARTITIONED",
    "SCENARIO_LIVE_DAEMON_03_PRUNE_DELETES_ONLY_STALE_DATED_FILES",
)


def test_audit_mirror_is_byte_identical(tmp_path: Path) -> None:
    """Mirror receives the byte-identical sanitized line in order."""
    primary = tmp_path / "primary.jsonl"
    mirror = tmp_path / "mirror" / "run.jsonl"
    audit = AuditLog(primary, mirror_path=mirror)
    audit.record("order_posted", symbol="BTCUSDT", price="100.0")
    audit.record("fill", symbol="BTCUSDT", qty="1", artifact_key="SECRET-VALUE")
    audit.close()
    assert primary.read_bytes() == mirror.read_bytes()
    assert len(primary.read_text(encoding="utf-8").splitlines()) == 2
    for line in (primary.read_bytes() + mirror.read_bytes()).decode().splitlines():
        assert "artifact_key" not in line
        assert "SECRET-VALUE" not in line


def test_audit_mirror_failure_is_non_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A dead mirror warns once; the primary write still lands."""
    import logging

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    primary = tmp_path / "primary.jsonl"
    audit = AuditLog(primary, mirror_path=blocker / "m.jsonl")
    with caplog.at_level(logging.WARNING, logger="src.live.audit"):
        audit.record("order_posted", symbol="BTCUSDT")
        audit.record("fill", symbol="BTCUSDT")
    assert len(primary.read_text(encoding="utf-8").splitlines()) == 2
    warnings = [r for r in caplog.records if "mirror" in r.message]
    assert len(warnings) == 1


def test_audit_mirror_write_failure_is_non_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """An opened mirror that fails on write warns once; primary still lands."""
    import logging

    primary = tmp_path / "primary2.jsonl"
    audit = AuditLog(primary, mirror_path=tmp_path / "mirror2" / "run.jsonl")
    audit.record("order_posted", symbol="BTCUSDT")

    class _BoomWriter:
        closed = False

        def write(self, data: str) -> None:
            raise OSError("disk full")

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    audit._mirror_handle = _BoomWriter()  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING, logger="src.live.audit"):
        audit.record("fill", symbol="BTCUSDT")
        audit.record("fill", symbol="ETHUSDT")
    assert len(primary.read_text(encoding="utf-8").splitlines()) == 3
    assert len([r for r in caplog.records if "mirror" in r.message]) == 1
