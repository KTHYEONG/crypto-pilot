"""SCENARIO_LIVE_01 / SCENARIO_LIVE_13: SHADOW 변이 억제와 네트워크 차단."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from src.live.audit import AUDIT_LOG_ROOT, AuditLog, default_audit_log_path
from src.live.rest import SHADOW_ALLOWED_MUTATIONS, BinanceFuturesRestClient, HttpResponse
from src.live.settings import ExecutionMode


class StubTransport:
    def __init__(self) -> None:
        self.call_count = 0

    def call(self, method: str, url: str, headers: dict[str, str]) -> HttpResponse:
        self.call_count += 1
        return HttpResponse(status_code=200, headers={}, body=b"{}")


def _shadow_client(tmp_path: Path, transport: StubTransport) -> BinanceFuturesRestClient:
    return BinanceFuturesRestClient(
        "https://fapi.binance.com",
        None,
        None,
        ExecutionMode.SHADOW,
        AuditLog(tmp_path / "audit.jsonl"),
        session=transport,
    )


def _audit_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.usefixtures("_block_network")
class TestShadowChoke:
    def test_SCENARIO_LIVE_01_shadow_blocks_all_mutations(self, tmp_path: Path) -> None:
        transport = StubTransport()
        audit_path = tmp_path / "audit.jsonl"
        client = BinanceFuturesRestClient(
            "https://fapi.binance.com",
            None,
            None,
            ExecutionMode.SHADOW,
            AuditLog(audit_path),
            session=transport,
        )

        suppressed_calls = [
            ("POST", "/fapi/v1/order"),
            ("POST", "/fapi/v1/leverage"),
            ("POST", "/fapi/v1/marginType"),
            ("DELETE", "/fapi/v1/order"),
        ]
        for method, path in suppressed_calls:
            response = client.request(method, path, {"symbol": "BTCUSDT"})
            assert response.status == "suppressed"
            assert response.method == method
            assert response.path == path

        assert transport.call_count == 0

        events = _audit_events(audit_path)
        assert [e["event"] for e in events] == ["suppressed"] * 4

        client.request("GET", "/fapi/v1/exchangeInfo")
        assert transport.call_count == 1

        assert frozenset() == SHADOW_ALLOWED_MUTATIONS

    def test_SCENARIO_LIVE_13_no_network_in_unit_tests(self) -> None:
        # conftest의 autouse 픽스처가 socket.connect를 예외 스텁으로 대체했다.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(AssertionError):
            sock.connect(("fapi.binance.com", 443))
        sock.close()

    def test_audit_log_default_path_is_under_project_logs(self) -> None:
        path = default_audit_log_path("shadow_cycle")
        assert path.is_relative_to(AUDIT_LOG_ROOT)
        assert "logs" in path.parts

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_01_SHADOW_BLOCKS_ALL_MUTATIONS",
    "SCENARIO_LIVE_13_NO_NETWORK_IN_UNIT_TESTS",
)
