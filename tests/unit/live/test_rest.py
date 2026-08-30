# ruff: noqa
"""SCENARIO_LIVE_01 / SCENARIO_LIVE_13: SHADOW 변이 억제와 네트워크 차단."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.live.audit import AUDIT_LOG_ROOT, AuditLog, default_audit_log_path
from src.live.rest import SHADOW_ALLOWED_MUTATIONS, BinanceFuturesRestClient, HttpResponse, parse_rate_limits
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

def test_SCENARIO_LIVE_31_RATE_LIMITS_PARSE_CANONICAL_BINANCE_SCHEMA() -> None:
    """SCENARIO_LIVE_31_RATE_LIMITS_PARSE_CANONICAL_BINANCE_SCHEMA: the parser
    reads the real rateLimitType/interval/intervalNum shape, not the legacy
    filterType/'1m' shorthand it never actually receives from Binance."""
    canonical = {
        "rateLimits": [
            {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
            {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
            {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
        ]
    }
    limits = parse_rate_limits(canonical)
    assert limits.request_weight_1m == 2400
    assert limits.orders_1m == 1200
    assert limits.orders_10s == 300

    legacy_shorthand = {
        "rateLimits": [
            {"filterType": "REQUEST_WEIGHT", "interval": "1m", "limit": 2400},
            {"filterType": "ORDERS", "interval": "1m", "limit": 1200},
            {"filterType": "ORDERS", "interval": "10s", "limit": 300},
        ]
    }
    with pytest.raises(DataIntegrityError):
        parse_rate_limits(legacy_shorthand)

    missing_orders_10s = {
        "rateLimits": [
            {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
            {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
        ]
    }
    with pytest.raises(DataIntegrityError):
        parse_rate_limits(missing_orders_10s)


def test_SCENARIO_REC_09_rest_endpoints_signed(tmp_path: Path) -> None:
    from urllib.parse import urlparse, parse_qs

    captured: list[str] = []

    class StubTransport:
        def call(self, method: str, url: str, headers: dict[str, str]):
            captured.append(url)
            parsed = urlparse(url)
            path = parsed.path
            if path == "/fapi/v1/userTrades":
                assert "fromId=7" in url
                assert "signature=" in url
                return HttpResponse(status_code=200, headers={}, body=b'[{"id":7}]')
            if path == "/fapi/v1/income":
                assert "signature=" in url
                return HttpResponse(status_code=200, headers={}, body=b'[]')
            if path == "/fapi/v1/premiumIndex":
                assert "symbol" not in parsed.query
                return HttpResponse(status_code=200, headers={}, body=b'[{"symbol":"BTCUSDT","markPrice":"100"}]')
            return HttpResponse(status_code=200, headers={}, body=b"{}")

    client = BinanceFuturesRestClient("https://fapi.binance.com", None, None, ExecutionMode.SHADOW, AuditLog(tmp_path / "a.jsonl"), session=StubTransport())
    # Need api secret for signed? Mock without secret may raise, use dummy
    from pydantic import SecretStr

    client._api_secret = SecretStr("secret")
    client._api_key = SecretStr("key")
    # user_trades
    res = client.user_trades("BTCUSDT", from_id=7)
    assert isinstance(res, list)
    # income
    res2 = client.income()
    assert isinstance(res2, list)
    # premium_index
    res3 = client.premium_index()
    assert isinstance(res3, dict)
    assert "BTCUSDT" in res3
    # error cases
    class BadTransport:
        def call(self, method, url, headers):
            parsed = urlparse(url)
            if parsed.path == "/fapi/v1/userTrades":
                return HttpResponse(status_code=200, headers={}, body=b'{"bad":1}')
            if parsed.path == "/fapi/v1/income":
                return HttpResponse(status_code=200, headers={}, body=b'{"bad":1}')
            if parsed.path == "/fapi/v1/premiumIndex":
                return HttpResponse(status_code=200, headers={}, body=b'{"bad":1}')
            return HttpResponse(status_code=200, headers={}, body=b"{}")

    client2 = BinanceFuturesRestClient("https://fapi.binance.com", SecretStr("k"), SecretStr("s"), ExecutionMode.SHADOW, AuditLog(tmp_path / "b.jsonl"), session=BadTransport())
    with pytest.raises(DataIntegrityError):
        client2.user_trades("BTCUSDT")
    with pytest.raises(DataIntegrityError):
        client2.income()
    with pytest.raises(DataIntegrityError):
        client2.premium_index()


def test_depth_endpoint_is_unsigned_and_targets_fapi_v1_depth(tmp_path: Path) -> None:
    from src.common.errors import DataIntegrityError
    from src.live.audit import AuditLog
    from src.live.rest import BinanceFuturesRestClient, HttpResponse
    from src.live.settings import ExecutionMode

    captured: list[str] = []

    class CapTransport:
        def call(self, method: str, url: str, headers: dict[str, str]) -> HttpResponse:
            captured.append(url)
            return HttpResponse(status_code=200, headers={}, body=b'{"lastUpdateId":1,"bids":[["100","1"]],"asks":[["101","1"]]}')

    client = BinanceFuturesRestClient(
        "https://fapi.binance.com", None, None, ExecutionMode.SHADOW, AuditLog(tmp_path / "a.jsonl"), session=CapTransport()
    )
    res = client.depth("BTCUSDT", limit=20)
    assert isinstance(res, dict)
    assert captured[0].__contains__("/fapi/v1/depth")
    assert "symbol=BTCUSDT" in captured[0]
    assert "limit=20" in captured[0]
    assert "signature=" not in captured[0]

    class BadTransport:
        def call(self, method: str, url: str, headers: dict[str, str]) -> HttpResponse:
            return HttpResponse(status_code=200, headers={}, body=b'{"lastUpdateId":1,"bids":[["100","1"]]}')

    client2 = BinanceFuturesRestClient(
        "https://fapi.binance.com", None, None, ExecutionMode.SHADOW, AuditLog(tmp_path / "b.jsonl"), session=BadTransport()
    )
    with pytest.raises(DataIntegrityError, match=".*"):
        client2.depth("BTCUSDT")


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_01_SHADOW_BLOCKS_ALL_MUTATIONS",
    "SCENARIO_LIVE_13_NO_NETWORK_IN_UNIT_TESTS",
    "SCENARIO_LIVE_31_RATE_LIMITS_PARSE_CANONICAL_BINANCE_SCHEMA",
    "SCENARIO_REC_09",
)
# SCENARIO_REC_09-rest-endpoints-signed
