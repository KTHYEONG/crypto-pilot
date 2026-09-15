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
        # Restricted CI sandboxes may deny socket construction itself; that is
        # an equally strong no-network guarantee.
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except PermissionError:
            return
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


def test_mutation_transport_failure_raises_status_unknown_without_resend(tmp_path, monkeypatch) -> None:
    from pydantic import SecretStr
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.rest import BinanceFuturesRestClient, HttpResponse, OrderStatusUnknown
    from src.live.settings import ExecutionMode

    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []
    monkeypatch.setattr("src.live.rest.time.sleep", sleeps.append)

    def _client(responder):
        class _Transport:
            def call(self, method, url, headers):
                calls.append((method, url.split("?")[0]))
                return responder(method, url)

        return BinanceFuturesRestClient(
            "https://fapi.binance.com", SecretStr("k"), SecretStr("s"), ExecutionMode.LIVE_TESTNET,
            AuditLog(tmp_path / "rest_audit.jsonl"), session=_Transport(),
        )

    order_params = {"symbol": "AAAUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "IOC",
                    "quantity": "1", "price": "100", "newClientOrderId": "mh20260914-ABCDEFGHIJ-0-0-0"}

    import pytest

    def _timeout(method, url):
        raise TimeoutError("read timed out")

    client = _client(_timeout)

    with pytest.raises(OrderStatusUnknown) as exc_info:
        client.new_order(order_params)

    assert calls == [("POST", "https://fapi.binance.com/fapi/v1/order")]
    assert exc_info.value.http_status == 0
    assert exc_info.value.path == "/fapi/v1/order"
    assert sleeps == []

import pytest as _pytest


@_pytest.mark.parametrize("status,body", [
    (503, b""),
    (500, b'{"code": -1000, "msg": "unknown"}'),
    (400, b'{"code": -1001, "msg": "disconnected"}'),
    (408, b'{"code": -1007, "msg": "timeout"}'),
])
def test_mutation_unknown_outcomes_raise_status_unknown_without_resend(tmp_path, monkeypatch, status, body) -> None:
    from pydantic import SecretStr
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.rest import BinanceFuturesRestClient, HttpResponse, OrderStatusUnknown
    from src.live.settings import ExecutionMode

    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []
    monkeypatch.setattr("src.live.rest.time.sleep", sleeps.append)

    def _client(responder):
        class _Transport:
            def call(self, method, url, headers):
                calls.append((method, url.split("?")[0]))
                return responder(method, url)

        return BinanceFuturesRestClient(
            "https://fapi.binance.com", SecretStr("k"), SecretStr("s"), ExecutionMode.LIVE_TESTNET,
            AuditLog(tmp_path / "rest_audit.jsonl"), session=_Transport(),
        )

    order_params = {"symbol": "AAAUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "IOC",
                    "quantity": "1", "price": "100", "newClientOrderId": "mh20260914-ABCDEFGHIJ-0-0-0"}

    import pytest

    client = _client(lambda method, url: HttpResponse(status_code=status, headers={}, body=body))

    with pytest.raises(OrderStatusUnknown) as exc_info:
        client.new_order(order_params)
    with pytest.raises(OrderStatusUnknown):
        client.cancel_order("AAAUSDT", "mh20260914-ABCDEFGHIJ-0-0-0")

    assert [method for method, _ in calls] == ["POST", "DELETE"]
    assert exc_info.value.http_status == status
    assert sleeps == []

import pytest as _pytest


@_pytest.mark.parametrize("status,body,code", [
    (429, b"", None),
    (400, b'{"code": -1003, "msg": "too many requests"}', -1003),
])
def test_mutation_rate_limits_are_rejected_without_retry(tmp_path, monkeypatch, status, body, code) -> None:
    from pydantic import SecretStr
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.rest import BinanceFuturesRestClient, HttpResponse, OrderStatusUnknown
    from src.live.settings import ExecutionMode

    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []
    monkeypatch.setattr("src.live.rest.time.sleep", sleeps.append)

    def _client(responder):
        class _Transport:
            def call(self, method, url, headers):
                calls.append((method, url.split("?")[0]))
                return responder(method, url)

        return BinanceFuturesRestClient(
            "https://fapi.binance.com", SecretStr("k"), SecretStr("s"), ExecutionMode.LIVE_TESTNET,
            AuditLog(tmp_path / "rest_audit.jsonl"), session=_Transport(),
        )

    order_params = {"symbol": "AAAUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "IOC",
                    "quantity": "1", "price": "100", "newClientOrderId": "mh20260914-ABCDEFGHIJ-0-0-0"}

    import pytest

    client = _client(lambda method, url: HttpResponse(status_code=status, headers={}, body=body))

    with pytest.raises(VenueError) as exc_info:
        client.new_order(order_params)

    assert len(calls) == 1
    assert exc_info.value.code == code
    assert exc_info.value.http_status == status
    assert sleeps == []

def test_get_requests_keep_retry_backoff(tmp_path, monkeypatch) -> None:
    from pydantic import SecretStr
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.rest import BinanceFuturesRestClient, HttpResponse, OrderStatusUnknown
    from src.live.settings import ExecutionMode

    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []
    monkeypatch.setattr("src.live.rest.time.sleep", sleeps.append)

    def _client(responder):
        class _Transport:
            def call(self, method, url, headers):
                calls.append((method, url.split("?")[0]))
                return responder(method, url)

        return BinanceFuturesRestClient(
            "https://fapi.binance.com", SecretStr("k"), SecretStr("s"), ExecutionMode.LIVE_TESTNET,
            AuditLog(tmp_path / "rest_audit.jsonl"), session=_Transport(),
        )

    order_params = {"symbol": "AAAUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "IOC",
                    "quantity": "1", "price": "100", "newClientOrderId": "mh20260914-ABCDEFGHIJ-0-0-0"}

    responses = [
        HttpResponse(status_code=500, headers={}, body=b'{"code": -1000, "msg": "unknown"}'),
        HttpResponse(status_code=200, headers={}, body=b'{"status": "NEW", "executedQty": "0"}'),
    ]
    client = _client(lambda method, url: responses.pop(0))

    payload = client.query_order("AAAUSDT", "mh20260914-ABCDEFGHIJ-0-0-0")

    assert payload["status"] == "NEW"
    assert [method for method, _ in calls] == ["GET", "GET"]
    assert sleeps == [1.0]

def test_mutation_clock_resync_is_the_only_resend(tmp_path, monkeypatch) -> None:
    from pydantic import SecretStr
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.rest import BinanceFuturesRestClient, HttpResponse, OrderStatusUnknown
    from src.live.settings import ExecutionMode

    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []
    monkeypatch.setattr("src.live.rest.time.sleep", sleeps.append)

    def _client(responder):
        class _Transport:
            def call(self, method, url, headers):
                calls.append((method, url.split("?")[0]))
                return responder(method, url)

        return BinanceFuturesRestClient(
            "https://fapi.binance.com", SecretStr("k"), SecretStr("s"), ExecutionMode.LIVE_TESTNET,
            AuditLog(tmp_path / "rest_audit.jsonl"), session=_Transport(),
        )

    order_params = {"symbol": "AAAUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "IOC",
                    "quantity": "1", "price": "100", "newClientOrderId": "mh20260914-ABCDEFGHIJ-0-0-0"}

    responses = [
        HttpResponse(status_code=400, headers={}, body=b'{"code": -1021, "msg": "timestamp outside recvWindow"}'),
        HttpResponse(status_code=200, headers={}, body=b'{"serverTime": 1}'),
        HttpResponse(status_code=200, headers={}, body=b'{"orderId": 7}'),
    ]
    client = _client(lambda method, url: responses.pop(0))

    payload = client.new_order(order_params)

    assert payload == {"orderId": 7}
    assert [method for method, _ in calls] == ["POST", "GET", "POST"]
    assert client.mode is ExecutionMode.LIVE_TESTNET

def test_keepalive_transport_sends_mutations_once_on_fresh_connection(monkeypatch) -> None:
    import http.client
    import pytest
    from src.live.rest import KeepAliveTransport

    created: list[object] = []
    requests: list[str] = []

    class _Conn:
        def __init__(self, host, port, timeout=None):
            created.append(self)

        def request(self, method, path, body=None, headers=None):
            requests.append(method)

        def getresponse(self):
            raise TimeoutError("read timed out")

        def close(self):
            return None

    monkeypatch.setattr(http.client, "HTTPSConnection", _Conn)
    transport = KeepAliveTransport("https://fapi.binance.com")

    with pytest.raises(TimeoutError):
        transport.call("POST", "https://fapi.binance.com/fapi/v1/order?x=1", {})
    assert requests == ["POST"]
    assert len(created) == 1

    with pytest.raises(TimeoutError):
        transport.call("GET", "https://fapi.binance.com/fapi/v1/order?x=1", {})
    assert requests == ["POST", "GET", "GET"]

def test_get_transport_failure_propagates_unchanged(tmp_path, monkeypatch) -> None:
    from pydantic import SecretStr
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.rest import BinanceFuturesRestClient, HttpResponse, OrderStatusUnknown
    from src.live.settings import ExecutionMode

    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []
    monkeypatch.setattr("src.live.rest.time.sleep", sleeps.append)

    def _client(responder):
        class _Transport:
            def call(self, method, url, headers):
                calls.append((method, url.split("?")[0]))
                return responder(method, url)

        return BinanceFuturesRestClient(
            "https://fapi.binance.com", SecretStr("k"), SecretStr("s"), ExecutionMode.LIVE_TESTNET,
            AuditLog(tmp_path / "rest_audit.jsonl"), session=_Transport(),
        )

    order_params = {"symbol": "AAAUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "IOC",
                    "quantity": "1", "price": "100", "newClientOrderId": "mh20260914-ABCDEFGHIJ-0-0-0"}

    import pytest

    def _reset(method, url):
        raise ConnectionResetError("peer reset")

    client = _client(_reset)

    with pytest.raises(ConnectionResetError):
        client.query_order("AAAUSDT", "mh20260914-ABCDEFGHIJ-0-0-0")

    assert calls == [("GET", "https://fapi.binance.com/fapi/v1/order")]




# --- halt_reason_persistence contract: new scenarios ---

def test_signed_query_round_trips_non_ascii_symbol(tmp_path: Path) -> None:
    import urllib.parse

    from pydantic import SecretStr

    # Given: a real Binance-listed non-ASCII perpetual (Chinese meme coin) held live
    client = BinanceFuturesRestClient(
        "https://fapi.binance.com", SecretStr("k"), SecretStr("s"),
        ExecutionMode.SHADOW, AuditLog(tmp_path / "a.jsonl"),
    )

    # When
    signed = client._signed_query({"symbol": "\u9f99\u867eUSDT", "side": "BUY"})

    # Then: percent-encoded UTF-8 round-trips exactly and a signature is appended
    query_part, _, sig_part = signed.rpartition("&signature=")
    assert len(sig_part) == 64
    parsed = dict(urllib.parse.parse_qsl(query_part))
    assert parsed["symbol"] == "\u9f99\u867eUSDT"
