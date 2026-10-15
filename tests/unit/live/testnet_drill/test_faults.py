"""Fault-injecting transport invariants."""

from __future__ import annotations

import pytest

from src.live.rest import HttpResponse
from src.live.testnet_drill.faults import FaultAction, FaultInjectingTransport, FaultRule
from src.live.testnet_drill.guard import DrillRefused
from tests.unit.live.testnet_drill._stubs import ScriptedTransport


class KillFiredError(Exception):
    """Raised by the injected kill stub instead of exiting the test process."""


def _time_transport() -> ScriptedTransport:
    return ScriptedTransport([("GET", "/fapi/v1/time", lambda *a: {"serverTime": 1700000000000})])


def _make_transport(inner: ScriptedTransport, rules: list[FaultRule], slept: list[float]) -> FaultInjectingTransport:
    def _kill(code: int) -> None:
        raise KillFiredError(f"exit={code}")

    return FaultInjectingTransport(
        inner,
        rules,
        base_url="https://testnet.binancefuture.com",
        sleep=slept.append,
        kill=_kill,  # type: ignore[arg-type]
    )


def test_empty_rules_are_transparent() -> None:
    inner = _time_transport()
    transport = _make_transport(inner, [], [])
    first = transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    second = transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    assert first.body == second.body
    assert b"serverTime" in first.body
    assert len(inner.calls) == 2
    assert len(transport.records) == 2
    assert all(r.delivered for r in transport.records)
    assert [r.seq for r in transport.records] == [1, 2]


def test_deliver_drop_reaches_venue_exactly_once() -> None:
    inner = ScriptedTransport([("POST", "/fapi/v1/order", lambda *a: {"orderId": 7})])
    transport = _make_transport(inner, [FaultRule(method="POST", path="/fapi/v1/order", action=FaultAction.DELIVER_DROP_RESPONSE)], [])
    with pytest.raises(TimeoutError, match="deliver_drop_response"):
        transport.call("POST", "https://testnet.binancefuture.com/fapi/v1/order?symbol=BTCUSDT", {})
    assert len(inner.calls) == 1
    (record,) = transport.records
    assert record.delivered is True
    assert b"7" in record.response_body


def test_drop_request_never_reaches_venue() -> None:
    inner = _time_transport()
    transport = _make_transport(inner, [FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.DROP_REQUEST)], [])
    with pytest.raises(TimeoutError, match="drop_request"):
        transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    assert inner.calls == []
    (record,) = transport.records
    assert record.delivered is False
    assert record.status_code is None


def test_synthetic_status_never_forwards() -> None:
    inner = _time_transport()
    rule = FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.SYNTHETIC_STATUS, status=418, body=b"{}", headers=(("Retry-After", "30"),))
    transport = _make_transport(inner, [rule], [])
    response = transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    assert inner.calls == []
    assert response.status_code == 418
    assert response.body == b"{}"
    (record,) = transport.records
    assert record.delivered is False
    assert record.status_code == 418


def test_occurrence_targeting() -> None:
    inner = ScriptedTransport([("DELETE", "/fapi/v1/order", lambda *a: {"status": "CANCELED"})])
    rule = FaultRule(method="DELETE", path="/fapi/v1/order", occurrence=2, action=FaultAction.DROP_REQUEST)
    transport = _make_transport(inner, [rule], [])
    url = "https://testnet.binancefuture.com/fapi/v1/order?symbol=BTCUSDT"
    transport.call("DELETE", url, {})
    with pytest.raises(TimeoutError, match="drop_request"):
        transport.call("DELETE", url, {})
    transport.call("DELETE", url, {})
    assert len(inner.calls) == 2


def test_rule_exhaustion() -> None:
    inner = ScriptedTransport([("DELETE", "/fapi/v1/order", lambda *a: {"status": "CANCELED"})])
    rule = FaultRule(method="DELETE", path="/fapi/v1/order", action=FaultAction.DROP_REQUEST, times=1)
    transport = _make_transport(inner, [rule], [])
    url = "https://testnet.binancefuture.com/fapi/v1/order?symbol=BTCUSDT"
    with pytest.raises(TimeoutError, match="drop_request"):
        transport.call("DELETE", url, {})
    transport.call("DELETE", url, {})
    assert len(inner.calls) == 1


def test_signature_stripped_from_records() -> None:
    inner = _time_transport()
    transport = _make_transport(inner, [], [])
    transport.call(
        "GET",
        "https://testnet.binancefuture.com/fapi/v1/time?timestamp=123&signature=abc&recvWindow=5000",
        {"X-MBX-APIKEY": "secret-key", "Content-Type": "application/json"},
    )
    (record,) = transport.records
    assert "signature=" not in record.url
    assert "timestamp=123" in record.url
    assert all(key.lower() != "x-mbx-apikey" for key, _ in record.request_headers)


def test_refuses_non_testnet_base_url() -> None:
    with pytest.raises(DrillRefused, match="base_url"):
        FaultInjectingTransport(_time_transport(), [], base_url="https://fapi.binance.com", sleep=lambda s: None, kill=lambda c: None)  # type: ignore[arg-type]


def test_kill_hook_is_injected() -> None:
    inner = ScriptedTransport([("POST", "/fapi/v1/order", lambda *a: {"orderId": 7})])
    transport = _make_transport(inner, [FaultRule(method="POST", path="/fapi/v1/order", action=FaultAction.KILL_AFTER_DELIVERY, exit_code=137)], [])
    with pytest.raises(KillFiredError, match="exit=137"):
        transport.call("POST", "https://testnet.binancefuture.com/fapi/v1/order?symbol=BTCUSDT", {})
    assert len(inner.calls) == 1
    (record,) = transport.records
    assert record.delivered is True


def test_kill_before_delivery_skips_venue() -> None:
    inner = _time_transport()
    transport = _make_transport(inner, [FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.KILL_BEFORE_DELIVERY)], [])
    with pytest.raises(KillFiredError):
        transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    assert inner.calls == []


def test_returning_kill_stub_fails_closed() -> None:
    inner = ScriptedTransport([("POST", "/fapi/v1/order", lambda *a: {"orderId": 7})])
    transport = FaultInjectingTransport(
        inner,
        [FaultRule(method="POST", path="/fapi/v1/order", action=FaultAction.KILL_AFTER_DELIVERY)],
        base_url="https://testnet.binancefuture.com",
        sleep=lambda s: None,
        kill=lambda code: None,  # type: ignore[arg-type]
    )
    with pytest.raises(AssertionError, match="unreachable"):
        transport.call("POST", "https://testnet.binancefuture.com/fapi/v1/order?symbol=BTCUSDT", {})
    before_transport = FaultInjectingTransport(
        _time_transport(),
        [FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.KILL_BEFORE_DELIVERY)],
        base_url="https://testnet.binancefuture.com",
        sleep=lambda s: None,
        kill=lambda code: None,  # type: ignore[arg-type]
    )
    with pytest.raises(AssertionError, match="unreachable"):
        before_transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})


def test_delay_keeps_original_timestamp() -> None:
    inner = _time_transport()
    slept: list[float] = []
    transport = _make_transport(inner, [FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.DELAY_DELIVER, delay_s=5.0)], slept)
    url = "https://testnet.binancefuture.com/fapi/v1/time?timestamp=123"
    transport.call("GET", url, {})
    assert slept == [5.0]
    assert inner.calls[0][1] == url


def test_corrupt_and_truncate_bodies() -> None:
    inner = ScriptedTransport([("GET", "/fapi/v1/time", lambda *a: HttpResponse(200, {}, b'{"serverTime":1}')), ("GET", "/fapi/v1/depth", lambda *a: HttpResponse(200, {}, b"0123456789"))])
    transport = _make_transport(
        inner,
        [
            FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.CORRUPT_BODY, body=b"corrupt"),
            FaultRule(method="GET", path="/fapi/v1/depth", action=FaultAction.TRUNCATE_BODY, truncate_to=4),
        ],
        [],
    )
    corrupted = transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    truncated = transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/depth", {})
    assert corrupted.body == b"corrupt"
    assert truncated.body == b"0123"
    assert transport.records[0].response_body == b'{"serverTime":1}'
    assert all(r.delivered for r in transport.records)


def test_explicit_pass_rule_forwards() -> None:
    inner = _time_transport()
    transport = _make_transport(inner, [FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.PASS)], [])
    response = transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    assert len(inner.calls) == 1
    assert b"serverTime" in response.body


def test_inner_exception_is_recorded_and_reraised() -> None:
    class _Broken:
        def call(self, method: str, url: str, headers: object) -> HttpResponse:
            raise OSError("down")

    transport = _make_transport(ScriptedTransport(), [], [])
    transport._inner = _Broken()
    with pytest.raises(OSError, match="down"):
        transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    (record,) = transport.records
    assert record.error == "OSError"
    assert record.delivered is None


def test_predicate_and_query_matching() -> None:
    inner = ScriptedTransport([("DELETE", "/fapi/v1/order", lambda *a: {"status": "CANCELED"})])
    rule = FaultRule(method="DELETE", path="/fapi/v1/order", params=(("symbol", "BTCUSDT"),), action=FaultAction.DROP_REQUEST, times=1)
    decoy = FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.DROP_REQUEST)
    transport = _make_transport(inner, [decoy, rule], [])
    transport.call("DELETE", "https://testnet.binancefuture.com/fapi/v1/order?symbol=ETHUSDT", {})
    with pytest.raises(TimeoutError, match="drop_request"):
        transport.call("DELETE", "https://testnet.binancefuture.com/fapi/v1/order?symbol=BTCUSDT", {})
    assert len(inner.calls) == 1


def test_rule_validation() -> None:
    with pytest.raises(ValueError, match="method"):
        FaultRule(method="", path="/fapi/v1/time")
    with pytest.raises(ValueError, match="signature"):
        FaultRule(method="GET", path="/fapi/v1/time", params=(("signature", "x"),))
    with pytest.raises(ValueError, match="1-based"):
        FaultRule(method="GET", path="/fapi/v1/time", occurrence=0)
    with pytest.raises(ValueError, match="times"):
        FaultRule(method="GET", path="/fapi/v1/time", times=0)
    with pytest.raises(ValueError, match="delay_s"):
        FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.DELAY_DELIVER, delay_s=-1.0)
    with pytest.raises(ValueError, match="truncate_to"):
        FaultRule(method="GET", path="/fapi/v1/time", action=FaultAction.TRUNCATE_BODY, truncate_to=-1)


def test_close_delegates() -> None:
    inner = _time_transport()
    transport = _make_transport(inner, [], [])
    transport.close()
    assert inner.closed is True


def test_request_host_cannot_escape_testnet() -> None:
    inner = _time_transport()
    transport = _make_transport(inner, [], [])
    with pytest.raises(DrillRefused, match="request_host"):
        transport.call("GET", "https://fapi.binance.com/fapi/v1/time", {})
    assert not inner.calls


def test_faulted_inner_exception_still_records_unknown_delivery() -> None:
    class Broken:
        def call(self, method: str, url: str, headers: object) -> HttpResponse:
            raise TimeoutError("unknown delivery")

    transport = _make_transport(ScriptedTransport(), [FaultRule("GET", "/fapi/v1/time", action=FaultAction.DELIVER_DROP_RESPONSE)], [])
    transport._inner = Broken()
    with pytest.raises(TimeoutError):
        transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {})
    assert transport.records[0].delivered is None


def test_recorded_body_is_redacted_without_changing_response() -> None:
    body = b'{"signature":"abc", "apiSecret":"secret-value", "echo":"key-value"}'
    inner = ScriptedTransport([("GET", "/fapi/v1/time", lambda *a: HttpResponse(200, {"X-MBX-APIKEY": "key-value", "Content-Type": "application/json"}, body))])
    transport = _make_transport(inner, [], [])
    response = transport.call("GET", "https://testnet.binancefuture.com/fapi/v1/time", {"X-MBX-APIKEY": "key-value"})
    assert response.body is body
    record = transport.records[0]
    assert b"secret-value" not in record.response_body
    assert b"key-value" not in record.response_body
    assert b'"abc"' not in record.response_body
    assert "key-value" not in str(record.response_headers)
