"""Hermetic contract tests for the native Binance forceOrder WebSocket feed.

All scenarios run against an in-process ``aiohttp`` WebSocket endpoint; no network is used.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import TestServer

from src.market_data.streams.liquidations import BinanceForceOrderFeed

_FORCE_ORDER = {"e": "forceOrder", "E": 1758531600000, "o": {"s": "BTCUSDT", "T": 1758531600000}}


class _StubWS:
    """Minimal WebSocket double driving feed error paths without a server."""

    def __init__(self) -> None:
        self.ping_exc: BaseException | None = None
        self.pong_exc: BaseException | None = None
        self.receive_exc: BaseException | None = None
        self.incoming: list[Any] = []
        self.closed = 0

    async def ping(self, *args: Any, **kwargs: Any) -> None:
        if self.ping_exc is not None:
            raise self.ping_exc

    async def pong(self, *args: Any, **kwargs: Any) -> None:
        if self.pong_exc is not None:
            raise self.pong_exc

    async def receive(self, *args: Any, **kwargs: Any) -> Any:
        if self.receive_exc is not None:
            raise self.receive_exc
        return self.incoming.pop(0)

    async def close(self) -> None:
        self.closed += 1


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


async def _run_with_server(handler) -> tuple[TestServer, ClientSession, str]:  # type: ignore[no-untyped-def]
    app = web.Application()
    app.router.add_get("/ws", handler)
    server = TestServer(app)
    await server.start_server()
    session = ClientSession()
    url = str(server.make_url("/ws")).replace("http://", "ws://")
    return server, session, url


def test_feed_text_frame_decodes_to_events() -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await ws.send_str(json.dumps(_FORCE_ORDER))
            await ws.receive()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            frame = await feed.receive(2.0)
            assert frame.kind == "events"
            assert frame.received_at is not None
            assert frame.received_at.tzinfo is not None
            assert frame.payloads == (_FORCE_ORDER,)
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())


def test_feed_client_ping_produces_alive_frame() -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            async for msg in ws:
                if msg.type == WSMsgType.PING:
                    await ws.pong(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=0.05, now_fn=_now)
            await asyncio.sleep(0.1)
            frame = await feed.receive(2.0)
            assert frame.kind == "alive"
            assert frame.received_at is not None
            assert frame.received_at.tzinfo is not None
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())


def test_feed_server_ping_is_answered() -> None:
    async def _scenario() -> None:
        pinged = asyncio.Event()

        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await asyncio.sleep(0.05)
            await ws.ping(b"x")
            pinged.set()
            await ws.receive()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            frame = await feed.receive(2.0)
            assert frame.kind == "alive"
            assert frame.received_at is not None
            assert pinged.is_set()
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())


def test_feed_server_close_reports_closed_and_close_is_idempotent() -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await asyncio.sleep(0.05)
            await ws.close()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            frame = await feed.receive(2.0)
            assert frame.kind == "closed"
            assert frame.received_at is None
            await feed.close()
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())


def test_feed_silence_reports_timeout() -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await asyncio.sleep(5.0)
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            frame = await feed.receive(0.05)
            assert frame.kind == "timeout"
            assert frame.received_at is None
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())


def test_feed_undecodable_text_frame_counts_as_alive(caplog: pytest.LogCaptureFixture) -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await ws.send_str("not json")
            await ws.receive()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            with caplog.at_level(logging.WARNING, logger="src.market_data.streams.liquidations"):
                frame = await feed.receive(2.0)
            assert frame.kind == "alive"
            assert frame.received_at is not None
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())
    assert any("BAD_FRAME" in record.message for record in caplog.records)


def test_feed_list_frame_decodes_to_ordered_events() -> None:
    async def _scenario() -> None:
        first = dict(_FORCE_ORDER)
        second = {"e": "forceOrder", "E": 1758531660000, "o": {"s": "ETHUSDT", "T": 1758531660000}}

        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await ws.send_str(json.dumps([first, second]))
            await ws.receive()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            frame = await feed.receive(2.0)
            assert frame.kind == "events"
            assert frame.payloads == (first, second)
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())


def test_feed_mixed_list_drops_non_mappings(caplog: pytest.LogCaptureFixture) -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await ws.send_str(json.dumps(["junk", 42, _FORCE_ORDER]))
            await ws.receive()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            with caplog.at_level(logging.WARNING, logger="src.market_data.streams.liquidations"):
                frame = await feed.receive(2.0)
            assert frame.kind == "events"
            assert frame.payloads == (_FORCE_ORDER,)
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())
    assert any("BAD_FRAME" in record.message for record in caplog.records)


def test_feed_empty_list_and_scalar_count_as_alive(caplog: pytest.LogCaptureFixture) -> None:
    async def _scenario() -> list[str]:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await ws.send_str("[]")
            await ws.send_str("42")
            await ws.receive()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            with caplog.at_level(logging.WARNING, logger="src.market_data.streams.liquidations"):
                first = await feed.receive(2.0)
                second = await feed.receive(2.0)
            await feed.close()
            return [first.kind, second.kind]
        finally:
            await session.close()
            await server.close()

    assert asyncio.run(_scenario()) == ["alive", "alive"]
    assert any("BAD_FRAME" in record.message for record in caplog.records)


def test_feed_binary_frame_decodes_to_events() -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await ws.send_bytes(json.dumps(_FORCE_ORDER).encode())
            await ws.receive()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            frame = await feed.receive(2.0)
            assert frame.kind == "events"
            assert frame.payloads == (_FORCE_ORDER,)
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())


def test_feed_garbled_binary_counts_as_alive(caplog: pytest.LogCaptureFixture) -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await ws.send_bytes(b"\xff\xfe\x00bad")
            await ws.receive()
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            with caplog.at_level(logging.WARNING, logger="src.market_data.streams.liquidations"):
                frame = await feed.receive(2.0)
            assert frame.kind == "alive"
            assert frame.received_at is not None
            await feed.close()
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())
    assert any("BAD_FRAME" in record.message for record in caplog.records)


def test_feed_receive_after_close_reports_closed() -> None:
    async def _scenario() -> None:
        async def _handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(autoping=False)
            await ws.prepare(request)
            await asyncio.sleep(5.0)
            return ws

        server, session, url = await _run_with_server(_handler)
        try:
            feed = await BinanceForceOrderFeed.connect(session, url=url, ping_interval_s=60.0, now_fn=_now)
            await feed.close()
            frame = await feed.receive(0.05)
            assert frame.kind == "closed"
            assert frame.received_at is None
        finally:
            await session.close()
            await server.close()

    asyncio.run(_scenario())


def test_feed_ping_failure_reports_closed() -> None:
    ws = _StubWS()
    ws.ping_exc = RuntimeError("boom")
    feed = BinanceForceOrderFeed(ws, ping_interval_s=0.0, now_fn=_now)  # type: ignore[arg-type]
    frame = asyncio.run(feed.receive(0.5))
    assert frame.kind == "closed"
    assert frame.received_at is None


def test_feed_ping_cancel_propagates() -> None:
    ws = _StubWS()
    ws.ping_exc = asyncio.CancelledError()
    feed = BinanceForceOrderFeed(ws, ping_interval_s=0.0, now_fn=_now)  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(feed.receive(0.5))


def test_feed_transport_error_reports_closed() -> None:
    ws = _StubWS()
    ws.receive_exc = RuntimeError("lost")
    feed = BinanceForceOrderFeed(ws, ping_interval_s=60.0, now_fn=_now)  # type: ignore[arg-type]
    frame = asyncio.run(feed.receive(0.5))
    assert frame.kind == "closed"
    assert frame.received_at is None


def test_feed_receive_cancel_propagates() -> None:
    ws = _StubWS()
    ws.receive_exc = asyncio.CancelledError()
    feed = BinanceForceOrderFeed(ws, ping_interval_s=60.0, now_fn=_now)  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(feed.receive(0.5))


def test_feed_pong_failure_reports_closed() -> None:
    ws = _StubWS()
    ws.pong_exc = RuntimeError("boom")
    ws.incoming.append(SimpleNamespace(type=WSMsgType.PING, data=b"x"))
    feed = BinanceForceOrderFeed(ws, ping_interval_s=60.0, now_fn=_now)  # type: ignore[arg-type]
    frame = asyncio.run(feed.receive(0.5))
    assert frame.kind == "closed"


def test_feed_pong_cancel_propagates() -> None:
    ws = _StubWS()
    ws.pong_exc = asyncio.CancelledError()
    ws.incoming.append(SimpleNamespace(type=WSMsgType.PING, data=b"x"))
    feed = BinanceForceOrderFeed(ws, ping_interval_s=60.0, now_fn=_now)  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(feed.receive(0.5))


def test_feed_unexpected_frame_type_reports_closed() -> None:
    ws = _StubWS()
    ws.incoming.append(SimpleNamespace(type=WSMsgType.CONTINUATION, data=b"frag"))
    feed = BinanceForceOrderFeed(ws, ping_interval_s=60.0, now_fn=_now)  # type: ignore[arg-type]
    frame = asyncio.run(feed.receive(0.5))
    assert frame.kind == "closed"
    assert frame.received_at is None


def test_feed_dispatch_failure_reports_closed() -> None:
    def _boom() -> pd.Timestamp:
        raise RuntimeError("clock boom")

    ws = _StubWS()
    ws.incoming.append(SimpleNamespace(type=WSMsgType.PONG, data=b""))
    feed = BinanceForceOrderFeed(ws, ping_interval_s=60.0, now_fn=_boom)  # type: ignore[arg-type]
    frame = asyncio.run(feed.receive(0.5))
    assert frame.kind == "closed"


def test_feed_dispatch_cancel_propagates() -> None:
    def _cancel() -> pd.Timestamp:
        raise asyncio.CancelledError

    ws = _StubWS()
    ws.incoming.append(SimpleNamespace(type=WSMsgType.PONG, data=b""))
    feed = BinanceForceOrderFeed(ws, ping_interval_s=60.0, now_fn=_cancel)  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(feed.receive(0.5))
