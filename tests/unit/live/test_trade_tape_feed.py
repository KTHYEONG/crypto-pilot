"""Paper tape feed invariants: through-only fills, bounded memory, malformed skip, readiness."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.live.trade_tape_feed import LiveTradeTape


def _msg(symbol: str, transact_ms: int, price: str, buyer_maker: bool) -> dict:
    return {
        "stream": f"{symbol.lower()}@aggTrade",
        "data": {
            "e": "aggTrade", "E": transact_ms, "s": symbol,
            "p": price, "q": "1.0", "T": transact_ms, "m": buyer_maker,
        },
    }


def _feed(messages: list, symbols: list[str], **kwargs):
    queue = list(messages)

    class _Conn:
        async def recv(self):
            if queue:
                return queue.pop(0)
            await asyncio.sleep(3600)
            return None

    @contextlib.asynccontextmanager
    async def _connect(url: str):
        _connect.seen_url = url
        yield _Conn()

    _connect.seen_url = ""
    kwargs.setdefault("stream_url", "wss://test/stream")
    kwargs.setdefault("max_trades_per_symbol", 100)
    kwargs.setdefault("stale_after_s", 60.0)
    tape = LiveTradeTape(symbols, connect=_connect, **kwargs)
    return tape, _connect


def _start(tape: LiveTradeTape, timeout: float = 5.0, shutdown=None):
    return tape.start(ready_timeout_s=timeout, shutdown=shutdown)


def test_buy_fills_only_on_seller_print_strictly_below() -> None:
    tape, _ = _feed(
        [_msg("XUSDT", 1000, "100.0", True), _msg("XUSDT", 1100, "99.9", True)],
        ["XUSDT"],
    )
    try:
        _start(tape)
        deadline = time.monotonic() + 5.0
        while tape.traded_through("XUSDT", "BUY", Decimal("100.0"), since_ms=0) is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert tape.traded_through("XUSDT", "BUY", Decimal("100.0"), since_ms=0) == 1100
    finally:
        tape.stop()


def test_sell_mirrors_with_buyer_aggressor_above() -> None:
    tape, _ = _feed([_msg("XUSDT", 10, "101.1", False)], ["XUSDT"])
    try:
        _start(tape)
        assert tape.traded_through("XUSDT", "SELL", Decimal("101.0"), since_ms=0) == 10
        assert tape.traded_through("XUSDT", "SELL", Decimal("102.0"), since_ms=0) is None
        assert tape.traded_through("XUSDT", "BUY", Decimal("101.0"), since_ms=0) is None
    finally:
        tape.stop()


def test_prints_before_since_ms_are_ignored() -> None:
    tape, _ = _feed([_msg("XUSDT", 900, "99.0", True)], ["XUSDT"])
    try:
        _start(tape)
        assert tape.traded_through("XUSDT", "BUY", Decimal("100.0"), since_ms=1000) is None
    finally:
        tape.stop()


def test_bounded_memory_drops_oldest() -> None:
    messages = [_msg("XUSDT", t, "99.0", True) for t in (1, 2, 3, 4, 5)]
    tape, _ = _feed(messages, ["XUSDT"], max_trades_per_symbol=3)
    try:
        _start(tape)
        deadline = time.monotonic() + 5.0
        while tape.traded_through("XUSDT", "BUY", Decimal("100.0"), since_ms=5) != 5:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert tape.traded_through("XUSDT", "BUY", Decimal("100.0"), since_ms=4) == 4
        assert tape.traded_through("XUSDT", "BUY", Decimal("100.0"), since_ms=0) == 3
        summary = tape.stop()
        assert summary.trades_received == 5
        assert summary.symbols_seen == 1
        assert summary.dropped_oldest == 2
    finally:
        with contextlib.suppress(Exception):
            tape.stop()


def test_malformed_messages_are_skipped() -> None:
    tape, _ = _feed(
        [
            "not-json",
            b"\xff\xfe-bad-bytes",
            json.dumps(_msg("XUSDT", 50, "99.0", True)),
            json.dumps(_msg("XUSDT", 51, "99.0", True)).encode(),
            _msg("XUSDT", 52, "99.0", True),
            {"nope": True},
            {"data": {"s": "XUSDT", "T": "bad", "p": "99.0", "m": True}},
            {"data": {"s": "XUSDT", "T": 53, "p": "bad", "m": True}},
            {"data": {"s": "XUSDT", "T": 54, "p": "99.0", "m": "yes"}},
            {"data": {"s": "XUSDT", "T": 55, "p": "0", "m": True}},
            {"data": {"s": "OTHERUSDT", "T": 56, "p": "1.0", "m": True}},
            {"stream": "x", "data": {"T": 57, "p": "1.0", "m": True}},
            "[1, 2]",
        ],
        ["XUSDT"],
    )
    try:
        readiness = _start(tape)
        assert readiness.ready is True
        deadline = time.monotonic() + 5.0
        while tape.traded_through("XUSDT", "BUY", Decimal("100.0"), since_ms=52) != 52:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert tape.traded_through("XUSDT", "BUY", Decimal("100.0"), since_ms=0) == 50
    finally:
        tape.stop()


def test_readiness_reports_missing_symbols() -> None:
    tape, _ = _feed([_msg("AUSDT", 10, "1.0", True)], ["AUSDT", "BUSDT"], stale_after_s=60.0)
    try:
        readiness = tape.start(ready_timeout_s=0.5)
        assert readiness.ready is False
        assert readiness.missing == ("BUSDT",)
        assert readiness.seen == 1
        assert readiness.required == 2
        assert readiness.waited_s <= 0.5 + 1.0
        assert tape.traded_through("BUSDT", "BUY", Decimal("1.0"), since_ms=0) is None
    finally:
        tape.stop()


def test_constructor_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max_trades_per_symbol"):
        LiveTradeTape(["XUSDT"], stream_url="wss://t", max_trades_per_symbol=0, stale_after_s=5.0)


def test_empty_symbols_start_ready_and_stop_clean() -> None:
    tape, _ = _feed([], [])
    readiness = tape.start(ready_timeout_s=1.0)
    assert readiness.ready is True
    assert readiness.required == 0
    assert readiness.missing == ()
    summary = tape.stop()
    assert summary.trades_received == 0
    assert summary.symbols_seen == 0


def test_stop_before_start_returns_zero_summary() -> None:
    tape, _ = _feed([], ["XUSDT"])
    summary = tape.stop()
    assert summary.trades_received == 0
    assert summary.reconnects == 0
    assert tape.stop() is summary


def test_invalid_timeout_never_raises() -> None:
    tape, _ = _feed([], ["XUSDT"])
    try:
        readiness = tape.start(ready_timeout_s=float("nan"))
        assert readiness.ready is False
        assert readiness.seen == 0
    finally:
        tape.stop()


def test_zero_timeout_returns_immediately_unready() -> None:
    tape, _ = _feed([_msg("XUSDT", 1, "1.0", True)], ["XUSDT"])
    try:
        readiness = tape.start(ready_timeout_s=0.0)
        assert readiness.ready is False
        assert readiness.waited_s == 0.0
    finally:
        tape.stop()


def test_second_start_reports_snapshot_without_wait() -> None:
    tape, _ = _feed([_msg("XUSDT", 1, "1.0", True)], ["XUSDT"])
    try:
        _start(tape)
        readiness = tape.start(ready_timeout_s=5.0)
        assert readiness.ready is True
        assert readiness.waited_s == 0.0
    finally:
        tape.stop()


def test_shutdown_request_stops_readiness_wait() -> None:
    tape, _ = _feed([], ["XUSDT", "YUSDT"])
    try:
        readiness = tape.start(ready_timeout_s=5.0, shutdown=SimpleNamespace(requested=True))
        assert readiness.ready is False
        assert readiness.missing == ("XUSDT", "YUSDT")
    finally:
        tape.stop()


def test_evil_shutdown_never_raises() -> None:
    class _Evil:
        @property
        def requested(self) -> bool:
            raise RuntimeError("boom")

    tape, _ = _feed([], ["XUSDT"])
    try:
        readiness = tape.start(ready_timeout_s=5.0, shutdown=_Evil())
        assert readiness.ready is False
    finally:
        tape.stop()


def test_traded_through_rejects_bad_inputs() -> None:
    tape, _ = _feed([_msg("XUSDT", 10, "99.0", True)], ["XUSDT"])
    try:
        _start(tape)
        assert tape.traded_through("XUSDT", "HOLD", Decimal("100.0"), since_ms=0) is None
        assert tape.traded_through("MISSING", "BUY", Decimal("100.0"), since_ms=0) is None
        assert tape.traded_through("XUSDT", "BUY", "bad-limit", since_ms=0) is None  # type: ignore[arg-type]
    finally:
        tape.stop()


def test_is_live_tracks_socket_health() -> None:
    tape, _ = _feed(
        [_msg("AUSDT", 10, "1.0", True)], ["AUSDT", "BUSDT"], stale_after_s=60.0,
    )
    assert tape.is_live("AUSDT", int(time.time() * 1000)) is False
    try:
        _start(tape)
        now_ms = int(time.time() * 1000)
        assert tape.is_live("AUSDT", now_ms) is True
        assert tape.is_live("BUSDT", now_ms) is True
        assert tape.is_live("AUSDT", "bad-now") is False  # type: ignore[arg-type]
    finally:
        tape.stop()


def test_is_live_goes_stale_without_messages() -> None:
    tape, _ = _feed([_msg("AUSDT", 10, "1.0", True)], ["AUSDT"], stale_after_s=0.05)
    try:
        _start(tape)
        time.sleep(0.15)
        assert tape.is_live("AUSDT", int(time.time() * 1000)) is False
    finally:
        tape.stop()


def test_reconnect_counts_and_recovers() -> None:
    queue = [_msg("XUSDT", 7, "99.0", True)]

    class _Conn:
        def __init__(self) -> None:
            self.calls = 0

        async def recv(self):
            self.calls += 1
            if queue:
                return queue.pop(0)
            if self.calls == 2:
                raise RuntimeError("recv boom")
            await asyncio.sleep(3600)
            return None

    @contextlib.asynccontextmanager
    async def _flaky(url: str):
        yield _Conn()

    tape = LiveTradeTape(
        ["XUSDT"], stream_url="wss://t", max_trades_per_symbol=10,
        stale_after_s=60.0, connect=_flaky,
    )
    try:
        readiness = tape.start(ready_timeout_s=5.0)
        assert readiness.ready is True
        deadline = time.monotonic() + 10.0
        while tape._reconnects < 1:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        summary = tape.stop()
        assert summary.reconnects >= 1
        assert summary.trades_received == 1
    finally:
        with contextlib.suppress(Exception):
            tape.stop()


def test_session_failure_is_observability_only(monkeypatch) -> None:
    async def _boom() -> None:
        raise RuntimeError("session boom")

    tape, _ = _feed([], ["XUSDT"])
    monkeypatch.setattr(tape, "_run_session", _boom)
    readiness = tape.start(ready_timeout_s=0.5)
    assert readiness.ready is False
    summary = tape.stop()
    assert summary.trades_received == 0


def test_stop_failure_is_observability_only() -> None:
    tape, _ = _feed([_msg("XUSDT", 1, "1.0", True)], ["XUSDT"])
    try:
        _start(tape)

        def _boom(timeout=None) -> None:
            raise RuntimeError("join boom")

        tape._thread.join = _boom  # type: ignore[method-assign]
        summary = tape.stop()
        assert summary.trades_received == 1
        assert tape.stop() is summary
    finally:
        with contextlib.suppress(Exception):
            tape.stop()


def test_stuck_thread_is_logged_not_raised(monkeypatch) -> None:
    import src.live.trade_tape_feed as tape_mod

    class _Conn:
        async def recv(self):
            await asyncio.sleep(3600)
            return None

    @contextlib.asynccontextmanager
    async def _hanging(url: str):
        yield _Conn()
        await asyncio.sleep(30)

    monkeypatch.setattr(tape_mod, "_STOP_JOIN_TIMEOUT_S", 0.05)
    tape = LiveTradeTape(
        ["XUSDT"], stream_url="wss://t", max_trades_per_symbol=10,
        stale_after_s=60.0, connect=_hanging,
    )
    try:
        tape.start(ready_timeout_s=0.0)
        deadline = time.monotonic() + 5.0
        while not tape._connected:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        summary = tape.stop()
        assert summary.trades_received == 0
    finally:
        with contextlib.suppress(Exception):
            tape.stop()
