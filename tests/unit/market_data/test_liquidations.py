"""Contract coverage for the liquidation WebSocket stream collector.

Covers: parse_liquidation (raw forceOrder + ccxt unified), compact daily
partition persistence + dedup, research loader, and the resilient native
forceOrder stream loop (flush/shutdown + reconnect + attested liveness).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
import pytest

from src.market_data.streams.liquidations import (
    FeedFrame,
    LiquidationEvent,
    append_liquidation_events,
    load_liquidation_events,
    parse_liquidation,
    run_liquidation_stream,
)

_RAW_MSG = {
    "info": {
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "o": "LIMIT",
            "f": "IOC",
            "q": "0.014",
            "p": "9910",
            "ap": "9910",
            "X": "FILLED",
            "l": "0.014",
            "z": "0.014",
            "T": 1568014460893,
        }
    }
}


def _raw(symbol: str, ms: int) -> dict[str, Any]:
    return {
        "e": "forceOrder",
        "E": ms,
        "o": {
            "s": symbol,
            "S": "SELL",
            "o": "LIMIT",
            "f": "IOC",
            "q": "0.5",
            "p": "60000",
            "ap": "60000",
            "X": "FILLED",
            "l": "0.5",
            "z": "0.5",
            "T": ms,
        },
    }


def test_parse_liquidation_from_raw_force_order_payload() -> None:
    ingested = pd.Timestamp("2026-09-01T00:00:00Z")
    ev = parse_liquidation(_RAW_MSG, ingested_at=ingested)
    assert ev is not None
    assert ev.symbol == "BTCUSDT"
    assert ev.side == "SELL"
    assert ev.order_type == "LIMIT"
    assert ev.time_in_force == "IOC"
    assert ev.orig_qty == pytest.approx(0.014)
    assert ev.price == pytest.approx(9910.0)
    assert ev.avg_price == pytest.approx(9910.0)
    assert ev.status == "FILLED"
    assert ev.filled_accum_qty == pytest.approx(0.014)
    assert ev.event_time == pd.Timestamp(1568014460893, unit="ms", tz="UTC")
    assert ev.ingested_at == ingested


#: 현행 ccxt(binanceusdm) watch_liquidations_for_symbols 가 실제로 내보내는 형태:
#: 주문 오브젝트가 info 로 평탄화되고 quoteValue/baseValue 는 None 이다.
_CCXT_FLAT_INFO_MSG = {
    "info": {
        "s": "BLESSUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC",
        "q": "31180", "p": "0.0107580", "ap": "0.0109660", "X": "FILLED",
        "l": "6249", "z": "31180", "T": 1788092214457, "ps": "BLESSUSDT", "st": 1,
    },
    "symbol": "BLESS/USDT:USDT",
    "contracts": 6249.0,
    "price": 0.010966,
    "side": "sell",
    "baseValue": None,
    "quoteValue": None,
    "timestamp": 1788092214457,
}


def test_parse_liquidation_from_ccxt_flat_info_payload() -> None:
    ev = parse_liquidation(_CCXT_FLAT_INFO_MSG, ingested_at=pd.Timestamp("2026-09-01T00:00:00Z"))
    assert ev is not None
    assert ev.symbol == "BLESSUSDT"
    assert ev.side == "SELL"
    assert ev.order_type == "LIMIT"
    assert ev.orig_qty == pytest.approx(31180.0)
    assert ev.avg_price == pytest.approx(0.010966)
    assert ev.status == "FILLED"
    assert ev.event_time == pd.Timestamp(1788092214457, unit="ms", tz="UTC")


def test_parse_liquidation_from_ccxt_unified_dict() -> None:
    unified = {
        "symbol": "ETH/USDT:USDT",
        "timestamp": 1568014460893,
        "price": 1600.0,
        "baseValue": 3.2,
        "info": {},
    }
    ev = parse_liquidation(unified, ingested_at=pd.Timestamp("2026-09-01T00:00:00Z"))
    assert ev is not None
    assert ev.symbol == "ETHUSDT"
    assert ev.price == pytest.approx(1600.0)
    assert ev.orig_qty == pytest.approx(3.2)
    assert ev.event_time == pd.Timestamp(1568014460893, unit="ms", tz="UTC")
    # Malformed message -> None, never raises.
    assert parse_liquidation({}, ingested_at=pd.Timestamp("2026-09-01T00:00:00Z")) is None


def _event(symbol: str, ms: int, price: float, qty: float, accum: float) -> LiquidationEvent:
    et = pd.Timestamp(ms, unit="ms", tz="UTC")
    return LiquidationEvent(
        symbol=symbol,
        event_time=et,
        ingested_at=et,
        side="SELL",
        order_type="LIMIT",
        time_in_force="IOC",
        orig_qty=qty,
        price=price,
        avg_price=price,
        status="FILLED",
        last_filled_qty=qty,
        filled_accum_qty=accum,
    )


def test_append_liquidation_events_daily_zstd_partition_and_dedup(tmp_path) -> None:
    d1 = pd.Timestamp("2026-09-01T12:00:00Z").value // 1_000_000
    d2 = pd.Timestamp("2026-09-02T09:00:00Z").value // 1_000_000
    events = [
        _event("BTCUSDT", d1, 100.0, 1.0, 1.0),
        _event("BTCUSDT", d1, 100.0, 1.0, 1.0),  # exact dup -> collapsed
        _event("BTCUSDT", d1, 101.0, 2.0, 2.0),  # distinct
        _event("ETHUSDT", d2, 50.0, 3.0, 3.0),   # distinct day
    ]
    append_liquidation_events(events, tmp_path)
    append_liquidation_events(events, tmp_path)  # re-run must not duplicate

    f1 = tmp_path / "liquidations_20260901.parquet"
    f2 = tmp_path / "liquidations_20260902.parquet"
    assert f1.exists()
    assert f2.exists()

    df1 = pd.read_parquet(f1)
    assert len(df1) == 2
    assert df1["price"].dtype == "float64"
    assert df1["orig_qty"].dtype == "float32"
    assert isinstance(df1["side"].dtype, pd.CategoricalDtype)
    assert len(pd.read_parquet(f2)) == 1


def test_load_liquidation_events_roundtrip_and_missing_dir(tmp_path) -> None:
    missing = tmp_path / "nope"
    assert load_liquidation_events(missing).empty

    ms = pd.Timestamp("2026-09-03T01:00:00Z").value // 1_000_000
    append_liquidation_events([_event("BTCUSDT", ms, 100.0, 1.0, 1.0)], tmp_path)
    loaded = load_liquidation_events(tmp_path)
    assert len(loaded) == 1
    assert str(loaded["event_time"].dt.tz) == "UTC"
    after = pd.Timestamp("2026-09-04T00:00:00Z")
    assert load_liquidation_events(tmp_path, since=after).empty


class _Flag:
    requested = False


class _ManualClock:
    """Deterministic monotonic clock advanced explicitly by the feed fake."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _events_frame(*msgs: Any, at: pd.Timestamp) -> FeedFrame:
    return FeedFrame(kind="events", received_at=at, payloads=tuple(msgs))


def _alive_frame(at: pd.Timestamp) -> FeedFrame:
    return FeedFrame(kind="alive", received_at=at)


_TIMEOUT = FeedFrame(kind="timeout", received_at=None)
_CLOSED = FeedFrame(kind="closed", received_at=None, detail="server close")


class _ScriptedFeed:
    """Scripted ``LiquidationFeed`` fake; sets shutdown when frames are exhausted."""

    def __init__(
        self,
        frames: list[FeedFrame],
        shutdown: _Flag,
        *,
        clock: _ManualClock | None = None,
        step_s: float = 0.0,
        shutdown_on_exhaust: bool = True,
    ) -> None:
        self._frames = list(frames)
        self._shutdown = shutdown
        self._clock = clock
        self._step_s = step_s
        self._shutdown_on_exhaust = shutdown_on_exhaust
        self.receive_timeouts: list[float] = []
        self.closed = 0

    async def receive(self, timeout_s: float) -> FeedFrame:
        self.receive_timeouts.append(float(timeout_s))
        if self._clock is not None and self._step_s:
            self._clock.advance(self._step_s)
        if not self._frames:
            if self._shutdown_on_exhaust:
                self._shutdown.requested = True
                return FeedFrame(kind="timeout", received_at=None)
            return FeedFrame(kind="closed", received_at=None, detail="exhausted")
        frame = self._frames.pop(0)
        if not self._frames and self._shutdown_on_exhaust:
            self._shutdown.requested = True
        return frame

    async def close(self) -> None:
        self.closed += 1


def _once(feed: _ScriptedFeed) -> Any:
    async def _factory() -> _ScriptedFeed:
        return feed

    return _factory


def test_run_liquidation_stream_persists_events_and_stops_on_shutdown(tmp_path) -> None:
    flag = _Flag()
    feed = _ScriptedFeed(
        [_events_frame(_RAW_MSG, _CCXT_FLAT_INFO_MSG, "junk", {}, at=pd.Timestamp("2026-09-22T10:00:00Z"))],
        flag,
    )
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_once(feed),
        )
    )
    files = list(tmp_path.glob("liquidations_*.parquet"))
    assert sum(len(pd.read_parquet(f)) for f in files) == 2
    assert feed.closed == 1


def test_run_liquidation_stream_observes_shutdown_without_events(tmp_path) -> None:
    flag = _Flag()
    feed = _ScriptedFeed([_CLOSED], flag)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_once(feed),
            receive_timeout_s=0.5,
        )
    )
    assert 1 <= len(feed.receive_timeouts) <= 2
    assert all(t == 0.5 for t in feed.receive_timeouts)


def test_run_liquidation_stream_attests_quiet_stretch_with_alive_frames(tmp_path) -> None:
    """Quiet-but-connected stretches stay attested without writing event files."""
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    tracker = CoverageTracker("liquidations", tmp_path)
    feed = _ScriptedFeed(
        [_alive_frame(t0), _alive_frame(t0 + pd.Timedelta(seconds=5)), _alive_frame(t0 + pd.Timedelta(seconds=10))],
        flag,
    )
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_once(feed),
            coverage=tracker,
        )
    )
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    assert len(out) == 1
    assert out.iloc[0]["start"] == t0
    assert out.iloc[0]["end"] == t0 + pd.Timedelta(seconds=10)
    assert list(tmp_path.glob("liquidations_*.parquet")) == []


def test_run_liquidation_stream_timeout_frames_never_attest(tmp_path) -> None:
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    tracker = CoverageTracker("liquidations", tmp_path)
    feed = _ScriptedFeed([_alive_frame(t0), _TIMEOUT, _TIMEOUT], flag)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_once(feed),
            coverage=tracker,
            liveness_timeout_s=3600.0,
        )
    )
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    assert out.empty or out["end"].max() <= t0


def test_run_liquidation_stream_liveness_timeout_forces_reconnect(tmp_path) -> None:
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    t1 = t0 + pd.Timedelta(seconds=5)
    t2 = t0 + pd.Timedelta(seconds=60)
    t3 = t0 + pd.Timedelta(seconds=65)
    first = _ScriptedFeed(
        [_alive_frame(t0), _alive_frame(t1), _TIMEOUT, _TIMEOUT, _TIMEOUT, _TIMEOUT, _TIMEOUT],
        flag,
        clock=clock,
        step_s=5.0,
        shutdown_on_exhaust=False,
    )
    second = _ScriptedFeed([_alive_frame(t2), _alive_frame(t3)], flag, clock=clock, step_s=5.0)
    feeds = [first, second]
    calls = {"n": 0}

    async def _factory() -> _ScriptedFeed:
        calls["n"] += 1
        return feeds.pop(0)

    tracker = CoverageTracker("liquidations", tmp_path)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            clock=clock,
            coverage=tracker,
            liveness_timeout_s=12.0,
        )
    )
    assert calls["n"] == 2
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    assert len(out) == 2
    assert out.iloc[0]["end"] == t1
    assert out.iloc[0]["end"] < out.iloc[1]["start"]


def test_run_liquidation_stream_healthy_connection_reconnects_immediately(tmp_path, monkeypatch) -> None:
    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    t1 = t0 + pd.Timedelta(seconds=5)
    first = _ScriptedFeed(
        [_events_frame(_raw("BTCUSDT", 1758531600000), at=t0), _CLOSED], flag, shutdown_on_exhaust=False
    )
    second = _ScriptedFeed([_events_frame(_raw("ETHUSDT", 1758531660000), at=t1)], flag)
    feeds = [first, second]
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _fake_sleep(seconds: float, *a: Any, **k: Any) -> None:
        sleeps.append(float(seconds))
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    async def _factory() -> _ScriptedFeed:
        return feeds.pop(0)

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
        )
    )
    assert sleeps == []
    files = list(tmp_path.glob("liquidations_*.parquet"))
    assert sum(len(pd.read_parquet(f)) for f in files) == 2


def test_run_liquidation_stream_consecutive_failures_back_off_exponentially(tmp_path, monkeypatch) -> None:
    flag = _Flag()
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _fake_sleep(seconds: float, *a: Any, **k: Any) -> None:
        sleeps.append(float(seconds))
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    factory_marks: list[int] = []

    async def _factory() -> _ScriptedFeed:
        factory_marks.append(len(sleeps))
        if len(factory_marks) <= 8:
            raise ConnectionError("ws dropped")
        return _ScriptedFeed([], flag)

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            max_backoff_s=60.0,
        )
    )
    assert len(factory_marks) == 9
    totals = [
        sum(sleeps[factory_marks[i]:factory_marks[i + 1]]) for i in range(8)
    ]
    assert totals == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert all(s <= 60.0 for s in sleeps)


def test_run_liquidation_stream_coverage_never_outruns_persisted_events(tmp_path, monkeypatch) -> None:
    """A failed event flush keeps the buffer and leaves coverage pending."""
    import src.market_data.streams.liquidations as liq_mod
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    t1 = t0 + pd.Timedelta(seconds=5)
    real_append = liq_mod.append_liquidation_events
    calls = {"n": 0}

    def _fail_once(events: Any, directory: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            assert list(tmp_path.rglob("*.jsonl")) == []
            raise OSError("disk full")
        return real_append(events, directory)

    monkeypatch.setattr(liq_mod, "append_liquidation_events", _fail_once)
    tracker = CoverageTracker("liquidations", tmp_path)
    feed = _ScriptedFeed(
        [
            _events_frame(_raw("BTCUSDT", 1758531600000), at=t0),
            _events_frame(_raw("BTCUSDT", 1758531660000), at=t1),
        ],
        flag,
    )
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=0.0,
            shutdown=flag,
            feed_factory=_once(feed),
            clock=clock,
            coverage=tracker,
        )
    )
    assert calls["n"] >= 2
    files = list(tmp_path.glob("liquidations_*.parquet"))
    assert sum(len(pd.read_parquet(f)) for f in files) == 2
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    assert len(out) == 1
    assert out.iloc[0]["start"] == t0
    assert out.iloc[0]["end"] == t1


def test_run_liquidation_stream_symbol_filter_keeps_liveness(tmp_path) -> None:
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    t1 = t0 + pd.Timedelta(seconds=5)
    tracker = CoverageTracker("liquidations", tmp_path)
    feed = _ScriptedFeed(
        [
            _events_frame(_raw("ETHUSDT", 1758531600000), at=t0),
            _events_frame(_raw("ETHUSDT", 1758531660000), at=t1),
        ],
        flag,
    )
    asyncio.run(
        run_liquidation_stream(
            symbols=["BTCUSDT"],
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_once(feed),
            coverage=tracker,
        )
    )
    assert list(tmp_path.glob("liquidations_*.parquet")) == []
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    assert len(out) == 1
    assert out.iloc[0]["start"] == t0
    assert out.iloc[0]["end"] == t1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"receive_timeout_s": 0.0},
        {"ping_interval_s": 0.0},
        {"liveness_timeout_s": 5.0, "ping_interval_s": 5.0},
        {"liveness_timeout_s": 1.0, "ping_interval_s": 5.0},
    ],
)
def test_run_liquidation_stream_rejects_invalid_timing(tmp_path, kwargs: dict[str, float]) -> None:
    flag = _Flag()
    opened = {"n": 0}

    async def _factory() -> _ScriptedFeed:
        opened["n"] += 1
        return _ScriptedFeed([], flag)

    with pytest.raises(ValueError, match="must be"):
        asyncio.run(
            run_liquidation_stream(
                symbols=None,
                directory=tmp_path,
                shutdown=flag,
                feed_factory=_factory,  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )
        )
    assert opened["n"] == 0


def test_run_liquidation_stream_receive_error_reconnects(tmp_path) -> None:
    """A ``receive`` transport error is treated as a closed connection."""
    flag = _Flag()
    calls = {"n": 0}

    class _Flaky:
        closed = 0

        async def receive(self, timeout_s: float) -> FeedFrame:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transport lost")
            flag.requested = True
            return FeedFrame(kind="timeout", received_at=None)

        async def close(self) -> None:
            self.closed += 1

    feed = _Flaky()

    async def _factory() -> _Flaky:
        return feed

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
        )
    )
    assert calls["n"] == 2
    assert feed.closed == 2


def test_run_liquidation_stream_receive_cancel_propagates(tmp_path) -> None:
    flag = _Flag()

    class _Cancelling:
        async def receive(self, timeout_s: float) -> FeedFrame:
            raise asyncio.CancelledError

        async def close(self) -> None:
            return None

    async def _factory() -> _Cancelling:
        return _Cancelling()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_liquidation_stream(
                symbols=None,
                directory=tmp_path,
                shutdown=flag,
                feed_factory=_factory,  # type: ignore[arg-type]
            )
        )


def test_run_liquidation_stream_silent_disconnect_backs_off(tmp_path) -> None:
    """A liveness death with no evidence backs off; shutdown mid-backoff stops the stream."""
    flag = _Flag()
    clock = _ManualClock()
    feed = _ScriptedFeed([_TIMEOUT, _TIMEOUT, _TIMEOUT], flag, clock=clock, step_s=5.0, shutdown_on_exhaust=False)
    calls = {"n": 0}

    async def _factory() -> _ScriptedFeed:
        calls["n"] += 1
        return feed

    async def _scenario() -> None:
        async def _stop() -> None:
            await asyncio.sleep(0.2)
            flag.requested = True

        task = asyncio.create_task(_stop())
        await run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            clock=clock,
            liveness_timeout_s=12.0,
        )
        await task

    asyncio.run(_scenario())
    assert calls["n"] == 1


def test_run_liquidation_stream_shutdown_during_backoff(tmp_path) -> None:
    """Shutdown observed mid-backoff stops the stream without opening a feed."""
    flag = _Flag()
    opened = {"n": 0}

    async def _factory() -> _ScriptedFeed:
        opened["n"] += 1
        raise ConnectionError("down")

    async def _scenario() -> None:
        async def _stop() -> None:
            await asyncio.sleep(1.5)
            flag.requested = True

        task = asyncio.create_task(_stop())
        await run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
        )
        await task

    asyncio.run(_scenario())
    assert opened["n"] == 2


def test_run_liquidation_stream_cancelled_sleep_propagates(tmp_path, monkeypatch) -> None:
    flag = _Flag()

    async def _factory() -> _ScriptedFeed:
        raise ConnectionError("down")

    async def _boom(seconds: float, *a: Any, **k: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", _boom)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_liquidation_stream(
                symbols=None,
                directory=tmp_path,
                shutdown=flag,
                feed_factory=_factory,  # type: ignore[arg-type]
            )
        )


def test_run_liquidation_stream_coverage_mark_failures_logged(tmp_path) -> None:
    """Coverage callback failures never stop the stream."""

    class _Rejecting:
        def mark_ok(self, ts: Any) -> None:
            raise RuntimeError("no")

        def mark_error(self, ts: Any) -> None:
            raise RuntimeError("no")

        def flush(self) -> Any:
            raise RuntimeError("no")

    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    feed = _ScriptedFeed([_alive_frame(t0), _CLOSED], flag)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_once(feed),
            coverage=_Rejecting(),  # type: ignore[arg-type]
        )
    )
    assert feed.closed == 1


def test_run_liquidation_stream_feed_close_failure_logged(tmp_path) -> None:
    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")

    class _BadClose(_ScriptedFeed):
        async def close(self) -> None:
            raise RuntimeError("close boom")

    feed = _BadClose([_events_frame(_raw("BTCUSDT", 1758531600000), at=t0)], flag)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_once(feed),
        )
    )
    assert sum(len(pd.read_parquet(f)) for f in tmp_path.glob("liquidations_*.parquet")) == 1


def test_run_liquidation_stream_broken_shutdown_flag_treated_as_not_requested(tmp_path) -> None:
    """A shutdown flag that raises on access never reads as requested."""

    class _Raising:
        @property
        def requested(self) -> bool:
            raise RuntimeError("boom")

    class _Yielding:
        async def receive(self, timeout_s: float) -> FeedFrame:
            await asyncio.sleep(0)
            return FeedFrame(kind="timeout", received_at=None)

        async def close(self) -> None:
            return None

    async def _factory() -> _Yielding:
        return _Yielding()

    async def _scenario() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                run_liquidation_stream(
                    symbols=None,
                    directory=tmp_path,
                    shutdown=_Raising(),
                    feed_factory=_factory,  # type: ignore[arg-type]
                    liveness_timeout_s=3600.0,
                ),
                0.5,
            )

    asyncio.run(_scenario())


def test_run_liquidation_stream_clock_failure_logged_and_stops(tmp_path) -> None:
    flag = _Flag()
    calls = {"n": 0}

    def _clock() -> float:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("clock boom")
        return 0.0

    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    feed = _ScriptedFeed([_events_frame(_raw("BTCUSDT", 1758531600000), at=t0)], flag)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            shutdown=flag,
            feed_factory=_once(feed),
            clock=_clock,
        )
    )
    assert calls["n"] >= 3


async def _run_default_factory_once(
    tmp_path: Any, monkeypatch: Any, flag: _Flag, close_exc: BaseException | None
) -> dict[str, int]:
    """Run the stream with the default factory against a local forceOrder endpoint."""
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    import src.market_data.streams.liquidations as liq_mod

    payload = _raw("BTCUSDT", 1758531600000)
    closed = {"n": 0}

    async def _handler(request: Any) -> Any:
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        await ws.send_str(json.dumps(payload))
        await asyncio.sleep(5.0)
        return ws

    app = web.Application()
    app.router.add_get("/ws", _handler)
    server = TestServer(app)
    await server.start_server()
    try:
        local_url = str(server.make_url("/ws")).replace("http://", "ws://")
        real_connect = liq_mod.BinanceForceOrderFeed.connect

        @classmethod
        async def _local_connect(cls: Any, session: Any, **kwargs: Any) -> Any:
            return await real_connect(session, url=local_url, **kwargs)

        monkeypatch.setattr(liq_mod.BinanceForceOrderFeed, "connect", _local_connect)
        real_close = aiohttp.ClientSession.close

        async def _counting_close(self: Any) -> None:
            closed["n"] += 1
            if close_exc is not None:
                raise close_exc
            await real_close(self)

        monkeypatch.setattr(aiohttp.ClientSession, "close", _counting_close)

        async def _stop() -> None:
            await asyncio.sleep(1.0)
            flag.requested = True

        task = asyncio.create_task(_stop())
        await run_liquidation_stream(symbols=None, directory=tmp_path, flush_interval_s=0.0, shutdown=flag)
        await task
        return closed
    finally:
        await server.close()


def test_run_liquidation_stream_default_factory_uses_owned_session(tmp_path, monkeypatch) -> None:
    """Without ``feed_factory`` the stream connects, persists, and closes its session."""
    flag = _Flag()

    async def _scenario() -> dict[str, int]:
        return await _run_default_factory_once(tmp_path, monkeypatch, flag, None)

    closed = asyncio.run(_scenario())
    assert closed["n"] == 1
    assert sum(len(pd.read_parquet(f)) for f in tmp_path.glob("liquidations_*.parquet")) >= 1


def test_run_liquidation_stream_default_factory_session_close_failure_logged(tmp_path, monkeypatch) -> None:
    """An owned-session close failure is logged and never raises."""
    flag = _Flag()

    async def _scenario() -> dict[str, int]:
        return await _run_default_factory_once(tmp_path, monkeypatch, flag, RuntimeError("close boom"))

    closed = asyncio.run(_scenario())
    assert closed["n"] == 1
    assert sum(len(pd.read_parquet(f)) for f in tmp_path.glob("liquidations_*.parquet")) >= 1


def test_run_liquidation_stream_default_factory_session_cancel_propagates(tmp_path, monkeypatch) -> None:
    """Cancellation during the owned-session close propagates."""
    flag = _Flag()

    async def _scenario() -> None:
        await _run_default_factory_once(tmp_path, monkeypatch, flag, asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_scenario())


def test_run_liquidation_stream_quiet_coverage_flushes_at_interval_not_per_frame(tmp_path, monkeypatch) -> None:
    """Quiet stretch: ~5 s ping frames must not write one coverage record each."""
    from src.market_data.streams import coverage as coverage_mod
    from src.market_data.streams.coverage import CoverageTracker

    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    frames = [_alive_frame(t0 + pd.Timedelta(seconds=5 * i)) for i in range(24)]  # 120 s of pings
    feed = _ScriptedFeed(frames, flag, clock=clock, step_s=5.0)
    writes: list[int] = []
    original = coverage_mod._append_intervals

    def _spy(*args, **kwargs):
        writes.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(coverage_mod, "_append_intervals", _spy)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=60.0,
            shutdown=flag,
            feed_factory=_once(feed),
            coverage=CoverageTracker("liquidations", tmp_path),
            clock=clock,
        )
    )
    # 120 s / 60 s 주기 → 몇 번(최종 flush 포함)이지 24번이 아니다.
    assert 1 <= len(writes) <= 4
