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
    LiquidationHealth,
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


def test_append_liquidation_events_hourly_zstd_partition_and_dedup(tmp_path) -> None:
    d1 = pd.Timestamp("2026-09-01T12:00:00Z").value // 1_000_000
    d2 = pd.Timestamp("2026-09-02T09:00:00Z").value // 1_000_000
    events = [
        _event("BTCUSDT", d1, 100.0, 1.0, 1.0),
        _event("BTCUSDT", d1, 100.0, 1.0, 1.0),  # exact dup -> collapsed
        _event("BTCUSDT", d1, 101.0, 2.0, 2.0),  # distinct
        _event("ETHUSDT", d2, 50.0, 3.0, 3.0),   # distinct hour
    ]
    append_liquidation_events(events, tmp_path)
    append_liquidation_events(events, tmp_path)  # re-run must not duplicate

    f1 = tmp_path / "liquidations_20260901_12.parquet"
    f2 = tmp_path / "liquidations_20260902_09.parquet"
    assert f1.exists()
    assert f2.exists()

    df1 = pd.read_parquet(f1)
    assert len(df1) == 2
    assert df1["price"].dtype == "float64"
    assert df1["orig_qty"].dtype == "float32"
    assert isinstance(df1["side"].dtype, pd.CategoricalDtype)
    assert len(pd.read_parquet(f2)) == 1


def test_append_liquidation_events_routes_by_utc_hour_boundary(tmp_path) -> None:
    before_ms = pd.Timestamp("2026-09-24T05:59:59.999Z").value // 1_000_000
    on_ms = pd.Timestamp("2026-09-24T06:00:00.000Z").value // 1_000_000
    written = append_liquidation_events(
        [_event("BTCUSDT", before_ms, 100.0, 1.0, 1.0), _event("BTCUSDT", on_ms, 100.0, 1.0, 1.0)],
        tmp_path,
    )
    assert written == [tmp_path / "liquidations_20260924_05.parquet", tmp_path / "liquidations_20260924_06.parquet"]
    assert len(pd.read_parquet(written[0])) == 1
    assert len(pd.read_parquet(written[1])) == 1


def test_append_liquidation_events_leaves_legacy_daily_file_untouched(tmp_path) -> None:
    legacy = tmp_path / "liquidations_20260924.parquet"
    ms = pd.Timestamp("2026-09-24T05:00:00Z").value // 1_000_000
    append_liquidation_events([_event("BTCUSDT", ms, 100.0, 1.0, 1.0)], tmp_path)
    hourly = tmp_path / "liquidations_20260924_05.parquet"
    # reshape the hourly file into a legacy daily layout
    legacy.write_bytes(hourly.read_bytes())
    before = legacy.read_bytes()
    mtime_before = legacy.stat().st_mtime_ns
    ms2 = pd.Timestamp("2026-09-24T06:30:00Z").value // 1_000_000
    append_liquidation_events([_event("ETHUSDT", ms2, 50.0, 2.0, 2.0)], tmp_path)
    assert legacy.read_bytes() == before
    assert legacy.stat().st_mtime_ns == mtime_before
    assert len(pd.read_parquet(tmp_path / "liquidations_20260924_06.parquet")) == 1


def test_load_liquidation_events_deduplicates_across_layouts(tmp_path) -> None:
    ms = pd.Timestamp("2026-09-24T05:00:00Z").value // 1_000_000
    append_liquidation_events([_event("BTCUSDT", ms, 100.0, 1.0, 1.0)], tmp_path)
    hourly = tmp_path / "liquidations_20260924_05.parquet"
    legacy = tmp_path / "liquidations_20260924.parquet"
    legacy.write_bytes(hourly.read_bytes())
    loaded = load_liquidation_events(tmp_path)
    assert len(loaded) == 1


def test_append_liquidation_events_quarantines_corrupt_hour(tmp_path) -> None:
    ms = pd.Timestamp("2026-09-24T05:00:00Z").value // 1_000_000
    append_liquidation_events([_event("BTCUSDT", ms, 100.0, 1.0, 1.0)], tmp_path)
    target = tmp_path / "liquidations_20260924_05.parquet"
    raw = target.read_bytes()
    target.write_bytes(raw[: len(raw) // 2])
    ms2 = pd.Timestamp("2026-09-24T05:30:00Z").value // 1_000_000
    written = append_liquidation_events([_event("ETHUSDT", ms2, 50.0, 2.0, 2.0)], tmp_path)
    assert written == [target]
    assert len(pd.read_parquet(target)) == 1
    quarantined = list((tmp_path / "_quarantine").glob("liquidations_20260924_05.parquet.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == raw[: len(raw) // 2]


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


def test_run_liquidation_stream_pong_only_connection_attests_nothing(tmp_path) -> None:
    """A pong-answering socket with no parseable events leaves no coverage behind."""
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
    assert out.empty
    assert list(tmp_path.glob("liquidations_*.parquet")) == []


def test_run_liquidation_stream_quiet_stretch_between_events_stays_attested(tmp_path) -> None:
    """Within one connection the attested segment spans first to latest event."""
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    t1 = t0 + pd.Timedelta(seconds=100)
    tracker = CoverageTracker("liquidations", tmp_path)
    feed = _ScriptedFeed(
        [
            _events_frame(_raw("BTCUSDT", 1758531600000), at=t0),
            _alive_frame(t0 + pd.Timedelta(seconds=30)),
            _alive_frame(t0 + pd.Timedelta(seconds=60)),
            _events_frame(_raw("BTCUSDT", 1758531700000), at=t1),
        ],
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
    assert out.iloc[0]["end"] == t1


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
            event_stall_timeout_s=7200.0,
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
    # pong-only connections carry no event evidence, so nothing is attested.
    assert out.empty


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
        {"max_pending_events": 0},
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
            event_stall_timeout_s=7200.0,
                ),
                0.5,
            )

    asyncio.run(_scenario())


def test_run_liquidation_stream_unexpected_error_reraised_after_cleanup(tmp_path) -> None:
    """An unexpected error persists buffered events, closes the feed, then propagates."""
    import pytest

    flag = _Flag()
    calls = {"n": 0}

    def _clock() -> float:
        calls["n"] += 1
        if calls["n"] == 6:
            raise RuntimeError("clock boom")
        return 0.0

    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    feed = _ScriptedFeed(
        [_events_frame(_raw("BTCUSDT", 1758531600000), at=t0), _alive_frame(t0)],
        flag,
        shutdown_on_exhaust=False,
    )
    with pytest.raises(RuntimeError, match="clock boom"):
        asyncio.run(
            run_liquidation_stream(
                symbols=None,
                directory=tmp_path,
                shutdown=flag,
                feed_factory=_once(feed),
                clock=_clock,
            )
        )
    assert calls["n"] >= 6
    assert sum(len(pd.read_parquet(f)) for f in tmp_path.glob("liquidations_*.parquet")) == 1
    assert feed.closed == 1


def test_liquidation_health_tracks_events_and_failed_connections(tmp_path) -> None:
    """Two connect failures then one event: counters rise, then reset with the event time."""
    flag = _Flag()
    t1 = pd.Timestamp("2026-09-22T10:05:00Z")
    health = LiquidationHealth()
    attempts = {"n": 0}

    async def _factory() -> _ScriptedFeed:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise ConnectionError("ws dropped")
        return _ScriptedFeed([_events_frame(_raw("BTCUSDT", 1758531600000), at=t1)], flag)

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            health=health,
        )
    )
    assert health.last_event_at == t1
    assert health.consecutive_failed_connections == 0
    assert health.last_connected_at is not None
    assert health.last_disconnect_reason is not None
    assert health.last_disconnect_reason.startswith("CONNECT_FAILED")


def test_liquidation_health_counts_failed_connect_attempts_before_events(tmp_path) -> None:
    """Before any event arrives, consecutive failures stay visible on the health record."""
    flag = _Flag()
    health = LiquidationHealth()
    attempts = {"n": 0}

    async def _factory() -> _ScriptedFeed:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise ConnectionError("ws dropped")
        feed = _ScriptedFeed([_CLOSED], flag)
        return feed

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            health=health,
        )
    )
    # 2 connect failures, then an eventless disconnect recorded without a counter bump at shutdown.
    assert health.consecutive_failed_connections == 2
    assert health.last_event_at is None
    assert health.last_connected_at is not None
    assert health.last_disconnect_reason is not None
    assert health.last_disconnect_reason.startswith("DISCONNECTED")


def test_liquidation_health_pong_only_connections_count_as_failed(tmp_path) -> None:
    """Two pong-only connections ending by EVENT_STALL count as two failed connections."""
    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    health = LiquidationHealth()

    def _pongs(n: int) -> list[FeedFrame]:
        return [_alive_frame(t0 + pd.Timedelta(seconds=5 * i)) for i in range(n)]

    first = _ScriptedFeed(_pongs(30), flag, clock=clock, step_s=5.0, shutdown_on_exhaust=False)
    second = _ScriptedFeed(_pongs(30), flag, clock=clock, step_s=5.0, shutdown_on_exhaust=False)
    third = _ScriptedFeed([], flag, clock=clock)
    feeds = [first, second, third]

    async def _factory() -> _ScriptedFeed:
        return feeds.pop(0)

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            clock=clock,
            event_stall_timeout_s=100.0,
            health=health,
        )
    )
    assert feeds == []
    assert health.consecutive_failed_connections == 2
    assert health.last_event_at is None
    assert health.last_disconnect_reason is not None
    assert health.last_disconnect_reason.startswith("EVENT_STALL")


def test_liquidation_health_heartbeat_entry_is_json_ready() -> None:
    """The heartbeat snapshot serializes with ISO-8601 UTC timestamps."""
    import json

    health = LiquidationHealth(
        last_event_at=pd.Timestamp("2026-09-22T10:00:00Z"),
        last_connected_at=pd.Timestamp("2026-09-22T09:59:00Z"),
        consecutive_failed_connections=2,
        last_disconnect_reason="EVENT_STALL: no parseable event for 100s",
    )
    entry = health.as_heartbeat_entry()
    assert json.dumps(entry)
    assert entry["last_event_at"] == "2026-09-22T10:00:00+00:00"
    assert entry["last_connected_at"] == "2026-09-22T09:59:00+00:00"
    assert entry["consecutive_failed_connections"] == 2
    assert isinstance(entry["consecutive_failed_connections"], int)
    empty = LiquidationHealth().as_heartbeat_entry()
    assert empty["last_event_at"] is None
    assert empty["last_connected_at"] is None
    assert json.dumps(empty)


def test_append_liquidation_events_backfills_legacy_hour_missing_event_time_ms(tmp_path) -> None:
    """An hour file without the dedup key column still merges instead of failing."""
    ms = pd.Timestamp("2026-09-24T05:00:00Z").value // 1_000_000
    append_liquidation_events([_event("BTCUSDT", ms, 100.0, 1.0, 1.0)], tmp_path)
    target = tmp_path / "liquidations_20260924_05.parquet"
    legacy = pd.read_parquet(target).drop(columns=["event_time_ms"])
    assert "event_time_ms" not in legacy.columns
    legacy.to_parquet(target, index=False, compression="zstd")
    ms2 = pd.Timestamp("2026-09-24T05:30:00Z").value // 1_000_000
    append_liquidation_events([_event("ETHUSDT", ms2, 50.0, 2.0, 2.0)], tmp_path)
    merged = pd.read_parquet(target)
    assert len(merged) == 2
    assert set(merged["symbol"]) == {"BTCUSDT", "ETHUSDT"}


def test_run_liquidation_stream_stall_fires_on_quiet_timeouts_after_pings(tmp_path) -> None:
    """Alive frames keep liveness fresh but never reset the event-stall timer."""
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    pings = [_alive_frame(t0 + pd.Timedelta(seconds=5 * i)) for i in range(18)]
    pings += [_TIMEOUT, _TIMEOUT, _TIMEOUT]
    first = _ScriptedFeed(pings, flag, clock=clock, step_s=5.0, shutdown_on_exhaust=False)
    second = _ScriptedFeed([], flag, clock=clock)
    feeds = [first, second]

    async def _factory() -> _ScriptedFeed:
        return feeds.pop(0)

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            coverage=CoverageTracker("liquidations", tmp_path),
            clock=clock,
            event_stall_timeout_s=100.0,
        )
    )
    assert feeds == []  # EVENT_STALL (not liveness) opened the second connection
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    assert out.empty


def test_run_liquidation_stream_unparseable_frames_neither_attest_nor_reset_stall(tmp_path) -> None:
    """Frames with zero parseable payloads cannot mask a broken subscription."""
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    junk = [_events_frame({"not": "forceOrder"}, "junk", at=t0 + pd.Timedelta(seconds=5 * i)) for i in range(30)]
    first = _ScriptedFeed(junk, flag, clock=clock, step_s=5.0, shutdown_on_exhaust=False)
    second = _ScriptedFeed([], flag, clock=clock)
    feeds = [first, second]

    async def _factory() -> _ScriptedFeed:
        return feeds.pop(0)

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            coverage=CoverageTracker("liquidations", tmp_path),
            clock=clock,
            event_stall_timeout_s=100.0,
        )
    )
    assert feeds == []  # EVENT_STALL opened the second connection
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    assert out.empty
    assert list(tmp_path.glob("liquidations_*.parquet")) == []


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
    """Event-driven coverage still flushes per interval, not once per frame."""
    from src.market_data.streams import coverage as coverage_mod
    from src.market_data.streams.coverage import CoverageTracker

    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    frames = [
        _events_frame(_raw("BTCUSDT", 1758531600000 + 5000 * i), at=t0 + pd.Timedelta(seconds=5 * i))
        for i in range(24)
    ]  # 120 s of events
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


def test_force_order_stream_url_uses_market_category_path() -> None:
    """Binance splits USD-M streams by category; forceOrder events only flow on the `market` path."""
    from src.market_data.streams.liquidations import FORCE_ORDER_STREAM_URL

    assert FORCE_ORDER_STREAM_URL == "wss://fstream.binance.com/market/ws/!forceOrder@arr"


def test_run_liquidation_stream_event_stall_closes_coverage_and_reconnects(tmp_path) -> None:
    """An eventless tail attests nothing: the first connection's segment ends at its last event."""
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    t0b = t0 + pd.Timedelta(seconds=5)
    headed = [
        _events_frame(_raw("BTCUSDT", 1758531600000), at=t0),
        _events_frame(_raw("BTCUSDT", 1758531605000), at=t0b),
    ]
    headed += [_alive_frame(t0 + pd.Timedelta(seconds=5 * i)) for i in range(2, 40)]  # ~200 s tail of pings
    first = _ScriptedFeed(headed, flag, clock=clock, step_s=5.0, shutdown_on_exhaust=False)
    t2 = t0 + pd.Timedelta(seconds=210)
    t2b = t2 + pd.Timedelta(seconds=5)
    second = _ScriptedFeed(
        [
            _events_frame(_raw("BTCUSDT", 1758535200000), at=t2),
            _events_frame(_raw("BTCUSDT", 1758535205000), at=t2b),
        ],
        flag, clock=clock, step_s=5.0,
    )
    feeds = [first, second]

    async def _factory() -> _ScriptedFeed:
        return feeds.pop(0)

    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=0.0,
            shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            coverage=CoverageTracker("liquidations", tmp_path),
            clock=clock,
            event_stall_timeout_s=100.0,
        )
    )
    assert feeds == []  # 정지 판정으로 두 번째 연결이 열렸다
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    # 첫 연결의 정지 구간은 마지막 이벤트(t0b) 이후로 이어 붙지 않는다.
    assert len(out) == 2
    assert out.iloc[0]["start"] == t0
    assert out.iloc[0]["end"] == t0b
    assert out.iloc[1]["start"] == t2
    assert out.iloc[1]["end"] == t2b


def test_run_liquidation_stream_healthy_event_flow_never_stalls(tmp_path) -> None:
    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    frames = []
    for i in range(30):
        at = t0 + pd.Timedelta(seconds=30 * i)
        frames.append(_events_frame(_raw("BTCUSDT", 1758535200000 + i * 1000), at=at))
        frames.append(_alive_frame(at + pd.Timedelta(seconds=15)))
    feed = _ScriptedFeed(frames, flag, clock=clock, step_s=15.0)
    opened = {"n": 0}

    async def _factory() -> _ScriptedFeed:
        opened["n"] += 1
        return feed

    asyncio.run(
        run_liquidation_stream(
            symbols=None, directory=tmp_path, flush_interval_s=0.0, shutdown=flag,
            feed_factory=_factory,  # type: ignore[arg-type]
            clock=clock, event_stall_timeout_s=100.0,
        )
    )
    assert opened["n"] == 1


def test_run_liquidation_stream_rejects_stall_not_above_liveness(tmp_path) -> None:
    with pytest.raises(ValueError, match="event_stall_timeout_s"):
        asyncio.run(
            run_liquidation_stream(
                symbols=None, directory=tmp_path, shutdown=_Flag(),
                liveness_timeout_s=15.0, event_stall_timeout_s=15.0,
            )
        )


def test_parse_liquidation_preserves_raw_order_with_unknown_fields() -> None:
    """The venue order object is kept verbatim, including future fields."""
    import json as _json

    first = {
        "e": "forceOrder",
        "E": 1758531600123,
        "o": {
            "s": "QNTUSDT", "ps": "QNTUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "1.2",
            "p": "88.5", "ap": "88.4", "X": "FILLED", "l": "1.2", "z": "1.2", "T": 1758531600100,
            "st": 1,
        },
    }
    ev = parse_liquidation(first, ingested_at=pd.Timestamp("2026-09-01T00:00:00Z"))
    assert ev is not None
    assert ev.raw_order_json is not None
    assert _json.loads(ev.raw_order_json) == first["o"]
    assert _json.loads(ev.raw_order_json)["ps"] == "QNTUSDT"
    assert _json.loads(ev.raw_order_json)["st"] == 1


def test_parse_liquidation_unified_fallback_has_no_raw_payload() -> None:
    """Events without a raw order object carry no raw payload."""
    unified = {
        "symbol": "ETH/USDT:USDT",
        "timestamp": 1568014460893,
        "price": 1600.0,
        "amount": 3.2,
    }
    ev = parse_liquidation(unified, ingested_at=pd.Timestamp("2026-09-01T00:00:00Z"))
    assert ev is not None
    assert ev.raw_order_json is None


def test_append_liquidation_events_merges_legacy_hour_without_raw_column(tmp_path) -> None:
    """Hour files written before the column gain nulls for old rows."""
    ms = pd.Timestamp("2026-09-24T05:00:00Z").value // 1_000_000
    append_liquidation_events([_event("BTCUSDT", ms, 100.0, 1.0, 1.0)], tmp_path)
    target = tmp_path / "liquidations_20260924_05.parquet"
    legacy = pd.read_parquet(target).drop(columns=["raw_order_json"])
    assert "raw_order_json" not in legacy.columns
    legacy.to_parquet(target, index=False, compression="zstd")
    ms2 = pd.Timestamp("2026-09-24T05:30:00Z").value // 1_000_000
    raw = _raw("ETHUSDT", ms2)
    ev2 = parse_liquidation(raw, ingested_at=pd.Timestamp("2026-09-24T05:31:00Z"))
    assert ev2 is not None
    assert ev2.raw_order_json is not None
    append_liquidation_events([ev2], tmp_path)
    merged = pd.read_parquet(target)
    assert "raw_order_json" in merged.columns
    assert len(merged) == 2
    old = merged[merged["symbol"] == "BTCUSDT"].iloc[0]
    assert pd.isna(old["raw_order_json"])
    fresh = merged[merged["symbol"] == "ETHUSDT"].iloc[0]
    assert isinstance(fresh["raw_order_json"], str)
    assert fresh["raw_order_json"]


def test_load_liquidation_events_reads_mixed_layouts(tmp_path) -> None:
    """Legacy files without the column still load alongside new files."""
    ms = pd.Timestamp("2026-09-24T05:00:00Z").value // 1_000_000
    append_liquidation_events([_event("BTCUSDT", ms, 100.0, 1.0, 1.0)], tmp_path)
    hourly = tmp_path / "liquidations_20260924_05.parquet"
    legacy = tmp_path / "liquidations_20260924.parquet"
    legacy_frame = pd.read_parquet(hourly).drop(columns=["raw_order_json"])
    legacy_frame.to_parquet(legacy, index=False, compression="zstd")
    hourly.unlink()
    ms2 = pd.Timestamp("2026-09-24T06:00:00Z").value // 1_000_000
    ev2 = parse_liquidation(
        _raw("ETHUSDT", ms2), ingested_at=pd.Timestamp("2026-09-24T06:01:00Z")
    )
    assert ev2 is not None
    append_liquidation_events([ev2], tmp_path)
    loaded = load_liquidation_events(tmp_path)
    assert len(loaded) == 2
    assert "raw_order_json" in loaded.columns
    assert loaded[loaded["symbol"] == "BTCUSDT"]["raw_order_json"].isna().all()
    assert loaded[loaded["symbol"] == "ETHUSDT"]["raw_order_json"].notna().all()


def test_parse_liquidation_raw_serialization_failure_keeps_event() -> None:
    """A non-serializable order value yields no raw payload without dropping the event."""
    msg = dict(_raw("BTCUSDT", 1758531600000))
    msg["o"] = dict(msg["o"], extra={"bad"})
    ev = parse_liquidation(msg, ingested_at=pd.Timestamp("2026-09-01T00:00:00Z"))
    assert ev is not None
    assert ev.raw_order_json is None
    assert ev.symbol == "BTCUSDT"


def test_liquidation_buffer_bounded_under_persistent_flush_failure(tmp_path, monkeypatch) -> None:
    """Unpersisted events stay bounded; drops and flush failures are counted."""
    import src.market_data.streams.liquidations as liq_mod
    from src.market_data.streams.liquidations import LiquidationHealth

    def _boom(events: Any, directory: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(liq_mod, "append_liquidation_events", _boom)
    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    msgs = [_raw("BTCUSDT", 1758531600000 + 1000 * i) for i in range(25)]
    feed = _ScriptedFeed([_events_frame(*msgs, at=t0)], flag)
    health = LiquidationHealth()
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=3600.0,
            shutdown=flag,
            feed_factory=_once(feed),
            health=health,
            max_pending_events=10,
        )
    )
    assert health.dropped_events_total == 15
    assert health.consecutive_flush_failures >= 1
    assert health.pending_events == 10
    assert list(tmp_path.glob("liquidations_*.parquet")) == []


def test_liquidation_dropped_span_never_attested(tmp_path, monkeypatch) -> None:
    """Coverage never certifies receipt times whose events were dropped."""
    import src.market_data.streams.liquidations as liq_mod
    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    real_append = liq_mod.append_liquidation_events
    calls = {"n": 0}

    def _fail_once(events: Any, directory: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_append(events, directory)

    monkeypatch.setattr(liq_mod, "append_liquidation_events", _fail_once)
    flag = _Flag()
    clock = _ManualClock()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    first = [_raw("BTCUSDT", 1758531600000 + 1000 * i) for i in range(25)]
    second = [_raw("BTCUSDT", 1758531700000 + 1000 * i) for i in range(5)]
    feed = _ScriptedFeed(
        [
            _events_frame(*first, at=t0),
            _events_frame(*second, at=t0 + pd.Timedelta(seconds=30)),
        ],
        flag,
        clock=clock,
        step_s=5.0,
    )
    tracker = CoverageTracker("liquidations", tmp_path)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=0.0,
            shutdown=flag,
            feed_factory=_once(feed),
            clock=clock,
            coverage=tracker,
            max_pending_events=10,
        )
    )
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    dropped_end = t0 + pd.Timedelta(seconds=14)
    if not out.empty:
        assert not ((out["start"] <= t0) & (out["end"] > t0)).any()
        assert not ((out["start"] <= dropped_end) & (out["end"] > dropped_end)).any()


def test_liquidation_successful_flush_resets_persistence_health(tmp_path, monkeypatch) -> None:
    """One success clears the failure streak, the backlog count, and stamps the write."""
    import src.market_data.streams.liquidations as liq_mod
    from src.market_data.streams.liquidations import LiquidationHealth

    real_append = liq_mod.append_liquidation_events
    calls = {"n": 0}

    def _fail_once(events: Any, directory: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_append(events, directory)

    monkeypatch.setattr(liq_mod, "append_liquidation_events", _fail_once)
    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    t1 = t0 + pd.Timedelta(seconds=5)
    feed = _ScriptedFeed(
        [
            _events_frame(_raw("BTCUSDT", 1758531600000), at=t0),
            _events_frame(_raw("BTCUSDT", 1758531605000), at=t1),
        ],
        flag,
    )
    health = LiquidationHealth()
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=0.0,
            shutdown=flag,
            feed_factory=_once(feed),
            health=health,
            max_pending_events=10,
        )
    )
    assert calls["n"] >= 2
    assert health.consecutive_flush_failures == 0
    assert health.pending_events == 0
    assert health.last_persisted_at is not None


def test_liquidation_coverage_discard_failure_logged(tmp_path, monkeypatch) -> None:
    """A discard failure is logged and never drops the stream."""
    import src.market_data.streams.liquidations as liq_mod
    from src.market_data.streams.liquidations import LiquidationHealth

    def _boom(events: Any, directory: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(liq_mod, "append_liquidation_events", _boom)

    class _RejectingCoverage:
        def mark_ok(self, ts: Any) -> None:
            return None

        def mark_error(self, ts: Any) -> None:
            return None

        def flush(self) -> Any:
            return []

        def discard_unflushed(self, ts: Any) -> None:
            raise OSError("coverage down")

    import logging as _logging

    flag = _Flag()
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    msgs = [_raw("BTCUSDT", 1758531600000 + 1000 * i) for i in range(25)]
    feed = _ScriptedFeed([_events_frame(*msgs, at=t0)], flag)
    health = LiquidationHealth()
    with _TestCapLog(_logging.getLogger("src.market_data.streams.liquidations")) as records:
        asyncio.run(
            run_liquidation_stream(
                symbols=None,
                directory=tmp_path,
                flush_interval_s=3600.0,
                shutdown=flag,
                feed_factory=_once(feed),
                health=health,
                coverage=_RejectingCoverage(),  # type: ignore[arg-type]
                max_pending_events=10,
            )
        )
    assert health.dropped_events_total == 15
    assert any("COVERAGE_MARK_FAILED" in message for message in records)


class _TestCapLog:
    """Minimal log capture without the pytest caplog fixture (usable in any context)."""

    def __init__(self, logger: Any) -> None:
        import logging as _logging

        self._logger = logger
        self._records: list[str] = []
        self._handler = _logging.Handler()
        self._handler.emit = lambda record: self._records.append(record.getMessage())  # type: ignore[method-assign]

    def __enter__(self) -> list[str]:
        self._logger.addHandler(self._handler)
        return self._records

    def __exit__(self, *args: Any) -> bool:
        self._logger.removeHandler(self._handler)
        return False
