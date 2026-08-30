"""Contract coverage for the liquidation WebSocket stream collector.

Covers: parse_liquidation (raw forceOrder + ccxt unified), compact daily
partition persistence + dedup, research loader, and the resilient async
stream loop (flush/shutdown + reconnect-without-dying).
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from src.market_data.streams.liquidations import (
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


class _StubExchange:
    """Minimal ccxt.pro-shaped stub driving one watch cycle."""

    def __init__(self, batches: list, shutdown, *, error_first: bool = False) -> None:
        self._batches = list(batches)
        self._shutdown = shutdown
        self._error_first = error_first
        self.calls = 0
        self.closed = 0

    async def watch_liquidations(self):  # noqa: D401 - stub
        self.calls += 1
        if self._error_first and self.calls == 1:
            raise ConnectionError("ws dropped")
        batch = self._batches.pop(0) if self._batches else []
        if not self._batches:
            self._shutdown.requested = True
        return batch

    async def close(self) -> None:
        self.closed += 1


class _Flag:
    requested = False


def test_run_liquidation_stream_flushes_and_stops_on_shutdown(tmp_path) -> None:
    flag = _Flag()
    msg2 = {
        "info": {
            "o": {
                "s": "ETHUSDT", "S": "BUY", "o": "LIMIT", "f": "IOC",
                "q": "2", "p": "50", "ap": "50", "X": "FILLED",
                "l": "2", "z": "2", "T": 1700000000000,
            }
        }
    }
    stub = _StubExchange([[_RAW_MSG, msg2]], flag)
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=0.0,
            shutdown=flag,
            exchange_factory=lambda: stub,
        )
    )
    files = list(tmp_path.glob("liquidations_*.parquet"))
    total = sum(len(pd.read_parquet(f)) for f in files)
    assert total == 2
    assert stub.closed == 1


def test_run_liquidation_stream_reconnects_on_error_without_dying(tmp_path, monkeypatch) -> None:
    flag = _Flag()
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _fake_sleep(seconds, *a, **k):
        sleeps.append(float(seconds))
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    stub = _StubExchange([[_RAW_MSG]], flag, error_first=True)
    # must not propagate the ConnectionError
    asyncio.run(
        run_liquidation_stream(
            symbols=None,
            directory=tmp_path,
            flush_interval_s=0.0,
            shutdown=flag,
            exchange_factory=lambda: stub,
        )
    )
    backoffs = [s for s in sleeps if s > 0]
    assert backoffs
    assert backoffs[0] == 1.0
    assert all(s <= 60.0 for s in backoffs)
    files = list(tmp_path.glob("liquidations_*.parquet"))
    assert sum(len(pd.read_parquet(f)) for f in files) == 1
