"""Contract coverage for the liquidation WebSocket stream collector.

Covers: parse_liquidation (raw forceOrder + ccxt unified), compact daily
partition persistence + dedup, research loader, and the resilient native
forceOrder stream loop (flush/shutdown + reconnect + attested liveness).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.market_data.streams.liquidations import (
    LiquidationEvent,
    append_liquidation_events,
    load_liquidation_events,
    parse_liquidation,
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


def test_append_keeps_earliest_ingested_and_non_null_raw(tmp_path) -> None:
    """Same event twice, 1 s apart, raw only on the later copy: earliest ingested wins with payload."""
    base_ms = pd.Timestamp("2026-09-24T05:00:00Z").value // 1_000_000
    early = _event("BTCUSDT", base_ms, 100.0, 1.0, 1.0)
    object.__setattr__(early, "raw_order_json", None)
    late = _event("BTCUSDT", base_ms, 100.0, 1.0, 1.0)
    late_event_time = pd.Timestamp("2026-09-24T05:00:00Z")
    late = parse_liquidation(
        {"e": "forceOrder", "E": base_ms, "o": {"s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "GTC",
         "q": "1.0", "p": "100.0", "ap": "100.0", "X": "FILLED", "l": "1.0", "z": "1.0", "T": base_ms}},
        ingested_at=late_event_time + pd.Timedelta(seconds=1),
    )
    assert late is not None
    assert late.raw_order_json is not None
    append_liquidation_events([late, early], tmp_path)
    merged = pd.read_parquet(tmp_path / "liquidations_20260924_05.parquet")
    assert len(merged) == 1
    assert merged.iloc[0]["ingested_at"] == pd.Timestamp(base_ms, unit="ms", tz="UTC")
    assert isinstance(merged.iloc[0]["raw_order_json"], str)
    assert merged.iloc[0]["raw_order_json"]


def test_parse_raw_ws_frame_shape(tmp_path) -> None:
    """The exact capture frame shape parses with ps/st preserved in raw_order_json."""
    import json as _json

    msg = {"e": "forceOrder", "E": 1758531600000,
           "o": {"s": "QNTUSDT", "S": "SELL", "o": "MARKET", "f": "IOC", "q": "2.0", "p": "50.0",
                 "ap": "50.0", "X": "FILLED", "l": "2.0", "z": "2.0", "T": 1758531600000,
                 "ps": "QNTUSDT", "st": 1}}
    event = parse_liquidation(msg, ingested_at=pd.Timestamp("2026-09-24T05:00:00Z"))
    assert event is not None
    assert event.raw_order_json is not None
    assert _json.loads(event.raw_order_json)["ps"] == "QNTUSDT"
    assert _json.loads(event.raw_order_json)["st"] == 1


def test_network_capture_code_is_gone() -> None:
    """Network symbols are deleted; the module does not import aiohttp."""
    import ast
    from pathlib import Path

    import src.market_data.streams.liquidations as module

    assert not hasattr(module, "run_liquidation_stream")
    assert not hasattr(module, "BinanceForceOrderFeed")
    tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "aiohttp" not in imported
