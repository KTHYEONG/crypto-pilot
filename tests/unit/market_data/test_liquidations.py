"""Contract coverage for the liquidation WebSocket stream collector.

Covers: parse_liquidation (strict raw forceOrder), compact hourly
partition persistence and deduplication.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
import pytest

from src.market_data.streams.liquidations import (
    LiquidationEvent,
    append_liquidation_events,
    parse_liquidation,
)

_RAW_MSG = {
    "e": "forceOrder",
    "E": 1568014460900,
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
    },
}

_REQUIRED_KEYS = ("s", "S", "o", "f", "q", "p", "ap", "X", "l", "z", "T")


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


def _base_order() -> dict[str, Any]:
    return {
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


def _frame_with(order: dict[str, Any]) -> dict[str, Any]:
    return {"e": "forceOrder", "E": 1568014460900, "o": order}


_INGESTED = pd.Timestamp("2026-09-01T00:00:00Z")


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
    assert ev.last_filled_qty == pytest.approx(0.014)
    assert ev.filled_accum_qty == pytest.approx(0.014)
    assert ev.event_time == pd.Timestamp(1568014460893, unit="ms", tz="UTC")
    assert ev.event_time != pd.Timestamp(1568014460900, unit="ms", tz="UTC")
    assert ev.ingested_at == ingested
    import json as _json

    assert _json.loads(ev.raw_order_json or "") == _RAW_MSG["o"]


@pytest.mark.parametrize("key", list(_REQUIRED_KEYS))
def test_parse_liquidation_missing_key_rejects(key: str) -> None:
    order = _base_order()
    del order[key]
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


@pytest.mark.parametrize("key", list(_REQUIRED_KEYS))
def test_parse_liquidation_none_key_rejects(key: str) -> None:
    order = _base_order()
    order[key] = None
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


@pytest.mark.parametrize("key", ["q", "p", "ap", "l", "z"])
@pytest.mark.parametrize("value", ["NaN", "nan", "Infinity", "-inf", "1e400", float("nan")])
def test_parse_liquidation_non_finite_rejects(key: str, value: Any) -> None:
    order = _base_order()
    order[key] = value
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


def test_parse_liquidation_inf_float_rejects() -> None:
    order = _base_order()
    order["q"] = float("inf")
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


@pytest.mark.parametrize("key", ["q", "p"])
@pytest.mark.parametrize("value", ["0", "0.0", "-0", "-1"])
def test_parse_liquidation_non_positive_qty_price_rejects(key: str, value: str) -> None:
    order = _base_order()
    order[key] = value
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


def test_parse_liquidation_legitimate_zero_fills_preserved() -> None:
    order = _base_order()
    order.update({"X": "EXPIRED", "ap": "0", "l": "0", "z": "0", "p": "60000"})
    ev = parse_liquidation(_frame_with(order), ingested_at=_INGESTED)
    assert ev is not None
    assert ev.avg_price == 0.0
    assert ev.last_filled_qty == 0.0
    assert ev.filled_accum_qty == 0.0
    assert ev.price == pytest.approx(60000.0)


def test_parse_liquidation_negative_zero_float_accepted() -> None:
    order = _base_order()
    order.update({"ap": -0.0, "l": -0.0, "z": -0.0})
    ev = parse_liquidation(_frame_with(order), ingested_at=_INGESTED)
    assert ev is not None
    assert ev.avg_price == 0.0


@pytest.mark.parametrize("key", ["ap", "l", "z"])
def test_parse_liquidation_negative_fill_rejects(key: str) -> None:
    order = _base_order()
    order[key] = "-0.001"
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


@pytest.mark.parametrize("key", ["q", "p", "ap", "l", "z"])
def test_parse_liquidation_bool_numeric_rejects(key: str) -> None:
    order = _base_order()
    order[key] = True
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


def test_parse_liquidation_bad_T_rejects() -> None:
    bad_values: list[Any] = [True, 1568014460893.0, "1.5e12", "-5", 0, 10**20, "NONCE", "", " 123", [1], {"a": 1}, None]
    for value in bad_values:
        order = _base_order()
        order["T"] = value
        assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None, value


def test_parse_liquidation_digit_string_T_accepted() -> None:
    order = _base_order()
    order["T"] = "1568014460893"
    ev = parse_liquidation(_frame_with(order), ingested_at=_INGESTED)
    assert ev is not None
    assert ev.event_time == pd.Timestamp(1568014460893, unit="ms", tz="UTC")


@pytest.mark.parametrize("key", ["q", "p", "ap", "l", "z"])
def test_parse_liquidation_oversized_json_integer_rejects(key: str, caplog: pytest.LogCaptureFixture) -> None:
    import json

    order = _base_order()
    order[key] = 10**400
    frame = json.loads(json.dumps(_frame_with(order)))
    with caplog.at_level(logging.DEBUG, logger="src.market_data.streams.liquidations"):
        assert parse_liquidation(frame, ingested_at=_INGESTED) is None
    records = [record for record in caplog.records if record.name == "src.market_data.streams.liquidations"]
    assert len(records) == 1
    assert records[0].getMessage() == f"[DATA] stage=parse_liquidation status=REJECTED reason=type field={key}"


def test_parse_liquidation_timestamp_digit_limit_rejects() -> None:
    order = _base_order()
    order["T"] = "1" * 5000
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


@pytest.mark.parametrize("timestamp", [10**14, "100000000000000", 9223372036855])
def test_parse_liquidation_timestamp_outside_storage_range_rejects(timestamp: Any) -> None:
    order = _base_order()
    order["T"] = timestamp
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


def test_parse_liquidation_last_storage_millisecond_persists(tmp_path) -> None:
    order = _base_order()
    order["T"] = 9223372036854
    event = parse_liquidation(_frame_with(order), ingested_at=_INGESTED)
    assert event is not None
    paths = append_liquidation_events([event], tmp_path)
    persisted = pd.read_parquet(paths[0])
    assert len(persisted) == 1
    assert persisted.iloc[0]["event_time_ms"] == order["T"]
    assert persisted.iloc[0]["event_time"] == event.event_time


@pytest.mark.parametrize("key", ["s", "S", "o", "f", "X"])
@pytest.mark.parametrize("value", [123, "", " BTCUSDT", "SELL ", 4.5, ["x"], {"a": 1}, True, None])
def test_parse_liquidation_bad_string_field_rejects(key: str, value: Any) -> None:
    order = _base_order()
    order[key] = value
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


@pytest.mark.parametrize("side", ["sell", "LONG", "Buy", ""])
def test_parse_liquidation_side_outside_enum_rejects(side: str) -> None:
    order = _base_order()
    order["S"] = side
    assert parse_liquidation(_frame_with(order), ingested_at=_INGESTED) is None


def test_parse_liquidation_buy_side_parses() -> None:
    order = _base_order()
    order["S"] = "BUY"
    ev = parse_liquidation(_frame_with(order), ingested_at=_INGESTED)
    assert ev is not None
    assert ev.side == "BUY"


def test_parse_liquidation_legacy_ccxt_shapes_reject() -> None:
    valid_order = _base_order()
    flat_info = dict(valid_order, ps="BLESSUSDT", st=1)
    legacy: list[Any] = [
        {"info": {"o": dict(valid_order)}},
        {"info": dict(flat_info), "symbol": "BLESS/USDT:USDT", "timestamp": 1788092214457},
        {"symbol": "ETH/USDT:USDT", "timestamp": 1568014460893, "price": 1600.0, "baseValue": 3.2, "info": {}},
        {"symbol": "ETH/USDT:USDT", "timestamp": 1568014460893, "price": 1600.0, "amount": 3.2},
        {},
        [],
    ]
    for msg in legacy:
        assert parse_liquidation(msg, ingested_at=_INGESTED) is None  # type: ignore[arg-type]


def test_parse_liquidation_shape_rejects() -> None:
    assert parse_liquidation("x", ingested_at=_INGESTED) is None  # type: ignore[arg-type]
    assert parse_liquidation({"e": "forceOrder", "o": [1]}, ingested_at=_INGESTED) is None
    assert parse_liquidation({"e": "forceOrder"}, ingested_at=_INGESTED) is None


def test_parse_liquidation_nat_ingestion_rejects() -> None:
    assert parse_liquidation(_frame_with(_base_order()), ingested_at=pd.NaT) is None


def test_parse_liquidation_never_raises_on_arbitrary_json_values() -> None:
    grid: list[Any] = [None, True, 0, -1, 1.5, float("inf"), "", "x", "1e400", [], {}, [1], {"a": 1}]
    for key in _REQUIRED_KEYS:
        for value in grid:
            order = _base_order()
            order[key] = value
            ev = parse_liquidation(_frame_with(order), ingested_at=_INGESTED)
            assert ev is None or isinstance(ev, LiquidationEvent)
            if ev is not None:
                assert ev.price > 0
                assert ev.orig_qty > 0
                for v in (ev.orig_qty, ev.price, ev.avg_price, ev.last_filled_qty, ev.filled_accum_qty):
                    assert math.isfinite(v)
    for value in grid:
        ev = parse_liquidation({"e": "forceOrder", "E": 1, "o": value}, ingested_at=_INGESTED)  # type: ignore[dict-item]
        assert ev is None or isinstance(ev, LiquidationEvent)


def test_parse_liquidation_rejection_logs_structured_debug(caplog: pytest.LogCaptureFixture) -> None:
    order = _base_order()
    order["q"] = "0"
    with caplog.at_level(logging.DEBUG, logger="src.market_data.streams.liquidations"):
        result = parse_liquidation(_frame_with(order), ingested_at=_INGESTED)
    assert result is None
    records = [r for r in caplog.records if r.name == "src.market_data.streams.liquidations"]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    text = records[0].getMessage()
    assert "[DATA] stage=parse_liquidation status=REJECTED reason=non_positive field=q" in text
    assert "BTCUSDT" not in text
    assert "0.014" not in text
    assert "raw_order" not in text


def test_parse_liquidation_persisted_schema_unchanged(tmp_path) -> None:
    ms1 = pd.Timestamp("2026-09-24T05:00:00Z").value // 1_000_000
    ms2 = pd.Timestamp("2026-09-24T05:00:30Z").value // 1_000_000
    ev1 = parse_liquidation(_raw("BTCUSDT", ms1), ingested_at=pd.Timestamp("2026-09-24T05:00:00Z"))
    ev2 = parse_liquidation(_raw("ETHUSDT", ms2), ingested_at=pd.Timestamp("2026-09-24T05:00:01Z"))
    assert ev1 is not None
    assert ev2 is not None
    append_liquidation_events([ev1, ev2], tmp_path)
    df = pd.read_parquet(tmp_path / "liquidations_20260924_05.parquet")
    assert list(df.columns) == [
        "symbol", "event_time", "ingested_at", "side", "order_type", "time_in_force",
        "orig_qty", "price", "avg_price", "status", "last_filled_qty",
        "filled_accum_qty", "event_time_ms", "raw_order_json",
    ]
    assert df["price"].dtype == "float64"
    assert df["avg_price"].dtype == "float64"
    assert df["orig_qty"].dtype == "float32"
    assert df["last_filled_qty"].dtype == "float32"
    assert df["filled_accum_qty"].dtype == "float32"
    for col in ("side", "status", "order_type", "time_in_force"):
        assert isinstance(df[col].dtype, pd.CategoricalDtype)
    assert str(df["raw_order_json"].dtype) == "string"
    assert str(df["event_time"].dt.tz) == "UTC"
    assert str(df["ingested_at"].dt.tz) == "UTC"


def test_parse_liquidation_deterministic() -> None:
    first = parse_liquidation(_RAW_MSG, ingested_at=_INGESTED)
    second = parse_liquidation(_RAW_MSG, ingested_at=_INGESTED)
    assert first is not None
    assert second is not None
    assert first == second


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


def test_append_liquidation_events_writes_utc_event_time(tmp_path) -> None:
    """One appended event lands in the UTC-hour file with tz-aware event time."""
    ms = pd.Timestamp("2026-09-03T01:00:00Z").value // 1_000_000
    append_liquidation_events([_event("BTCUSDT", ms, 100.0, 1.0, 1.0)], tmp_path)
    df = pd.read_parquet(tmp_path / "liquidations_20260903_01.parquet")
    assert len(df) == 1
    assert str(df["event_time"].dt.tz) == "UTC"


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
