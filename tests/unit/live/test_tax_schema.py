"""Tax-ledger schema invariants: Decimal round trip, fail-closed load, venue parsing."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.tax_ledger import (
    TaxLedgerCorruptError,
    TaxRecord,
    append_tax_records,
    collect_tax_records,
    read_tax_ledger,
    reconcile_cycle_cash,
    simulated_tax_records,
    TaxWatermark,
)
from src.live.tax_schema import tax_record_to_row

NOW = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def _trade(rid: str, *, symbol: str = "BTCUSDT", side: str = "BUY",
           qty: str = "1", price: str = "100", when: str = "2027-03-01 00:00",
           source: str = "simulated", mode: str = "paper", venue_id: int = 1,
           position_side: str = "", pnl: str = "0") -> TaxRecord:
    return TaxRecord(
        record_id=rid,
        kind="TRADE",
        event_time=pd.Timestamp(when, tz="UTC"),
        symbol=symbol,
        side=side,
        quantity=Decimal(qty),
        price=Decimal(price),
        quote_qty=Decimal(qty) * Decimal(price),
        fee=Decimal("0"),
        fee_asset="USDT",
        realized_pnl=Decimal(pnl),
        income_asset="USDT",
        is_maker=False,
        venue_id=venue_id,
        source=source,
        mode=mode,
        position_side=position_side,
    )


def _venue_trade(tid: int, **over) -> dict:
    entry = {
        "id": tid,
        "symbol": "BTCUSDT",
        "price": "100",
        "qty": "1",
        "quoteQty": "100",
        "commission": "0.1",
        "commissionAsset": "USDT",
        "realizedPnl": "0",
        "time": 1000,
        "buyer": True,
        "maker": False,
        "positionSide": "BOTH",
    }
    entry.update(over)
    return entry


class _StubClient:
    def __init__(self, trades=None, incomes=None) -> None:
        self._trades = trades or {}
        self._incomes = incomes or []

    def user_trades(self, symbol, from_id=None, limit=1000):
        rows = self._trades.get(symbol, [])
        if from_id is not None:
            rows = [t for t in rows if int(t["id"]) >= int(from_id)]
        return list(rows[:limit])

    def income(self, start_time_ms=None, end_time_ms=None, limit=1000):
        return [e for e in self._incomes
                if int(e["time"]) >= int(start_time_ms) and int(e["time"]) <= int(end_time_ms)][:limit]


def _collect(client, symbols, watermark, **over):
    kw = {
        "income_page_limit": 1000,
        "trades_page_limit": 1000,
        "income_window": pd.Timedelta(days=7),
        "income_overlap": pd.Timedelta(hours=6),
        "income_retention": pd.Timedelta(days=90),
        "max_pages": 200,
    }
    kw.update(over)
    found: list = []
    records, new_wm = collect_tax_records(
        client, symbols, watermark, "live_testnet", now=NOW, settlement_asset="USDT",
        issues=found, **kw
    )
    return records, new_wm, found


def test_decimal_round_trip_is_bit_exact(tmp_path: Path) -> None:
    import dataclasses

    rec = _trade("simulated:TRADE:journal:1", qty="0.001", price="67890.12", venue_id=1)
    rec = dataclasses.replace(rec, fee=Decimal("0.00000001"))
    ledger_dir = tmp_path / "tax"
    append_tax_records([rec], ledger_dir)
    (loaded,) = read_tax_ledger(ledger_dir)
    assert loaded == rec
    shard = next(ledger_dir.glob("tax_ledger_*.jsonl"))
    assert '"fee": "0.00000001"' in shard.read_text(encoding="utf-8")


def test_legacy_json_numbers_load_exactly(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    row = tax_record_to_row(_trade("L1"))
    row["fee"] = 0.1
    row["quote_qty"] = 100.0
    del row["income_type"]
    del row["position_side"]
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (loaded,) = read_tax_ledger(ledger_dir)
    assert loaded.fee == Decimal("0.1")
    assert loaded.income_type == ""
    assert loaded.position_side == ""


def test_float_amount_rejected_at_construction() -> None:
    import dataclasses

    with pytest.raises(TypeError):
        dataclasses.replace(_trade("X"), quantity=1.0)


@pytest.mark.parametrize(
    "override",
    [
        {"quantity": "abc"},
        {"fee": "oops"},
        {"realized_pnl": None},
        {"side": "SHORT"},
        {"quantity": None},
    ],
    ids=["bad_quantity", "bad_fee", "null_pnl", "bad_side", "null_quantity"],
)
def test_malformed_rows_fail_closed_with_line_number(tmp_path: Path, override: dict) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    good = tax_record_to_row(_trade("G1"))
    bad = tax_record_to_row(_trade("G2"))
    bad.update(override)
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(
        json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8"
    )
    with pytest.raises(TaxLedgerCorruptError) as exc_info:
        read_tax_ledger(ledger_dir)
    assert exc_info.value.line_number == 2


def test_non_finite_json_constant_is_corruption(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    row = tax_record_to_row(_trade("G1"))
    line = json.dumps(row).replace('"100"', "NaN", 1)
    assert "NaN" in line
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(line + "\n", encoding="utf-8")
    with pytest.raises(TaxLedgerCorruptError):
        read_tax_ledger(ledger_dir)


def test_naive_event_time_rejected(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    row = tax_record_to_row(_trade("G1"))
    row["event_time"] = "2027-01-01T00:00:00"
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(TaxLedgerCorruptError):
        read_tax_ledger(ledger_dir)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    row = tax_record_to_row(_trade("G1"))
    row["tax_rate"] = "0.22"
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(TaxLedgerCorruptError):
        read_tax_ledger(ledger_dir)


@pytest.mark.parametrize("position_side", ["LONG", None])
def test_venue_trade_must_be_one_way(tmp_path: Path, position_side) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    row = tax_record_to_row(
        _trade("venue:TRADE:9", source="venue", mode="live_testnet", position_side="BOTH", venue_id=9)
    )
    if position_side is None:
        del row["position_side"]
    else:
        row["position_side"] = position_side
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(TaxLedgerCorruptError):
        read_tax_ledger(ledger_dir)


def test_venue_trade_both_loads(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    row = tax_record_to_row(
        _trade("venue:TRADE:9", source="venue", mode="live_testnet", position_side="BOTH", venue_id=9)
    )
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (loaded,) = read_tax_ledger(ledger_dir)
    assert loaded.position_side == "BOTH"


def test_simulated_trade_carries_no_venue_pnl(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    with pytest.raises(DataIntegrityError):
        append_tax_records([_trade("S1", pnl="1")], ledger_dir)
    assert list(ledger_dir.glob("tax_ledger_*.jsonl")) == []


@pytest.mark.parametrize(
    ("source", "mode"),
    [("venue", "paper"), ("simulated", "live_mainnet")],
    ids=["venue_paper", "simulated_live"],
)
def test_source_and_mode_must_be_compatible(source: str, mode: str) -> None:
    import dataclasses

    from src.live.tax_schema import validate_tax_record

    if source == "venue":
        rec = _trade("S1", source="venue", mode="live_testnet", position_side="BOTH", venue_id=1)
    else:
        rec = _trade("S1")
    rec = dataclasses.replace(rec, mode=mode)
    with pytest.raises(DataIntegrityError):
        validate_tax_record(rec)


def test_append_validates_whole_batch_before_writing(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    with pytest.raises(DataIntegrityError):
        append_tax_records([_trade("S1"), _trade("S2", pnl="1")], ledger_dir)
    assert list(ledger_dir.glob("tax_ledger_*.jsonl")) == []


def test_duplicate_identical_record_loads_once(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    row = tax_record_to_row(_trade("D1", when="2027-01-15 00:00"))
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (ledger_dir / "tax_ledger_202702.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(read_tax_ledger(ledger_dir)) == 1


def test_conflicting_duplicate_is_corruption(tmp_path: Path) -> None:
    import dataclasses

    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    first = tax_record_to_row(_trade("D1", when="2027-01-15 00:00"))
    second = tax_record_to_row(
        dataclasses.replace(_trade("D1", when="2027-01-15 00:00"), fee=Decimal("0.01"))
    )
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(json.dumps(first) + "\n", encoding="utf-8")
    (ledger_dir / "tax_ledger_202702.jsonl").write_text(json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(TaxLedgerCorruptError) as exc_info:
        read_tax_ledger(ledger_dir)
    assert exc_info.value.line_number == 1
    assert "tax_ledger_202702" in str(exc_info.value.path)


def test_same_millisecond_fills_ordered_by_venue_id(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    rows = [
        tax_record_to_row(_trade("venue:TRADE:10", source="venue", mode="live_testnet",
                                 position_side="BOTH", venue_id=10, when="2027-01-15 00:00")),
        tax_record_to_row(_trade("venue:TRADE:9", source="venue", mode="live_testnet",
                                 position_side="BOTH", venue_id=9, when="2027-01-15 00:00")),
    ]
    (ledger_dir / "tax_ledger_202701.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    loaded = read_tax_ledger(ledger_dir)
    assert [r.record_id for r in loaded] == ["venue:TRADE:9", "venue:TRADE:10"]


def test_only_tax_shards_are_read(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    (ledger_dir / "other.jsonl").write_text(
        json.dumps(tax_record_to_row(_trade("D1"))) + "\n", encoding="utf-8"
    )
    assert read_tax_ledger(ledger_dir) == ()
    assert read_tax_ledger(tmp_path / "absent") == ()


def test_venue_numerics_parsed_exactly() -> None:
    client = _StubClient(trades={"BTCUSDT": [_venue_trade(4, commission="0.00000001")]})
    records, _, issues = _collect(
        client, ["BTCUSDT"], TaxWatermark(last_trade_id={}, last_collected_at=None)
    )
    assert issues == []
    (rec,) = records
    assert rec.fee == Decimal("0.00000001")
    assert rec.position_side == "BOTH"


@pytest.mark.parametrize(
    "entry",
    [
        _venue_trade(6, price=100.5),
        _venue_trade(6, positionSide="SHORT"),
        {k: v for k, v in _venue_trade(6).items() if k != "positionSide"},
        _venue_trade(6, commissionAsset=""),
    ],
    ids=["float_price", "hedge_side", "missing_position_side", "missing_fee_asset"],
)
def test_venue_float_or_hedge_row_is_a_parse_issue(entry: dict) -> None:
    client = _StubClient(trades={"BTCUSDT": [_venue_trade(5), entry, _venue_trade(7)]})
    records, new_wm, found = _collect(
        client, ["BTCUSDT"], TaxWatermark(last_trade_id={"BTCUSDT": 4}, last_collected_at=None)
    )
    assert sorted(r.venue_id for r in records) == [5, 7]
    assert len(found) == 1
    assert found[0].stage == "parse"
    assert new_wm.last_trade_id == {"BTCUSDT": 5}


def test_simulated_fee_equals_paper_cash_formula() -> None:
    from src.live.ledger import _fill_cash_delta

    ts = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    fill = SimpleNamespace(
        fill_id="journal:3",
        quantity_delta=Decimal("0.003"),
        fill_price=Decimal("12345.6"),
        fee_bps=4.5,
        timestamp=ts,
        symbol="BTCUSDT",
        liquidity="taker",
    )
    (rec,) = simulated_tax_records([fill], "paper")
    cash_before = Decimal("2000")
    cash_after = cash_before + _fill_cash_delta("BUY", Decimal("0.003"), Decimal("12345.6"), 4.5)
    result = reconcile_cycle_cash(
        cash_before, cash_after, [rec], [], tolerance_usdt=Decimal("0")
    )
    assert result.difference == Decimal("0")
    assert result.within_tolerance is True


def _funding(rid: str = "F1", **over) -> TaxRecord:
    kw = {
        "record_id": rid,
        "kind": "FUNDING_FEE",
        "event_time": pd.Timestamp("2027-03-01 08:00", tz="UTC"),
        "symbol": "BTCUSDT",
        "side": "",
        "quantity": Decimal("2"),
        "price": Decimal("100"),
        "quote_qty": Decimal("200"),
        "fee": Decimal("0"),
        "fee_asset": "USDT",
        "realized_pnl": Decimal("-0.2"),
        "income_asset": "USDT",
        "is_maker": False,
        "venue_id": 0,
        "source": "simulated",
        "mode": "paper",
    }
    kw.update(over)
    return TaxRecord(**kw)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("0.1"), Decimal("0.1")),
        (5, Decimal("5")),
        ("0.00000001", Decimal("0.00000001")),
    ],
    ids=["decimal", "int", "string"],
)
def test_parse_tax_decimal_accepts_exact_forms(value: object, expected: Decimal) -> None:
    from src.live.tax_schema import parse_tax_decimal

    assert parse_tax_decimal(value, field="fee") == expected


@pytest.mark.parametrize(
    "value",
    [True, 0.1, None, "", " 1", "1,000", "abc", "NaN", "Infinity", "-Infinity", Decimal("NaN"), [1]],
    ids=["bool", "float", "none", "empty", "padded", "grouped", "alpha", "nan", "inf", "ninf",
         "decimal_nan", "list"],
)
def test_parse_tax_decimal_rejects_everything_else(value: object) -> None:
    from src.live.tax_schema import parse_tax_decimal

    with pytest.raises(DataIntegrityError):
        parse_tax_decimal(value, field="fee")


@pytest.mark.parametrize(
    ("field", "value"),
    [("is_maker", 1), ("venue_id", True), ("venue_id", "1"), ("quantity", 1)],
    ids=["maker_int", "venue_bool", "venue_str", "quantity_int"],
)
def test_constructor_rejects_mistyped_fields(field: str, value: object) -> None:
    import dataclasses

    with pytest.raises(TypeError):
        dataclasses.replace(_trade("X"), **{field: value})


def test_constructor_rejects_naive_event_time() -> None:
    import dataclasses

    with pytest.raises(TypeError):
        dataclasses.replace(_trade("X"), event_time=pd.Timestamp("2027-01-01 00:00"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_id", ""), ("kind", "BOGUS"), ("source", "backtest"),
        ("symbol", ""), ("quantity", Decimal("0")), ("price", Decimal("0")),
        ("quote_qty", Decimal("-1")), ("income_asset", ""),
        ("income_type", "FUNDING_FEE"),
    ],
    ids=["empty_id", "bad_kind", "bad_source", "empty_symbol", "zero_qty", "zero_price",
         "neg_quote", "empty_income_asset", "trade_income_type"],
)
def test_validate_rejects_each_trade_rule(field: str, value: object) -> None:
    import dataclasses

    from src.live.tax_schema import validate_tax_record

    with pytest.raises(DataIntegrityError):
        validate_tax_record(dataclasses.replace(_trade("V1"), **{field: value}))


def test_validate_rejects_unsatisfiable_fee_asset() -> None:
    import dataclasses

    from src.live.tax_schema import validate_tax_record

    with pytest.raises(DataIntegrityError):
        validate_tax_record(
            dataclasses.replace(_trade("V1"), fee=Decimal("0.01"), fee_asset="")
        )


def test_validate_rejects_simulated_position_side() -> None:
    import dataclasses

    from src.live.tax_schema import validate_tax_record

    with pytest.raises(DataIntegrityError):
        validate_tax_record(dataclasses.replace(_trade("V1"), position_side="BOTH"))


def test_validate_rejects_non_finite_amount() -> None:
    import dataclasses

    from src.live.tax_schema import validate_tax_record

    with pytest.raises(DataIntegrityError):
        validate_tax_record(dataclasses.replace(_trade("V1"), quantity=Decimal("NaN")))


@pytest.mark.parametrize(
    ("field", "value"),
    [("side", "BUY"), ("position_side", "BOTH"), ("income_asset", "")],
    ids=["income_side", "income_position_side", "empty_income_asset"],
)
def test_validate_rejects_each_income_rule(field: str, value: object) -> None:
    import dataclasses

    from src.live.tax_schema import validate_tax_record

    with pytest.raises(DataIntegrityError):
        validate_tax_record(dataclasses.replace(_funding(), **{field: value}))


def test_validate_rejects_venue_income_without_type() -> None:
    import dataclasses

    from src.live.tax_schema import validate_tax_record

    rec = dataclasses.replace(
        _funding(), source="venue", mode="live_testnet", income_type="FUNDING_FEE", venue_id=7,
        record_id="venue:FUNDING_FEE:7",
    )
    validate_tax_record(rec)
    with pytest.raises(DataIntegrityError):
        validate_tax_record(dataclasses.replace(rec, income_type=""))
    with pytest.raises(DataIntegrityError):
        validate_tax_record(dataclasses.replace(rec, symbol=""))


def test_validate_rejects_symbolic_funding_without_symbol() -> None:
    import dataclasses

    from src.live.tax_schema import validate_tax_record

    with pytest.raises(DataIntegrityError):
        validate_tax_record(dataclasses.replace(_funding(), symbol=""))


def test_from_row_rejects_non_object() -> None:
    from src.live.tax_schema import tax_record_from_row

    with pytest.raises(DataIntegrityError):
        tax_record_from_row(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_from_row_rejects_missing_key() -> None:
    from src.live.tax_schema import tax_record_from_row

    row = tax_record_to_row(_trade("G1"))
    del row["mode"]
    with pytest.raises(DataIntegrityError):
        tax_record_from_row(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [("symbol", 5), ("is_maker", 1), ("venue_id", True), ("venue_id", 1.5),
     ("event_time", 12345), ("event_time", "not-a-time")],
    ids=["symbol_int", "maker_int", "venue_bool", "venue_float", "time_int", "time_garbage"],
)
def test_from_row_rejects_mistyped_fields(field: str, value: object) -> None:
    from src.live.tax_schema import tax_record_from_row

    row = tax_record_to_row(_trade("G1"))
    row[field] = value
    with pytest.raises(DataIntegrityError):
        tax_record_from_row(row)


def test_simulated_fill_with_unnumeric_fee_bps_fails_closed() -> None:
    ts = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    fill = SimpleNamespace(
        fill_id="journal:3",
        quantity_delta=Decimal("1"),
        fill_price=Decimal("100"),
        fee_bps="eight",
        timestamp=ts,
        symbol="BTCUSDT",
        liquidity="taker",
    )
    with pytest.raises(DataIntegrityError):
        simulated_tax_records([fill], "paper")


@pytest.mark.parametrize(
    "field",
    ["kind", "symbol", "side", "fee_asset", "income_asset", "source", "mode", "income_type", "position_side"],
)
def test_append_rejects_non_string_fields_before_touching_shard(tmp_path: Path, field: str) -> None:
    import dataclasses

    ledger_dir = tmp_path / "tax"
    append_tax_records([_trade("existing")], ledger_dir)
    shard = ledger_dir / "tax_ledger_202703.jsonl"
    shard.write_bytes(shard.read_bytes() + b'{"record_id": "torn')
    before = shard.read_bytes()
    invalid = dataclasses.replace(_trade("invalid"), **{field: None})

    with pytest.raises(DataIntegrityError, match=field):
        append_tax_records([_trade("valid"), invalid], ledger_dir)

    assert shard.read_bytes() == before
    assert [record.record_id for record in read_tax_ledger(ledger_dir)] == ["existing"]


@pytest.mark.parametrize("field", ["genesis_at", "income_covered_from", "collected_through"])
@pytest.mark.parametrize("value", [pd.NaT, pd.Timestamp("2027-01-01"), pd.Timestamp("2027-01-01", tz="Asia/Seoul")])
def test_coverage_rejects_invalid_timestamp(field, value):
    from dataclasses import replace

    from src.live.tax_schema import TaxCoverage

    coverage = TaxCoverage(None, {}, None, None, ())
    with pytest.raises(ValueError, match=field):
        replace(coverage, **{field: value})


@pytest.mark.parametrize("positions", [{"": Decimal("1")}, {"BTCUSDT": Decimal("NaN")},
                                       {"BTCUSDT": Decimal("0")}, {"BTCUSDT": 1}])
def test_coverage_rejects_invalid_genesis_positions(positions):
    from src.live.tax_schema import TaxCoverage

    with pytest.raises(ValueError, match="genesis_positions"):
        TaxCoverage(None, positions, None, None, ())


def test_coverage_validates_gap_intervals():
    from src.live.tax_schema import TaxCoverage

    start = pd.Timestamp("2027-01-01T00:00Z")
    end = pd.Timestamp("2027-01-02T00:00Z")
    for gaps in [((pd.NaT, end),), ((start, pd.NaT),), ((end, start),),
                 ((start, end), (start, end)), ((end, end), (start, start))]:
        with pytest.raises(ValueError, match="income_gaps"):
            TaxCoverage(None, {}, None, None, gaps)
    coverage = TaxCoverage(start, {"BTCUSDT": Decimal("-1")}, start, end, ((start, end),))
    assert coverage.income_gaps == ((start, end),)
