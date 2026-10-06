"""Part-3 collection universe, settlement asset, coverage persistence and gap detection."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.tax_ledger import (
    TaxLedgerCorruptError,
    TaxWatermark,
    VenuePositionSnapshot,
    collect_and_persist_live_tax,
    collect_tax_records,
    load_tax_watermark,
    save_tax_watermark,
    tax_collection_symbols,
)
from src.live.tax_summary import TaxSummaryConfig, summarize_tax_year

NOW = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


class _FakeClient:
    def __init__(self, trades=None, incomes=None, fail_trades=(), fail_income_calls=()) -> None:
        self._trades = {s: sorted(r, key=lambda t: int(t["id"])) for s, r in (trades or {}).items()}
        self._incomes = sorted(incomes or [], key=lambda e: (int(e["time"]), int(e["tranId"])))
        self._fail_trades = set(fail_trades)
        self._fail_income = set(fail_income_calls)
        self._n = 0
        self.trades_calls: list = []

    def user_trades(self, symbol, from_id=None, limit=1000):
        self.trades_calls.append(symbol)
        if symbol in self._fail_trades:
            raise RuntimeError(f"venue timeout for {symbol}")
        rows = self._trades.get(symbol, [])
        if from_id is not None:
            rows = [t for t in rows if int(t["id"]) >= int(from_id)]
        return list(rows[:limit])

    def income(self, start_time_ms=None, end_time_ms=None, limit=1000):
        self._n += 1
        if self._n in self._fail_income:
            raise RuntimeError("venue timeout for income")
        return [e for e in self._incomes
                if int(e["time"]) >= int(start_time_ms) and int(e["time"]) <= int(end_time_ms)][:limit]


def _trade(tid: int, sym: str = "BTCUSDT", ts: int = 1000, **over) -> dict:
    row = {
        "id": tid, "symbol": sym, "price": "100", "qty": "1", "quoteQty": "100",
        "commission": "0.1", "commissionAsset": "USDT", "realizedPnl": "0",
        "time": ts, "buyer": True, "maker": False, "positionSide": "BOTH",
    }
    row.update(over)
    return row


def _income(tid: int, sym: str = "BTCUSDT", ts: int = 1000, itype: str = "FUNDING_FEE") -> dict:
    return {"tranId": tid, "incomeType": itype, "income": "1.5", "asset": "USDT", "symbol": sym, "time": ts}


def _collect(client, symbols, watermark, *, now=NOW, issues=None, **over):
    kw = {
        "income_page_limit": 1000, "trades_page_limit": 1000,
        "income_window": pd.Timedelta(days=7), "income_overlap": pd.Timedelta(hours=6),
        "income_retention": pd.Timedelta(days=90), "max_pages": 200,
    }
    kw.update(over)
    found = issues if issues is not None else []
    records, new_wm = collect_tax_records(
        client, symbols, watermark, "live_testnet", now=now, settlement_asset="USDT",
        issues=found, **kw,
    )
    return records, new_wm, found


def _empty_watermark() -> TaxWatermark:
    return TaxWatermark(last_trade_id={}, last_collected_at=None)


def _settings():
    from src.live.settings import LiveSettings

    return LiveSettings()


def _snapshot(at: pd.Timestamp, positions=None) -> VenuePositionSnapshot:
    return VenuePositionSnapshot(taken_at=at, positions=dict(positions or {}))


def test_universe_keeps_every_known_symbol() -> None:
    watermark = TaxWatermark(
        last_trade_id={"AAAUSDT": 5}, last_collected_at=None,
        known_trade_symbols=frozenset({"BBBUSDT"}),
    )
    assert tax_collection_symbols((), watermark, {"CCCUSDT": Decimal("1"), "DDDUSDT": Decimal("0")}) == (
        "AAAUSDT", "BBBUSDT", "CCCUSDT",
    )


def test_universe_rejects_empty_symbols() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tax_collection_symbols([""], _empty_watermark(), {})
    stale = TaxWatermark(last_trade_id={"": 5}, last_collected_at=None)
    with pytest.raises(ValueError, match="non-empty"):
        tax_collection_symbols((), stale, {})
    with pytest.raises(ValueError, match="non-empty"):
        tax_collection_symbols((), _empty_watermark(), {"": Decimal("1")})


def test_income_observed_symbol_enters_universe() -> None:
    incomes = [
        _income(1, sym="EEEUSDT", ts=_ms(NOW - pd.Timedelta(days=2)), itype="COMMISSION"),
        _income(2, sym="", ts=_ms(NOW - pd.Timedelta(days=1)), itype="TRANSFER"),
    ]
    _, new_wm, _ = _collect(_FakeClient(incomes=incomes), [], _empty_watermark())
    assert new_wm.known_trade_symbols == frozenset({"EEEUSDT"})


def test_failed_fetch_keeps_symbol_known() -> None:
    client = _FakeClient(fail_trades=("AAAUSDT",))
    _, new_wm, found = _collect(client, ["AAAUSDT"], _empty_watermark())
    assert "AAAUSDT" in new_wm.known_trade_symbols
    assert "AAAUSDT" not in new_wm.last_trade_id
    assert [(i.stream, i.stage) for i in found] == [("trades:AAAUSDT", "fetch")]


def test_pnl_asset_is_the_settlement_asset() -> None:
    client = _FakeClient({"BTCUSDT": [_trade(1, commissionAsset="BNB", realizedPnl="1.5")]})
    records, _, _ = _collect(client, ["BTCUSDT"], _empty_watermark())
    assert len(records) == 1
    assert records[0].income_asset == "USDT"
    assert records[0].fee_asset == "BNB"


def test_first_collection_anchors_income_coverage() -> None:
    _, first_wm, _ = _collect(_FakeClient(), [], _empty_watermark())
    assert first_wm.income_covered_from == NOW - pd.Timedelta(days=90)
    _, second_wm, _ = _collect(_FakeClient(), [], first_wm, now=NOW + pd.Timedelta(days=1))
    assert second_wm.income_covered_from == first_wm.income_covered_from


def test_legacy_watermark_never_fabricates_coverage() -> None:
    watermark = TaxWatermark(last_trade_id={}, last_collected_at=NOW - pd.Timedelta(days=10))
    _, new_wm, _ = _collect(_FakeClient(), [], watermark, now=NOW)
    assert new_wm.income_covered_from is None


def test_failed_first_collection_records_no_coverage() -> None:
    client = _FakeClient(fail_income_calls=(1,))
    _, new_wm, found = _collect(client, [], _empty_watermark())
    assert new_wm.income_covered_from is None
    assert new_wm.last_collected_at is None
    assert [i.stage for i in found] == ["fetch"]


def test_retention_gap_persisted_and_merged(tmp_path: Path) -> None:
    tax_dir = tmp_path / "tax"
    old = NOW - pd.Timedelta(days=120)
    watermark = TaxWatermark(last_trade_id={}, last_collected_at=old)
    failing = _FakeClient(fail_income_calls=(1,))
    _, failed_wm, _ = _collect(failing, [], watermark)
    floor1 = NOW - pd.Timedelta(days=90)
    assert failed_wm.last_collected_at == old
    assert failed_wm.income_gaps == ((old, floor1),)
    save_tax_watermark(tax_dir / "watermark.json", failed_wm)
    assert load_tax_watermark(tax_dir / "watermark.json") == failed_wm
    later = NOW + pd.Timedelta(days=1)
    _, third_wm, _ = _collect(_FakeClient(), [], failed_wm, now=later)
    assert third_wm.income_gaps == ((old, later - pd.Timedelta(days=90)),)


def test_genesis_recorded_once(tmp_path: Path) -> None:
    tax_dir = tmp_path / "tax"
    t0 = NOW
    collect_and_persist_live_tax(
        _FakeClient(), (), tax_dir, "live_testnet", now=t0,
        settings=_settings(), venue_snapshot=_snapshot(t0),
    )
    collect_and_persist_live_tax(
        _FakeClient(), (), tax_dir, "live_testnet", now=t0 + pd.Timedelta(days=1),
        settings=_settings(), venue_snapshot=_snapshot(t0 + pd.Timedelta(days=1), {"AAAUSDT": Decimal("2")}),
    )
    saved = load_tax_watermark(tax_dir / "watermark.json")
    assert saved.genesis_at == t0
    assert dict(saved.genesis_positions) == {}


def test_no_snapshot_no_genesis(tmp_path: Path) -> None:
    tax_dir = tmp_path / "tax"
    collect_and_persist_live_tax(
        _FakeClient(), (), tax_dir, "live_testnet", now=NOW,
        settings=_settings(), venue_snapshot=None,
    )
    assert load_tax_watermark(tax_dir / "watermark.json").genesis_at is None


def test_non_flat_genesis_reported(tmp_path: Path) -> None:
    tax_dir = tmp_path / "tax"
    _, issues = collect_and_persist_live_tax(
        _FakeClient(), (), tax_dir, "live_testnet", now=NOW,
        settings=_settings(), venue_snapshot=_snapshot(NOW, {"AAAUSDT": Decimal("-0.5")}),
    )
    assert ("genesis", "genesis_not_flat") in [(i.stream, i.stage) for i in issues]
    saved = load_tax_watermark(tax_dir / "watermark.json")
    assert dict(saved.genesis_positions) == {"AAAUSDT": Decimal("-0.5")}


def test_append_failure_records_no_genesis(tmp_path: Path) -> None:
    tax_dir = tmp_path / "tax"
    tax_dir.mkdir(parents=True)
    (tax_dir / "tax_ledger_202611.jsonl").write_text("{not json}\n", encoding="utf-8")
    before = list(tax_dir.iterdir())
    client = _FakeClient({"BTCUSDT": [_trade(1, ts=_ms(pd.Timestamp("2026-11-15", tz="UTC")))]})
    with pytest.raises(TaxLedgerCorruptError):
        collect_and_persist_live_tax(
            client, ["BTCUSDT"], tax_dir, "live_testnet", now=NOW,
            settings=_settings(), venue_snapshot=_snapshot(NOW),
        )
    assert not (tax_dir / "watermark.json").exists()
    assert sorted(p.name for p in tax_dir.iterdir()) == sorted(p.name for p in before)


def test_legacy_watermark_json_loads(tmp_path: Path) -> None:
    path = tmp_path / "watermark.json"
    path.write_text(json.dumps({
        "last_trade_id": {"AAAUSDT": 3},
        "last_collected_at": "2026-09-01T00:00:00+00:00",
        "last_income_id": 9,
    }), encoding="utf-8")
    watermark = load_tax_watermark(path)
    assert watermark.last_trade_id == {"AAAUSDT": 3}
    assert watermark.known_trade_symbols == frozenset()
    assert watermark.income_covered_from is None
    assert watermark.income_gaps == ()
    assert watermark.coverage().genesis_at is None


@pytest.mark.parametrize("payload", [
    {"last_trade_id": {}, "last_collected_at": None,
     "income_gaps": [["2026-09-02T00:00:00+00:00", "2026-09-01T00:00:00+00:00"]]},
    {"last_trade_id": {}, "last_collected_at": None, "genesis_positions": {"AAAUSDT": "0"}},
    {"last_trade_id": {}, "last_collected_at": None, "known_trade_symbols": "AAAUSDT"},
    {"last_trade_id": {}, "last_collected_at": 123},
    {"last_trade_id": {}, "last_collected_at": "not-a-time"},
    {"last_trade_id": {}, "last_collected_at": None, "income_covered_from": "2026-09-01T00:00:00"},
    {"last_trade_id": {}, "last_collected_at": None, "income_gaps": {}},
    {"last_trade_id": {}, "last_collected_at": None,
     "income_gaps": [["2026-09-01T00:00:00+00:00"]]},
    {"last_trade_id": {}, "last_collected_at": None, "known_trade_symbols": ["AAAUSDT", 5]},
    {"last_trade_id": {}, "last_collected_at": None, "genesis_positions": []},
    {"last_trade_id": {}, "last_collected_at": None, "genesis_positions": {"AAAUSDT": "bogus"}},
    {"last_trade_id": {}, "last_collected_at": None, "genesis_positions": {"": "1"}},
    {"last_trade_id": {"AAAUSDT": "x"}, "last_collected_at": None},
])
def test_malformed_coverage_fields_fail_closed(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "watermark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_tax_watermark(path)


def test_naive_legacy_timestamp_localizes_to_utc(tmp_path: Path) -> None:
    path = tmp_path / "watermark.json"
    path.write_text(json.dumps({"last_trade_id": {}, "last_collected_at": "2026-09-01T00:00:00"}), encoding="utf-8")
    watermark = load_tax_watermark(path)
    assert watermark.last_collected_at == pd.Timestamp("2026-09-01T00:00:00", tz="UTC")


def test_disjoint_retention_gap_appends() -> None:
    old_gap = (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC"))
    stale_at = NOW - pd.Timedelta(days=120)
    watermark = TaxWatermark(last_trade_id={}, last_collected_at=stale_at, income_gaps=(old_gap,))
    _, new_wm, _ = _collect(_FakeClient(), [], watermark)
    assert new_wm.income_gaps == (old_gap, (stale_at, NOW - pd.Timedelta(days=90)))


def _collect_daily(tax_dir: Path, start: pd.Timestamp, end: pd.Timestamp, skip=()) -> None:
    day = start
    while day <= end:
        if day not in skip:
            collect_and_persist_live_tax(
                _FakeClient(), (), tax_dir, "live_testnet", now=day,
                settings=_settings(),
                venue_snapshot=_snapshot(day) if day == start else None,
            )
        day += pd.Timedelta(days=1)


def _summary_config():
    return TaxSummaryConfig.from_settings(_settings())


def test_coverage_drives_summary_completeness(tmp_path: Path) -> None:
    tax_dir = tmp_path / "tax"
    start = pd.Timestamp("2026-11-01", tz="UTC")
    _collect_daily(tax_dir, start, pd.Timestamp("2028-01-02", tz="UTC"))
    coverage = load_tax_watermark(tax_dir / "watermark.json").coverage()
    summary = summarize_tax_year(2027, tax_dir, source="venue", config=_summary_config(), coverage=coverage)
    assert summary["reconciliation"]["status"] == "reconciled"


def test_collection_outage_marks_year_incomplete(tmp_path: Path) -> None:
    tax_dir = tmp_path / "tax"
    start = pd.Timestamp("2026-11-01", tz="UTC")
    gap_start = pd.Timestamp("2027-03-01", tz="UTC")
    gap_end = pd.Timestamp("2027-07-01", tz="UTC")
    skipped = []
    day = gap_start
    while day < gap_end:
        skipped.append(day)
        day += pd.Timedelta(days=1)
    _collect_daily(tax_dir, start, pd.Timestamp("2028-01-02", tz="UTC"), skip=set(skipped))
    coverage = load_tax_watermark(tax_dir / "watermark.json").coverage()
    assert len(coverage.income_gaps) == 1
    summary = summarize_tax_year(2027, tax_dir, source="venue", config=_summary_config(), coverage=coverage)
    assert summary["reconciliation"]["status"] == "incomplete"
    codes = {issue["code"] for issue in summary["reconciliation"]["issues"]}
    assert "income_gap" in codes
