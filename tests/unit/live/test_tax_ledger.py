"""Tax ledger durability and lossless venue collection (live edge 07)."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.tax_ledger import (
    INCOME_TYPE_KIND,
    TaxCollectionIssue,
    TaxLedgerCorruptError,
    TaxRecord,
    TaxWatermark,
    append_tax_records,
    classify_income_type,
    collect_and_persist_live_tax,
    collect_tax_records,
    load_tax_records,
    load_tax_watermark,
    save_tax_watermark,
    simulated_tax_records,
    summarize_tax_year,
)

NOW = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


class WindowedFakeClient:
    """Venue stub with window-sliced income and id-paged trades.

    income() filters rows to [start_time_ms, end_time_ms] and returns the first
    ``limit`` rows in (time, tranId) order; user_trades() filters by id.
    """

    def __init__(
        self,
        trades_by_symbol: dict | None = None,
        incomes: list | None = None,
        failing_trade_symbols: tuple = (),
        fail_income_calls: tuple = (),
    ) -> None:
        self._trades = {
            sym: sorted(rows, key=lambda t: int(t["id"]))
            for sym, rows in (trades_by_symbol or {}).items()
        }
        self._incomes = sorted(incomes or [], key=lambda e: (int(e["time"]), int(e["tranId"])))
        self._failing_trade_symbols = set(failing_trade_symbols)
        self._fail_income_calls = set(fail_income_calls)
        self._income_n = 0
        self.trades_calls: list = []
        self.income_calls: list = []

    def user_trades(self, symbol, from_id=None, limit=1000):
        self.trades_calls.append((symbol, from_id, limit))
        if symbol in self._failing_trade_symbols:
            raise RuntimeError(f"venue timeout for {symbol}")
        trades = self._trades.get(symbol, [])
        if from_id is not None:
            trades = [t for t in trades if int(t["id"]) >= int(from_id)]
        return list(trades[:limit])

    def income(self, start_time_ms=None, end_time_ms=None, limit=1000):
        self.income_calls.append((start_time_ms, end_time_ms, limit))
        self._income_n += 1
        if self._income_n in self._fail_income_calls:
            raise RuntimeError("venue timeout for income")
        rows = [
            e
            for e in self._incomes
            if int(e["time"]) >= int(start_time_ms) and int(e["time"]) <= int(end_time_ms)
        ]
        return list(rows[:limit])


def _venue_trade(tid: int, price: str = "100", sym: str = "BTCUSDT", ts: int = 1000) -> dict:
    return {
        "id": tid,
        "symbol": sym,
        "price": price,
        "qty": "1",
        "quoteQty": "100",
        "commission": "0.1",
        "commissionAsset": "USDT",
        "realizedPnl": "0",
        "time": ts,
        "buyer": True,
        "maker": False,
    }


def _venue_income(
    tid: int,
    income: str = "1.5",
    sym: str = "BTCUSDT",
    ts: int = 1000,
    itype: str = "FUNDING_FEE",
    asset: str = "USDT",
) -> dict:
    return {
        "tranId": tid,
        "incomeType": itype,
        "income": income,
        "asset": asset,
        "symbol": sym,
        "time": ts,
    }


def _collect(client, symbols, watermark, *, now=NOW, issues=None, **over):
    kw = {
        "income_page_limit": 1000,
        "trades_page_limit": 1000,
        "income_window": pd.Timedelta(days=7),
        "income_overlap": pd.Timedelta(hours=6),
        "income_retention": pd.Timedelta(days=90),
        "max_pages": 200,
    }
    kw.update(over)
    found = issues if issues is not None else []
    records, new_wm = collect_tax_records(
        client, symbols, watermark, "live_testnet", now=now, issues=found, **kw
    )
    return records, new_wm, found


def _tax_record(rid: str, when: str = "2026-09-15 00:00", **over) -> TaxRecord:
    kw = {
        "record_id": rid,
        "kind": "TRADE",
        "event_time": pd.Timestamp(when, tz="UTC"),
        "symbol": "AAAUSDT",
        "side": "BUY",
        "quantity": 1.0,
        "price": 100.0,
        "quote_qty": 100.0,
        "fee": 0.0,
        "fee_asset": "USDT",
        "realized_pnl": 0.0,
        "income_asset": "USDT",
        "is_maker": False,
        "venue_id": 1,
        "source": "simulated",
        "mode": "paper",
    }
    kw.update(over)
    return TaxRecord(**kw)


def _sim_fill(fill_id, qty, price, ts, symbol="BTCUSDT", fee_bps=8.0, liquidity="taker"):
    return SimpleNamespace(
        fill_id=fill_id,
        quantity_delta=qty,
        fill_price=price,
        fee_bps=fee_bps,
        timestamp=ts,
        symbol=symbol,
        liquidity=liquidity,
    )


def _default_settings():
    from src.live.settings import LiveSettings

    return LiveSettings()


# --- migrated pre-existing coverage -------------------------------------------


def test_collect_trades_watermark_idempotent(tmp_path: Path):
    client = WindowedFakeClient({"BTCUSDT": [_venue_trade(1), _venue_trade(2), _venue_trade(3)]})
    wm = TaxWatermark(last_trade_id={}, last_collected_at=None)
    records1, wm1, _ = _collect(client, ["BTCUSDT"], wm)
    assert len(records1) == 3
    records2, _wm2, _ = _collect(client, ["BTCUSDT"], wm1)
    assert client.trades_calls[-1][1] == 4
    assert len(records2) == 0
    ledger_dir = tmp_path / "tax"
    append_tax_records(records1, ledger_dir)
    append_tax_records(records2, ledger_dir)
    append_tax_records(records1, ledger_dir)
    assert len(load_tax_records(ledger_dir)) == 3


def test_summarize_rejects_mixed_sources(tmp_path: Path):
    ledger_dir = tmp_path / "tax2"
    venue_rec = _tax_record(
        "venue:TRADE:1",
        when="2027-01-15 00:00",
        symbol="BTCUSDT",
        source="venue",
        mode="live_testnet",
        venue_id=1,
    )
    sim_rec = _tax_record("simulated:TRADE:journal:1", when="2027-06-15 00:00", symbol="BTCUSDT", venue_id=1)
    append_tax_records([venue_rec, sim_rec], ledger_dir)
    with pytest.raises(DataIntegrityError):
        summarize_tax_year(2027, ledger_dir, source="venue")
    ledger_dir2 = tmp_path / "tax3"
    append_tax_records([sim_rec], ledger_dir2)
    summary = summarize_tax_year(2027, ledger_dir2, source="simulated")
    assert summary["source"] == "simulated"


def test_summarize_moving_average_and_fifo_cost_basis(tmp_path: Path):
    ledger_dir = tmp_path / "tax4"
    recs = [
        _tax_record("venue:TRADE:1", when="2027-01-10 00:00", symbol="BTCUSDT", source="venue",
                    mode="live_testnet", venue_id=1, fee=0.0),
        _tax_record("venue:TRADE:2", when="2027-02-10 00:00", symbol="BTCUSDT", source="venue",
                    mode="live_testnet", venue_id=2, price=200.0, quote_qty=200.0, fee=0.0),
        _tax_record("venue:TRADE:3", when="2027-03-10 00:00", symbol="BTCUSDT", source="venue",
                    mode="live_testnet", venue_id=3, side="SELL", price=300.0, quote_qty=300.0, fee=0.0),
    ]
    append_tax_records(recs, ledger_dir)
    summ_ma = summarize_tax_year(2027, ledger_dir, cost_basis="moving_average", source="venue")
    assert summ_ma["per_symbol"]["BTCUSDT"]["acquisition_cost"] == pytest.approx(300.0)
    assert summ_ma["per_symbol"]["BTCUSDT"]["disposal_proceeds"] == pytest.approx(300.0)
    assert summ_ma["per_symbol"]["BTCUSDT"]["closing_quantity"] == pytest.approx(1.0)
    assert summ_ma["per_symbol"]["BTCUSDT"]["closing_cost_basis"] == pytest.approx(150.0)
    summ_fifo = summarize_tax_year(2027, ledger_dir, cost_basis="fifo", source="venue")
    assert summ_fifo["per_symbol"]["BTCUSDT"]["closing_cost_basis"] == pytest.approx(200.0)
    for k in ("tax_rate", "income_type", "deduction"):
        assert k not in summ_ma
        assert k not in summ_fifo


def test_funding_tax_record_ids_are_deterministic(tmp_path: Path) -> None:
    """Same events twice yield identical ids of the pinned form; pnl equals amount, fee is 0."""
    from src.live.ledger import FundingEvent
    from src.live.tax_ledger import funding_tax_records

    epoch = pd.Timestamp("2026-09-01 08:00", tz="UTC")
    events = (
        FundingEvent(
            symbol="AAAUSDT",
            epoch=epoch,
            rate=Decimal("0.001"),
            quantity=Decimal("2"),
            price=Decimal("100"),
            amount=Decimal("-0.2"),
            price_source="trade_close_1h",
        ),
    )
    first = funding_tax_records(events, run_id="run1", mode="paper")
    second = funding_tax_records(events, run_id="run1", mode="paper")
    epoch_ms = int(epoch.timestamp() * 1000)
    assert [r.record_id for r in first] == [f"simulated:FUNDING_FEE:run1:AAAUSDT:{epoch_ms}"]
    assert [r.record_id for r in second] == [r.record_id for r in first]
    assert first[0].kind == "FUNDING_FEE"
    assert first[0].realized_pnl == float(Decimal("-0.2"))
    assert first[0].fee == 0.0


def test_append_tax_records_skips_ids_already_in_shard(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    first_written = append_tax_records([_tax_record("X")], ledger_dir)
    assert len(first_written) == 1
    written = append_tax_records([_tax_record("X"), _tax_record("Y")], ledger_dir)
    assert written == first_written
    lines = (ledger_dir / "tax_ledger_202609.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert sorted(json.loads(line)["record_id"] for line in lines) == ["X", "Y"]


def test_append_tax_records_deduplicates_within_one_batch(tmp_path: Path) -> None:
    rec = _tax_record("Y")
    ledger_dir = tmp_path / "tax"
    append_tax_records([rec, rec], ledger_dir)
    lines = (ledger_dir / "tax_ledger_202609.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_append_tax_records_replay_after_crash_writes_nothing_new(tmp_path: Path) -> None:
    rec = _tax_record("Y")
    ledger_dir = tmp_path / "tax"
    append_tax_records([rec], ledger_dir)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    before = shard.read_bytes()
    assert append_tax_records([rec], ledger_dir) == []
    assert shard.read_bytes() == before


def test_append_tax_records_corrupt_shard_fails_closed(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    shard.write_text("not json\n", encoding="utf-8")
    with pytest.raises(TaxLedgerCorruptError) as exc_info:
        append_tax_records([_tax_record("Y")], ledger_dir)
    assert exc_info.value.path == shard
    assert exc_info.value.line_number == 1
    assert shard.read_text(encoding="utf-8") == "not json\n"


def test_append_tax_records_rejects_shard_line_without_record_id(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    shard.write_text(
        "\n".join(
            [
                "",
                json.dumps({"kind": "TRADE", "event_time": "2026-09-15T00:00:00+00:00"}),
                json.dumps(
                    {
                        "record_id": "K", "kind": "TRADE", "event_time": "2026-09-15T00:00:00+00:00",
                        "symbol": "AAAUSDT", "side": "BUY", "quantity": 1.0, "price": 100.0,
                        "quote_qty": 100.0, "fee": 0.0, "fee_asset": "USDT", "realized_pnl": 0.0,
                        "income_asset": "USDT", "is_maker": False, "venue_id": 1,
                        "source": "simulated", "mode": "paper",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError, match="record_id"):
        append_tax_records([_tax_record("Y")], ledger_dir)
    # The loader also fails closed instead of silently skipping the bad line.
    with pytest.raises(TaxLedgerCorruptError):
        load_tax_records(ledger_dir)


def test_append_tax_records_month_routing_preserved(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    written = append_tax_records(
        [_tax_record("S", when="2026-09-15 00:00"), _tax_record("O", when="2026-10-02 00:00")], ledger_dir
    )
    assert sorted(p.name for p in written) == ["tax_ledger_202609.jsonl", "tax_ledger_202610.jsonl"]


def test_reconcile_cycle_cash_matches_trades_fees_and_funding() -> None:
    from src.live.tax_ledger import reconcile_cycle_cash

    def _trade(rid: str, side: str, quote: float, fee: float) -> TaxRecord:
        return _tax_record(rid, side=side, price=quote, quote_qty=quote, fee=fee)

    def _funding(rid: str, pnl: float) -> TaxRecord:
        return _tax_record(rid, when="2026-09-15 08:00", kind="FUNDING_FEE", side="", realized_pnl=pnl, venue_id=0)

    cash_before = Decimal("2100")
    cash_after = Decimal("2100") - Decimal("100") - Decimal("0.02") + Decimal("50") - Decimal("0.01") - Decimal("0.5")
    result = reconcile_cycle_cash(
        cash_before,
        cash_after,
        [_trade("T1", "BUY", 100.0, 0.02), _trade("T2", "SELL", 50.0, 0.01)],
        [_funding("F1", -0.5)],
        tolerance_usdt=Decimal("0.01"),
    )
    assert result.within_tolerance is True
    assert result.difference == Decimal("0")


def test_reconcile_cycle_cash_flags_unexplained_cash() -> None:
    from src.live.tax_ledger import reconcile_cycle_cash

    trade = _tax_record("T1", fee=0.02)
    result = reconcile_cycle_cash(
        Decimal("2100"),
        Decimal("2100") - Decimal("100") - Decimal("0.02") - Decimal("0.53"),
        [trade],
        [],
        tolerance_usdt=Decimal("0.01"),
    )
    assert result.within_tolerance is False
    assert result.difference == Decimal("-0.53")


def test_reconcile_cycle_cash_rejects_wrong_record_kinds() -> None:
    from src.live.tax_ledger import reconcile_cycle_cash

    funding = _tax_record("F1", when="2026-09-15 08:00", kind="FUNDING_FEE", side="",
                          realized_pnl=-0.5, venue_id=0)
    with pytest.raises(ValueError, match="TRADE"):
        reconcile_cycle_cash(Decimal("2100"), Decimal("2100"), [funding], [], tolerance_usdt=Decimal("0.01"))
    trade = _tax_record("T1", fee=0.02)
    with pytest.raises(ValueError, match="FUNDING_FEE"):
        reconcile_cycle_cash(Decimal("2100"), Decimal("2100"), [], [trade], tolerance_usdt=Decimal("0.01"))
    bad_side = _tax_record("T9", side="HOLD", fee=0.02, venue_id=9)
    with pytest.raises(ValueError, match="BUY or SELL"):
        reconcile_cycle_cash(Decimal("2100"), Decimal("2100"), [bad_side], [], tolerance_usdt=Decimal("0.01"))


def test_reconcile_cycle_cash_tolerates_empty_side_zero_quote() -> None:
    """A zero-quantity fill maps to side "" with zero quote; only its fee counts."""
    from src.live.tax_ledger import reconcile_cycle_cash

    flat = _tax_record("T0", side="", quantity=0.0, quote_qty=0.0, fee=0.0, venue_id=0)
    result = reconcile_cycle_cash(
        Decimal("2100"), Decimal("2100"), [flat], [], tolerance_usdt=Decimal("0.01")
    )
    assert result.within_tolerance is True
    assert result.difference == Decimal("0")


def test_simulated_tax_records_unique_id_across_cycles(tmp_path: Path) -> None:
    """Paper/shadow simulated records must survive multi-day read-time dedup."""
    from src.live.fills import FillEvent

    ledger_dir = tmp_path / "tax"

    def _fill(day: str, sym: str, qty: str, seq: int) -> FillEvent:
        ts = pd.Timestamp(f"2026-0{day}", tz="UTC")
        return FillEvent(
            decision_time=ts, timestamp=ts, symbol=sym, quantity_delta=Decimal(qty),
            fill_price=Decimal("100"), fee_bps=8.0, reason="immediate_taker",
            pre_trade_equity=Decimal("2000"), liquidity="taker", mode="paper",
            run_id=ts.strftime("%Y%m%d"), leg_index=0, client_order_id="c",
            fill_id=f"journal:{seq}",
        )

    seq = 0
    for day in ("3-01", "3-02", "3-03"):
        fills = []
        for sym, qty in (("BTCUSDT", "1"), ("ETHUSDT", "2")):
            seq += 1
            fills.append(_fill(day, sym, qty, seq))
        recs = simulated_tax_records(fills, "paper")
        append_tax_records(recs, ledger_dir)

    loaded = load_tax_records(ledger_dir, year=2026)
    # 3 cycles x 2 symbols = 6 distinct records, none dropped by record_id dedup
    assert len(loaded) == 6
    assert loaded["record_id"].nunique() == 6
    assert sorted(loaded["event_time"].dt.strftime("%Y-%m-%d").unique().tolist()) == [
        "2026-03-01", "2026-03-02", "2026-03-03",
    ]


def test_persist_writes_records_before_watermark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """collect_and_persist appends records before saving the watermark (crash-safe replay)."""
    import src.live.tax_ledger as tax_mod

    client = WindowedFakeClient({"BTCUSDT": [_venue_trade(1), _venue_trade(2), _venue_trade(3)]})
    tax_dir = tmp_path / "tax"

    def _boom(path, watermark):
        raise RuntimeError("crash before watermark save")

    monkeypatch.setattr(tax_mod, "save_tax_watermark", _boom)
    with pytest.raises(RuntimeError, match="crash before watermark save"):
        collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=NOW,
                                     settings=_default_settings())
    assert len(load_tax_records(tax_dir)) == 3
    assert not (tax_dir / "watermark.json").exists()
    monkeypatch.undo()

    new_rows, issues = collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=NOW,
                                                    settings=_default_settings())
    assert new_rows == 0
    assert issues == ()
    assert len(load_tax_records(tax_dir)) == 3
    saved = json.loads((tax_dir / "watermark.json").read_text(encoding="utf-8"))
    assert saved["last_trade_id"] == {"BTCUSDT": 3}


def test_persist_append_failure_keeps_old_watermark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the append raises, the exception propagates and watermark.json is untouched."""
    import src.live.tax_ledger as tax_mod

    client = WindowedFakeClient({"BTCUSDT": [_venue_trade(8)]})
    tax_dir = tmp_path / "tax"
    tax_dir.mkdir(parents=True)
    save_tax_watermark(
        tax_dir / "watermark.json",
        TaxWatermark(last_trade_id={"BTCUSDT": 7}, last_collected_at=None),
    )
    before = (tax_dir / "watermark.json").read_bytes()

    def _boom(records, ledger_dir):
        raise RuntimeError("shard unavailable")

    monkeypatch.setattr(tax_mod, "append_tax_records", _boom)
    with pytest.raises(RuntimeError, match="shard unavailable"):
        collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=NOW,
                                     settings=_default_settings())
    assert (tax_dir / "watermark.json").read_bytes() == before


def test_collect_unparseable_trade_holds_watermark_below_it() -> None:
    """Bad trade id 6: records 5 and 7 returned, watermark stops at 5, one parse issue."""
    client = WindowedFakeClient(
        {"BTCUSDT": [_venue_trade(5), _venue_trade(6, price="not-a-number"), _venue_trade(7)]}
    )
    watermark = TaxWatermark(last_trade_id={"BTCUSDT": 4}, last_collected_at=None)
    records, new_watermark, found = _collect(client, ["BTCUSDT"], watermark)
    assert sorted(r.venue_id for r in records) == [5, 7]
    assert new_watermark.last_trade_id == {"BTCUSDT": 5}
    assert len(found) == 1
    assert found[0].stream == "trades:BTCUSDT"
    assert found[0].stage == "parse"
    assert "6" in found[0].detail


def test_collect_unparseable_income_holds_time_watermark() -> None:
    """Bad income row: good rows still returned, time watermark unchanged, one parse issue."""
    t0, t1, t2 = _ms(NOW - pd.Timedelta(days=3)), _ms(NOW - pd.Timedelta(days=2)), _ms(NOW - pd.Timedelta(days=1))
    client = WindowedFakeClient(
        {},
        incomes=[
            _venue_income(10, sym="", ts=t0),
            _venue_income(11, income="bad", sym="", ts=t1),
            _venue_income(12, sym="", ts=t2),
        ],
    )
    prior = NOW - pd.Timedelta(days=10)
    watermark = TaxWatermark(last_trade_id={}, last_collected_at=prior)
    records, new_watermark, found = _collect(client, [], watermark)
    assert sorted(r.venue_id for r in records) == [10, 12]
    assert new_watermark.last_collected_at is not None
    assert new_watermark.last_collected_at <= pd.Timestamp(t1, unit="ms", tz="UTC")
    assert len(found) == 1
    assert found[0].stream == "income"
    assert found[0].stage == "parse"
    assert "11" in found[0].detail


def test_collect_trade_fetch_failure_keeps_stream_watermark_and_reports() -> None:
    """ETH fetch raises: ETH watermark unchanged, BTC advances, one fetch issue."""
    client = WindowedFakeClient(
        {"BTCUSDT": [_venue_trade(1), _venue_trade(2)]}, failing_trade_symbols=("ETHUSDT",)
    )
    watermark = TaxWatermark(last_trade_id={"BTCUSDT": 0, "ETHUSDT": 3}, last_collected_at=None)
    records, new_watermark, found = _collect(client, ["BTCUSDT", "ETHUSDT"], watermark)
    assert {r.symbol for r in records} == {"BTCUSDT"}
    assert new_watermark.last_trade_id == {"BTCUSDT": 2, "ETHUSDT": 3}
    assert len(found) == 1
    assert found[0].stream == "trades:ETHUSDT"
    assert found[0].stage == "fetch"


def test_persist_corrupt_watermark_fails_closed(tmp_path: Path) -> None:
    """Unparseable watermark.json: DataIntegrityError, no client call, no shard written."""
    tax_dir = tmp_path / "tax"
    tax_dir.mkdir(parents=True)
    (tax_dir / "watermark.json").write_text("{not json", encoding="utf-8")
    client = WindowedFakeClient({"BTCUSDT": [_venue_trade(1)]})
    with pytest.raises(DataIntegrityError):
        collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=NOW,
                                     settings=_default_settings())
    assert client.trades_calls == []
    assert client.income_calls == []
    assert list(tax_dir.glob("tax_ledger_*.jsonl")) == []


def test_persist_absent_watermark_starts_empty(tmp_path: Path) -> None:
    """No watermark file: collection runs against an empty watermark and saves progress."""
    client = WindowedFakeClient({"BTCUSDT": [_venue_trade(1)]})
    tax_dir = tmp_path / "tax"
    new_rows, issues = collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=NOW,
                                                    settings=_default_settings())
    assert new_rows == 1
    assert issues == ()
    assert len(load_tax_records(tax_dir)) == 1
    saved = json.loads((tax_dir / "watermark.json").read_text(encoding="utf-8"))
    assert saved["last_trade_id"] == {"BTCUSDT": 1}


def test_persist_honors_settings_page_limits(tmp_path: Path) -> None:
    """Small page limits from settings still collect every row via paging."""
    from src.live.settings import LiveSettings

    base = NOW - pd.Timedelta(days=2)
    incomes = [_venue_income(i, sym="", ts=_ms(base + pd.Timedelta(hours=i))) for i in range(3)]
    client = WindowedFakeClient({}, incomes=incomes)
    tax_dir = tmp_path / "tax"
    settings = LiveSettings(tax_income_page_limit=2, tax_trades_page_limit=2)
    new_rows, issues = collect_and_persist_live_tax(client, [], tax_dir, "live_testnet", now=NOW,
                                                    settings=settings)
    assert new_rows == 3
    assert issues == ()
    assert len(load_tax_records(tax_dir)) == 3


def test_watermark_round_trips_atomically(tmp_path: Path) -> None:
    """save then load returns equal values with no temp file left behind."""
    watermark = TaxWatermark(
        last_trade_id={"BTCUSDT": 3},
        last_collected_at=pd.Timestamp("2026-01-02 00:00:00", tz="UTC"),
    )
    path = tmp_path / "tax" / "watermark.json"
    save_tax_watermark(path, watermark)
    assert load_tax_watermark(path) == watermark
    assert list((tmp_path / "tax").glob("*.tmp")) == []


def test_persist_refetched_records_are_deduplicated(tmp_path: Path) -> None:
    """Same venue records on two runs: the second run writes 0 rows."""
    client = WindowedFakeClient({"BTCUSDT": [_venue_trade(1), _venue_trade(2)]})
    tax_dir = tmp_path / "tax"
    first_rows, _ = collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=NOW,
                                                 settings=_default_settings())
    assert first_rows == 2
    second_rows, _ = collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=NOW,
                                                  settings=_default_settings())
    assert second_rows == 0
    assert len(load_tax_records(tax_dir)) == 2


class _TypeErrorClient:
    """Client failing every fetch with TypeError."""

    def user_trades(self, *args, **kwargs):
        raise TypeError("unexpected keyword argument 'from_id'")

    def income(self, *args, **kwargs):
        raise TypeError("unexpected keyword argument 'start_time_ms'")


def test_collect_typeerror_fetch_failures_keep_watermarks_and_report() -> None:
    """TypeError on every fetch: fetch issues, watermarks unchanged."""
    client = _TypeErrorClient()
    prior = NOW - pd.Timedelta(days=4)
    watermark = TaxWatermark(last_trade_id={"BTCUSDT": 4}, last_collected_at=prior)
    records, new_watermark, found = _collect(client, ["BTCUSDT"], watermark)
    assert records == ()
    assert new_watermark.last_trade_id == {"BTCUSDT": 4}
    assert new_watermark.last_collected_at == prior
    assert [(i.stream, i.stage) for i in found] == [
        ("trades:BTCUSDT", "fetch"),
        ("income", "fetch"),
    ]


def test_collect_clean_income_advances_time_watermark() -> None:
    """No income failures: the time watermark advances to now."""
    t0, t1 = _ms(NOW - pd.Timedelta(days=2)), _ms(NOW - pd.Timedelta(days=1))
    client = WindowedFakeClient([], incomes=[_venue_income(10, sym="", ts=t0), _venue_income(12, sym="", ts=t1)])
    watermark = TaxWatermark(last_trade_id={}, last_collected_at=NOW - pd.Timedelta(days=5))
    records, new_watermark, found = _collect(client, [], watermark)
    assert [r.venue_id for r in records] == [10, 12]
    assert [r.record_id for r in records] == ["venue:FUNDING_FEE:10", "venue:FUNDING_FEE:12"]
    assert new_watermark.last_collected_at == NOW
    assert found == []


def test_watermark_rejects_non_object_and_malformed_fields(tmp_path: Path) -> None:
    """Non-dict JSON and bad field types fail closed with DataIntegrityError."""
    non_object = tmp_path / "non_object.json"
    non_object.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_tax_watermark(non_object)

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"last_trade_id": {"BTCUSDT": "abc"}}', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_tax_watermark(malformed)

    naive = tmp_path / "naive.json"
    naive.write_text(
        '{"last_trade_id": {}, "last_collected_at": "2026-01-02 00:00:00"}',
        encoding="utf-8",
    )
    assert load_tax_watermark(naive) == TaxWatermark(
        last_trade_id={},
        last_collected_at=pd.Timestamp("2026-01-02 00:00:00", tz="UTC"),
    )


def test_collect_holds_time_watermark_on_income_failure() -> None:
    """Fetch/parse income failures keep the time watermark; a clean run advances to now."""
    prior = pd.Timestamp("2026-09-20T00:00:00Z")
    now = pd.Timestamp("2026-09-24T00:00:00Z")
    bad_ts = _ms(pd.Timestamp("2026-09-22T00:00:00Z"))

    fails = WindowedFakeClient({}, incomes=[], fail_income_calls=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    broken = WindowedFakeClient(
        {}, incomes=[_venue_income(11, income="not-a-number", sym="", ts=bad_ts)]
    )
    ok = WindowedFakeClient({}, incomes=[])

    wm = TaxWatermark(last_trade_id={}, last_collected_at=prior)
    _, fail_wm, fail_issues = _collect(fails, [], wm, now=now)
    assert fail_wm.last_collected_at == prior
    assert [i.stream for i in fail_issues] == ["income"]
    _, parse_wm, parse_issues = _collect(broken, [], wm, now=now)
    assert parse_wm.last_collected_at == prior
    assert [i.stream for i in parse_issues] == ["income"]
    _, ok_wm, _ = _collect(ok, [], wm, now=now)
    assert ok_wm.last_collected_at == now


# --- shard durability invariants ----------------------------------------------


def test_append_torn_tail_truncated_and_new_rows_added(tmp_path: Path) -> None:
    """Partial last line without trailing newline is dropped; prior rows plus 2 new rows remain."""
    ledger_dir = tmp_path / "tax"
    append_tax_records([_tax_record("A"), _tax_record("B")], ledger_dir)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    shard.write_bytes(shard.read_bytes() + b'{"record_id": "PARTIAL')
    written = append_tax_records([_tax_record("C"), _tax_record("D")], ledger_dir)
    assert written == [shard]
    lines = shard.read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["record_id"] for line in lines) == ["A", "B", "C", "D"]


def test_append_torn_replay_written_once(tmp_path: Path) -> None:
    """A torn tail that is a prefix of R is regenerated exactly once on replay."""
    from src.live.tax_ledger import _record_to_row

    ledger_dir = tmp_path / "tax"
    prior = _tax_record("A")
    append_tax_records([prior], ledger_dir)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    replay = _tax_record("R", venue_id=9)
    prefix = json.dumps(_record_to_row(replay), ensure_ascii=False)[:30].encode()
    assert not prefix.endswith(b"\n")
    shard.write_bytes(shard.read_bytes() + prefix)
    append_tax_records([replay], ledger_dir)
    lines = shard.read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["record_id"] for line in lines) == ["A", "R"]


def test_append_midfile_corrupt_fails_closed_without_mutation(tmp_path: Path) -> None:
    """Corrupt line followed by a valid line: TaxLedgerCorruptError, no shard bytes change."""
    ledger_dir = tmp_path / "tax"
    append_tax_records(
        [_tax_record("S0", when="2026-09-15 00:00"), _tax_record("O0", when="2026-10-02 00:00")], ledger_dir
    )
    sept = ledger_dir / "tax_ledger_202609.jsonl"
    octb = ledger_dir / "tax_ledger_202610.jsonl"
    valid_row = json.dumps(
        {"record_id": "S1", "kind": "TRADE", "event_time": "2026-09-16T00:00:00+00:00"}
    )
    sept.write_bytes(b"not json\n" + valid_row.encode() + b"\n")
    before = {sept: sept.read_bytes(), octb: octb.read_bytes()}
    batch = [_tax_record("S9", when="2026-09-20 00:00"), _tax_record("O9", when="2026-10-03 00:00")]
    with pytest.raises(TaxLedgerCorruptError) as exc_info:
        append_tax_records(batch, ledger_dir)
    assert exc_info.value.path == sept
    assert exc_info.value.line_number == 1
    assert sept.read_bytes() == before[sept]
    assert octb.read_bytes() == before[octb]


def test_append_complete_but_invalid_last_line_is_corruption(tmp_path: Path) -> None:
    """A newline-terminated bad last line is corruption, never a torn tail."""
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    shard.write_bytes(b'{"record_id": "K"}\n{bad}\n')
    before = shard.read_bytes()
    with pytest.raises(TaxLedgerCorruptError) as exc_info:
        append_tax_records([_tax_record("Y")], ledger_dir)
    assert exc_info.value.line_number == 2
    assert shard.read_bytes() == before


def test_loader_raises_on_midfile_corruption(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    shard.write_bytes(b'{"record_id": "K"}\nnot json\n')
    with pytest.raises(TaxLedgerCorruptError):
        load_tax_records(ledger_dir)


def test_loader_ignores_torn_tail_without_mutation(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    append_tax_records([_tax_record("A"), _tax_record("B")], ledger_dir)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    shard.write_bytes(shard.read_bytes() + b'{"record_id": "TORN')
    before = shard.read_bytes()
    df = load_tax_records(ledger_dir)
    assert sorted(df["record_id"].tolist()) == ["A", "B"]
    assert shard.read_bytes() == before


def test_append_fsyncs_shards_and_new_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each touched shard fd is fsynced; a newly created shard fsyncs its directory."""
    import src.live.tax_ledger as tax_mod

    real_fsync = os.fsync
    synced_paths: list[str] = []

    def _spy(fd: int) -> None:
        try:
            synced_paths.append(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            synced_paths.append(str(fd))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    fsynced_dirs: list[Path] = []
    real_fsync_dir = tax_mod._fsync_dir

    def _dir_spy(dir_path: Path) -> None:
        fsynced_dirs.append(Path(dir_path))
        real_fsync_dir(dir_path)

    monkeypatch.setattr(tax_mod, "_fsync_dir", _dir_spy)
    ledger_dir = tmp_path / "tax"
    sept, octb = append_tax_records(
        [_tax_record("S", when="2026-09-15 00:00"), _tax_record("O", when="2026-10-02 00:00")], ledger_dir
    )
    assert str(sept) in synced_paths
    assert str(octb) in synced_paths
    assert fsynced_dirs == [ledger_dir, ledger_dir]


def test_watermark_save_crash_keeps_old_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """os.replace raising leaves the old watermark file intact."""
    path = tmp_path / "tax" / "watermark.json"
    save_tax_watermark(path, TaxWatermark(last_trade_id={"BTCUSDT": 1}, last_collected_at=NOW))
    before = path.read_bytes()

    def _boom(src, dst):
        raise RuntimeError("power loss during replace")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(RuntimeError, match="power loss"):
        save_tax_watermark(
            path, TaxWatermark(last_trade_id={"BTCUSDT": 2}, last_collected_at=NOW)
        )
    assert path.read_bytes() == before


def test_watermark_ignores_legacy_last_income_id(tmp_path: Path) -> None:
    """A watermark.json with last_income_id loads fine and re-saves without the key."""
    path = tmp_path / "watermark.json"
    path.write_text(
        json.dumps(
            {
                "last_trade_id": {"BTCUSDT": 3},
                "last_income_id": 99,
                "last_collected_at": "2026-01-02T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    assert load_tax_watermark(path) == TaxWatermark(
        last_trade_id={"BTCUSDT": 3},
        last_collected_at=pd.Timestamp("2026-01-02 00:00:00", tz="UTC"),
    )
    save_tax_watermark(path, load_tax_watermark(path))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "last_income_id" not in saved
    assert saved["last_trade_id"] == {"BTCUSDT": 3}


# --- lossless collection invariants -------------------------------------------


def test_collect_income_paged_past_venue_limit() -> None:
    """2,500 rows with page_limit 1000: all returned, watermark reaches now."""
    base = NOW - pd.Timedelta(days=3)
    incomes = [
        _venue_income(i, sym="", ts=_ms(base + pd.Timedelta(seconds=100 * i))) for i in range(2500)
    ]
    client = WindowedFakeClient({}, incomes=incomes)
    records, new_wm, found = _collect(
        client, [], TaxWatermark(last_trade_id={}, last_collected_at=None), income_page_limit=1000
    )
    assert len(records) == 2500
    assert len({r.record_id for r in records}) == 2500
    assert new_wm.last_collected_at == NOW
    assert found == []


def test_collect_cross_type_tranid_stored_once_each(tmp_path: Path) -> None:
    """tranId is per-type: TRANSFER 900 and FUNDING_FEE 500 are independent rows."""
    t0, t1 = _ms(NOW - pd.Timedelta(hours=2)), _ms(NOW - pd.Timedelta(hours=1))
    incomes = [
        _venue_income(900, sym="", ts=t0, itype="TRANSFER"),
        _venue_income(500, sym="", ts=t1, itype="FUNDING_FEE"),
    ]
    client = WindowedFakeClient({}, incomes=incomes)
    wm = TaxWatermark(last_trade_id={}, last_collected_at=None)
    first, wm1, _ = _collect(client, [], wm)
    assert {r.record_id for r in first} == {"venue:TRANSFER:900", "venue:FUNDING_FEE:500"}
    ledger_dir = tmp_path / "tax"
    append_tax_records(first, ledger_dir)
    second, _wm2, _ = _collect(client, [], wm1)
    append_tax_records(second, ledger_dir)
    df = load_tax_records(ledger_dir)
    assert len(df) == 2
    assert sorted(df["record_id"].tolist()) == ["venue:FUNDING_FEE:500", "venue:TRANSFER:900"]


def test_collect_unknown_income_type_preserved_with_classify_issue() -> None:
    """Unknown types keep their raw id/kind and emit exactly one classify issue."""
    ts = _ms(NOW - pd.Timedelta(days=1))
    client = WindowedFakeClient(
        {},
        incomes=[
            _venue_income(7, sym="", ts=ts, itype="NEW_VENUE_THING"),
            _venue_income(7, sym="", ts=ts, itype="TRANSFER"),
        ],
    )
    records, _wm, found = _collect(client, [], TaxWatermark(last_trade_id={}, last_collected_at=None))
    by_id = {r.record_id: r for r in records}
    assert set(by_id) == {"venue:NEW_VENUE_THING:7", "venue:TRANSFER:7"}
    assert by_id["venue:NEW_VENUE_THING:7"].kind == "UNCLASSIFIED"
    assert by_id["venue:NEW_VENUE_THING:7"].income_type == "NEW_VENUE_THING"
    assert by_id["venue:TRANSFER:7"].kind == "TRANSFER"
    assert [(i.stream, i.stage) for i in found] == [("income", "classify")]
    assert "NEW_VENUE_THING" in found[0].detail


def test_classify_income_type_table() -> None:
    """Explicit venue mapping, verbatim upper-case match, Binance spelling kept."""
    assert classify_income_type("REALIZED_PNL") == ("REALIZED_PNL", True)
    assert classify_income_type("DELIVERED_SETTELMENT") == ("REALIZED_PNL", True)
    assert classify_income_type("FUNDING_FEE") == ("FUNDING_FEE", True)
    assert classify_income_type("COMMISSION") == ("COMMISSION", True)
    assert classify_income_type("transfer") == ("TRANSFER", True)
    assert classify_income_type("INTERNAL_TRANSFER") == ("TRANSFER", True)
    assert classify_income_type("INSURANCE_CLEAR") == ("UNCLASSIFIED", True)
    assert classify_income_type("COMMISSION_REBATE") == ("UNCLASSIFIED", True)
    assert classify_income_type("NEW_VENUE_THING") == ("UNCLASSIFIED", False)
    assert set(INCOME_TYPE_KIND) >= {
        "REALIZED_PNL", "DELIVERED_SETTELMENT", "FUNDING_FEE", "COMMISSION", "TRANSFER",
        "INSURANCE_CLEAR", "COMMISSION_REBATE",
    }
    with pytest.raises(DataIntegrityError):
        classify_income_type("")


def test_summarize_delivery_settlement_counts_as_realized(tmp_path: Path) -> None:
    """DELIVERED_SETTELMENT rows land in realized_pnl, not in unclassified_income."""
    ts = _ms(NOW - pd.Timedelta(days=10))
    client = WindowedFakeClient({}, incomes=[_venue_income(3, income="12.5", sym="", ts=ts,
                                                            itype="DELIVERED_SETTELMENT")])
    records, _, _ = _collect(client, [], TaxWatermark(last_trade_id={}, last_collected_at=None))
    assert records[0].kind == "REALIZED_PNL"
    ledger_dir = tmp_path / "tax"
    append_tax_records(records, ledger_dir)
    summary = summarize_tax_year(2025, ledger_dir, source="venue")
    assert summary["total"]["realized_pnl"] == pytest.approx(12.5)
    assert "DELIVERED_SETTELMENT" not in summary["unclassified_income"]


def test_summarize_unclassified_income_itemised_by_raw_type(tmp_path: Path) -> None:
    """TRANSFER/UNCLASSIFIED rows are broken out per raw income_type."""
    ts = _ms(NOW - pd.Timedelta(days=10))
    client = WindowedFakeClient(
        {},
        incomes=[
            _venue_income(1, income="3.0", sym="", ts=ts, itype="INSURANCE_CLEAR"),
            _venue_income(2, income="1.0", sym="", ts=ts + 1, itype="COMMISSION_REBATE"),
        ],
    )
    records, _, _ = _collect(client, [], TaxWatermark(last_trade_id={}, last_collected_at=None))
    ledger_dir = tmp_path / "tax"
    append_tax_records(records, ledger_dir)
    summary = summarize_tax_year(2025, ledger_dir, source="venue")
    assert summary["unclassified_income"] == {"INSURANCE_CLEAR": pytest.approx(3.0),
                                              "COMMISSION_REBATE": pytest.approx(1.0)}


def test_collect_income_fetch_failure_keeps_time_watermark() -> None:
    """Second page raising: page-1 rows returned, fetch issue, watermark not past page 1."""
    base = NOW - pd.Timedelta(days=2)
    incomes = [_venue_income(i, sym="", ts=_ms(base + pd.Timedelta(hours=i))) for i in range(5)]
    client = WindowedFakeClient({}, incomes=incomes, fail_income_calls=(2,))
    prior = NOW - pd.Timedelta(days=5)
    records, new_wm, found = _collect(
        client, [], TaxWatermark(last_trade_id={}, last_collected_at=prior),
        income_page_limit=2, income_overlap=pd.Timedelta(0),
    )
    assert [r.venue_id for r in records] == [0, 1]
    assert [(i.stream, i.stage) for i in found] == [("income", "fetch")]
    assert new_wm.last_collected_at is not None
    assert new_wm.last_collected_at <= pd.Timestamp(incomes[2]["time"], unit="ms", tz="UTC")


def test_collect_page_budget_resumes_over_two_calls(tmp_path: Path) -> None:
    """max_pages=1 with 1,500 rows: first call page_caps, two appends store 1,500 once."""
    base = NOW - pd.Timedelta(days=3)
    incomes = [
        _venue_income(i, sym="", ts=_ms(base + pd.Timedelta(seconds=100 * i))) for i in range(1500)
    ]
    client = WindowedFakeClient({}, incomes=incomes)
    wm = TaxWatermark(last_trade_id={}, last_collected_at=base - pd.Timedelta(hours=1))
    first, wm1, found1 = _collect(client, [], wm, income_page_limit=1000, max_pages=1)
    assert len(first) == 1000
    assert [i.stage for i in found1] == ["page_cap"]
    assert wm1.last_collected_at is not None
    assert wm1.last_collected_at <= pd.Timestamp(incomes[1000]["time"], unit="ms", tz="UTC")
    assert wm1.last_collected_at > wm.last_collected_at
    ledger_dir = tmp_path / "tax"
    append_tax_records(first, ledger_dir)
    # 지속적인 페이지 압박(max_pages=1)에서도 워터마크는 매 호출 전진하고 결국 전량을 모은다.
    current = wm1
    for _ in range(5):
        more, nxt, _ = _collect(client, [], current, income_page_limit=1000, max_pages=1)
        append_tax_records(more, ledger_dir)
        assert nxt.last_collected_at is not None
        assert current.last_collected_at is not None
        assert nxt.last_collected_at >= current.last_collected_at
        current = nxt
    df = load_tax_records(ledger_dir)
    assert len(df) == 1500
    assert df["record_id"].nunique() == 1500


def test_collect_stalled_page_detected() -> None:
    """1,000 rows sharing one ms: page_cap 'stalled', no loop, watermark held at that ms."""
    ts = _ms(NOW - pd.Timedelta(days=1))
    incomes = [_venue_income(i, sym="", ts=ts) for i in range(1000)]
    client = WindowedFakeClient({}, incomes=incomes)
    prior = pd.Timestamp(ts, unit="ms", tz="UTC")
    records, new_wm, found = _collect(
        client, [], TaxWatermark(last_trade_id={}, last_collected_at=prior),
        income_page_limit=1000, income_overlap=pd.Timedelta(0),
    )
    assert len(records) == 1000
    assert [(i.stream, i.stage) for i in found] == [("income", "page_cap")]
    assert "stalled" in found[0].detail
    assert new_wm.last_collected_at is not None
    assert new_wm.last_collected_at <= prior


def test_collect_retention_gap_surfaced() -> None:
    """Watermark older than retention: retention_gap issue, collection restarts at the floor."""
    old_ts = _ms(NOW - pd.Timedelta(days=100))
    fresh_ts = _ms(NOW - pd.Timedelta(days=1))
    client = WindowedFakeClient(
        {},
        incomes=[
            _venue_income(1, sym="", ts=old_ts),
            _venue_income(2, sym="", ts=fresh_ts),
        ],
    )
    prior = NOW - pd.Timedelta(days=120)
    records, new_wm, found = _collect(
        client, [], TaxWatermark(last_trade_id={}, last_collected_at=prior),
        income_retention=pd.Timedelta(days=90),
    )
    assert [r.venue_id for r in records] == [2]
    assert [(i.stream, i.stage) for i in found] == [("income", "retention_gap")]
    assert new_wm.last_collected_at == NOW


def test_collect_conflicting_duplicate_id_reported() -> None:
    """Same record_id with different payload: first kept, id_conflict, watermark capped."""
    ts = _ms(NOW - pd.Timedelta(days=1))
    client = WindowedFakeClient(
        {},
        incomes=[
            _venue_income(5, income="1.0", sym="", ts=ts),
            _venue_income(5, income="2.0", sym="", ts=ts),
        ],
    )
    records, new_wm, found = _collect(client, [], TaxWatermark(last_trade_id={}, last_collected_at=None))
    assert len(records) == 1
    assert records[0].record_id == "venue:FUNDING_FEE:5"
    assert records[0].realized_pnl == pytest.approx(1.0)
    assert [(i.stream, i.stage) for i in found] == [("income", "id_conflict")]
    assert new_wm.last_collected_at == pd.Timestamp(ts, unit="ms", tz="UTC")


def test_collect_legacy_funding_id_deduplicates(tmp_path: Path) -> None:
    """Existing venue:FUNDING_FEE:42 row: re-fetching tranId 42 adds no duplicate."""
    ts = _ms(NOW - pd.Timedelta(days=10))
    legacy = _tax_record(
        "venue:FUNDING_FEE:42", when="2025-12-22 00:00", kind="FUNDING_FEE", side="",
        quantity=0.0, price=0.0, quote_qty=0.0, realized_pnl=1.5, symbol="",
        source="venue", mode="live_testnet", venue_id=42, income_type="FUNDING_FEE",
    )
    ledger_dir = tmp_path / "tax"
    append_tax_records([legacy], ledger_dir)
    client = WindowedFakeClient({}, incomes=[_venue_income(42, sym="", ts=ts)])
    records, _, _ = _collect(client, [], TaxWatermark(last_trade_id={}, last_collected_at=None))
    assert [r.record_id for r in records] == ["venue:FUNDING_FEE:42"]
    assert append_tax_records(records, ledger_dir) == []
    assert len(load_tax_records(ledger_dir)) == 1


# --- simulated fill identity ---------------------------------------------------


def test_simulated_tax_records_use_fill_identity(tmp_path: Path) -> None:
    """Two attempts of one decision day yield distinct fill-based ids."""
    ts = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    first = simulated_tax_records(
        [_sim_fill("journal:7", Decimal("1"), Decimal("100"), ts)], "paper"
    )
    second = simulated_tax_records(
        [_sim_fill("journal:12", Decimal("1"), Decimal("100"), ts)], "paper"
    )
    assert [r.record_id for r in first] == ["simulated:TRADE:journal:7"]
    assert [r.record_id for r in second] == ["simulated:TRADE:journal:12"]
    ledger_dir = tmp_path / "tax"
    append_tax_records(first, ledger_dir)
    append_tax_records(second, ledger_dir)
    assert len(load_tax_records(ledger_dir)) == 2


def test_simulated_tax_records_rededup_after_crash(tmp_path: Path) -> None:
    """Re-emitting the same journal range after a crash changes nothing on disk."""
    ts = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    fills = [_sim_fill("journal:7", Decimal("1"), Decimal("100"), ts)]
    ledger_dir = tmp_path / "tax"
    append_tax_records(simulated_tax_records(fills, "paper"), ledger_dir)
    before = (ledger_dir / "tax_ledger_202603.jsonl").read_bytes()
    append_tax_records(simulated_tax_records(fills, "paper"), ledger_dir)
    assert (ledger_dir / "tax_ledger_202603.jsonl").read_bytes() == before
    assert len(load_tax_records(ledger_dir)) == 1


def test_simulated_tax_records_nan_price_fails_closed() -> None:
    """NaN fill price: DataIntegrityError, never a zero-priced record."""
    ts = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    with pytest.raises(DataIntegrityError):
        simulated_tax_records([_sim_fill("journal:7", Decimal("1"), float("nan"), ts)], "paper")


def test_simulated_tax_records_missing_fill_id_fails_closed() -> None:
    """Missing fill_id: DataIntegrityError."""
    ts = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    fill = _sim_fill("journal:7", Decimal("1"), Decimal("100"), ts)
    del fill.fill_id
    with pytest.raises(DataIntegrityError):
        simulated_tax_records([fill], "paper")


def test_simulated_tax_records_naive_timestamp_fails_closed() -> None:
    """Naive timestamp: DataIntegrityError, never stamped with wall-clock time."""
    with pytest.raises(DataIntegrityError):
        simulated_tax_records(
            [_sim_fill("journal:7", Decimal("1"), Decimal("100"), pd.Timestamp("2026-03-01 00:00"))],
            "paper",
        )


def test_simulated_tax_records_zero_quantity_skipped_and_sorted() -> None:
    """Zero-qty fills produce no record; output is sorted by event time."""
    early = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    late = pd.Timestamp("2026-03-02 00:00", tz="UTC")
    records = simulated_tax_records(
        [
            _sim_fill("journal:9", Decimal("1"), Decimal("100"), late),
            _sim_fill("journal:8", Decimal("0"), Decimal("100"), early),
            _sim_fill("journal:7", Decimal("-2"), Decimal("100"), early),
        ],
        "paper",
    )
    assert [r.record_id for r in records] == ["simulated:TRADE:journal:7", "simulated:TRADE:journal:9"]
    assert records[0].side == "SELL"
    assert records[1].side == "BUY"
    assert [r.event_time for r in records] == sorted(r.event_time for r in records)


def test_collect_issues_carry_fetch_parse_classify_page_cap_stages() -> None:
    """Issue dataclass exposes the six contracted stages across streams."""
    assert TaxCollectionIssue(stream="income", stage="retention_gap", detail="x").stage == "retention_gap"
    for stage in ("fetch", "parse", "classify", "page_cap", "retention_gap", "id_conflict"):
        issue = TaxCollectionIssue(stream="income", stage=stage, detail="d")
        assert issue.stage == stage


# --- strict venue parsing & paging edges ---------------------------------------


class _PagedClient:
    """Returns scripted payloads: trades per symbol (one list per call) and income pages."""

    def __init__(self, trade_pages=None, income_pages=None) -> None:
        self._trade_pages = {k: list(v) for k, v in (trade_pages or {}).items()}
        self._income_pages = list(income_pages or [])

    def user_trades(self, symbol, from_id=None, limit=1000):
        pages = self._trade_pages.get(symbol, [])
        return pages.pop(0) if pages else []

    def income(self, start_time_ms=None, end_time_ms=None, limit=1000):
        return self._income_pages.pop(0) if self._income_pages else []


def _bad_trade(**change):
    entry = _venue_trade(5)
    for key, value in change.items():
        if value is _MISSING:
            entry.pop(key)
        else:
            entry[key] = value
    return entry


_MISSING = object()


@pytest.mark.parametrize(
    "entry",
    [
        _bad_trade(price=_MISSING),
        _bad_trade(qty="x"),
        _bad_trade(commission="nan"),
        _bad_trade(time=_MISSING),
        _bad_trade(buyer="yes"),
        _bad_trade(time="soon"),
        "not-an-object",
    ],
    ids=["missing_price", "bad_qty", "nan_fee", "missing_time", "non_bool_buyer", "bad_time", "non_object"],
)
def test_malformed_venue_trade_is_reported_never_defaulted(entry) -> None:
    """A trade row missing or garbling a required field yields a parse issue and no record (no 0-price or now-stamped row)."""
    client = _PagedClient(trade_pages={"BTCUSDT": [[_venue_trade(4), entry]]})

    records, wm, issues = _collect(client, ["BTCUSDT"], TaxWatermark(last_trade_id={}, last_collected_at=NOW))

    assert [r.record_id for r in records] == ["venue:TRADE:4"]
    assert any(i.stage == "parse" and i.stream == "trades:BTCUSDT" for i in issues)
    if isinstance(entry, dict) and "id" in entry:
        assert wm.last_trade_id["BTCUSDT"] == 4  # stops before the failed id so it is re-read


@pytest.mark.parametrize(
    "entry",
    [{"incomeType": "FUNDING_FEE", "income": "1", "asset": "USDT", "time": 1000}, ["x"], {**_venue_income(9), "income": "abc"}],
    ids=["missing_tranId", "non_object", "bad_income"],
)
def test_malformed_venue_income_is_reported(entry) -> None:
    client = _PagedClient(income_pages=[[entry]])

    records, wm, issues = _collect(client, [], TaxWatermark(last_trade_id={}, last_collected_at=NOW - pd.Timedelta(hours=1)))

    assert records == ()
    assert any(i.stage == "parse" and i.stream == "income" for i in issues)
    assert wm.last_collected_at <= NOW - pd.Timedelta(hours=1)


def test_trades_page_budget_resumes_by_from_id() -> None:
    """A full trades page advances fromId; the page cap stops paging and the next call resumes without loss."""
    trades = {"BTCUSDT": [_venue_trade(i) for i in (1, 2, 3)]}
    client = WindowedFakeClient(trades)
    wm0 = TaxWatermark(last_trade_id={}, last_collected_at=NOW)

    first, wm1, issues = _collect(client, ["BTCUSDT"], wm0, trades_page_limit=2, max_pages=1)
    assert [r.record_id for r in first] == ["venue:TRADE:1", "venue:TRADE:2"]
    assert any(i.stage == "page_cap" and i.stream == "trades:BTCUSDT" for i in issues)
    second, _, _ = _collect(client, ["BTCUSDT"], wm1, trades_page_limit=2, max_pages=5)
    assert [r.record_id for r in second] == ["venue:TRADE:3"]


def test_non_list_payloads_are_fetch_issues() -> None:
    client = _PagedClient(trade_pages={"BTCUSDT": [{"code": -1000}]}, income_pages=[{"code": -1000}])
    start = NOW - pd.Timedelta(hours=1)

    records, wm, issues = _collect(client, ["BTCUSDT"], TaxWatermark(last_trade_id={"BTCUSDT": 7}, last_collected_at=start))

    assert records == ()
    assert {(i.stream, i.stage) for i in issues} >= {("trades:BTCUSDT", "fetch"), ("income", "fetch")}
    assert wm.last_trade_id["BTCUSDT"] == 7
    assert wm.last_collected_at <= start


def test_future_watermark_fetches_nothing_and_clamps_to_now() -> None:
    """A watermark ahead of the wall clock (clock step) triggers no income request and is clamped to now."""
    client = WindowedFakeClient({}, incomes=[_venue_income(1, ts=_ms(NOW))])

    records, wm, _ = _collect(client, [], TaxWatermark(last_trade_id={}, last_collected_at=NOW + pd.Timedelta(days=1)))

    assert records == ()
    assert client.income_calls == []
    assert wm.last_collected_at == NOW


@pytest.mark.parametrize(
    "fill",
    [
        _sim_fill("journal:7", Decimal("NaN"), Decimal("100"), pd.Timestamp("2026-03-01", tz="UTC")),
        _sim_fill("journal:7", Decimal("1"), Decimal("100"), pd.Timestamp("2026-03-01", tz="UTC"), fee_bps=-1.0),
        _sim_fill("journal:x", Decimal("1"), Decimal("100"), pd.Timestamp("2026-03-01", tz="UTC")),
        _sim_fill("journal:7", Decimal("1"), Decimal("100"), None),
    ],
    ids=["nan_qty", "negative_fee", "non_journal_id", "missing_timestamp"],
)
def test_simulated_tax_records_invalid_fill_fails_closed(fill) -> None:
    with pytest.raises(DataIntegrityError):
        simulated_tax_records([fill], "paper")


def test_non_utf8_shard_line_is_corruption(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir()
    (ledger_dir / "tax_ledger_202609.jsonl").write_bytes(b"\xff\xfe\n")

    with pytest.raises(DataIntegrityError, match="line 1"):
        load_tax_records(ledger_dir)


def test_load_tax_records_first_seen_wins_and_legacy_income_type(tmp_path: Path) -> None:
    """A record_id duplicated across shards loads once; legacy rows without income_type load as ''."""
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir()
    row = {"record_id": "L1", "kind": "TRADE", "event_time": "2026-09-15T00:00:00+00:00", "realized_pnl": 0.0}
    (ledger_dir / "tax_ledger_202609.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    other_year = {**row, "record_id": "L0", "event_time": "2025-12-31T23:00:00+00:00"}
    (ledger_dir / "tax_ledger_202610.jsonl").write_text(
        json.dumps({**row, "realized_pnl": 9.0}) + "\n" + json.dumps(other_year) + "\n", encoding="utf-8"
    )

    df = load_tax_records(ledger_dir, year=2026)

    assert df["record_id"].tolist() == ["L1"]
    assert df["realized_pnl"].tolist() == [0.0]
    assert df["income_type"].tolist() == [""]


def test_year_filter_rejects_record_without_parseable_time(tmp_path: Path) -> None:
    """A record whose year cannot be determined fails closed instead of silently leaving the yearly total."""
    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir()
    (ledger_dir / "tax_ledger_202609.jsonl").write_text(
        json.dumps({"record_id": "B1", "kind": "TRADE", "event_time": "garbage"}) + "\n"
        + json.dumps({"record_id": "B2", "kind": "TRADE"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError, match="line 1"):
        load_tax_records(ledger_dir, year=2026)
    (ledger_dir / "tax_ledger_202609.jsonl").write_text(
        json.dumps({"record_id": "B2", "kind": "TRADE"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(DataIntegrityError, match="line 1"):
        load_tax_records(ledger_dir, year=2026)
