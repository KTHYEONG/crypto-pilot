"""Tax ledger scenarios."""

from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.tax_ledger import (
    TaxRecord,
    TaxWatermark,
    append_tax_records,
    collect_tax_records,
    load_tax_records,
    summarize_tax_year,
)


class StubTaxClient:
    def __init__(self):
        self.trades = [
            {"id": 1, "symbol": "BTCUSDT", "price": "100", "qty": "1", "quoteQty": "100", "commission": "0.1", "commissionAsset": "USDT", "realizedPnl": "0", "time": 1000, "buyer": True, "maker": False},
            {"id": 2, "symbol": "BTCUSDT", "price": "200", "qty": "1", "quoteQty": "200", "commission": "0.1", "commissionAsset": "USDT", "realizedPnl": "0", "time": 2000, "buyer": True, "maker": False},
            {"id": 3, "symbol": "BTCUSDT", "price": "300", "qty": "1", "quoteQty": "300", "commission": "0.1", "commissionAsset": "USDT", "realizedPnl": "0", "time": 3000, "buyer": False, "maker": False},
        ]
        self.last_from = None

    def user_trades(self, symbol, from_id=None, limit=1000):
        self.last_from = from_id
        if from_id is None:
            return self.trades
        return [t for t in self.trades if t["id"] >= from_id]

    def income(self, start_time_ms=None, income_type=None, limit=1000):
        return []


def test_SCENARIO_REC_06_tax_watermark_idempotent(tmp_path: Path):
    client = StubTaxClient()
    wm = TaxWatermark(last_trade_id={}, last_income_id=0, last_collected_at=None)
    now = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    records1, wm1 = collect_tax_records(client, ["BTCUSDT"], wm, "live_testnet", now=now)
    assert len(records1) == 3
    # second call should pass fromId 4
    records2, wm2 = collect_tax_records(client, ["BTCUSDT"], wm1, "live_testnet", now=now)
    assert client.last_from == 4
    assert len(records2) == 0
    # append both and load
    ledger_dir = tmp_path / "tax"
    append_tax_records(records1, ledger_dir)
    append_tax_records(records2, ledger_dir)
    # also append again records1 to test duplicate handling
    append_tax_records(records1, ledger_dir)
    df = load_tax_records(ledger_dir)
    assert len(df) == 3


def test_SCENARIO_REC_07_tax_source_purity_fail_closed(tmp_path: Path):
    ledger_dir = tmp_path / "tax2"
    # create venue and simulated records same year
    venue_rec = TaxRecord(
        record_id="venue:TRADE:1", kind="TRADE", event_time=pd.Timestamp("2027-01-15 00:00:00", tz="UTC"), symbol="BTCUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0, fee=0.1, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT", is_maker=False, venue_id=1, source="venue", mode="live_testnet"
    )
    sim_rec = TaxRecord(
        record_id="simulated:TRADE:-1", kind="TRADE", event_time=pd.Timestamp("2027-06-15 00:00:00", tz="UTC"), symbol="BTCUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0, fee=0.1, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT", is_maker=False, venue_id=-1, source="simulated", mode="paper"
    )
    append_tax_records([venue_rec, sim_rec], ledger_dir)
    with pytest.raises(DataIntegrityError):
        summarize_tax_year(2027, ledger_dir, source="venue")
    # simulated only year
    ledger_dir2 = tmp_path / "tax3"
    append_tax_records([sim_rec], ledger_dir2)
    summary = summarize_tax_year(2027, ledger_dir2, source="simulated")
    assert summary["source"] == "simulated"


def test_SCENARIO_REC_08_moving_average_cost_basis(tmp_path: Path):
    ledger_dir = tmp_path / "tax4"
    # buy 100 qty1, buy 200 qty1, sell 300 qty1
    recs = [
        TaxRecord(record_id="venue:TRADE:1", kind="TRADE", event_time=pd.Timestamp("2027-01-10 00:00:00", tz="UTC"), symbol="BTCUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0, fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT", is_maker=False, venue_id=1, source="venue", mode="live_testnet"),
        TaxRecord(record_id="venue:TRADE:2", kind="TRADE", event_time=pd.Timestamp("2027-02-10 00:00:00", tz="UTC"), symbol="BTCUSDT", side="BUY", quantity=1.0, price=200.0, quote_qty=200.0, fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT", is_maker=False, venue_id=2, source="venue", mode="live_testnet"),
        TaxRecord(record_id="venue:TRADE:3", kind="TRADE", event_time=pd.Timestamp("2027-03-10 00:00:00", tz="UTC"), symbol="BTCUSDT", side="SELL", quantity=1.0, price=300.0, quote_qty=300.0, fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT", is_maker=False, venue_id=3, source="venue", mode="live_testnet"),
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
# SCENARIO_REC_06-tax-watermark-idempotent
# SCENARIO_REC_07-tax-source-purity-fail-closed
# SCENARIO_REC_08-moving-average-cost-basis


def test_funding_tax_record_ids_are_deterministic(tmp_path: Path) -> None:
    """Same events twice yield identical ids of the pinned form; pnl equals amount, fee is 0."""
    from decimal import Decimal

    import pandas as pd

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
    from src.live.tax_ledger import TaxRecord, append_tax_records

    def _rec(rid: str) -> TaxRecord:
        return TaxRecord(
            record_id=rid, kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
            symbol="AAAUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0,
            fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
            is_maker=False, venue_id=1, source="simulated", mode="paper",
        )

    ledger_dir = tmp_path / "tax"
    first_written = append_tax_records([_rec("X")], ledger_dir)
    assert len(first_written) == 1
    written = append_tax_records([_rec("X"), _rec("Y")], ledger_dir)
    assert written == first_written
    lines = (ledger_dir / "tax_ledger_202609.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert sorted(__import__("json").loads(line)["record_id"] for line in lines) == ["X", "Y"]


def test_append_tax_records_deduplicates_within_one_batch(tmp_path: Path) -> None:
    from src.live.tax_ledger import TaxRecord, append_tax_records

    rec = TaxRecord(
        record_id="Y", kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
        symbol="AAAUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0,
        fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
        is_maker=False, venue_id=1, source="simulated", mode="paper",
    )
    ledger_dir = tmp_path / "tax"
    append_tax_records([rec, rec], ledger_dir)
    lines = (ledger_dir / "tax_ledger_202609.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_append_tax_records_replay_after_crash_writes_nothing_new(tmp_path: Path) -> None:
    from src.live.tax_ledger import TaxRecord, append_tax_records

    rec = TaxRecord(
        record_id="Y", kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
        symbol="AAAUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0,
        fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
        is_maker=False, venue_id=1, source="simulated", mode="paper",
    )
    ledger_dir = tmp_path / "tax"
    append_tax_records([rec], ledger_dir)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    before = shard.read_bytes()
    assert append_tax_records([rec], ledger_dir) == []
    assert shard.read_bytes() == before


def test_append_tax_records_corrupt_shard_fails_closed(tmp_path: Path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.tax_ledger import TaxRecord, append_tax_records

    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir(parents=True)
    shard = ledger_dir / "tax_ledger_202609.jsonl"
    shard.write_text("not json\n", encoding="utf-8")
    rec = TaxRecord(
        record_id="Y", kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
        symbol="AAAUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0,
        fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
        is_maker=False, venue_id=1, source="simulated", mode="paper",
    )
    with pytest.raises(DataIntegrityError):
        append_tax_records([rec], ledger_dir)
    assert shard.read_text(encoding="utf-8") == "not json\n"


def test_append_tax_records_rejects_shard_line_without_record_id(tmp_path: Path) -> None:
    import json

    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.tax_ledger import TaxRecord, append_tax_records, load_tax_records

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
    rec = TaxRecord(
        record_id="Y", kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
        symbol="AAAUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0,
        fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
        is_maker=False, venue_id=1, source="simulated", mode="paper",
    )
    with pytest.raises(DataIntegrityError, match="record_id"):
        append_tax_records([rec], ledger_dir)
    assert "K" in load_tax_records(ledger_dir)["record_id"].tolist()


def test_append_tax_records_month_routing_preserved(tmp_path: Path) -> None:
    from src.live.tax_ledger import TaxRecord, append_tax_records

    def _rec(rid: str, when: str) -> TaxRecord:
        return TaxRecord(
            record_id=rid, kind="TRADE", event_time=pd.Timestamp(when, tz="UTC"),
            symbol="AAAUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0,
            fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
            is_maker=False, venue_id=1, source="simulated", mode="paper",
        )

    ledger_dir = tmp_path / "tax"
    written = append_tax_records(
        [_rec("S", "2026-09-15 00:00"), _rec("O", "2026-10-02 00:00")], ledger_dir
    )
    assert sorted(p.name for p in written) == ["tax_ledger_202609.jsonl", "tax_ledger_202610.jsonl"]


def test_reconcile_cycle_cash_matches_trades_fees_and_funding() -> None:
    from decimal import Decimal

    import pandas as pd

    from src.live.tax_ledger import TaxRecord, reconcile_cycle_cash

    def _trade(rid: str, side: str, quote: float, fee: float) -> TaxRecord:
        return TaxRecord(
            record_id=rid, kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
            symbol="AAAUSDT", side=side, quantity=1.0, price=quote, quote_qty=quote,
            fee=fee, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
            is_maker=False, venue_id=1, source="simulated", mode="paper",
        )

    def _funding(rid: str, pnl: float) -> TaxRecord:
        return TaxRecord(
            record_id=rid, kind="FUNDING_FEE", event_time=pd.Timestamp("2026-09-15 08:00", tz="UTC"),
            symbol="AAAUSDT", side="", quantity=1.0, price=100.0, quote_qty=100.0,
            fee=0.0, fee_asset="USDT", realized_pnl=pnl, income_asset="USDT",
            is_maker=False, venue_id=0, source="simulated", mode="paper",
        )

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
    from decimal import Decimal

    import pandas as pd

    from src.live.tax_ledger import TaxRecord, reconcile_cycle_cash

    trade = TaxRecord(
        record_id="T1", kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
        symbol="AAAUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0,
        fee=0.02, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
        is_maker=False, venue_id=1, source="simulated", mode="paper",
    )
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
    from decimal import Decimal

    import pandas as pd
    import pytest

    from src.live.tax_ledger import TaxRecord, reconcile_cycle_cash

    funding = TaxRecord(
        record_id="F1", kind="FUNDING_FEE", event_time=pd.Timestamp("2026-09-15 08:00", tz="UTC"),
        symbol="AAAUSDT", side="", quantity=1.0, price=100.0, quote_qty=100.0,
        fee=0.0, fee_asset="USDT", realized_pnl=-0.5, income_asset="USDT",
        is_maker=False, venue_id=0, source="simulated", mode="paper",
    )
    with pytest.raises(ValueError, match="TRADE"):
        reconcile_cycle_cash(Decimal("2100"), Decimal("2100"), [funding], [], tolerance_usdt=Decimal("0.01"))
    trade = TaxRecord(
        record_id="T1", kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
        symbol="AAAUSDT", side="BUY", quantity=1.0, price=100.0, quote_qty=100.0,
        fee=0.02, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
        is_maker=False, venue_id=1, source="simulated", mode="paper",
    )
    with pytest.raises(ValueError, match="FUNDING_FEE"):
        reconcile_cycle_cash(Decimal("2100"), Decimal("2100"), [], [trade], tolerance_usdt=Decimal("0.01"))
    bad_side = TaxRecord(
        record_id="T9", kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
        symbol="AAAUSDT", side="HOLD", quantity=1.0, price=100.0, quote_qty=100.0,
        fee=0.02, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
        is_maker=False, venue_id=9, source="simulated", mode="paper",
    )
    with pytest.raises(ValueError, match="BUY or SELL"):
        reconcile_cycle_cash(Decimal("2100"), Decimal("2100"), [bad_side], [], tolerance_usdt=Decimal("0.01"))


def test_reconcile_cycle_cash_tolerates_empty_side_zero_quote() -> None:
    """A zero-quantity fill maps to side "" with zero quote; only its fee counts."""
    from decimal import Decimal

    import pandas as pd

    from src.live.tax_ledger import TaxRecord, reconcile_cycle_cash

    flat = TaxRecord(
        record_id="T0", kind="TRADE", event_time=pd.Timestamp("2026-09-15 00:00", tz="UTC"),
        symbol="AAAUSDT", side="", quantity=0.0, price=100.0, quote_qty=0.0,
        fee=0.0, fee_asset="USDT", realized_pnl=0.0, income_asset="USDT",
        is_maker=False, venue_id=0, source="simulated", mode="paper",
    )
    result = reconcile_cycle_cash(
        Decimal("2100"), Decimal("2100"), [flat], [], tolerance_usdt=Decimal("0.01")
    )
    assert result.within_tolerance is True
    assert result.difference == Decimal("0")


def test_simulated_tax_records_unique_id_across_cycles(tmp_path: Path) -> None:
    """Paper/shadow simulated records must survive multi-day read-time dedup."""
    from src.live.fills import FillEvent
    from decimal import Decimal
    from src.live.tax_ledger import simulated_tax_records

    ledger_dir = tmp_path / "tax"

    def _fill(day: str, sym: str, qty: str) -> FillEvent:
        ts = pd.Timestamp(f"2026-0{day}", tz="UTC")
        return FillEvent(
            decision_time=ts, timestamp=ts, symbol=sym, quantity_delta=Decimal(qty),
            fill_price=Decimal("100"), fee_bps=8.0, reason="immediate_taker",
            pre_trade_equity=Decimal("2000"), liquidity="taker", mode="paper",
            run_id=ts.strftime("%Y%m%d"), leg_index=0, client_order_id="c",
        )

    for day in ("3-01", "3-02", "3-03"):
        recs = simulated_tax_records([_fill(day, "BTCUSDT", "1"), _fill(day, "ETHUSDT", "2")], "paper")
        append_tax_records(recs, ledger_dir)

    loaded = load_tax_records(ledger_dir, year=2026)
    # 3 cycles x 2 symbols = 6 distinct records, none dropped by record_id dedup
    assert len(loaded) == 6
    assert loaded["record_id"].nunique() == 6
    assert sorted(loaded["event_time"].dt.strftime("%Y-%m-%d").unique().tolist()) == [
        "2026-03-01", "2026-03-02", "2026-03-03",
    ]


class _LiveTaxClient:
    """Configurable venue stub: per-symbol trades, income list, or fetch failures."""

    def __init__(
        self,
        trades_by_symbol: dict | None = None,
        incomes: list | None = None,
        failing_trade_symbols: tuple = (),
        failing_income: bool = False,
    ) -> None:
        self._trades = trades_by_symbol or {}
        self._incomes = incomes if incomes is not None else []
        self._failing_trade_symbols = set(failing_trade_symbols)
        self._failing_income = failing_income
        self.calls: list[str] = []

    def user_trades(self, symbol, from_id=None, limit=1000):
        self.calls.append(f"user_trades:{symbol}")
        if symbol in self._failing_trade_symbols:
            raise RuntimeError(f"venue timeout for {symbol}")
        trades = self._trades.get(symbol, [])
        if from_id is None:
            return list(trades)
        return [t for t in trades if t["id"] >= from_id]

    def income(self, start_time_ms=None, income_type=None, limit=1000):
        self.calls.append("income")
        if self._failing_income:
            raise RuntimeError("venue timeout for income")
        return list(self._incomes)


def _venue_trade(tid: int, price: str = "100", sym: str = "BTCUSDT", ts: int = 1000) -> dict:
    return {
        "id": tid, "symbol": sym, "price": price, "qty": "1", "quoteQty": "100",
        "commission": "0.1", "commissionAsset": "USDT", "realizedPnl": "0",
        "time": ts, "buyer": True, "maker": False,
    }


def _venue_income(tid: int, income: str = "1.5", sym: str = "BTCUSDT") -> dict:
    return {
        "tranId": tid, "incomeType": "FUNDING_FEE", "income": income,
        "asset": "USDT", "symbol": sym, "time": 1000,
    }


def test_live_tax_records_written_before_watermark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """collect_and_persist appends records before saving the watermark (crash-safe replay)."""
    import src.live.tax_ledger as tax_mod
    from src.live.tax_ledger import collect_and_persist_live_tax, load_tax_records

    client = _LiveTaxClient({"BTCUSDT": [_venue_trade(1), _venue_trade(2), _venue_trade(3)]})
    tax_dir = tmp_path / "tax"
    now = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

    def _boom(path, watermark):
        raise RuntimeError("crash before watermark save")

    monkeypatch.setattr(tax_mod, "save_tax_watermark", _boom)
    with pytest.raises(RuntimeError, match="crash before watermark save"):
        collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=now)
    assert len(load_tax_records(tax_dir)) == 3
    assert not (tax_dir / "watermark.json").exists()
    monkeypatch.undo()

    new_rows, issues = collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=now)
    assert new_rows == 0
    assert issues == ()
    assert len(load_tax_records(tax_dir)) == 3
    import json

    saved = json.loads((tax_dir / "watermark.json").read_text(encoding="utf-8"))
    assert saved["last_trade_id"] == {"BTCUSDT": 3}


def test_live_tax_append_failure_keeps_old_watermark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the append raises, the exception propagates and watermark.json is untouched."""
    import src.live.tax_ledger as tax_mod
    from src.live.tax_ledger import TaxWatermark, collect_and_persist_live_tax, save_tax_watermark

    client = _LiveTaxClient({"BTCUSDT": [_venue_trade(8)]})
    tax_dir = tmp_path / "tax"
    tax_dir.mkdir(parents=True)
    save_tax_watermark(
        tax_dir / "watermark.json",
        TaxWatermark(last_trade_id={"BTCUSDT": 7}, last_income_id=0, last_collected_at=None),
    )
    before = (tax_dir / "watermark.json").read_bytes()

    def _boom(records, ledger_dir):
        raise RuntimeError("shard unavailable")

    monkeypatch.setattr(tax_mod, "append_tax_records", _boom)
    with pytest.raises(RuntimeError, match="shard unavailable"):
        collect_and_persist_live_tax(
            client, ["BTCUSDT"], tax_dir, "live_testnet",
            now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        )
    assert (tax_dir / "watermark.json").read_bytes() == before


def test_live_tax_unparseable_trade_holds_watermark_below_it() -> None:
    """Bad trade id 6: records 5 and 7 returned, watermark stops at 5, one parse issue."""
    from src.live.tax_ledger import TaxWatermark, collect_tax_records

    client = _LiveTaxClient(
        {"BTCUSDT": [_venue_trade(5), _venue_trade(6, price="not-a-number"), _venue_trade(7)]}
    )
    watermark = TaxWatermark(last_trade_id={"BTCUSDT": 4}, last_income_id=0, last_collected_at=None)
    found: list = []
    records, new_watermark = collect_tax_records(
        client, ["BTCUSDT"], watermark, "live_testnet",
        now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), issues=found,
    )
    assert sorted(r.venue_id for r in records) == [5, 7]
    assert new_watermark.last_trade_id == {"BTCUSDT": 5}
    assert len(found) == 1
    assert found[0].stream == "trades:BTCUSDT"
    assert found[0].stage == "parse"
    assert "6" in found[0].detail


def test_live_tax_unparseable_income_holds_income_watermark() -> None:
    """Bad income id 11: income watermark stops at 10, one parse issue for income."""
    from src.live.tax_ledger import TaxWatermark, collect_tax_records

    client = _LiveTaxClient(
        {},
        incomes=[_venue_income(10), _venue_income(11, income="bad"), _venue_income(12)],
    )
    watermark = TaxWatermark(last_trade_id={}, last_income_id=9, last_collected_at=None)
    found: list = []
    records, new_watermark = collect_tax_records(
        client, ["BTCUSDT"], watermark, "live_testnet",
        now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), issues=found,
    )
    assert [r.venue_id for r in records] == [10, 12]
    assert new_watermark.last_income_id == 10
    assert len(found) == 1
    assert found[0].stream == "income"
    assert found[0].stage == "parse"
    assert "11" in found[0].detail


def test_live_tax_fetch_failure_keeps_stream_watermark_and_reports() -> None:
    """ETH fetch raises: ETH watermark unchanged, BTC advances, one fetch issue."""
    from src.live.tax_ledger import TaxWatermark, collect_tax_records

    client = _LiveTaxClient(
        {"BTCUSDT": [_venue_trade(1), _venue_trade(2)]}, failing_trade_symbols=("ETHUSDT",)
    )
    watermark = TaxWatermark(
        last_trade_id={"BTCUSDT": 0, "ETHUSDT": 3}, last_income_id=0, last_collected_at=None
    )
    found: list = []
    records, new_watermark = collect_tax_records(
        client, ["BTCUSDT", "ETHUSDT"], watermark, "live_testnet",
        now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), issues=found,
    )
    assert {r.symbol for r in records} == {"BTCUSDT"}
    assert new_watermark.last_trade_id == {"BTCUSDT": 2, "ETHUSDT": 3}
    assert len(found) == 1
    assert found[0].stream == "trades:ETHUSDT"
    assert found[0].stage == "fetch"


def test_live_tax_corrupt_watermark_fails_closed(tmp_path: Path) -> None:
    """Unparseable watermark.json: DataIntegrityError, no client call, no shard written."""
    from src.live.tax_ledger import collect_and_persist_live_tax

    tax_dir = tmp_path / "tax"
    tax_dir.mkdir(parents=True)
    (tax_dir / "watermark.json").write_text("{not json", encoding="utf-8")
    client = _LiveTaxClient({"BTCUSDT": [_venue_trade(1)]})
    with pytest.raises(DataIntegrityError):
        collect_and_persist_live_tax(
            client, ["BTCUSDT"], tax_dir, "live_testnet",
            now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        )
    assert client.calls == []
    assert list(tax_dir.glob("tax_ledger_*.jsonl")) == []


def test_live_tax_absent_watermark_starts_empty(tmp_path: Path) -> None:
    """No watermark file: collection runs against an empty watermark and saves progress."""
    import json

    from src.live.tax_ledger import collect_and_persist_live_tax, load_tax_records

    client = _LiveTaxClient({"BTCUSDT": [_venue_trade(1)]})
    tax_dir = tmp_path / "tax"
    new_rows, issues = collect_and_persist_live_tax(
        client, ["BTCUSDT"], tax_dir, "live_testnet",
        now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
    )
    assert new_rows == 1
    assert issues == ()
    assert len(load_tax_records(tax_dir)) == 1
    saved = json.loads((tax_dir / "watermark.json").read_text(encoding="utf-8"))
    assert saved["last_trade_id"] == {"BTCUSDT": 1}


def test_live_tax_watermark_round_trips_atomically(tmp_path: Path) -> None:
    """save then load returns equal values with no temp file left behind."""
    import pandas as pd

    from src.live.tax_ledger import TaxWatermark, load_tax_watermark, save_tax_watermark

    watermark = TaxWatermark(
        last_trade_id={"BTCUSDT": 3},
        last_income_id=9,
        last_collected_at=pd.Timestamp("2026-01-02 00:00:00", tz="UTC"),
    )
    path = tmp_path / "tax" / "watermark.json"
    save_tax_watermark(path, watermark)
    assert load_tax_watermark(path) == watermark
    assert list((tmp_path / "tax").glob("*.tmp")) == []


def test_live_tax_refetched_records_are_deduplicated(tmp_path: Path) -> None:
    """Same venue records on two runs: the second run writes 0 rows."""
    from src.live.tax_ledger import collect_and_persist_live_tax, load_tax_records

    client = _LiveTaxClient({"BTCUSDT": [_venue_trade(1), _venue_trade(2)]})
    tax_dir = tmp_path / "tax"
    now = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    first_rows, _ = collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=now)
    assert first_rows == 2
    second_rows, _ = collect_and_persist_live_tax(client, ["BTCUSDT"], tax_dir, "live_testnet", now=now)
    assert second_rows == 0
    assert len(load_tax_records(tax_dir)) == 2


class _LegacySignatureClient:
    """Old-style venue client accepting only positional calls (triggers TypeError fallback)."""

    def __init__(self, trades: list, incomes: list) -> None:
        self._trades = trades
        self._incomes = incomes

    def user_trades(self, symbol):  # noqa: ANN001, ANN202 - legacy positional-only signature
        return [t for t in self._trades if t.get("symbol", symbol) == symbol]

    def income(self):  # noqa: ANN202 - legacy positional-only signature
        return list(self._incomes)


class _TypeErrorClient:
    """Client failing every fetch with TypeError (fallback also fails)."""

    def user_trades(self, *args, **kwargs):
        raise TypeError("unexpected keyword argument 'from_id'")

    def income(self, *args, **kwargs):
        raise TypeError("unexpected keyword argument 'start_time_ms'")


def test_live_tax_legacy_signature_falls_back_and_reports_unparseable_ids() -> None:
    """Positional fallback succeeds; id-less records are reported without moving the watermark."""
    from src.live.tax_ledger import TaxWatermark, collect_tax_records

    client = _LegacySignatureClient(
        [_venue_trade(5), {"id": "bad", "symbol": "BTCUSDT"}],
        [_venue_income(10), {"tranId": "bad"}],
    )
    watermark = TaxWatermark(last_trade_id={}, last_income_id=9, last_collected_at=None)
    found: list = []
    records, new_watermark = collect_tax_records(
        client, ["BTCUSDT"], watermark, "live_testnet",
        now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), issues=found,
    )
    assert sorted(r.venue_id for r in records) == [5, 10]
    assert new_watermark.last_trade_id == {"BTCUSDT": 5}
    assert new_watermark.last_income_id == 10
    assert [(i.stream, i.stage) for i in found] == [
        ("trades:BTCUSDT", "parse"),
        ("income", "parse"),
    ]


def test_live_tax_typeerror_fetch_failures_keep_watermarks_and_report() -> None:
    """TypeError on both the call and the fallback: fetch issues, watermarks unchanged."""
    from src.live.tax_ledger import TaxWatermark, collect_tax_records

    client = _TypeErrorClient()
    watermark = TaxWatermark(
        last_trade_id={"BTCUSDT": 4}, last_income_id=9, last_collected_at=None
    )
    found: list = []
    records, new_watermark = collect_tax_records(
        client, ["BTCUSDT"], watermark, "live_testnet",
        now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), issues=found,
    )
    assert records == ()
    assert new_watermark.last_trade_id == {"BTCUSDT": 4}
    assert new_watermark.last_income_id == 9
    assert [(i.stream, i.stage) for i in found] == [
        ("trades:BTCUSDT", "fetch"),
        ("income", "fetch"),
    ]


def test_live_tax_clean_income_advances_income_watermark() -> None:
    """No income failures: the watermark advances to the highest parsed income id."""
    from src.live.tax_ledger import TaxWatermark, collect_tax_records

    client = _LiveTaxClient([], incomes=[_venue_income(10), _venue_income(12)])
    watermark = TaxWatermark(last_trade_id={}, last_income_id=9, last_collected_at=None)
    found: list = []
    records, new_watermark = collect_tax_records(
        client, [], watermark, "live_testnet",
        now=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), issues=found,
    )
    assert [r.venue_id for r in records] == [10, 12]
    assert new_watermark.last_income_id == 12
    assert found == []


def test_live_tax_watermark_rejects_non_object_and_malformed_fields(tmp_path: Path) -> None:
    """Non-dict JSON and bad field types fail closed with DataIntegrityError."""
    import pandas as pd

    from src.live.tax_ledger import TaxWatermark, load_tax_watermark

    non_object = tmp_path / "non_object.json"
    non_object.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_tax_watermark(non_object)

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"last_trade_id": {"BTCUSDT": "abc"}, "last_income_id": 0}', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_tax_watermark(malformed)

    naive = tmp_path / "naive.json"
    naive.write_text(
        '{"last_trade_id": {}, "last_income_id": 0, "last_collected_at": "2026-01-02 00:00:00"}',
        encoding="utf-8",
    )
    assert load_tax_watermark(naive) == TaxWatermark(
        last_trade_id={},
        last_income_id=0,
        last_collected_at=pd.Timestamp("2026-01-02 00:00:00", tz="UTC"),
    )


def test_collect_tax_records_holds_income_time_watermark_on_income_failure() -> None:
    """수입 조회·파싱 실패 시 시간 워터마크가 전진하면 실패 구간이 재조회되지 않는다."""
    import pandas as pd

    from src.live.tax_ledger import TaxCollectionIssue, TaxWatermark, collect_tax_records

    prior = pd.Timestamp("2026-09-20T00:00:00Z")
    now = pd.Timestamp("2026-09-24T00:00:00Z")

    class _FetchFails:
        def user_trades(self, symbol, from_id=None):
            return []

        def income(self, start_time_ms=None):
            raise ConnectionError("down")

    class _ParseFails(_FetchFails):
        def income(self, start_time_ms=None):
            return [{"tranId": 11, "incomeType": "FUNDING_FEE", "income": "not-a-number", "asset": "USDT", "time": 1790000000000}]

    class _Ok(_FetchFails):
        def income(self, start_time_ms=None):
            return []

    wm = TaxWatermark(last_trade_id={}, last_income_id=10, last_collected_at=prior)
    for client in (_FetchFails(), _ParseFails()):
        issues: list[TaxCollectionIssue] = []
        _, new_wm = collect_tax_records(client, ["BTCUSDT"], wm, "live", now=now, issues=issues)
        assert new_wm.last_collected_at == prior
        assert [i.stream for i in issues] == ["income"]
    _, ok_wm = collect_tax_records(_Ok(), ["BTCUSDT"], wm, "live", now=now)
    assert ok_wm.last_collected_at == now
