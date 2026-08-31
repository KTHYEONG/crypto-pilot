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
