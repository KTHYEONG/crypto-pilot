"""Microstructure scenarios."""

import math
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.live.microstructure import (
    BookQuote,
    MicrostructureRecord,
    append_microstructure,
    build_microstructure_records,
    fetch_book_quotes,
)


class StubClient:
    def __init__(self):
        self.ticker_calls = 0
        self.tickers_calls = 0

    def book_tickers(self):
        self.tickers_calls += 1
        return {
            "A": {"symbol": "A", "bidPrice": "100.0", "askPrice": "100.2", "bidQty": "1", "askQty": "2"},
            "B": {"symbol": "B", "bidPrice": "100.0", "askPrice": "100.2", "bidQty": "1", "askQty": "2"},
            "C": {"symbol": "C", "bidPrice": "100.0", "askPrice": "100.2", "bidQty": "1", "askQty": "2"},
        }

    def book_ticker(self, symbol):
        self.ticker_calls += 1
        return {"symbol": symbol, "bidPrice": "100.0", "askPrice": "100.2"}


def test_SCENARIO_REC_04_quotes_single_batch_call():
    client = StubClient()
    quotes = fetch_book_quotes(client, ["A", "B", "C"])
    assert client.tickers_calls == 1
    assert client.ticker_calls == 0
    q = quotes["A"]
    assert q.mid == Decimal("100.1")
    # spread = (0.2 /100.1)*1e4 ≈19.98
    assert 19.98 < q.spread_bps < 19.99
    # bid==ask==0 => nan
    qb = BookQuote(symbol="X", bid=Decimal(0), ask=Decimal(0), bid_qty=Decimal(0), ask_qty=Decimal(0))
    assert math.isnan(qb.spread_bps)


def test_SCENARIO_REC_05_microstructure_record_shape(tmp_path: Path):
    dt = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    quotes = {
        "A": BookQuote(symbol="A", bid=Decimal("100"), ask=Decimal("101"), bid_qty=Decimal("1"), ask_qty=Decimal("2")),
    }
    records = build_microstructure_records(dt, "paper", quotes, premium=None)
    r = records[0]
    assert r.mark_price is None
    assert r.index_price is None
    assert r.last_funding_rate is None
    assert r.next_funding_time is None
    assert isinstance(r.bid, float)
    assert isinstance(r.ask, float)
    assert isinstance(r.bid_qty, float)
    assert isinstance(r.ask_qty, float)
    assert isinstance(r.mid, float)
    assert isinstance(r.spread_bps, float)
    # append and check columns
    written = append_microstructure(records, tmp_path / "micro")
    assert written
    df = pd.read_parquet(written[0])
    expected_cols = {f.name for f in MicrostructureRecord.__dataclass_fields__.values()} if hasattr(MicrostructureRecord, "__dataclass_fields__") else set()
    # MicrostructureRecord fields
    import dataclasses

    fields = {f.name for f in dataclasses.fields(MicrostructureRecord)}
    assert set(df.columns) == fields
# SCENARIO_REC_04-quotes-single-batch-call
# SCENARIO_REC_05-microstructure-record-shape
