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


def test_parse_book_quote_rejects_zero_bid() -> None:
    """Spec 02: a zero bid (empty book side) is unusable."""
    from src.live.microstructure import parse_book_quote

    assert parse_book_quote("X", {"bidPrice": "0.0", "askPrice": "89550.0"}) is None


def test_parse_book_quote_rejects_zero_both_sides() -> None:
    """Spec 02: zero on both sides is rejected."""
    from src.live.microstructure import parse_book_quote

    assert parse_book_quote("X", {"bidPrice": "0.00", "askPrice": "0.00"}) is None


def test_parse_book_quote_rejects_crossed_and_locked() -> None:
    """Spec 02: crossed and locked books are rejected."""
    from src.live.microstructure import parse_book_quote

    assert parse_book_quote("X", {"bidPrice": "101", "askPrice": "100"}) is None
    assert parse_book_quote("X", {"bidPrice": "100", "askPrice": "100"}) is None


def test_parse_book_quote_rejects_missing_price() -> None:
    """Spec 02: a missing price is rejected, never defaulted to 0."""
    from src.live.microstructure import parse_book_quote

    assert parse_book_quote("X", {"askPrice": "100.1"}) is None
    assert parse_book_quote("X", {}) is None
    assert parse_book_quote("X", None) is None


def test_parse_book_quote_rejects_non_finite() -> None:
    """Spec 02: NaN and Infinity prices are rejected."""
    from src.live.microstructure import parse_book_quote

    assert parse_book_quote("X", {"bidPrice": "NaN", "askPrice": "100.1"}) is None
    assert parse_book_quote("X", {"bidPrice": "100", "askPrice": "Infinity"}) is None
    assert parse_book_quote("X", {"bidPrice": "100", "askPrice": "100.1", "bidQty": "oops"}) is None


def test_parse_book_quote_accepts_valid_with_default_qty() -> None:
    """Spec 02: a valid quote passes with zero quantities and a positive mid."""
    from decimal import Decimal

    from src.live.microstructure import parse_book_quote

    quote = parse_book_quote("X", {"bidPrice": "100", "askPrice": "100.1"})
    assert quote is not None
    assert quote.bid_qty == Decimal(0)
    assert quote.ask_qty == Decimal(0)
    assert quote.mid == Decimal("100.05")
    assert quote.spread_bps > 0
    legacy = parse_book_quote("X", {"bid": "100", "ask": "100.1", "bidQty": "2", "askQty": "3"})
    assert legacy is not None
    assert legacy.bid_qty == Decimal(2)


def test_fetch_book_quotes_drops_only_invalid_symbols() -> None:
    """Spec 02: batch fetch keeps the valid symbol and drops the zero-bid one."""
    from src.live.microstructure import fetch_book_quotes

    class _Client:
        def book_tickers(self):
            return {
                "BTCUSDT": {"bidPrice": "100", "askPrice": "100.1"},
                "BTCUSDT_260925": {"bidPrice": "0.0", "askPrice": "100.1"},
            }

    result = fetch_book_quotes(_Client(), ["BTCUSDT", "BTCUSDT_260925"])
    assert set(result) == {"BTCUSDT"}


def test_parse_book_quote_rejects_non_finite_qty() -> None:
    """Spec 02: a present non-finite quantity makes the quote invalid."""
    from src.live.microstructure import parse_book_quote

    assert parse_book_quote("X", {"bidPrice": "100", "askPrice": "100.1", "bidQty": "NaN"}) is None


def test_fetch_book_quotes_list_payload_drops_invalid() -> None:
    """Spec 02: list batch payloads validate every entry the same way."""
    from src.live.microstructure import fetch_book_quotes

    class _Client:
        def book_tickers(self):
            return [
                {"symbol": "BTCUSDT", "bidPrice": "100", "askPrice": "100.1"},
                {"symbol": "BTCUSDT_260925", "bidPrice": "0.0", "askPrice": "100.1"},
            ]

    result = fetch_book_quotes(_Client(), ["BTCUSDT", "BTCUSDT_260925"])
    assert set(result) == {"BTCUSDT"}
