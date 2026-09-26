"""Invariant guards for earliest-receipt snapshot partition merges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.market_data.streams.snapshots import parse_book_ticker_payload, write_hourly_partition


def _row(symbol: str, captured: str, fetched_ms: int | None, bid: str = "70000") -> dict[str, object]:
    return {"captured_at": pd.Timestamp(captured), "symbol": symbol, "exchange_time_ms": 1,
            "fetched_at_ms": fetched_ms, "bid_px": float(bid), "bid_qty": 1.0,
            "ask_px": 70001.0, "ask_qty": 1.0}


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_merge_keeps_earliest_fetched_at(tmp_path: Path) -> None:
    """A later receipt never replaces the earliest observation of a grid instant."""
    target = tmp_path / "book_ticker" / "20260926" / "10.parquet"
    write_hourly_partition(_frame([_row("BTCUSDT", "2026-09-26T10:00:00Z", 200)]), tmp_path, "book_ticker")
    write_hourly_partition(_frame([_row("BTCUSDT", "2026-09-26T10:00:00Z", 100)]), tmp_path, "book_ticker")
    assert pd.read_parquet(target).iloc[0]["fetched_at_ms"] == 100
    write_hourly_partition(_frame([_row("BTCUSDT", "2026-09-26T10:00:00Z", 300)]), tmp_path, "book_ticker")
    assert pd.read_parquet(target).iloc[0]["fetched_at_ms"] == 100


def test_legacy_null_fetched_at_loses(tmp_path: Path) -> None:
    """A legacy row without receipt loses to the first real receipt."""
    target = tmp_path / "book_ticker" / "20260926" / "10.parquet"
    write_hourly_partition(_frame([_row("BTCUSDT", "2026-09-26T10:00:00Z", None)]), tmp_path, "book_ticker")
    write_hourly_partition(_frame([_row("BTCUSDT", "2026-09-26T10:00:00Z", 100)]), tmp_path, "book_ticker")
    assert pd.read_parquet(target).iloc[0]["fetched_at_ms"] == 100


def test_parser_output_merges_by_grid() -> None:
    """Parser frames carry the dtypes the merge relies on."""
    parsed = parse_book_ticker_payload(
        [{"symbol": "BTCUSDT", "bidPrice": "1", "bidQty": "1", "askPrice": "2", "askQty": "1",
          "time": 1758679200000}],
        captured_at=pd.Timestamp("2026-09-26T10:00:00Z"),
        fetched_at=pd.Timestamp("2026-09-26T10:00:05Z"), max_rejected_fraction=0.05,
    )
    assert str(parsed.frame["fetched_at_ms"].dtype) == "Int64"
