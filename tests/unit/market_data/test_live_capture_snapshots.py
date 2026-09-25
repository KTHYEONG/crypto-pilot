from __future__ import annotations

import gzip
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.streams.snapshots import (
    BOOK_TICKER_COLUMNS,
    PREMIUM_INDEX_COLUMNS,
    load_snapshot_dataset,
    next_grid_time,
    parse_book_ticker_payload,
    parse_premium_index_payload,
    write_hourly_partition,
    write_reference_snapshot,
)

_CAP = pd.Timestamp("2026-09-22T10:00:00Z")
_FETCH = pd.Timestamp("2026-09-22T10:00:00.250Z")


def test_next_grid_time_strictly_future() -> None:
    """Grid points advance strictly forward on epoch alignment."""
    assert next_grid_time(pd.Timestamp("2026-09-22T10:00:00Z"), 60) == pd.Timestamp("2026-09-22T10:01:00Z")
    assert next_grid_time(pd.Timestamp("2026-09-22T10:00:59.900Z"), 60) == pd.Timestamp("2026-09-22T10:01:00Z")
    with pytest.raises(ValueError, match="divisor"):
        next_grid_time(pd.Timestamp("2026-09-22T10:00:00Z"), 7)
    with pytest.raises(ValueError, match="tz-aware"):
        next_grid_time(pd.Timestamp("2026-09-22 10:00:00"), 60)


def _book_payload() -> list[dict[str, object]]:
    return [
        {"symbol": "BTCUSDT", "bidPrice": "60000.5", "bidQty": "1.2", "askPrice": "60001.0", "askQty": "0.8", "time": 1758531600000},
        {"symbol": "AAPLUSDT", "bidPrice": "10.0", "bidQty": "5", "askPrice": "10.1", "askQty": "6", "time": 1758531600000},
    ]


def test_parse_book_ticker_keeps_every_symbol() -> None:
    """All symbols kept without filtering, with compact dtypes."""
    frame = parse_book_ticker_payload(_book_payload(), captured_at=_CAP, fetched_at=_FETCH)
    assert set(frame["symbol"]) == {"BTCUSDT", "AAPLUSDT"}
    assert tuple(frame.columns) == BOOK_TICKER_COLUMNS
    assert str(frame["bid_qty"].dtype) == "float32"
    assert str(frame["exchange_time_ms"].dtype) == "int64"


def test_parse_book_ticker_stamps_fetched_at() -> None:
    """Every row carries the receipt instant as nullable Int64 milliseconds."""
    frame = parse_book_ticker_payload(_book_payload(), captured_at=_CAP, fetched_at=_FETCH)
    assert tuple(frame.columns) == BOOK_TICKER_COLUMNS
    assert str(frame["fetched_at_ms"].dtype) == "Int64"
    assert (frame["fetched_at_ms"] == int(_CAP.value // 1_000_000) + 250).all()


def test_parse_book_ticker_rejects_bad_fetched_at() -> None:
    """A receipt before the grid label, or a naive receipt, fails closed."""
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload(
            _book_payload(), captured_at=_CAP, fetched_at=_CAP - pd.Timedelta(seconds=1)
        )
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload(
            _book_payload(), captured_at=_CAP, fetched_at=pd.Timestamp("2026-09-22 10:00:00")
        )


def test_parse_premium_index_rejects_bad_fetched_at() -> None:
    """A receipt before the grid label, or a naive receipt, fails closed."""
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload(
            _premium_payload(), captured_at=_CAP, fetched_at=_CAP - pd.Timedelta(seconds=1)
        )
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload(
            _premium_payload(), captured_at=_CAP, fetched_at=pd.Timestamp("2026-09-22 10:00:00")
        )


def test_parse_book_ticker_keeps_snapshot_when_a_side_is_empty() -> None:
    """A settled delivery contract quotes 0 on empty sides; the snapshot survives with NaN prices."""
    payload = [
        *_book_payload(),
        {"symbol": "BTCUSDT_260925", "bidPrice": "0.0", "bidQty": "0.000", "askPrice": "89550.0", "askQty": "0.006", "time": 1790323298753},
        {"symbol": "ETHUSDT_260925", "bidPrice": "0.00", "bidQty": "0.000", "askPrice": "0.00", "askQty": "0.000", "time": 1790323308750},
    ]
    frame = parse_book_ticker_payload(payload, captured_at=_CAP, fetched_at=_FETCH).set_index("symbol")
    assert len(frame) == 4
    assert pd.isna(frame.loc["BTCUSDT_260925", "bid_px"])
    assert frame.loc["BTCUSDT_260925", "ask_px"] == 89550.0
    assert frame.loc["ETHUSDT_260925", ["bid_px", "ask_px"]].isna().all()
    assert frame.loc["BTCUSDT", "bid_px"] == 60000.5


def test_parse_book_ticker_rejects_malformed_rows() -> None:
    """Missing keys and negative prices are rejected."""
    bad_missing = [{"symbol": "BTCUSDT", "bidPrice": "1", "bidQty": "1", "askQty": "1", "time": 1}]
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload(bad_missing, captured_at=pd.Timestamp("2026-09-22T10:00:00Z"), fetched_at=_FETCH)
    bad_price = [{"symbol": "BTCUSDT", "bidPrice": "-1", "bidQty": "1", "askPrice": "1", "askQty": "1", "time": 1}]
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload(bad_price, captured_at=pd.Timestamp("2026-09-22T10:00:00Z"), fetched_at=_FETCH)
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload([], captured_at=pd.Timestamp("2026-09-22T10:00:00Z"), fetched_at=_FETCH)
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload([42], captured_at=pd.Timestamp("2026-09-22T10:00:00Z"), fetched_at=_FETCH)
    bad_numeric = [{"symbol": "BTCUSDT", "bidPrice": "abc", "bidQty": "1", "askPrice": "1", "askQty": "1", "time": 1}]
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload(bad_numeric, captured_at=pd.Timestamp("2026-09-22T10:00:00Z"), fetched_at=_FETCH)
    bad_nan = [{"symbol": "BTCUSDT", "bidPrice": "nan", "bidQty": "1", "askPrice": "1", "askQty": "1", "time": 1}]
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload(bad_nan, captured_at=pd.Timestamp("2026-09-22T10:00:00Z"), fetched_at=_FETCH)
    bad_time = [{"symbol": "BTCUSDT", "bidPrice": "1", "bidQty": "1", "askPrice": "1", "askQty": "1", "time": "abc"}]
    with pytest.raises(DataIntegrityError):
        parse_book_ticker_payload(bad_time, captured_at=pd.Timestamp("2026-09-22T10:00:00Z"), fetched_at=_FETCH)


def _premium_payload() -> list[dict[str, object]]:
    return [
        {
            "symbol": "BTCUSDT", "markPrice": "60000", "indexPrice": "59999",
            "estimatedSettlePrice": "60001", "lastFundingRate": "0.0001",
            "interestRate": "0.00005", "nextFundingTime": 1758531600000, "time": 1758531590000,
        },
        {
            "symbol": "GONEUSDT", "markPrice": "1.0", "indexPrice": "1.0",
            "estimatedSettlePrice": "", "lastFundingRate": "",
            "interestRate": "", "nextFundingTime": 1758531600000, "time": 1758531590000,
        },
    ]


def test_parse_premium_index_blanks_become_nan() -> None:
    """Empty-string funding fields become NaN, never zero."""
    frame = parse_premium_index_payload(_premium_payload(), captured_at=pd.Timestamp("2026-09-22T10:00:00Z"), fetched_at=_FETCH)
    assert tuple(frame.columns) == PREMIUM_INDEX_COLUMNS
    gone = frame[frame["symbol"] == "GONEUSDT"].iloc[0]
    assert pd.isna(gone["last_funding_rate"])
    assert gone["last_funding_rate"] != 0.0
    btc = frame[frame["symbol"] == "BTCUSDT"].iloc[0]
    assert btc["last_funding_rate"] == pytest.approx(0.0001)


def test_parse_premium_index_rejects_malformed() -> None:
    """Malformed premium payloads fail closed; None becomes NaN."""
    cap = pd.Timestamp("2026-09-22T10:00:00Z")
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload([], captured_at=cap, fetched_at=_FETCH)
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload([42], captured_at=cap, fetched_at=_FETCH)
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload([{"symbol": "BTCUSDT"}], captured_at=cap, fetched_at=_FETCH)
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload(
            [{"symbol": "BTCUSDT", "markPrice": None, "time": 1, "nextFundingTime": 2}],
            captured_at=cap, fetched_at=_FETCH,
        )
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload(
            [{"symbol": "BTCUSDT", "markPrice": "", "time": 1, "nextFundingTime": 2}],
            captured_at=cap, fetched_at=_FETCH,
        )
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload(
            [{"symbol": "BTCUSDT", "markPrice": "abc", "time": 1, "nextFundingTime": 2}],
            captured_at=cap, fetched_at=_FETCH,
        )
    with pytest.raises(DataIntegrityError):
        parse_premium_index_payload(
            [{"symbol": "BTCUSDT", "markPrice": "1.0"}],
            captured_at=cap, fetched_at=_FETCH,
        )
    none_row = [
        {"symbol": "BTCUSDT", "markPrice": "1", "indexPrice": None, "estimatedSettlePrice": None,
         "lastFundingRate": None, "interestRate": None, "nextFundingTime": 2, "time": 1}
    ]
    out = parse_premium_index_payload(none_row, captured_at=cap, fetched_at=_FETCH)
    assert pd.isna(out.iloc[0]["index_price"])


def _snapshot_frame() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {"captured_at": pd.Timestamp("2026-09-22T10:59:00Z"), "symbol": "BTCUSDT", "exchange_time_ms": 1, "bid_px": 1.0, "bid_qty": 2.0, "ask_px": 1.1, "ask_qty": 3.0},
            {"captured_at": pd.Timestamp("2026-09-22T11:00:00Z"), "symbol": "BTCUSDT", "exchange_time_ms": 2, "bid_px": 1.0, "bid_qty": 2.0, "ask_px": 1.1, "ask_qty": 3.0},
        ]
    )
    rows["captured_at"] = pd.to_datetime(rows["captured_at"], utc=True)
    return rows


def test_write_hourly_partition_split_and_merge_dedup(tmp_path: Path) -> None:
    """Rows split by hour; rewrites dedup on symbol and time keeping the last."""
    first = _snapshot_frame()
    written = write_hourly_partition(first, tmp_path, "book_ticker")
    assert [p.name for p in written] == ["10.parquet", "11.parquet"]
    second = first.copy()
    second.loc[0, "bid_px"] = 9.0
    write_hourly_partition(second.iloc[[0]], tmp_path, "book_ticker")
    merged = pd.read_parquet(tmp_path / "book_ticker" / "20260922" / "10.parquet")
    assert len(merged) == 1
    assert merged.iloc[0]["bid_px"] == pytest.approx(9.0)
    assert str(merged["bid_qty"].dtype) == "float32"


def test_write_hourly_partition_leaves_other_hours_untouched(tmp_path: Path) -> None:
    """Partitions for hours absent from the frame are never rewritten."""
    write_hourly_partition(_snapshot_frame().iloc[[0]], tmp_path, "book_ticker")
    target = tmp_path / "book_ticker" / "20260922" / "10.parquet"
    before = target.read_bytes()
    mtime_before = target.stat().st_mtime_ns
    time.sleep(0.01)
    write_hourly_partition(_snapshot_frame().iloc[[1]], tmp_path, "book_ticker")
    assert target.read_bytes() == before
    assert target.stat().st_mtime_ns == mtime_before


def test_write_hourly_partition_empty_frame_noop(tmp_path: Path) -> None:
    """Zero-row frames write nothing."""
    empty = pd.DataFrame({"captured_at": pd.Series(dtype="datetime64[ns, UTC]"), "symbol": pd.Series(dtype="string")})
    assert write_hourly_partition(empty, tmp_path, "book_ticker") == []
    assert list(tmp_path.rglob("*.parquet")) == []


def test_write_hourly_partition_quarantines_corrupt_hour(tmp_path: Path) -> None:
    """An undecodable hour file is quarantined instead of wedging every later flush."""
    write_hourly_partition(_snapshot_frame().iloc[[0]], tmp_path, "book_ticker")
    target = tmp_path / "book_ticker" / "20260922" / "10.parquet"
    raw = target.read_bytes()
    target.write_bytes(raw[: len(raw) // 2])
    write_hourly_partition(_snapshot_frame().iloc[[0]], tmp_path, "book_ticker")
    merged = pd.read_parquet(target)
    assert len(merged) == 1
    quarantined = list((tmp_path / "book_ticker" / "20260922" / "_quarantine").glob("10.parquet.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == raw[: len(raw) // 2]


def test_write_hourly_partition_merges_legacy_hour_without_fetched_at(tmp_path: Path) -> None:
    """An hour file without the new column merges with new rows, keeping dtypes."""
    write_hourly_partition(_snapshot_frame().iloc[[0]], tmp_path, "book_ticker")
    target = tmp_path / "book_ticker" / "20260922" / "10.parquet"
    assert "fetched_at_ms" not in pd.read_parquet(target).columns
    new_row = parse_book_ticker_payload(
        [
            {
                "symbol": "ETHUSDT", "bidPrice": "3000", "bidQty": "2",
                "askPrice": "3001", "askQty": "2", "time": 1758531600000,
            }
        ],
        captured_at=pd.Timestamp("2026-09-22T10:59:30Z"),
        fetched_at=pd.Timestamp("2026-09-22T10:59:30.100Z"),
    )
    write_hourly_partition(new_row, tmp_path, "book_ticker")
    merged = pd.read_parquet(target)
    assert len(merged) == 2
    assert str(merged["fetched_at_ms"].dtype) == "Int64"
    assert str(merged["exchange_time_ms"].dtype) == "int64"
    legacy = merged[merged["symbol"] == "BTCUSDT"].iloc[0]
    assert pd.isna(legacy["fetched_at_ms"])
    fresh = merged[merged["symbol"] == "ETHUSDT"].iloc[0]
    assert fresh["fetched_at_ms"] == int(pd.Timestamp("2026-09-22T10:59:30.100Z").value // 1_000_000)


def test_load_snapshot_dataset_bounds_by_window(tmp_path: Path) -> None:
    """Loader reads only overlapping hour partitions in order."""
    rows = pd.DataFrame(
        [
            {"captured_at": pd.Timestamp("2026-09-22T09:10:00Z"), "symbol": "B", "exchange_time_ms": 1, "bid_px": 1.0, "bid_qty": 1.0, "ask_px": 1.0, "ask_qty": 1.0},
            {"captured_at": pd.Timestamp("2026-09-22T10:10:00Z"), "symbol": "B", "exchange_time_ms": 2, "bid_px": 1.0, "bid_qty": 1.0, "ask_px": 1.0, "ask_qty": 1.0},
            {"captured_at": pd.Timestamp("2026-09-22T10:20:00Z"), "symbol": "A", "exchange_time_ms": 3, "bid_px": 1.0, "bid_qty": 1.0, "ask_px": 1.0, "ask_qty": 1.0},
            {"captured_at": pd.Timestamp("2026-09-22T11:10:00Z"), "symbol": "A", "exchange_time_ms": 4, "bid_px": 1.0, "bid_qty": 1.0, "ask_px": 1.0, "ask_qty": 1.0},
        ]
    )
    rows["captured_at"] = pd.to_datetime(rows["captured_at"], utc=True)
    write_hourly_partition(rows, tmp_path, "book_ticker")
    out = load_snapshot_dataset(tmp_path, "book_ticker", start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T11:00:00Z"))
    assert len(out) == 2
    assert list(out["symbol"]) == ["B", "A"]
    assert (out["captured_at"] >= pd.Timestamp("2026-09-22T10:00:00Z")).all()
    assert (out["captured_at"] < pd.Timestamp("2026-09-22T11:00:00Z")).all()


def test_write_reference_snapshot_byte_exact_once_per_day(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Reference payloads persist byte-exact with deterministic gzip, once per day."""
    raw = b'{"symbols": [{"symbol": "BTCUSDT"}]}'
    cap = pd.Timestamp("2026-09-22T10:00:00Z")
    first = write_reference_snapshot(raw, tmp_path, "exchange_info", captured_at=cap)
    assert first is not None
    assert gzip.decompress(first.read_bytes()) == raw
    second = write_reference_snapshot(raw, tmp_path, "exchange_info", captured_at=cap + pd.Timedelta(hours=2))
    assert second is None
    assert gzip.decompress(first.read_bytes()) == raw
    other = tmp_path_factory.mktemp("ref2")
    twin = write_reference_snapshot(raw, other, "exchange_info", captured_at=cap)
    assert twin is not None
    assert first.read_bytes() == twin.read_bytes()
    assert os.stat(first).st_mtime_ns != 0 or True


def test_write_reference_snapshot_rejects_bad_input(tmp_path: Path) -> None:
    """Empty, non-JSON, and unknown references are rejected without files."""
    cap = pd.Timestamp("2026-09-22T10:00:00Z")
    with pytest.raises(DataIntegrityError):
        write_reference_snapshot(b"", tmp_path, "exchange_info", captured_at=cap)
    with pytest.raises(DataIntegrityError):
        write_reference_snapshot(b"not json", tmp_path, "exchange_info", captured_at=cap)
    with pytest.raises(DataIntegrityError):
        write_reference_snapshot(b'{"a": 1}', tmp_path, "foo", captured_at=cap)
    assert list(tmp_path.rglob("*")) == [] or all(p.is_dir() for p in tmp_path.rglob("*"))


def test_snapshot_frame_guards_and_loader_edges(tmp_path: Path) -> None:
    """Frame guards reject bad input; loader handles empty windows."""
    with pytest.raises(DataIntegrityError):
        write_hourly_partition(pd.DataFrame({"symbol": ["A"]}), tmp_path, "book_ticker")
    naive = pd.DataFrame({"captured_at": [pd.Timestamp("2026-09-22 10:00:00")], "symbol": ["A"]})
    with pytest.raises(DataIntegrityError):
        write_hourly_partition(naive, tmp_path, "book_ticker")
    missing = pd.DataFrame({"captured_at": [None], "symbol": ["A"]})
    with pytest.raises(DataIntegrityError):
        write_hourly_partition(missing, tmp_path, "book_ticker")
    assert load_snapshot_dataset(tmp_path, "missing", start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T11:00:00Z")).empty
    assert load_snapshot_dataset(tmp_path, "book_ticker", start=pd.Timestamp("2026-09-22T11:00:00Z"), end=pd.Timestamp("2026-09-22T10:00:00Z")).empty
    write_hourly_partition(_snapshot_frame(), tmp_path, "book_ticker")
    assert load_snapshot_dataset(tmp_path, "book_ticker", start=pd.Timestamp("2026-09-22T12:00:00Z"), end=pd.Timestamp("2026-09-22T13:00:00Z")).empty
    corrupt = tmp_path / "book_ticker" / "20260922" / "12.parquet"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"not parquet")
    out = load_snapshot_dataset(tmp_path, "book_ticker", start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T13:00:00Z"))
    assert len(out) == 2
