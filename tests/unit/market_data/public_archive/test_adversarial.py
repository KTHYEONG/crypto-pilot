from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance.vision import BinanceVisionDownloader
from src.market_data.public_archive.derive import derive_agg_trades, derive_metrics, slice_agg_trades
from src.market_data.public_archive.store import load_coverage, merge_partition, partition_path

DAY = date(2024, 1, 15)
START = int(pd.Timestamp(DAY, tz="UTC").value // 10**6)


def metrics(time: object = "2024-01-15 00:00:00", value: object = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "create_time": [time], "symbol": ["BTCUSDT"],
        "sum_open_interest": [value], "sum_open_interest_value": [1.0],
        "count_toptrader_long_short_ratio": [1.0], "sum_toptrader_long_short_ratio": [1.0],
        "count_long_short_ratio": [1.0], "sum_taker_long_short_vol_ratio": [1.0],
    })


def trades() -> pd.DataFrame:
    return pd.DataFrame({
        "agg_trade_id": [2, 1, 3], "price": [110.0, 100.0, 120.0],
        "quantity": [2.0, 1.0, 3.0], "first_trade_id": [3, 1, 6],
        "last_trade_id": [5, 2, 6], "transact_time": [START + 1_000, START, START + 180_000],
        "is_buyer_maker": [True, False, False],
    })


@pytest.mark.parametrize(("time", "value"), [("bad", 1), ("2024-01-15", "bad"), ("2024-01-15", np.inf), ("2024-01-15 00:00:00.000001", 1)])
def test_metrics_invalid_values_fail_closed(time, value) -> None:
    with pytest.raises(DataIntegrityError):
        derive_metrics(metrics(time, value), symbol="BTCUSDT", day=DAY)


def test_trade_order_vwap_and_bar_local_variance() -> None:
    out = derive_agg_trades(trades(), symbol="BTCUSDT", day=DAY)
    assert out["n_trades"].tolist() == [5, 1]
    assert out["vwap"].tolist() == pytest.approx([320 / 3, 120])
    assert out["rv_trade"].tolist() == pytest.approx([np.log(110 / 100) ** 2, 0])
    frame = trades().astype({"transact_time": "str", "agg_trade_id": "str"})
    pd.testing.assert_frame_equal(out, derive_agg_trades(frame, symbol="BTCUSDT", day=DAY))


def test_trade_invalid_ranges_and_slice_values() -> None:
    frame = trades()
    frame.loc[0, "last_trade_id"] = 2
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(frame, symbol="BTCUSDT", day=DAY)
    frame = trades()
    frame.loc[0, "price"] = np.inf
    with pytest.raises(DataIntegrityError):
        slice_agg_trades(frame, day=DAY, start_hour_utc=0, end_hour_utc=1)
    frame = trades()
    frame.loc[0, "price"] = 1e308
    frame.loc[0, "quantity"] = 1e308
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(frame, symbol="BTCUSDT", day=DAY)


def test_empty_rerun_removes_only_target_day(tmp_path) -> None:
    first = pd.DataFrame({"timestamp": [pd.Timestamp(DAY, tz="UTC")], "value": [1.0]})
    second = first.copy()
    second["timestamp"] += pd.Timedelta(days=1)
    merge_partition(tmp_path, "metrics", "BTCUSDT", DAY, first)
    merge_partition(tmp_path, "metrics", "BTCUSDT", date(2024, 1, 16), second)
    assert merge_partition(tmp_path, "metrics", "BTCUSDT", DAY, pd.DataFrame()) == 1
    pd.testing.assert_frame_equal(pd.read_parquet(partition_path(tmp_path, "metrics", "BTCUSDT", "202401")), second.reset_index(drop=True))


def test_existing_month_corruption_is_not_overwritten(tmp_path) -> None:
    path = partition_path(tmp_path, "metrics", "BTCUSDT", "202401")
    path.parent.mkdir(parents=True)
    pd.DataFrame({"timestamp": [pd.Timestamp("2024-02-01", tz="UTC")]}).to_parquet(path)
    before = path.read_bytes()
    with pytest.raises(DataIntegrityError):
        merge_partition(tmp_path, "metrics", "BTCUSDT", DAY, pd.DataFrame({"timestamp": [pd.Timestamp(DAY, tz="UTC")]}))
    assert path.read_bytes() == before


@pytest.mark.parametrize(("field", "value"), [("status", "bogus"), ("dataset", "aggTrades"), ("recorded_at", "NaT"), ("rows", -1)])
def test_ledger_schema_fails_closed(tmp_path, field, value) -> None:
    path = tmp_path / "_coverage" / "metrics.jsonl"
    path.parent.mkdir()
    row = {"dataset": "metrics", "symbol": "BTCUSDT", "day": DAY.isoformat(), "status": "ok", "raw_sha256": None, "rows": 1, "recorded_at": "2024-01-17T00:00:00+00:00", "detail": None}
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(DataIntegrityError):
        load_coverage(tmp_path, "metrics")


def test_listing_stalled_pagination_fails_closed(monkeypatch) -> None:
    downloader = BinanceVisionDownloader()
    body = b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>true</IsTruncated></ListBucketResult>'
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda *args, **kwargs: body)
    assert downloader.list_object_keys(prefix="data/futures/um/daily/metrics/") is None


@pytest.mark.parametrize("body", ["[]\n", "\n"])
def test_nonrecord_ledger_lines_fail_closed(tmp_path, body) -> None:
    path = tmp_path / "_coverage" / "metrics.jsonl"
    path.parent.mkdir()
    path.write_text(body)
    with pytest.raises(DataIntegrityError):
        load_coverage(tmp_path, "metrics")
