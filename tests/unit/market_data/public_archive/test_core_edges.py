from __future__ import annotations

import urllib.error
from datetime import date

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance.vision import BinanceVisionDownloader
from src.market_data.public_archive import derive as derive_mod
from src.market_data.public_archive.catalog import DATASETS, daily_key, key_day
from src.market_data.public_archive.derive import (
    derive_agg_trades,
    derive_book_depth,
    derive_metrics,
    slice_agg_trades,
)
from src.market_data.public_archive.fetch import fetch_verified_zip, read_single_csv
from src.market_data.public_archive.store import (
    CoverageRow,
    load_coverage,
    merge_partition,
    partition_path,
)

_DAY = date(2024, 1, 15)
_START_MS = int(pd.Timestamp(_DAY, tz="UTC").value // 10**6)


def _vision_page_keys(keys: list[str], truncated: bool, next_marker: str | None = None) -> bytes:
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    parts = [f'<ListBucketResult xmlns="{ns}">']
    parts.extend(f"<Key>{k}</Key>" for k in keys)
    parts.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
    if next_marker is not None:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts.append("</ListBucketResult>")
    return "".join(parts).encode()


def _downloader(monkeypatch: pytest.MonkeyPatch) -> BinanceVisionDownloader:
    d = BinanceVisionDownloader()
    monkeypatch.setattr(d, "min_request_interval_seconds", 0.0)
    monkeypatch.setattr(d, "_next_request_monotonic", 0.0)
    return d


def test_fetch_archive_bytes_ok_and_404(monkeypatch) -> None:
    d = _downloader(monkeypatch)
    monkeypatch.setattr(d, "_read_url_bytes", lambda url, timeout=None: b"body")
    assert d.fetch_archive_bytes("daily", "metrics", "f.zip") == b"body"

    def _raise(url: str, timeout=None):
        raise urllib.error.HTTPError(url, 404, "x", {}, None)

    monkeypatch.setattr(d, "_read_url_bytes", _raise)
    assert d.fetch_archive_bytes("daily", "metrics", "missing.zip") is None


def test_fetch_archive_bytes_reraises_other_errors(monkeypatch) -> None:
    d = _downloader(monkeypatch)

    def _raise(url: str, timeout=None):
        raise urllib.error.HTTPError(url, 500, "x", {}, None)

    monkeypatch.setattr(d, "_read_url_bytes", _raise)
    with pytest.raises(urllib.error.HTTPError):
        d.fetch_archive_bytes("daily", "x.zip")


def test_list_object_keys_paginates_and_sorts(monkeypatch) -> None:
    d = _downloader(monkeypatch)
    first = _vision_page_keys(["data/futures/um/daily/metrics/ZZZUSDT/x.zip"], True)
    second = _vision_page_keys(["data/futures/um/daily/metrics/AAAUSDT/x.zip"], False)
    calls = {"n": 0}

    def _fake(url: str, timeout=None):
        calls["n"] += 1
        return first if calls["n"] == 1 else second

    monkeypatch.setattr(d, "_read_url_bytes", _fake)
    out = d.list_object_keys(prefix="data/futures/um/daily/metrics/")
    assert out == sorted([
        "data/futures/um/daily/metrics/ZZZUSDT/x.zip",
        "data/futures/um/daily/metrics/AAAUSDT/x.zip",
    ])


def test_list_object_keys_failure_returns_none(monkeypatch) -> None:
    d = _downloader(monkeypatch)

    def _boom(url: str, timeout=None):
        raise OSError("boom")

    monkeypatch.setattr(d, "_read_url_bytes", _boom)
    assert d.list_object_keys(prefix="data/futures/um/daily/metrics/") is None


def test_catalog_roundtrip() -> None:
    key = daily_key("bookDepth", "BTCUSDT", _DAY)
    assert key_day(key) == _DAY
    assert set(DATASETS) == {"metrics", "bookDepth", "aggTrades"}
    with pytest.raises(DataIntegrityError):
        key_day("not-a-key")


def test_derive_metrics_edges() -> None:
    cols = [
        "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ]
    with pytest.raises(DataIntegrityError):
        derive_metrics(pd.DataFrame({"a": [1]}), symbol="BTCUSDT", day=_DAY)
    empty = pd.DataFrame(columns=cols)
    out = derive_metrics(empty, symbol="BTCUSDT", day=_DAY)
    assert list(out.columns)[:2] == ["timestamp", "sum_open_interest"]
    base = pd.DataFrame({c: [0] * 1 for c in cols})
    base["create_time"] = [_START_MS]
    base["symbol"] = ["ETHUSDT"]
    with pytest.raises(DataIntegrityError):
        derive_metrics(base, symbol="BTCUSDT", day=_DAY)
    null_ts = base.copy()
    null_ts["symbol"] = ["BTCUSDT"]
    null_ts["create_time"] = [None]
    with pytest.raises(DataIntegrityError):
        derive_metrics(null_ts, symbol="BTCUSDT", day=_DAY)
    outside = base.copy()
    outside["symbol"] = ["BTCUSDT"]
    outside["create_time"] = [_START_MS + 86_400_000]
    with pytest.raises(DataIntegrityError):
        derive_metrics(outside, symbol="BTCUSDT", day=_DAY)
    many = pd.DataFrame({
        "create_time": [_START_MS] * 289,
        "symbol": ["BTCUSDT"] * 289,
        "sum_open_interest": [1.0] * 289,
        "sum_open_interest_value": [1.0] * 289,
        "count_toptrader_long_short_ratio": [1.0] * 289,
        "sum_toptrader_long_short_ratio": [1.0] * 289,
        "count_long_short_ratio": [1.0] * 289,
        "sum_taker_long_short_vol_ratio": [1.0] * 289,
    })
    assert len(derive_metrics(many, symbol="BTCUSDT", day=_DAY)) == 1
    dup_same = pd.DataFrame({
        "create_time": [_START_MS, _START_MS],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "sum_open_interest": [1.0, 1.0],
        "sum_open_interest_value": [1.0, 1.0],
        "count_toptrader_long_short_ratio": [1.0, 1.0],
        "sum_toptrader_long_short_ratio": [1.0, 1.0],
        "count_long_short_ratio": [1.0, 1.0],
        "sum_taker_long_short_vol_ratio": [1.0, 1.0],
    })
    assert len(derive_metrics(dup_same, symbol="BTCUSDT", day=_DAY)) == 1
    dup_conflict = dup_same.copy()
    dup_conflict.loc[1, "sum_open_interest"] = 2.0
    with pytest.raises(DataIntegrityError):
        derive_metrics(dup_conflict, symbol="BTCUSDT", day=_DAY)
    str_ts = dup_same.iloc[:1].copy()
    str_ts["create_time"] = ["2024-01-15 00:00:00"]
    assert len(derive_metrics(str_ts, symbol="BTCUSDT", day=_DAY)) == 1


def test_derive_book_depth_edges() -> None:
    empty = pd.DataFrame(columns=["timestamp", "percentage", "depth", "notional"])
    out = derive_book_depth(empty, symbol="BTCUSDT", day=_DAY)
    assert "depth_p1" in out.columns
    with pytest.raises(DataIntegrityError):
        derive_book_depth(pd.DataFrame({"a": [1]}), symbol="BTCUSDT", day=_DAY)
    bad = pd.DataFrame({
        "timestamp": [_START_MS], "percentage": [1.0], "depth": [-1.0], "notional": [1.0],
    })
    with pytest.raises(DataIntegrityError):
        derive_book_depth(bad, symbol="BTCUSDT", day=_DAY)
    null_depth = pd.DataFrame({
        "timestamp": [_START_MS], "percentage": [1.0], "depth": ["bad"], "notional": [1.0],
    })
    with pytest.raises(DataIntegrityError):
        derive_book_depth(null_depth, symbol="BTCUSDT", day=_DAY)
    null_stamp = pd.DataFrame({
        "timestamp": [None], "percentage": [1.0], "depth": [1.0], "notional": [1.0],
    })
    with pytest.raises(DataIntegrityError):
        derive_book_depth(null_stamp, symbol="BTCUSDT", day=_DAY)
    assert derive_mod._band_tag("-5") == "m5"
    with pytest.raises(DataIntegrityError):
        derive_mod._band_tag("nope")


def _agg_base(**overrides: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "agg_trade_id": [0],
        "price": [100.0],
        "quantity": [1.0],
        "first_trade_id": [0],
        "last_trade_id": [0],
        "transact_time": [_START_MS + 1_000],
        "is_buyer_maker": [False],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_derive_agg_trades_edges() -> None:
    empty = pd.DataFrame(columns=[
        "agg_trade_id", "price", "quantity", "first_trade_id",
        "last_trade_id", "transact_time", "is_buyer_maker",
    ])
    assert len(derive_agg_trades(empty, symbol="BTCUSDT", day=_DAY)) == 0
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(pd.DataFrame({"a": [1]}), symbol="BTCUSDT", day=_DAY)
    with pytest.raises(DataIntegrityError):
        slice_agg_trades(pd.DataFrame({"a": [1]}), day=_DAY, start_hour_utc=0, end_hour_utc=1)
    with pytest.raises(ValueError, match="slice window"):
        slice_agg_trades(pd.DataFrame(), day=_DAY, start_hour_utc=5, end_hour_utc=5)
    null_frame = pd.DataFrame({
        "agg_trade_id": [0], "price": [1.0], "quantity": [1.0],
        "first_trade_id": [0], "last_trade_id": [0],
        "transact_time": [None], "is_buyer_maker": [True],
    })
    with pytest.raises(DataIntegrityError):
        slice_agg_trades(null_frame, day=_DAY, start_hour_utc=0, end_hour_utc=1)
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(_agg_base(price=["bad"]), symbol="BTCUSDT", day=_DAY)
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(_agg_base(transact_time=[None]), symbol="BTCUSDT", day=_DAY)
    int_flags = _agg_base(is_buyer_maker=[1])
    assert len(derive_agg_trades(int_flags, symbol="BTCUSDT", day=_DAY)) == 1
    int_zero = _agg_base(is_buyer_maker=[0])
    assert len(derive_agg_trades(int_zero, symbol="BTCUSDT", day=_DAY)) == 1
    str_true = _agg_base(is_buyer_maker=["True"])
    assert len(derive_agg_trades(str_true, symbol="BTCUSDT", day=_DAY)) == 1
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(
            _agg_base(agg_trade_id=["bad"]), symbol="BTCUSDT", day=_DAY
        )
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(_agg_base(is_buyer_maker=[2]), symbol="BTCUSDT", day=_DAY)
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(_agg_base(is_buyer_maker=["maybe"]), symbol="BTCUSDT", day=_DAY)
    empty_slice = pd.DataFrame(columns=[
        "agg_trade_id", "price", "quantity", "transact_time", "is_buyer_maker",
    ])
    out = slice_agg_trades(empty_slice, day=_DAY, start_hour_utc=0, end_hour_utc=1)
    assert list(out.columns) == [
        "transact_time_ms", "agg_trade_id", "price", "quantity", "is_buyer_maker",
    ]


def test_fetch_edges() -> None:
    import hashlib
    import io as _io
    import zipfile as _zf

    d = BinanceVisionDownloader()
    payload = _io.BytesIO()
    with _zf.ZipFile(payload, "w") as zf:
        zf.writestr("a.csv", "x\n1\n")
    raw = payload.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    d.fetch_archive_bytes = lambda *p: raw if p[-1] == "f" else None  # type: ignore[method-assign]
    with pytest.raises(DataIntegrityError):
        fetch_verified_zip(d, "data/futures/um/daily/metrics/BTCUSDT/f")
    d.fetch_archive_bytes = lambda *p: raw if p[-1] in ("f", "f.CHECKSUM") else None  # type: ignore[method-assign]
    with pytest.raises(DataIntegrityError):
        fetch_verified_zip(d, "data/futures/um/daily/metrics/BTCUSDT/f")
    header_zip = _io.BytesIO()
    with _zf.ZipFile(header_zip, "w") as zf:
        zf.writestr("BTCUSDT-metrics-2024-01-15.csv", "create_time,symbol\n1,BTCUSDT\n")
    frame = read_single_csv(header_zip.getvalue())
    assert list(frame.columns) == ["create_time", "symbol"]
    noheader_zip = _io.BytesIO()
    with _zf.ZipFile(noheader_zip, "w") as zf:
        zf.writestr("d.csv", "1,2\n3,4\n")
    frame2 = read_single_csv(noheader_zip.getvalue())
    assert len(frame2) == 2


def test_store_edges(tmp_path) -> None:
    assert load_coverage(tmp_path, "metrics") == {}
    assert partition_path(tmp_path, "metrics", "BTCUSDT", "202401").name == "202401.parquet"
    assert partition_path(tmp_path, "metrics", "BTCUSDT", "202401", kind="slice").parent.name == "BTCUSDT"
    assert merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY, pd.DataFrame()) == 0
    good = pd.DataFrame({
        "timestamp": [pd.Timestamp(_DAY, tz="UTC").as_unit("ms")],
        "value": [1.0],
    })
    with pytest.raises(DataIntegrityError):
        merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY, pd.DataFrame({"x": [1]}))
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY, good)
    row = CoverageRow(
        dataset="metrics", symbol="BTCUSDT", day=_DAY, status="ok",
        raw_sha256="abc", rows=1,
        recorded_at=pd.Timestamp("2024-01-17", tz="UTC"), detail="d",
    )
    from src.market_data.public_archive.store import append_coverage as _append

    _append(tmp_path, row)
    loaded = load_coverage(tmp_path, "metrics")
    assert loaded[("BTCUSDT", _DAY)].raw_sha256 == "abc"
    ledger = tmp_path / "_coverage" / "metrics.jsonl"
    ledger.write_text(ledger.read_text() + "\n[1,2]\n")
    with pytest.raises(DataIntegrityError):
        load_coverage(tmp_path, "metrics")
    ledger.write_text('{"dataset": "metrics"}\n')
    with pytest.raises(DataIntegrityError):
        load_coverage(tmp_path, "metrics")


def test_store_slice_and_partition_errors(tmp_path) -> None:
    slice_frame = pd.DataFrame({
        "transact_time_ms": [_START_MS + 1_000],
        "agg_trade_id": [7],
        "price": [100.0],
        "quantity": [1.0],
        "is_buyer_maker": [True],
    })
    assert merge_partition(
        tmp_path, "aggTrades", "BTCUSDT", _DAY, slice_frame, kind="slice"
    ) == 1
    back = pd.read_parquet(tmp_path / "aggTrades_slice" / "BTCUSDT" / "202401.parquet")
    assert back["transact_time_ms"].iloc[0] == _START_MS + 1_000
    day2 = date(2024, 1, 16)
    start2 = int(pd.Timestamp(day2, tz="UTC").value // 10**6)
    slice_day2 = pd.DataFrame({
        "transact_time_ms": [start2 + 1_000],
        "agg_trade_id": [9],
        "price": [101.0],
        "quantity": [2.0],
        "is_buyer_maker": [False],
    })
    assert merge_partition(
        tmp_path, "aggTrades", "BTCUSDT", day2, slice_day2, kind="slice"
    ) == 2
    bad_part = tmp_path / "metrics" / "BTCUSDT" / "202401.parquet"
    bad_part.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"nope": [1]}).to_parquet(bad_part)
    with pytest.raises(DataIntegrityError):
        merge_partition(
            tmp_path, "metrics", "BTCUSDT", _DAY,
            pd.DataFrame({
                "timestamp": [pd.Timestamp(_DAY, tz="UTC").as_unit("ms")],
                "value": [1.0],
            }),
        )
