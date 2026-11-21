from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.store import (
    CoverageRow,
    append_coverage,
    load_coverage,
    merge_partition,
)

_DAY1 = date(2024, 1, 15)
_DAY2 = date(2024, 1, 16)


def _frame(day: date, values: list[float]) -> pd.DataFrame:
    stamps = [pd.Timestamp(day, tz="UTC").as_unit("ms") + pd.Timedelta(minutes=5 * i) for i in range(len(values))]
    return pd.DataFrame({"timestamp": stamps, "value": values})


def test_merge_is_idempotent(tmp_path: Path) -> None:
    frame = _frame(_DAY1, [1.0, 2.0])
    first = merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, frame)
    second = merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, frame)
    assert first == second == 2
    back = pd.read_parquet(tmp_path / "metrics" / "BTCUSDT" / "202401.parquet")
    assert len(back) == 2


def test_merge_replaces_only_that_day(tmp_path: Path) -> None:
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, _frame(_DAY1, [1.0]))
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY2, _frame(_DAY2, [9.0]))
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, _frame(_DAY1, [2.0]))
    back = pd.read_parquet(tmp_path / "metrics" / "BTCUSDT" / "202401.parquet")
    assert sorted(back["value"].tolist()) == [2.0, 9.0]


def _coverage_row(status: str, recorded_at: str) -> CoverageRow:
    return CoverageRow(
        dataset="metrics",
        symbol="BTCUSDT",
        day=_DAY1,
        status=status,  # type: ignore[arg-type]
        raw_sha256=None,
        rows=1,
        recorded_at=pd.Timestamp(recorded_at, tz="UTC"),
        detail=None,
    )


def test_coverage_last_row_wins(tmp_path: Path) -> None:
    append_coverage(tmp_path, _coverage_row("failed", "2024-01-17 00:00"))
    append_coverage(tmp_path, _coverage_row("ok", "2024-01-17 01:00"))
    loaded = load_coverage(tmp_path, "metrics")
    assert loaded[("BTCUSDT", _DAY1)].status == "ok"


def test_coverage_malformed_line_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "_coverage"
    ledger.mkdir(parents=True, exist_ok=True)
    (ledger / "metrics.jsonl").write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_coverage(tmp_path, "metrics")


def test_month_mismatch_rejected(tmp_path: Path) -> None:
    merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, _frame(_DAY1, [1.0]))
    before = (tmp_path / "metrics" / "BTCUSDT" / "202401.parquet").read_bytes()
    with pytest.raises(DataIntegrityError):
        merge_partition(tmp_path, "metrics", "BTCUSDT", _DAY1, _frame(date(2024, 2, 1), [5.0]))
    assert (tmp_path / "metrics" / "BTCUSDT" / "202401.parquet").read_bytes() == before


def _ok_row(day: date, rows: int = 10) -> CoverageRow:
    return CoverageRow(
        dataset="metrics",
        symbol="BTCUSDT",
        day=day,
        status="ok",
        raw_sha256=None,
        rows=rows,
        recorded_at=pd.Timestamp("2024-01-17", tz="UTC"),
        detail=None,
    )


def test_archive_lock_invalid_wait(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import archive_lock

    with pytest.raises(ValueError, match="wait_s"), archive_lock(tmp_path, wait_s=-1):
        pass
    with pytest.raises(ValueError, match="wait_s"), archive_lock(tmp_path, wait_s=float("inf")):
        pass


def _spawn_holder(root: Path, hold_s: float, holder: str = "child"):  # type: ignore[no-untyped-def]
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        from pathlib import Path
        import time
        from src.market_data.public_archive.store import archive_lock
        root = Path({str(root)!r})
        with archive_lock(root, wait_s=0, holder={holder!r}):
            print("ready", flush=True)
            time.sleep({hold_s})
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)  # noqa: S603
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


def test_archive_lock_excludes_second_process(tmp_path: Path) -> None:
    import time

    from src.market_data.public_archive.store import ArchiveBusyError, archive_lock

    proc = _spawn_holder(tmp_path, 3.0, holder="child-holder")
    try:
        deadline = time.monotonic() + 10
        seen = ""
        while time.monotonic() < deadline:
            try:
                with archive_lock(tmp_path, wait_s=0):
                    pass
            except ArchiveBusyError as exc:
                seen = str(exc)
                break
            time.sleep(0.05)
        assert seen != ""
        assert "child-holder" in seen
        with pytest.raises(ArchiveBusyError), archive_lock(tmp_path, wait_s=0):
            pass
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    with archive_lock(tmp_path, wait_s=0):
        pass


def test_archive_lock_waits_then_succeeds(tmp_path: Path) -> None:
    import time

    from src.market_data.public_archive.store import archive_lock

    proc = _spawn_holder(tmp_path, 0.5, holder="waiter")
    try:
        start = time.monotonic()
        with archive_lock(tmp_path, wait_s=3):
            elapsed = time.monotonic() - start
        assert elapsed >= 0.45
    finally:
        proc.wait(timeout=10)


def test_nested_acquire_fails_fast(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import ArchiveBusyError, archive_lock

    with archive_lock(tmp_path, wait_s=0), pytest.raises(ArchiveBusyError), archive_lock(tmp_path, wait_s=10):
        pass


def test_lock_released_on_exception(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import archive_lock

    with pytest.raises(RuntimeError), archive_lock(tmp_path, wait_s=0):
        raise RuntimeError("boom")
    with archive_lock(tmp_path, wait_s=0):
        pass


def test_lock_io_failure_is_not_reported_as_contention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import store

    def fail(*args: object) -> None:
        raise OSError("lock filesystem failure")

    with monkeypatch.context() as patch:
        patch.setattr(store.fcntl, "flock", fail)
        with pytest.raises(OSError, match="filesystem failure"), store.archive_lock(tmp_path, wait_s=30):
            pass
    with store.archive_lock(tmp_path, wait_s=0):
        pass


def test_lock_metadata_failure_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import store

    def fail(*args: object) -> None:
        raise OSError("metadata failure")

    with monkeypatch.context() as patch:
        patch.setattr(store.os, "fsync", fail)
        with pytest.raises(OSError, match="metadata failure"), store.archive_lock(tmp_path, wait_s=0):
            pass
    with store.archive_lock(tmp_path, wait_s=0):
        pass


def test_sealed_month_detection(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import sealed_months

    cov = {("BTCUSDT", date(2023, 3, 5)): _ok_row(date(2023, 3, 5))}
    assert ("derived", "BTCUSDT", "202303") in sealed_months(tmp_path, "metrics", cov)
    cov_zero = {("BTCUSDT", date(2023, 3, 5)): _ok_row(date(2023, 3, 5), rows=0)}
    assert sealed_months(tmp_path, "metrics", cov_zero) == frozenset()
    part = tmp_path / "metrics" / "BTCUSDT" / "202303.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"stub")
    assert sealed_months(tmp_path, "metrics", cov) == frozenset()


def test_sealed_slice_only_when_derived_present(tmp_path: Path) -> None:
    from src.market_data.public_archive.store import sealed_months

    cov = {("BTCUSDT", date(2023, 3, 5)): _ok_row(date(2023, 3, 5))}
    derived = tmp_path / "aggTrades" / "BTCUSDT" / "202303.parquet"
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_bytes(b"stub")
    sealed = sealed_months(tmp_path, "aggTrades", cov)
    assert ("derived", "BTCUSDT", "202303") not in sealed
    assert ("slice", "BTCUSDT", "202303") in sealed


def test_archive_lock_busy_without_readable_holder(tmp_path: Path, monkeypatch) -> None:
    from pathlib import Path as _Path

    from src.market_data.public_archive.store import ArchiveBusyError, archive_lock

    proc = _spawn_holder(tmp_path, 2.0, holder="x")
    try:
        import time as _time

        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            try:
                with archive_lock(tmp_path, wait_s=0):
                    pass
            except ArchiveBusyError:
                break
            _time.sleep(0.05)
        monkeypatch.setattr(_Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
        with pytest.raises(ArchiveBusyError, match="Another process holds"), archive_lock(tmp_path, wait_s=0):
            pass
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# --- raw tier -------------------------------------------------------------------------------------------------------

_RAW_AGG_HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"


def _raw_zip(body: str) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("data.csv", body)
    return buffer.getvalue()


def test_raw_partition_paths_and_rejections(tmp_path: Path) -> None:
    from src.market_data.public_archive import store as store_mod

    assert store_mod.raw_partition_path(tmp_path, "aggTrades", "BTCUSDT", "202401") == (
        tmp_path / "raw" / "aggTrades" / "BTCUSDT" / "202401.parquet"
    )
    assert store_mod.raw_partition_path(tmp_path, "metrics", "BTCUSDT", "2024-01-02") == (
        tmp_path / "raw" / "metrics" / "BTCUSDT" / "2024-01-02.csv.zst"
    )
    with pytest.raises(ValueError, match="no raw tier"):
        store_mod.raw_partition_path(tmp_path, "klines", "BTCUSDT", "2024-01-02")


def test_raw_agg_trades_parquet_table_equal_to_csv(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.market_data.public_archive import fetch, store as store_mod

    body = f"{_RAW_AGG_HEADER}\n1,100.0,2.5,1,2,1704067200000,true\n2,0.00123456,3.0,3,4,1704067201000,false\n"
    zip_bytes = _raw_zip(body)
    path = tmp_path / "raw-day.parquet"
    report = store_mod.write_raw_agg_trades(
        path, fetch.iter_agg_trade_batches(zip_bytes), source_zip_sha256="s" * 64
    )
    assert report.rows == 2
    assert report.content_sha256 is None
    assert report.bytes_written == path.stat().st_size
    store_mod.verify_raw_agg_trades(path, fetch.iter_agg_trade_batches(zip_bytes))
    expected = pa.Table.from_batches(list(fetch.iter_agg_trade_batches(zip_bytes)))
    assert pq.read_table(path).equals(expected)
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"


def test_raw_prices_round_trip_to_csv_decimals(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from src.market_data.public_archive import fetch, store as store_mod

    prices = ["100.0", "0.00123456", "2.5", "12345.6789"]
    lines = [f"{i},{p},1.0,{i},{i},1704067200000,true" for i, p in enumerate(prices, start=1)]
    zip_bytes = _raw_zip("\n".join([_RAW_AGG_HEADER, *lines]) + "\n")
    path = tmp_path / "raw-day.parquet"
    store_mod.write_raw_agg_trades(path, fetch.iter_agg_trade_batches(zip_bytes), source_zip_sha256="s" * 64)
    table = pq.read_table(path)
    assert [repr(value) for value in table.column("price").to_pylist()] == [repr(float(p)) for p in prices]
    assert [str(value) for value in table.column("price").to_pylist()] == prices


def test_raw_agg_verify_detects_mismatch(tmp_path: Path) -> None:
    from src.market_data.public_archive import fetch, store as store_mod

    good = _raw_zip(f"{_RAW_AGG_HEADER}\n1,100.0,2.5,1,2,1704067200000,true\n")
    other = _raw_zip(f"{_RAW_AGG_HEADER}\n1,100.5,2.5,1,2,1704067200000,true\n")
    path = tmp_path / "raw-day.parquet"
    store_mod.write_raw_agg_trades(path, fetch.iter_agg_trade_batches(good), source_zip_sha256="s" * 64)
    with pytest.raises(DataIntegrityError):
        store_mod.verify_raw_agg_trades(path, fetch.iter_agg_trade_batches(other))


def test_raw_agg_empty_source_writes_no_file(tmp_path: Path) -> None:
    from src.market_data.public_archive import fetch, store as store_mod

    path = tmp_path / "raw-day.parquet"
    report = store_mod.write_raw_agg_trades(
        path, fetch.iter_agg_trade_batches(_raw_zip(f"{_RAW_AGG_HEADER}\n")), source_zip_sha256="s" * 64
    )
    assert report.rows == 0
    assert not path.exists()


def test_raw_csv_zst_byte_exact(tmp_path: Path) -> None:
    import hashlib

    import pyarrow as pa

    from src.market_data.public_archive import store as store_mod

    csv_bytes = b"create_time,symbol,x\n2024-01-02 00:00:00,BTCUSDT,1\n2024-01-02 00:05:00,BTCUSDT,2\n"
    path = tmp_path / "2024-01-02.csv.zst"
    report = store_mod.write_raw_csv(path, csv_bytes, source_zip_sha256="s" * 64)
    assert report.rows == 2
    assert report.content_sha256 == hashlib.sha256(csv_bytes).hexdigest()
    with pa.input_stream(str(path), compression="zstd") as stream:
        assert stream.read() == csv_bytes
    store_mod.verify_raw_csv(path, csv_bytes)
    with pytest.raises(DataIntegrityError):
        store_mod.verify_raw_csv(path, csv_bytes + b"x")


def test_coverage_rows_backward_compatible(tmp_path: Path) -> None:
    import dataclasses
    import json as _json

    from src.market_data.public_archive import store as store_mod

    legacy = {
        "dataset": "metrics", "day": "2024-01-02", "detail": None, "raw_sha256": "a" * 64,
        "recorded_at": "2024-01-03T00:00:00+00:00", "rows": 3, "status": "ok", "symbol": "BTCUSDT",
    }
    path = store_mod._coverage_path(tmp_path, "metrics")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")
    loaded = store_mod.load_coverage(tmp_path, "metrics")[("BTCUSDT", date(2024, 1, 2))]
    assert loaded.raw_tier is False
    assert loaded.raw_sha256_content is None
    row = dataclasses.replace(_ok_row(date(2024, 1, 3), rows=4), raw_tier=True, raw_sha256_content="b" * 64)
    store_mod.append_coverage(tmp_path, row)
    reloaded = store_mod.load_coverage(tmp_path, "metrics")[("BTCUSDT", date(2024, 1, 3))]
    assert reloaded.raw_tier is True
    assert reloaded.raw_sha256_content == "b" * 64


def test_sealed_raw_partition_when_absent(tmp_path: Path) -> None:
    from src.market_data.public_archive import store as store_mod

    row = store_mod.CoverageRow(
        dataset="aggTrades", symbol="BTCUSDT", day=date(2024, 1, 2), status="ok", raw_sha256="a" * 64,
        rows=3, recorded_at=pd.Timestamp("2024-01-03", tz="UTC"), detail=None, raw_tier=True,
    )
    coverage = {("BTCUSDT", row.day): row}
    assert ("raw", "BTCUSDT", "202401") in store_mod.sealed_months(tmp_path, "aggTrades", coverage)
    target = store_mod.raw_partition_path(tmp_path, "aggTrades", "BTCUSDT", "202401")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stub")
    assert ("raw", "BTCUSDT", "202401") not in store_mod.sealed_months(tmp_path, "aggTrades", coverage)


def test_streamed_raw_memory_bounded(tmp_path: Path) -> None:
    import pyarrow as pa

    from src.market_data.public_archive import store as store_mod

    path = tmp_path / "raw.parquet"
    tmp = path.with_name(path.name + ".tmp")
    seen_streaming: list[bool] = []

    def batches():
        for index in range(20):
            if index:
                seen_streaming.append(tmp.exists())  # earlier batches already on disk while later ones are produced
            yield pa.RecordBatch.from_pydict({
                "agg_trade_id": pa.array([index], pa.int64()),
                "price": pa.array([100.0], pa.float64()),
                "quantity": pa.array([1.0], pa.float64()),
                "first_trade_id": pa.array([index], pa.int64()),
                "last_trade_id": pa.array([index], pa.int64()),
                "transact_time": pa.array([1704067200000 + index], pa.int64()),
                "is_buyer_maker": pa.array([True], pa.bool_()),
            })

    report = store_mod.write_raw_agg_trades(path, batches(), source_zip_sha256="s" * 64)
    assert report.rows == 20
    assert all(seen_streaming)


def test_raw_agg_skips_empty_batches_and_fails_on_short_or_long_source(tmp_path: Path) -> None:
    import pyarrow as pa

    from src.market_data.public_archive import fetch, store as store_mod

    two = _raw_zip(f"{_RAW_AGG_HEADER}\n1,100.0,2.5,1,2,1704067200000,true\n2,100.5,1.0,2,2,1704067201000,false\n")
    one = _raw_zip(f"{_RAW_AGG_HEADER}\n1,100.0,2.5,1,2,1704067200000,true\n")
    path = tmp_path / "raw-day.parquet"
    empty = pa.RecordBatch.from_pylist([], schema=next(fetch.iter_agg_trade_batches(two)).schema)
    report = store_mod.write_raw_agg_trades(
        path, [empty, *fetch.iter_agg_trade_batches(one)], source_zip_sha256="s" * 64
    )
    assert report.rows == 1
    with pytest.raises(DataIntegrityError, match="longer than source"):
        store_mod.verify_raw_agg_trades(path, [])
    path2 = tmp_path / "raw-two.parquet"
    store_mod.write_raw_agg_trades(path2, fetch.iter_agg_trade_batches(two), source_zip_sha256="s" * 64)
    with pytest.raises(DataIntegrityError, match="shorter than source"):
        store_mod.verify_raw_agg_trades(path2, [*fetch.iter_agg_trade_batches(two), *fetch.iter_agg_trade_batches(one)])


def test_raw_agg_write_failure_leaves_no_tmp(tmp_path: Path) -> None:
    import pyarrow as pa

    from src.market_data.public_archive import fetch, store as store_mod

    good = next(fetch.iter_agg_trade_batches(_raw_zip(f"{_RAW_AGG_HEADER}\n1,100.0,2.5,1,2,1704067200000,true\n")))

    def batches():
        yield good
        raise RuntimeError("stream broke")

    path = tmp_path / "raw-day.parquet"
    with pytest.raises(RuntimeError):
        store_mod.write_raw_agg_trades(path, batches(), source_zip_sha256="s" * 64)
    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    assert pa is not None


def test_raw_csv_failed_publish_leaves_no_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import store as store_mod

    def fail(tmp, path):
        raise OSError("publish failed")

    monkeypatch.setattr(store_mod, "_publish", fail)
    path = tmp_path / "2024-01-02.csv.zst"
    with pytest.raises(OSError, match="publish failed"):
        store_mod.write_raw_csv(path, b"a,b\n1,2\n", source_zip_sha256="s" * 64)
    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
