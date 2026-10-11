from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.catalog import daily_key
from src.market_data.public_archive.service import CollectionPlan, execute_plan, plan_collection
from src.market_data.public_archive.store import load_coverage

_METRICS_HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_taker_long_short_vol_ratio"
)

_AGG_HEADER = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
    "transact_time,is_buyer_maker"
)


def _zip_csv(body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("data.csv", body)
    return buffer.getvalue()


def _metrics_row(symbol: str, day: date, hour: int = 0) -> str:
    stamp = (pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S")
    return f"{stamp},{symbol},1,2,3,4,5,6"


def _metrics_zip(symbol: str, day: date) -> bytes:
    return _zip_csv(f"{_METRICS_HEADER}\n{_metrics_row(symbol, day)}\n")


def _agg_zip(symbol: str, day: date) -> bytes:
    ms = int(pd.Timestamp(day, tz="UTC").value // 10**6) + 23 * 3_600_000 + 30 * 60_000
    return _zip_csv(f"{_AGG_HEADER}\n7,100.0,2.0,1,2,{ms},true\n")


class FakeDownloader:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.blobs: dict[str, bytes] = {}
        self.fetch_calls: list[str] = []
        self.fail_listing = False

    def add_zip(self, full_key: str, payload: bytes) -> None:
        digest = hashlib.sha256(payload).hexdigest()
        stripped = full_key.removeprefix("data/futures/um/")
        self.keys.extend([full_key, full_key + ".CHECKSUM"])
        self.blobs[stripped] = payload
        self.blobs[stripped + ".CHECKSUM"] = f"{digest}  {full_key}\n".encode()

    def add_corrupt(self, full_key: str, payload: bytes) -> None:
        stripped = full_key.removeprefix("data/futures/um/")
        self.keys.extend([full_key, full_key + ".CHECKSUM"])
        self.blobs[stripped] = payload
        self.blobs[stripped + ".CHECKSUM"] = ("0" * 64 + f"  {full_key}\n").encode()

    def list_object_keys(self, *, prefix: str, timeout: int | None = None) -> list[str] | None:
        del timeout
        if self.fail_listing:
            return None
        return [k for k in self.keys if k.startswith(prefix)]

    def fetch_archive_bytes(self, *parts: str) -> bytes | None:
        key = "/".join(parts)
        self.fetch_calls.append(key)
        return self.blobs.get(key)

    def list_all_symbols(self) -> list[str]:
        return ["BTCUSDT"]


def _metrics_fake(symbols: list[str], days: list[date], *, corrupt: set[date] | None = None) -> FakeDownloader:
    fake = FakeDownloader()
    for symbol in symbols:
        for day in days:
            key = daily_key("metrics", symbol, day)
            if corrupt and day in corrupt:
                fake.add_corrupt(key, _metrics_zip(symbol, day))
            else:
                fake.add_zip(key, _metrics_zip(symbol, day))
    return fake


_NOW = pd.Timestamp("2024-02-01", tz="UTC")


def test_listing_failure_downloads_nothing(tmp_path: Path) -> None:
    fake = _metrics_fake(["BTCUSDT"], [date(2024, 1, 15)])
    fake.fail_listing = True
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=date(2024, 1, 15), end=date(2024, 1, 15),
    )
    assert plans is None
    assert fake.fetch_calls == []


def test_second_run_is_noop(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16)]
    fake = _metrics_fake(["BTCUSDT"], days)
    kwargs = {
        "root": tmp_path, "datasets": ["metrics"], "symbols": {"metrics": ["BTCUSDT"]},
        "start": days[0], "end": days[-1],
    }
    plans = plan_collection(fake, **kwargs)  # type: ignore[arg-type]
    assert plans is not None
    assert sum(len(p.days) for p in plans) == 2
    report = execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.ok, report.failed) == (2, 0)
    before = len(fake.fetch_calls)
    again = plan_collection(fake, **kwargs)  # type: ignore[arg-type]
    assert again == []
    execute_plan(fake, again, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert len(fake.fetch_calls) == before


def test_unit_failure_isolated(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)]
    fake = _metrics_fake(["BTCUSDT"], days, corrupt={date(2024, 1, 16)})
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=days[0], end=days[-1],
    )
    assert plans is not None
    report = execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.ok, report.failed) == (2, 1)
    coverage = load_coverage(tmp_path, "metrics")
    assert coverage[("BTCUSDT", days[0])].status == "ok"
    assert coverage[("BTCUSDT", days[1])].status == "failed"
    assert coverage[("BTCUSDT", days[2])].status == "ok"
    part = pd.read_parquet(tmp_path / "metrics" / "BTCUSDT" / "202401.parquet")
    assert sorted(pd.to_datetime(part["timestamp"], utc=True).dt.date.unique().tolist()) == [days[0], days[2]]


def test_today_never_planned(tmp_path: Path) -> None:
    today = pd.Timestamp.now(tz="UTC").date()
    fake = _metrics_fake(["BTCUSDT"], [today])
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=today, end=today,
    )
    assert plans == []


def test_unlisted_day_not_recorded(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 17)]
    fake = _metrics_fake(["BTCUSDT"], days)
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=date(2024, 1, 15), end=date(2024, 1, 17),
    )
    assert plans is not None
    assert [p.days for p in plans] == [(days[0], days[1])]
    execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    coverage = load_coverage(tmp_path, "metrics")
    assert ("BTCUSDT", date(2024, 1, 16)) not in coverage


def test_slice_and_bars_from_one_download(tmp_path: Path) -> None:
    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))
    plans = plan_collection(
        fake, root=tmp_path, datasets=["aggTrades"], symbols={"aggTrades": ["BTCUSDT"]},
        start=day, end=day, slice_hours=(23, 24), slice_symbols=["BTCUSDT"],
    )
    assert plans is not None
    assert plans[0].slice_hours == (23, 24)
    report = execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.ok, report.failed) == (1, 0)
    assert (tmp_path / "aggTrades" / "BTCUSDT" / "202401.parquet").exists()
    assert (tmp_path / "aggTrades_slice" / "BTCUSDT" / "202401.parquet").exists()
    zips = [c for c in fake.fetch_calls if c.endswith(".zip")]
    assert zips == [daily_key("aggTrades", "BTCUSDT", day).removeprefix("data/futures/um/")]


def test_plan_rejects_bad_inputs(tmp_path: Path) -> None:
    fake = FakeDownloader()
    with pytest.raises(DataIntegrityError):
        plan_collection(
            fake, root=tmp_path, datasets=["nope"], symbols={},  # type: ignore[arg-type]
            start=date(2024, 1, 15), end=date(2024, 1, 16),
        )
    with pytest.raises(ValueError, match="slice window"):
        plan_collection(
            fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
            start=date(2024, 1, 15), end=date(2024, 1, 16), slice_hours=(5, 5),
        )


def test_plan_resumes_after_last_ok(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16)]
    fake = _metrics_fake(["BTCUSDT"], days)
    first = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=days[0], end=days[-1],
    )
    assert first is not None
    execute_plan(fake, first, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    resumed = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=None, end=days[-1],
    )
    assert resumed == []
    fresh = plan_collection(
        fake, root=tmp_path, datasets=["bookDepth"], symbols={"bookDepth": ["BTCUSDT"]},
        start=None, end=date(2020, 1, 1),
    )
    assert fresh == []


def test_fetch_miss_counts_without_recording(tmp_path: Path) -> None:
    old = CollectionPlan(dataset="metrics", symbol="BTCUSDT", days=(date(2024, 1, 15),), slice_hours=None)
    report = execute_plan(FakeDownloader(), [old], root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.absent, report.pending) == (1, 0)
    assert load_coverage(tmp_path, "metrics") == {}
    recent = CollectionPlan(dataset="metrics", symbol="BTCUSDT", days=(date(2024, 1, 30),), slice_hours=None)
    report = execute_plan(FakeDownloader(), [recent], root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.absent, report.pending) == (0, 1)
    assert load_coverage(tmp_path, "metrics") == {}


def test_workers_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        execute_plan(FakeDownloader(), [], root=tmp_path, workers=0)
    with pytest.raises(ValueError, match="workers"):
        execute_plan(FakeDownloader(), [], root=tmp_path, workers=9)


def test_merge_failure_isolated_and_partition_kept(tmp_path: Path) -> None:
    day = date(2024, 1, 15)
    path = tmp_path / "metrics" / "BTCUSDT" / "202401.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"value": [1.0]}).to_parquet(path)
    before = path.read_bytes()
    fake = _metrics_fake(["BTCUSDT"], [day])
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=day, end=day,
    )
    assert plans is not None
    report = execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.ok, report.failed) == (0, 1)
    assert path.read_bytes() == before


def test_month_partition_accumulates(tmp_path: Path) -> None:
    fake = _metrics_fake(["BTCUSDT"], [date(2024, 1, 15), date(2024, 1, 16)])
    for day in (date(2024, 1, 15), date(2024, 1, 16)):
        plans = plan_collection(
            fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
            start=day, end=day,
        )
        assert plans is not None
        execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    part = pd.read_parquet(tmp_path / "metrics" / "BTCUSDT" / "202401.parquet")
    assert len(part) == 2


def test_bookdepth_and_unknown_dataset(tmp_path: Path) -> None:
    day = date(2024, 1, 15)
    stamp = pd.Timestamp(day, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    body = f"timestamp,percentage,depth,notional\n{stamp},1.0,10.0,100.0\n"
    fake = FakeDownloader()
    fake.add_zip(daily_key("bookDepth", "BTCUSDT", day), _zip_csv(body))
    plans = plan_collection(
        fake, root=tmp_path, datasets=["bookDepth"], symbols={"bookDepth": ["BTCUSDT"]},
        start=day, end=day,
    )
    assert plans is not None
    report = execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.ok, report.failed) == (1, 0)
    odd = CollectionPlan(dataset="nope", symbol="BTCUSDT", days=(day,), slice_hours=None)
    fake.add_zip(daily_key("nope", "BTCUSDT", day), _zip_csv(body))
    report = execute_plan(fake, [odd], root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert report.failed == 1


def test_merge_batch_guards(tmp_path: Path) -> None:
    from src.market_data.public_archive.service import _merge_day_batch

    day = date(2024, 1, 15)
    with pytest.raises(DataIntegrityError, match="sort column"):
        _merge_day_batch(tmp_path, "metrics", "BTCUSDT", [(day, pd.DataFrame({"x": [1.0]}))])
    bad = pd.DataFrame({
        "timestamp": [pd.Timestamp(date(2024, 2, 1), tz="UTC").as_unit("ms")],
        "value": [1.0],
    })
    with pytest.raises(DataIntegrityError, match="outside partition month"):
        _merge_day_batch(tmp_path, "metrics", "BTCUSDT", [(day, bad)])


def test_empty_derive_keeps_existing(tmp_path: Path) -> None:
    day1, day2 = date(2024, 1, 15), date(2024, 1, 16)
    fake = _metrics_fake(["BTCUSDT"], [day1])
    key = daily_key("metrics", "BTCUSDT", day2)
    stripped = key.removeprefix("data/futures/um/")
    empty_payload = _zip_csv(f"{_METRICS_HEADER}\n")
    fake.keys.extend([key, key + ".CHECKSUM"])
    fake.blobs[stripped] = empty_payload
    fake.blobs[stripped + ".CHECKSUM"] = (
        f"{hashlib.sha256(empty_payload).hexdigest()}  {key}\n".encode()
    )
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=day1, end=day2,
    )
    assert plans is not None
    report = execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.ok, report.failed) == (2, 0)
    coverage = load_coverage(tmp_path, "metrics")
    assert coverage[("BTCUSDT", day2)].rows == 0
    part = pd.read_parquet(tmp_path / "metrics" / "BTCUSDT" / "202401.parquet")
    assert len(part) == 1


def test_empty_derive_on_fresh_root_writes_nothing(tmp_path: Path) -> None:
    day = date(2024, 1, 15)
    fake = FakeDownloader()
    key = daily_key("metrics", "BTCUSDT", day)
    stripped = key.removeprefix("data/futures/um/")
    empty_payload = _zip_csv(f"{_METRICS_HEADER}\n")
    fake.keys.extend([key, key + ".CHECKSUM"])
    fake.blobs[stripped] = empty_payload
    fake.blobs[stripped + ".CHECKSUM"] = (
        f"{hashlib.sha256(empty_payload).hexdigest()}  {key}\n".encode()
    )
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=day, end=day,
    )
    assert plans is not None
    report = execute_plan(fake, plans, root=tmp_path, workers=1, now_fn=lambda: _NOW)
    assert (report.ok, report.failed) == (1, 0)
    assert not (tmp_path / "metrics" / "BTCUSDT" / "202401.parquet").exists()


def test_headerless_aggtrades_supported(tmp_path: Path) -> None:
    day = date(2024, 1, 15)
    fake = FakeDownloader()
    ms = int(pd.Timestamp(day, tz="UTC").value // 10**6) + 23 * 3_600_000
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _zip_csv(f"7,100,2,1,2,{ms},true\n"))
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), (23, 24))], root=tmp_path)
    assert report.ok == 1
    assert len(pd.read_parquet(tmp_path / "aggTrades_slice/BTCUSDT/202401.parquet")) == 1


def test_unlisted_recent_days_count_pending_without_coverage(tmp_path: Path) -> None:
    end = date(2024, 1, 17)
    fake = FakeDownloader()
    plans = plan_collection(fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]}, start=date(2024, 1, 15), end=end)
    assert plans is not None
    report = execute_plan(fake, plans, root=tmp_path)
    assert report.pending == 3
    assert report.planned_days == 0
    assert load_coverage(tmp_path, "metrics") == {}


def test_listing_exception_fails_closed(tmp_path: Path) -> None:
    class _Broken(FakeDownloader):
        def list_object_keys(self, **kwargs):
            raise OSError("listing failed")

    fake = _Broken()
    plans = plan_collection(fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]}, start=date(2024, 1, 15), end=date(2024, 1, 15))
    assert plans is None
    assert fake.fetch_calls == []


def test_checksum_failure_record_survives_batch_merge_failure(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16)]
    fake = _metrics_fake(["BTCUSDT"], days, corrupt={days[0]})
    path = tmp_path / "metrics/BTCUSDT/202401.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"invalid": [1]}).to_parquet(path)
    report = execute_plan(fake, [CollectionPlan("metrics", "BTCUSDT", tuple(days), None)], root=tmp_path)
    assert report.failed == 2
    assert len(load_coverage(tmp_path, "metrics")) == 2


def test_coverage_write_failure_does_not_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import store

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(store, "append_coverage", fail)
    day = date(2024, 1, 15)
    fake = _metrics_fake(["BTCUSDT"], [day])
    report = execute_plan(fake, [CollectionPlan("metrics", "BTCUSDT", (day,), None)], root=tmp_path)
    assert report.failed == 1
    assert report.ok == 0


def test_slice_streaming_preserves_other_days_and_replaces_retry(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16)]
    fake = FakeDownloader()
    for day in days:
        fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))
        report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), (23, 24))], root=tmp_path)
        assert report.ok == 1
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (days[0],), (23, 24))], root=tmp_path)
    assert report.ok == 1
    frame = pd.read_parquet(tmp_path / "aggTrades_slice/BTCUSDT/202401.parquet")
    assert len(frame) == 2
    assert frame.transact_time_ms.is_monotonic_increasing


def test_empty_slice_stream(tmp_path: Path) -> None:
    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), (0, 1))], root=tmp_path)
    assert report.ok == 1
    assert pd.read_parquet(tmp_path / "aggTrades_slice/BTCUSDT/202401.parquet").empty


def test_slice_staging_failure_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import service

    def fail(*args, **kwargs):
        raise OSError("staging unavailable")

    monkeypatch.setattr(service, "write_parquet_atomic", fail)
    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), (23, 24))], root=tmp_path)
    assert report.failed == 1
    assert load_coverage(tmp_path, "aggTrades")[("BTCUSDT", day)].status == "failed"


def test_slice_writer_failure_keeps_previous_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import service

    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))
    plan = CollectionPlan("aggTrades", "BTCUSDT", (day,), (23, 24))
    execute_plan(fake, [plan], root=tmp_path)
    path = tmp_path / "aggTrades_slice/BTCUSDT/202401.parquet"
    before = path.read_bytes()
    writer = service.pq.ParquetWriter

    class _BrokenWriter:
        def __init__(self, *args, **kwargs):
            self.writer = writer(*args, **kwargs)

        def write_table(self, table):
            raise OSError("write failed")

        def close(self):
            self.writer.close()

    merge = service._merge_slice_files

    def broken_merge(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(service.pq, "ParquetWriter", _BrokenWriter)
            merge(*args, **kwargs)

    monkeypatch.setattr(service, "_merge_slice_files", broken_merge)
    report = execute_plan(fake, [plan], root=tmp_path)
    assert report.failed == 1
    assert path.read_bytes() == before


def test_slice_merge_failure_rolls_back_new_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import service

    def fail(*args, **kwargs):
        raise OSError("slice merge failed")

    monkeypatch.setattr(service, "_merge_slice_files", fail)
    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), (23, 24))], root=tmp_path)
    assert report.failed == 1
    assert not (tmp_path / "aggTrades/BTCUSDT/202401.parquet").exists()
    assert not (tmp_path / "aggTrades_slice/BTCUSDT/202401.parquet").exists()


def test_heartbeat_updates_during_blocked_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import threading

    from src.market_data.public_archive import fetch

    day = date(2024, 1, 15)
    fake = _metrics_fake(["BTCUSDT"], [day])
    original = fetch.fetch_verified_zip
    heartbeat_seen = threading.Event()

    def slow(downloader, key):
        initial = json.loads((tmp_path / "_status/collect_heartbeat.json").read_text())["ts"]
        for _ in range(100):
            if heartbeat_seen.wait(0.01):
                break
            stamp = json.loads((tmp_path / "_status/collect_heartbeat.json").read_text())["ts"]
            if stamp != initial:
                heartbeat_seen.set()
        return original(downloader, key)

    monkeypatch.setattr(fetch, "fetch_verified_zip", slow)
    report = execute_plan(fake, [CollectionPlan("metrics", "BTCUSDT", (day,), None)], root=tmp_path, heartbeat_s=0.01)
    assert report.ok == 1
    assert heartbeat_seen.is_set()


def test_missing_ledger_incremental_window(tmp_path: Path) -> None:
    from datetime import timedelta

    end = date(2024, 1, 31)
    days = [end - timedelta(days=i) for i in range(30)]
    fake = _metrics_fake(["BTCUSDT"], days)
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=None, end=end, incremental_window_days=14,
    )
    assert plans is not None
    planned = sorted({day for plan in plans for day in plan.days})
    assert planned
    assert min(planned) >= end - timedelta(days=14)


def test_negative_incremental_window_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incremental_window_days"):
        plan_collection(
            FakeDownloader(), root=tmp_path, datasets=["metrics"], symbols={},
            start=None, end=date(2024, 1, 31), incremental_window_days=-1,
        )


def test_ledger_cursor_overrides_window(tmp_path: Path) -> None:
    from datetime import timedelta

    from src.market_data.public_archive import store

    end = date(2024, 1, 31)
    days = [end - timedelta(days=i) for i in range(30)]
    fake = _metrics_fake(["BTCUSDT"], days)
    cutoff = date(2024, 1, 5)
    for day in sorted(d for d in days if d <= cutoff):
        store.append_coverage(
            tmp_path,
            store.CoverageRow(
                dataset="metrics", symbol="BTCUSDT", day=day, status="ok",
                raw_sha256="0" * 64, rows=1, recorded_at=_NOW, detail=None,
            ),
        )
    plans = plan_collection(
        fake, root=tmp_path, datasets=["metrics"], symbols={"metrics": ["BTCUSDT"]},
        start=None, end=end, incremental_window_days=14,
    )
    assert plans is not None
    planned = sorted({day for plan in plans for day in plan.days})
    assert planned
    assert min(planned) == cutoff + timedelta(days=1)
