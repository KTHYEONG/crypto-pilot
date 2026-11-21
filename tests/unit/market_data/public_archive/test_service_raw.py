from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.catalog import daily_key
from src.market_data.public_archive.service import CollectionPlan, execute_plan
from src.market_data.public_archive.store import load_coverage
from tests.unit.market_data.public_archive._service_support import (
    _AGG_HEADER,
    _NOW,
    FakeDownloader,
    _agg_rows_zip,
    _agg_zip,
    _metrics_fake,
    _metrics_zip,
    _zip_csv,
)

# --- raw tier -------------------------------------------------------------------------------------------------------


def _raw_coverage(root: Path, dataset: str, symbol: str, day: date):
    return load_coverage(root, dataset)[(symbol, day)]


def test_raw_tier_written_with_derived_for_aggtrades(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from src.market_data.public_archive import fetch

    day = date(2024, 1, 15)
    payload = _agg_zip("BTCUSDT", day)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), payload)
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), None)], root=tmp_path)
    assert report.ok == 1
    raw_path = tmp_path / "raw" / "aggTrades" / "BTCUSDT" / "202401.parquet"
    expected = pa.Table.from_batches(list(fetch.iter_agg_trade_batches(payload)))
    assert pq.read_table(raw_path).equals(expected)
    row = _raw_coverage(tmp_path, "aggTrades", "BTCUSDT", day)
    assert row.raw_tier is True
    assert row.rows == 1
    assert row.raw_sha256_content is None


def test_raw_csv_written_for_metrics_byte_exact(tmp_path: Path) -> None:
    import io
    import zipfile

    import pyarrow as pa

    day = date(2024, 1, 15)
    fake = _metrics_fake(["BTCUSDT"], [day])
    report = execute_plan(fake, [CollectionPlan("metrics", "BTCUSDT", (day,), None)], root=tmp_path)
    assert report.ok == 1
    with zipfile.ZipFile(io.BytesIO(_metrics_zip("BTCUSDT", day))) as zf:
        expected = zf.read("data.csv")
    raw_path = tmp_path / "raw" / "metrics" / "BTCUSDT" / f"{day.isoformat()}.csv.zst"
    with pa.input_stream(str(raw_path), compression="zstd") as stream:
        assert stream.read() == expected
    row = _raw_coverage(tmp_path, "metrics", "BTCUSDT", day)
    assert row.raw_tier is True
    assert row.raw_sha256_content is not None


def test_raw_failure_rolls_back_derived_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import service as service_mod

    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(service_mod, "_merge_day_files", fail)
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), None)], root=tmp_path)
    assert report.ok == 0
    assert report.failed == 1
    assert not (tmp_path / "aggTrades" / "BTCUSDT" / "202401.parquet").exists()
    assert not (tmp_path / "raw" / "aggTrades" / "BTCUSDT" / "202401.parquet").exists()
    assert _raw_coverage(tmp_path, "aggTrades", "BTCUSDT", day).status == "failed"


def test_derived_failure_leaves_no_raw_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import store as store_mod

    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))

    def fail(*args, **kwargs):
        raise OSError("derived write failed")

    monkeypatch.setattr(store_mod, "merge_day_batch", fail)
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), None)], root=tmp_path)
    assert report.failed == 1
    assert not (tmp_path / "raw").exists() or not any((tmp_path / "raw").rglob("*.parquet"))


def test_typed_reader_gap_keeps_derived_without_raw(tmp_path: Path) -> None:
    day = date(2024, 1, 15)
    payload = _agg_rows_zip(day, [(1_000, 100.0, 2.0, " true "), (23 * 3_600_000, 101.0, 1.0, " false ")])
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), payload)
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), None)], root=tmp_path)
    assert report.ok == 1
    row = _raw_coverage(tmp_path, "aggTrades", "BTCUSDT", day)
    assert row.status == "ok"
    assert row.raw_tier is False


def _settle_without_raw(root: Path, dataset: str, symbol: str, day: date, rows: int) -> None:
    from dataclasses import replace as _replace

    from src.market_data.public_archive import store as store_mod

    base = store_mod.CoverageRow(
        dataset=dataset, symbol=symbol, day=day, status="ok", raw_sha256=None, rows=rows,
        recorded_at=pd.Timestamp("2024-01-16", tz="UTC"), detail=None,
    )
    store_mod.append_coverage(root, _replace(base, raw_tier=False))


def test_raw_only_backfill_leaves_derived_untouched(tmp_path: Path) -> None:
    import shutil

    from src.market_data.public_archive import service as service_mod

    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_zip("BTCUSDT", day))
    execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), (23, 24))], root=tmp_path)
    derived = tmp_path / "aggTrades" / "BTCUSDT" / "202401.parquet"
    slice_path = tmp_path / "aggTrades_slice" / "BTCUSDT" / "202401.parquet"
    before = (derived.read_bytes(), slice_path.read_bytes(), derived.stat().st_mtime_ns, slice_path.stat().st_mtime_ns)
    shutil.rmtree(tmp_path / "raw")
    _settle_without_raw(tmp_path, "aggTrades", "BTCUSDT", day, rows=1)

    plans = service_mod.plan_raw_backfill(root=tmp_path, dataset="aggTrades", symbols={"BTCUSDT": {day}})
    assert [p.raw_only for p in plans] == [True]
    report = execute_plan(fake, plans, root=tmp_path)
    assert report.ok == 1
    after = (derived.read_bytes(), slice_path.read_bytes(), derived.stat().st_mtime_ns, slice_path.stat().st_mtime_ns)
    assert after == before
    assert (tmp_path / "raw" / "aggTrades" / "BTCUSDT" / "202401.parquet").exists()
    row = _raw_coverage(tmp_path, "aggTrades", "BTCUSDT", day)
    assert row.raw_tier is True
    assert row.rows == 1
    assert row.status == "ok"
    assert service_mod.plan_raw_backfill(root=tmp_path, dataset="aggTrades", symbols={"BTCUSDT": {day}}) == []


def test_raw_backfill_daily_budget_stops_before_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import catalog as catalog_mod
    from src.market_data.public_archive import service as service_mod

    day = date(2024, 1, 15)
    fake = _metrics_fake(["BTCUSDT"], [day])
    execute_plan(fake, [CollectionPlan("metrics", "BTCUSDT", (day,), None)], root=tmp_path)
    raw_dir = tmp_path / "raw"
    shutil.rmtree(raw_dir)
    _settle_without_raw(tmp_path, "metrics", "BTCUSDT", day, rows=1)
    monkeypatch.setattr(catalog_mod, "RAW_BACKFILL_MAX_GIB_PER_DAY", 0)
    plans = service_mod.plan_raw_backfill(root=tmp_path, dataset="metrics", symbols={"BTCUSDT": {day}})
    fake.fetch_calls.clear()
    report = execute_plan(fake, plans, root=tmp_path)
    assert report.stopped_at_deadline is True
    assert report.remaining_days == 1
    assert report.ok == 0
    assert fake.fetch_calls == []
    assert not raw_dir.exists()
    assert _raw_coverage(tmp_path, "metrics", "BTCUSDT", day).raw_tier is False


def test_raw_backfill_refuses_datasets_without_raw_tier(tmp_path: Path) -> None:
    from src.market_data.public_archive import service as service_mod

    with pytest.raises(DataIntegrityError):
        service_mod.plan_raw_backfill(root=tmp_path, dataset="klines", symbols={})


def test_empty_aggtrades_day_completes_raw_tier_without_file(tmp_path: Path) -> None:
    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _zip_csv(f"{_AGG_HEADER}\n"))
    report = execute_plan(fake, [CollectionPlan("aggTrades", "BTCUSDT", (day,), None)], root=tmp_path)
    assert report.ok == 1
    assert not any((tmp_path / "raw").rglob("*.parquet")) if (tmp_path / "raw").exists() else True
    row = _raw_coverage(tmp_path, "aggTrades", "BTCUSDT", day)
    assert row.raw_tier is True
    assert row.rows == 0


def test_raw_stage_exception_fails_only_that_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import service as service_mod

    day = date(2024, 1, 15)
    fake = _metrics_fake(["BTCUSDT"], [day])

    def boom(*args, **kwargs):
        raise OSError("staging unavailable")

    monkeypatch.setattr(service_mod, "_stage_raw", boom)
    report = execute_plan(fake, [CollectionPlan("metrics", "BTCUSDT", (day,), None)], root=tmp_path)
    assert report.ok == 0
    assert report.failed == 1
    assert _raw_coverage(tmp_path, "metrics", "BTCUSDT", day).status == "failed"
    assert not (tmp_path / "metrics" / "BTCUSDT").exists()


def _budget_file(root: Path, payload: object) -> None:
    import json as _json

    path = root / "_status" / "raw_backfill_budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else _json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "payload",
    ["{not json", {"utc_day": "2024-02-01", "bytes": -5}, {"utc_day": "2024-02-01", "bytes": "x"}, {"other": 1}],
    ids=["unparseable", "negative", "wrong_type", "missing_key"],
)
def test_malformed_raw_budget_fails_closed(tmp_path: Path, payload: object) -> None:
    from src.common.errors import DataIntegrityError as _DIE

    day = date(2024, 1, 15)
    _budget_file(tmp_path, payload)
    plans = [CollectionPlan("metrics", "BTCUSDT", (day,), None, raw_only=True)]
    with pytest.raises(_DIE):
        execute_plan(_metrics_fake(["BTCUSDT"], [day]), plans, root=tmp_path, now_fn=lambda: _NOW)


def test_raw_budget_spent_today_stops_before_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import catalog as catalog_mod

    day = date(2024, 1, 15)
    monkeypatch.setattr(catalog_mod, "RAW_BACKFILL_MAX_GIB_PER_DAY", 1)
    _budget_file(tmp_path, {"utc_day": "2024-02-01", "bytes": 1 << 30})
    fake = _metrics_fake(["BTCUSDT"], [day])
    plans = [CollectionPlan("metrics", "BTCUSDT", (day,), None, raw_only=True)]
    report = execute_plan(fake, plans, root=tmp_path, now_fn=lambda: _NOW)
    assert report.stopped_at_deadline is True
    assert fake.fetch_calls == []


def test_raw_budget_resets_on_new_utc_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import catalog as catalog_mod

    day = date(2024, 1, 15)
    monkeypatch.setattr(catalog_mod, "RAW_BACKFILL_MAX_GIB_PER_DAY", 1)
    _budget_file(tmp_path, {"utc_day": "2024-01-31", "bytes": 1 << 30})
    fake = _metrics_fake(["BTCUSDT"], [day])
    plans = [CollectionPlan("metrics", "BTCUSDT", (day,), None)]
    report = execute_plan(fake, plans, root=tmp_path, now_fn=lambda: _NOW)
    assert report.ok == 1


def _agg_padded_zip(day: date) -> bytes:
    return _agg_rows_zip(day, [(1_000, 100.0, 2.0, " true "), (23 * 3_600_000, 101.0, 1.0, " false ")])


def test_raw_only_typed_reader_gap_writes_nothing(tmp_path: Path) -> None:
    from src.market_data.public_archive import service as service_mod

    day = date(2024, 1, 15)
    fake = FakeDownloader()
    fake.add_zip(daily_key("aggTrades", "BTCUSDT", day), _agg_padded_zip(day))
    _settle_without_raw(tmp_path, "aggTrades", "BTCUSDT", day, rows=2)
    plans = service_mod.plan_raw_backfill(root=tmp_path, dataset="aggTrades", symbols={"BTCUSDT": {day}})
    report = execute_plan(fake, plans, root=tmp_path)
    assert report.ok == 0
    assert not (tmp_path / "raw").exists()
    assert _raw_coverage(tmp_path, "aggTrades", "BTCUSDT", day).raw_tier is False


def test_raw_only_stage_exception_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import service as service_mod

    day = date(2024, 1, 15)
    fake = _metrics_fake(["BTCUSDT"], [day])
    _settle_without_raw(tmp_path, "metrics", "BTCUSDT", day, rows=1)

    def boom(*args, **kwargs):
        raise OSError("staging unavailable")

    monkeypatch.setattr(service_mod, "_stage_raw", boom)
    plans = service_mod.plan_raw_backfill(root=tmp_path, dataset="metrics", symbols={"BTCUSDT": {day}})
    report = execute_plan(fake, plans, root=tmp_path)
    assert report.failed == 1
    row = _raw_coverage(tmp_path, "metrics", "BTCUSDT", day)
    assert row.status == "ok"
    assert row.raw_tier is False


def test_missing_raw_partition_of_raw_tier_row_seals_month(tmp_path: Path) -> None:
    import tempfile as _tf

    from src.market_data.public_archive import service as service_mod
    from src.market_data.public_archive import store as store_mod

    day = date(2024, 1, 15)
    store_mod.append_coverage(tmp_path, store_mod.CoverageRow(
        dataset="aggTrades", symbol="BTCUSDT", day=day, status="ok", raw_sha256=None, rows=1,
        recorded_at=pd.Timestamp("2024-01-16", tz="UTC"), detail=None, raw_tier=True,
    ))
    plan = CollectionPlan(dataset="aggTrades", symbol="BTCUSDT", days=(day,), slice_hours=None)
    runner = service_mod._PlanRunner(FakeDownloader(), [plan], root=tmp_path, heartbeat_s=60.0, now_fn=lambda: _NOW)
    with _tf.TemporaryDirectory(dir=tmp_path) as staging:
        staged = Path(staging) / "raw-day.parquet"
        staged.write_bytes(b"stub")
        item = service_mod._RawStaged(path=staged, report=store_mod.RawWriteReport(1, "s" * 64, None, 4))
        detail = runner.commit_month(plan, [day], Path(staging), [], [], [(day, item)])
    assert detail == "sealed_month"
    assert not store_mod.raw_partition_path(tmp_path, "aggTrades", "BTCUSDT", "202401").exists()
