"""Invariants for the read-only research mirror inventory."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.market_data.public_archive.store import CoverageRow, append_coverage
from src.market_data.services.data_inventory import DatasetInventory, build_inventory, write_inventory

TODAY = date(2026, 10, 22)


def _rows(root: Path) -> dict[str, DatasetInventory]:
    return {row.name: row for row in build_inventory(root, today=TODAY)}


def _hour_dir(root: Path, dataset: str, stamp: str, count: int) -> Path:
    path = root / "live_capture" / dataset / stamp
    path.mkdir(parents=True)
    for hour in range(count):
        name = f"part_{hour:02d}.parquet" if dataset == "exec_depth" else f"{hour:02d}.parquet"
        (path / name).write_bytes(b"x")
    return path


def test_hourly_datasets_flag_incomplete_days(tmp_path: Path) -> None:
    _hour_dir(tmp_path, "book_ticker", "20261020", 24)
    _hour_dir(tmp_path, "book_ticker", "20261021", 23)
    row = _rows(tmp_path)["book_ticker"]
    assert row.present_days == 2
    assert row.incomplete_days == (date(2026, 10, 21),)


def test_todays_partial_day_is_not_flagged(tmp_path: Path) -> None:
    _hour_dir(tmp_path, "book_ticker", "20261021", 24)
    _hour_dir(tmp_path, "book_ticker", "20261022", 9)
    row = _rows(tmp_path)["book_ticker"]
    assert row.present_days == 2
    assert row.incomplete_days == ()


def test_calendar_gaps_are_reported(tmp_path: Path) -> None:
    for stamp in ("20261015", "20261016", "20261018"):
        _hour_dir(tmp_path, "exec_depth", stamp, 2)
    row = _rows(tmp_path)["exec_depth"]
    assert row.missing_days == (date(2026, 10, 17),)
    assert row.present_days == 3


def test_malformed_names_never_count(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _hour_dir(tmp_path, "book_ticker", "2026-10-20", 24)
    _hour_dir(tmp_path, "book_ticker", "foo", 24)
    _hour_dir(tmp_path, "book_ticker", "20261340", 24)
    with caplog.at_level(logging.INFO):
        row = _rows(tmp_path)["book_ticker"]
    assert row.present_days == 0
    assert row.first_day is None
    assert "malformed=3" in caplog.text


def test_paper_audit_unions_runs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    for run, days in {
        "r1": ("2026-10-20", "2026-10-21"),
        "r2": ("2026-10-21", "2026-10-22"),
    }.items():
        audit = tmp_path / "state" / "runs" / run / "audit"
        audit.mkdir(parents=True)
        for day in days:
            (audit / f"{day}.jsonl").write_text("{}\n", encoding="utf-8")
        (audit / "notes.txt").write_text("x", encoding="utf-8")
        (audit / "nope.jsonl").write_text("{}\n", encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        row = _rows(tmp_path)["paper_audit"]
    assert row.present_days == 3
    assert (row.first_day, row.last_day) == (date(2026, 10, 20), date(2026, 10, 22))
    assert "runs=r1,r2" in caplog.text
    assert "malformed=4" in caplog.text


def test_day_files_and_liquidation_hours_scan_names(tmp_path: Path) -> None:
    raw = tmp_path / "live_capture" / "raw" / "archive" / "book_ticker"
    raw.mkdir(parents=True)
    (raw / "20261020.jsonl.xz").write_bytes(b"x")
    (raw / "bad.jsonl.xz").write_bytes(b"x")
    liquidations = tmp_path / "futures" / "liquidations"
    liquidations.mkdir(parents=True)
    (liquidations / "liquidations_20261020_00.parquet").write_bytes(b"x")
    (liquidations / "liquidations_20261020_01.parquet").write_bytes(b"x")
    (liquidations / "unexpected.parquet").write_bytes(b"x")
    rows = _rows(tmp_path)
    assert rows["raw_book_ticker"].present_days == 1
    assert rows["raw_book_ticker"].bytes == 1
    assert rows["liquidations"].present_days == 1
    assert rows["liquidations"].incomplete_days == (date(2026, 10, 20),)


def test_legacy_daily_liquidation_partitions_count_as_complete_days(tmp_path: Path) -> None:
    liquidations = tmp_path / "futures" / "liquidations"
    liquidations.mkdir(parents=True)
    (liquidations / "liquidations_20260923.parquet").write_bytes(b"xx")
    (liquidations / "liquidations_20260924.parquet").write_bytes(b"xx")
    for hour in range(24):
        (liquidations / f"liquidations_20260925_{hour:02d}.parquet").write_bytes(b"x")
    (liquidations / "liquidations_2026092.parquet").write_bytes(b"x")
    row = _rows(tmp_path)["liquidations"]
    assert row.first_day == date(2026, 9, 23)
    assert row.present_days == 3
    assert row.missing_days == ()
    assert row.incomplete_days == ()
    assert row.bytes == 28


def _coverage(root: Path, symbol: str, day: date, status: str) -> CoverageRow:
    return CoverageRow(
        dataset="metrics",
        symbol=symbol,
        day=day,
        status=status,  # type: ignore[arg-type]
        raw_sha256=None,
        rows=1,
        recorded_at=pd.Timestamp(datetime(2026, 10, 23, tzinfo=None)),
        detail=None,
    )


def test_archive_coverage_uses_ok_rows_only(tmp_path: Path) -> None:
    root = tmp_path / "public_archive"
    append_coverage(root, _coverage(root, "A", date(2026, 10, 20), "ok"))
    append_coverage(root, _coverage(root, "A", date(2026, 10, 21), "failed"))
    append_coverage(root, _coverage(root, "B", date(2026, 10, 21), "absent"))
    append_coverage(root, _coverage(root, "B", date(2026, 10, 22), "ok"))
    partition = root / "metrics" / "A" / "2026-10.parquet"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"12345")
    row = _rows(tmp_path)["archive_metrics"]
    assert row.present_days == 2
    assert row.symbols == 2
    assert row.missing_days == (date(2026, 10, 21),)
    assert row.bytes == 5


def test_absent_dataset_yields_an_empty_row(tmp_path: Path) -> None:
    row = _rows(tmp_path)["liquidations"]
    assert row.first_day is None
    assert row.present_days == 0
    assert row.bytes == 0


def test_non_matching_entry_kinds_are_ignored(tmp_path: Path) -> None:
    book = tmp_path / "live_capture" / "book_ticker"
    book.mkdir(parents=True)
    (book / "README").write_text("x", encoding="utf-8")
    raw = tmp_path / "live_capture" / "raw" / "archive" / "book_ticker"
    raw.mkdir(parents=True)
    (raw / "20261020.jsonl.xz").write_bytes(b"x")
    (raw / "nested").mkdir()
    liquidations = tmp_path / "futures" / "liquidations"
    liquidations.mkdir(parents=True)
    (liquidations / "nested").mkdir()
    runs = tmp_path / "state" / "runs"
    runs.mkdir(parents=True)
    (runs / "README").write_text("x", encoding="utf-8")
    (runs / "no_audit").mkdir()
    audit = runs / "r1" / "audit"
    audit.mkdir(parents=True)
    (audit / "2026-10-20.jsonl").write_text("{}\n", encoding="utf-8")
    (audit / "subdir").mkdir()
    rows = _rows(tmp_path)
    assert rows["book_ticker"].present_days == 0
    assert rows["raw_book_ticker"].present_days == 1
    assert rows["liquidations"].present_days == 0
    assert rows["paper_audit"].present_days == 1


def test_inventory_is_read_only(tmp_path: Path) -> None:
    path = _hour_dir(tmp_path, "book_ticker", "20261020", 24)
    for file in path.iterdir():
        os.chmod(file, 0o444)
    before = {file: file.stat().st_mtime_ns for file in path.iterdir()}
    build_inventory(tmp_path, today=TODAY)
    after = {file: file.stat().st_mtime_ns for file in path.iterdir()}
    assert before == after


def test_write_inventory_is_atomic_sorted_iso_json(tmp_path: Path) -> None:
    _hour_dir(tmp_path, "book_ticker", "20261020", 24)
    rows = build_inventory(tmp_path, today=TODAY)
    target = tmp_path / "inventory.json"
    write_inventory(rows, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [row["name"] for row in payload] == [row.name for row in rows]
    book = next(row for row in payload if row["name"] == "book_ticker")
    assert book["first_day"] == "2026-10-20"
    assert book["missing_days"] == []
    assert json.dumps(payload, sort_keys=True, indent=2) + "\n" == target.read_text(encoding="utf-8")
    write_inventory(rows[:1], target)
    assert [row["name"] for row in json.loads(target.read_text(encoding="utf-8"))] == [rows[0].name]
    assert not list(tmp_path.glob(".inventory.json.*"))


def test_write_inventory_discards_the_temporary_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.market_data.services import data_inventory as module

    def broken(*_: object, **__: object) -> None:
        raise RuntimeError("serialisation failed")

    monkeypatch.setattr(module.json, "dump", broken)
    with pytest.raises(RuntimeError, match="serialisation failed"):
        write_inventory([], tmp_path / "inventory.json")
    assert not list(tmp_path.glob(".inventory.json.*"))


def test_invalid_partition_names_cannot_make_a_day_complete(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    book = _hour_dir(tmp_path, "book_ticker", "20261020", 23)
    for name in ("24.parquet", "notes.txt", "23.parquet.partial", "23.parquet\n"):
        (book / name).write_bytes(b"invalid")
    _hour_dir(tmp_path, "book_ticker", "20261021", 0)
    depth = _hour_dir(tmp_path, "exec_depth", "20261020", 0)
    (depth / "unrelated.parquet").write_bytes(b"invalid")
    raw = tmp_path / "live_capture/raw/archive/book_ticker"
    raw.mkdir(parents=True)
    for name in ("20261020.jsonl.xz.partial", "20261021.txt"):
        (raw / name).write_bytes(b"invalid")
    reference = tmp_path / "live_capture/reference/exchange_info"
    reference.mkdir(parents=True)
    (reference / "20261020.json.gz").write_bytes(b"valid")
    (reference / "20261021.json.gz.tmp").write_bytes(b"invalid")
    liquidations = tmp_path / "futures/liquidations"
    liquidations.mkdir(parents=True)
    for hour in range(23):
        (liquidations / f"liquidations_20261020_{hour:02d}.parquet").write_bytes(b"x")
    for hour in (24, 99):
        (liquidations / f"liquidations_20261020_{hour}.parquet").write_bytes(b"invalid")
    audit = tmp_path / "state/runs/r1/audit"
    audit.mkdir(parents=True)
    (audit / "20261020.jsonl").write_bytes(b"invalid")
    (audit / "2026-13-40.jsonl").write_bytes(b"invalid")
    with caplog.at_level(logging.INFO):
        rows = _rows(tmp_path)
    assert rows["book_ticker"].present_days == 1
    assert rows["book_ticker"].incomplete_days == (date(2026, 10, 20),)
    assert rows["book_ticker"].bytes == 23
    assert rows["exec_depth"].present_days == 0
    assert rows["raw_book_ticker"].present_days == 0
    assert rows["ref_exchange_info"].present_days == 1
    assert rows["liquidations"].incomplete_days == (date(2026, 10, 20),)
    assert rows["liquidations"].bytes == 23
    assert rows["paper_audit"].present_days == 0
    assert "malformed=4" in caplog.text


def test_only_the_last_day_is_exempt_when_today_is_present(tmp_path: Path) -> None:
    _hour_dir(tmp_path, "book_ticker", "20261022", 1)
    _hour_dir(tmp_path, "book_ticker", "20261023", 1)
    row = _rows(tmp_path)["book_ticker"]
    assert row.incomplete_days == (TODAY, date(2026, 10, 23))


def test_maximum_calendar_date_does_not_overflow(tmp_path: Path) -> None:
    _hour_dir(tmp_path, "exec_depth", "99991231", 1)
    row = _rows(tmp_path)["exec_depth"]
    assert row.last_day == date.max
    assert row.missing_days == ()
