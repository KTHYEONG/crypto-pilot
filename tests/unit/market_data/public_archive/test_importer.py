from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive import importer, store


def _write(root: Path, days: tuple[int, ...], *, dataset: str = "metrics", empty_slice: bool = False) -> None:
    for number in days:
        day = date(2024, 1, number)
        stamp = pd.Timestamp(day, tz="UTC")
        store.merge_partition(root, dataset, "BTCUSDT", day, pd.DataFrame({"timestamp": [stamp], "value": [number]}))
        if dataset == "aggTrades":
            frame = pd.DataFrame(
                {"transact_time_ms": pd.Series([] if empty_slice else [stamp.value // 10**6], dtype="int64")}
            )
            path = store.partition_path(root, dataset, "BTCUSDT", "202401", kind="slice")
            path.parent.mkdir(parents=True, exist_ok=True)
            if empty_slice:
                frame.to_parquet(path, index=False)
            else:
                store.merge_partition(root, dataset, "BTCUSDT", day, frame, kind="slice")
        store.append_coverage(root, store.CoverageRow(dataset, "BTCUSDT", day, "ok", str(number), 1, stamp, None))


def test_merge_conserves_ok_days_and_source(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, (2, 3))
    _write(dst, (1, 2))
    source_before = {p.relative_to(src): p.read_bytes() for p in src.rglob("*") if p.is_file()}
    target = store.partition_path(dst, "metrics", "BTCUSDT", "202401")
    frame = pd.read_parquet(target)
    frame.loc[frame.value == 2, "value"] = 200
    frame.to_parquet(target)
    report = importer.import_archive(src, dst, datasets=["metrics", "metrics"])
    assert report == importer.ImportReport(0, 1, 0, 0, 1)
    assert pd.read_parquet(target).value.tolist() == [1, 200, 3]
    assert {p.relative_to(src): p.read_bytes() for p in src.rglob("*") if p.is_file()} == source_before
    assert importer.import_archive(src, dst, datasets=["metrics"]).days_recorded == 0


def test_paired_copy_empty_slice_and_conflict(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, (1,), dataset="aggTrades", empty_slice=True)
    report = importer.import_archive(src, dst, datasets=["aggTrades"])
    assert report == importer.ImportReport(2, 0, 0, 0, 1)
    for kind in ("derived", "slice"):
        assert (
            store.partition_path(src, "aggTrades", "BTCUSDT", "202401", kind=kind).read_bytes()
            == store.partition_path(dst, "aggTrades", "BTCUSDT", "202401", kind=kind).read_bytes()
        )
    _write(src, (2,), dataset="aggTrades", empty_slice=True)
    assert importer.import_archive(src, dst, datasets=["aggTrades"]).months_merged == 2
    row = store.load_coverage(src, "aggTrades")["BTCUSDT", date(2024, 1, 1)]
    store.append_coverage(src, replace(row, raw_sha256="conflict"))
    before = store.partition_path(dst, "aggTrades", "BTCUSDT", "202401").read_bytes()
    assert importer.import_archive(src, dst, datasets=["aggTrades"]).months_skipped_conflict == 2
    assert store.partition_path(dst, "aggTrades", "BTCUSDT", "202401").read_bytes() == before


def test_sealed_missing_pair_and_missing_day(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, (1,), dataset="aggTrades")
    store.partition_path(src, "aggTrades", "BTCUSDT", "202401", kind="slice").unlink()
    assert importer.import_archive(src, dst, datasets=["aggTrades"]).days_recorded == 0
    assert store.load_coverage(dst, "aggTrades") == {}
    _write(src, (1,), dataset="aggTrades")
    _write(dst, (1,), dataset="aggTrades")
    store.partition_path(dst, "aggTrades", "BTCUSDT", "202401", kind="slice").unlink()
    assert importer.import_archive(src, dst, datasets=["aggTrades"]).months_skipped_sealed == 1
    # A nonempty coverage record without corresponding derived rows cannot be imported.
    path = store.partition_path(src, "aggTrades", "BTCUSDT", "202401")
    pd.read_parquet(path).iloc[:0].to_parquet(path)
    assert importer.import_archive(src, tmp_path / "fresh", datasets=["aggTrades"]).days_recorded == 0


@pytest.mark.parametrize("existing", [False, True])
def test_pair_write_failure_restores_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, (2,), dataset="aggTrades")
    if existing:
        _write(dst, (1,), dataset="aggTrades")
    before = {p.relative_to(dst): p.read_bytes() for p in dst.rglob("*") if p.is_file()}
    original = Path.replace

    def fail_slice(path: Path, target: Path) -> Path:
        if path.name == "slice.parquet":
            raise OSError("injected slice commit failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_slice)
    with pytest.raises(OSError, match="slice commit"):
        importer.import_archive(src, dst, datasets=["aggTrades"])
    assert {p.relative_to(dst): p.read_bytes() for p in dst.rglob("*") if p.is_file()} == before


def test_invalid_roots_datasets_symbols_and_rows(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    with pytest.raises(ValueError, match="disjoint"):
        importer.import_archive(src, src, datasets=[])
    with pytest.raises(FileNotFoundError):
        importer.import_archive(src, dst, datasets=[])
    _write(src, (1,))
    with pytest.raises(DataIntegrityError, match="dataset"):
        importer.import_archive(src, dst, datasets=["unknown"])
    row = next(iter(store.load_coverage(src, "metrics").values()))
    store.append_coverage(src, replace(row, symbol="../../BTCUSDT"))
    with pytest.raises(DataIntegrityError, match="symbol"):
        importer.import_archive(src, dst, datasets=["metrics"])


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"value": [1]}), "sort column"),
        (pd.DataFrame({"timestamp": pd.to_datetime(["2024-02-01"], utc=True)}), "outside"),
        (pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02", "2024-01-01"], utc=True)}), "sorted"),
    ],
)
def test_malformed_source_never_commits(tmp_path: Path, frame: pd.DataFrame, message: str) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, (1,))
    frame.to_parquet(store.partition_path(src, "metrics", "BTCUSDT", "202401"))
    with pytest.raises(DataIntegrityError, match=message):
        importer.import_archive(src, dst, datasets=["metrics"])
    assert not dst.exists()


def test_copy_failure_leaves_no_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, (1,))

    def fail(*args: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(importer.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="copy failed"):
        importer.import_archive(src, dst, datasets=["metrics"])
    assert not list(dst.rglob("*.parquet"))


def test_copied_partitions_are_readable_by_the_backup_user(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, (1,), dataset="aggTrades")
    importer.import_archive(src, dst, datasets=["aggTrades"])
    for kind in ("derived", "slice"):
        mode = store.partition_path(dst, "aggTrades", "BTCUSDT", "202401", kind=kind).stat().st_mode
        assert mode & 0o044 == 0o044
