from __future__ import annotations

import heapq
import os
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError
from src.market_data.public_archive import catalog, store


@dataclass(frozen=True, slots=True)
class ImportReport:
    months_copied: int
    months_merged: int
    months_skipped_sealed: int
    months_skipped_conflict: int
    days_recorded: int


def _month_of(day: date) -> str:
    return day.strftime("%Y%m")


def _chunks(path: Path, kind: str, month: str) -> Iterator[tuple[date, pa.Table]]:
    key = store._sort_key(kind)
    previous: object | None = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        frame = batch.to_pandas()
        if key not in frame:
            raise DataIntegrityError(f"partition missing sort column: {key}")
        values = frame[key]
        if not values.is_monotonic_increasing or (previous is not None and len(values) and values.iloc[0] < previous):
            raise DataIntegrityError("partition rows are not sorted")
        if len(values):
            previous = values.iloc[-1]
        days = values.map(lambda value: store._row_day(value, kind))
        if any(_month_of(day) != month for day in days):
            raise DataIntegrityError("partition row outside partition month")
        for day, group in frame.groupby(days, sort=True):
            yield day, pa.Table.from_pandas(group, preserve_index=False)


def _stage_merge(source: Path, dest: Path, staged: Path, kind: str, month: str, wanted: set[date]) -> None:
    def kept() -> Iterator[tuple[date, pa.Table]]:
        if dest.exists():
            yield from (item for item in _chunks(dest, kind, month) if item[0] not in wanted)

    incoming = (item for item in _chunks(source, kind, month) if item[0] in wanted)
    with pq.ParquetWriter(staged, pq.read_schema(source), compression="zstd") as writer:
        for _, table in heapq.merge(kept(), incoming, key=lambda item: item[0]):
            writer.write_table(table.cast(writer.schema))
    with staged.open("rb") as handle:
        os.fsync(handle.fileno())


def _copy_atomic(src: Path, dest: Path) -> None:
    import contextlib

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".import-", dir=str(dest.parent))
    os.close(fd)
    try:
        shutil.copyfile(src, tmp_name)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _commit(staged: dict[Path, Path], directory: Path) -> None:
    backups: dict[Path, Path | None] = {}
    committed: list[Path] = []
    for index, dest in enumerate(staged):
        dest.parent.mkdir(parents=True, exist_ok=True)
        backup = directory / f"{index}.backup"
        if dest.exists():
            os.link(dest, backup)
            backups[dest] = backup
        else:
            backups[dest] = None
    try:
        for dest, path in staged.items():
            path.replace(dest)
            committed.append(dest)
    except BaseException:
        for dest in reversed(committed):
            original = backups[dest]
            if original is None:
                dest.unlink()
            else:
                original.replace(dest)
        raise


def _write_month(
    paths: dict[str, Path], dest: Path, dataset: str, symbol: str, month: str, wanted: set[date]
) -> tuple[int, int]:
    dest.mkdir(parents=True, exist_ok=True)
    copied = merged = 0
    with tempfile.TemporaryDirectory(prefix=".import-", dir=dest) as directory:
        staging = Path(directory)
        staged: dict[Path, Path] = {}
        for kind, path in paths.items():
            target = store.partition_path(dest, dataset, symbol, month, kind=kind)  # type: ignore[arg-type]
            candidate = staging / f"{kind}.parquet"
            # An empty slice is legitimate; validate sort order and month even for copies.
            for _ in _chunks(path, kind, month):
                pass
            if target.exists():
                _stage_merge(path, target, candidate, kind, month, wanted)
                merged += 1
            else:
                _copy_atomic(path, candidate)
                copied += 1
            staged[target] = candidate
        _commit(staged, staging)
    return copied, merged


def import_archive(source: Path, dest: Path, *, datasets: Sequence[str]) -> ImportReport:
    """Merge source ok days without replacing dest ok days; caller holds dest's archive_lock."""
    source, dest = source.resolve(), dest.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError("source and dest archive roots must be disjoint")
    if not source.is_dir():
        raise FileNotFoundError(source)
    unknown = set(datasets) - catalog.DATASETS.keys()
    if unknown:
        raise DataIntegrityError(f"unknown archive datasets: {sorted(unknown)!r}")
    copied = merged = skipped_sealed = skipped_conflict = recorded = 0
    for dataset in dict.fromkeys(datasets):
        src_cov = store.load_coverage(source, dataset)
        dst_cov = store.load_coverage(dest, dataset)
        sealed = store.sealed_months(dest, dataset, dst_cov)
        groups: dict[tuple[str, str], dict[date, store.CoverageRow]] = {}
        for (symbol, day), row in src_cov.items():
            if row.status == "ok":
                if not catalog.is_perpetual_usdt(symbol) or Path(symbol).name != symbol:
                    raise DataIntegrityError(f"invalid archive symbol: {symbol!r}")
                groups.setdefault((symbol, _month_of(day)), {})[day] = row
        for (symbol, month), rows in sorted(groups.items()):
            kinds = ("derived", "slice") if dataset == "aggTrades" else ("derived",)
            sealed_count = sum((kind, symbol, month) in sealed for kind in kinds)
            if sealed_count:
                skipped_sealed += sealed_count
                continue
            overlap = [
                row for day, row in rows.items() if (symbol, day) in dst_cov and dst_cov[symbol, day].status == "ok"
            ]
            if any(dst_cov[symbol, row.day].raw_sha256 != row.raw_sha256 for row in overlap):
                skipped_conflict += len(kinds)
                continue
            wanted = set(rows) - {row.day for row in overlap}
            if not wanted:
                continue
            paths = {kind: store.partition_path(source, dataset, symbol, month, kind=kind) for kind in kinds}  # type: ignore[arg-type]
            if any(not path.is_file() for path in paths.values()):
                continue
            present = {day for day, _ in _chunks(paths["derived"], "derived", month)}
            wanted = {day for day in wanted if rows[day].rows == 0 or day in present}
            if not wanted:
                continue
            month_copied, month_merged = _write_month(paths, dest, dataset, symbol, month, wanted)
            for day in sorted(wanted):
                store.append_coverage(dest, replace(rows[day], recorded_at=pd.Timestamp.now(tz="UTC")))
            copied += month_copied
            merged += month_merged
            recorded += len(wanted)
    return ImportReport(copied, merged, skipped_sealed, skipped_conflict, recorded)
