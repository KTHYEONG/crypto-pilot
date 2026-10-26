"""Per-day completeness inventory for the local research mirror of VPS data."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from logging import getLogger
from pathlib import Path
from typing import Literal

from src.market_data.public_archive.store import load_coverage

logger = getLogger("DataInventory")

DayRule = Literal["day_dir", "day_file", "liquidation_hour_file", "run_audit_file", "archive_coverage"]

_LIQUIDATION_NAME = re.compile(r"liquidations_([0-9]{8})_([01][0-9]|2[0-3])\.parquet")
# Partitions written before the hourly layout hold one whole UTC day each; retention reads both layouts.
_LEGACY_LIQUIDATION_NAME = re.compile(r"liquidations_([0-9]{8})\.parquet")
_HOURS_PER_DAY = 24
_HOUR_NAME = re.compile(r"([01][0-9]|2[0-3])\.parquet")
_DEPTH_NAME = re.compile(r"part_.+\.parquet")


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Declared on-disk shape of one live dataset."""

    name: str
    rel_path: str
    day_rule: DayRule
    expected_units_per_day: int | None
    replaceable: bool
    source: str


INVENTORY_LAYOUTS: tuple[DatasetLayout, ...] = (
    DatasetLayout("book_ticker", "live_capture/book_ticker", "day_dir", 24, False, "bookTicker websocket snapshots"),
    DatasetLayout("premium_index", "live_capture/premium_index", "day_dir", 24, False, "premiumIndex websocket snapshots"),
    DatasetLayout("exec_depth", "live_capture/exec_depth", "day_dir", None, False, "execution-window depth capture"),
    DatasetLayout("raw_book_ticker", "live_capture/raw/archive/book_ticker", "day_file", None, False, "raw bookTicker websocket archive"),
    DatasetLayout("raw_force_order", "live_capture/raw/archive/force_order", "day_file", None, False, "raw forceOrder websocket archive"),
    DatasetLayout("raw_premium_index", "live_capture/raw/archive/premium_index", "day_file", None, False, "raw premiumIndex websocket archive"),
    DatasetLayout("ref_exchange_info", "live_capture/reference/exchange_info", "day_file", None, False, "daily exchangeInfo REST snapshot"),
    DatasetLayout("ref_funding_info", "live_capture/reference/funding_info", "day_file", None, False, "daily fundingInfo REST snapshot"),
    DatasetLayout("ref_asset_index", "live_capture/reference/asset_index", "day_file", None, False, "daily assetIndex REST snapshot"),
    DatasetLayout("liquidations", "futures/liquidations", "liquidation_hour_file", 24, False, "forceOrder-derived liquidations"),
    DatasetLayout("paper_audit", "state/runs", "run_audit_file", None, False, "paper run audit ledgers"),
    DatasetLayout("archive_metrics", "public_archive", "archive_coverage", None, True, "Binance Vision metrics"),
    DatasetLayout("archive_bookDepth", "public_archive", "archive_coverage", None, True, "Binance Vision bookDepth"),
    DatasetLayout("archive_aggTrades", "public_archive", "archive_coverage", None, True, "Binance Vision aggTrades"),
)


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    """Day coverage, gaps and size of one dataset under a mirror root."""

    name: str
    replaceable: bool
    first_day: date | None
    last_day: date | None
    present_days: int
    missing_days: tuple[date, ...]
    incomplete_days: tuple[date, ...]
    bytes: int
    symbols: int | None


@dataclass(frozen=True, slots=True)
class _DayScan:
    present: dict[date, int]
    bytes: int
    malformed: int
    ok_symbols: set[str] | None = None


def _parse_compact(value: str) -> date | None:
    if len(value) != 8 or not value.isdigit():
        return None
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError:
        return None


def _parse_iso_stem(name: str) -> date | None:
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\.jsonl", name) is None:
        return None
    try:
        return date.fromisoformat(name[: -len(".jsonl")])
    except ValueError:
        return None


def _entry_size(entry: os.DirEntry[str]) -> int:
    return entry.stat(follow_symlinks=False).st_size


def _sum_files(path: Path, *, depth: int) -> int:
    try:
        entries = os.scandir(path)
    except FileNotFoundError:
        return 0
    total = 0
    with entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False):
                total += _entry_size(entry)
            elif depth > 0 and entry.is_dir(follow_symlinks=False):
                total += _sum_files(Path(entry.path), depth=depth - 1)
    return total


def _scan_day_dirs(path: Path, *, hourly: bool) -> _DayScan:
    present: dict[date, int] = {}
    total_bytes = 0
    malformed = 0
    try:
        entries = os.scandir(path)
    except FileNotFoundError:
        return _DayScan({}, 0, 0)
    with entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            day = _parse_compact(entry.name)
            if day is None:
                malformed += 1
                continue
            units = 0
            with os.scandir(entry.path) as files:
                for file_entry in files:
                    if file_entry.is_file(follow_symlinks=False):
                        pattern = _HOUR_NAME if hourly else _DEPTH_NAME
                        if pattern.fullmatch(file_entry.name) is None:
                            malformed += 1
                            continue
                        units += 1
                        total_bytes += _entry_size(file_entry)
            if units:
                present[day] = units
    return _DayScan(present, total_bytes, malformed)


def _scan_day_files(path: Path, *, suffix: str) -> _DayScan:
    present: dict[date, int] = {}
    total_bytes = 0
    malformed = 0
    try:
        entries = os.scandir(path)
    except FileNotFoundError:
        return _DayScan({}, 0, 0)
    with entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            day = _parse_compact(entry.name.removesuffix(suffix)) if entry.name.endswith(suffix) else None
            if day is None:
                malformed += 1
                continue
            present[day] = present.get(day, 0) + 1
            total_bytes += _entry_size(entry)
    return _DayScan(present, total_bytes, malformed)


def _scan_liquidation_hours(path: Path) -> _DayScan:
    present: dict[date, int] = {}
    total_bytes = 0
    malformed = 0
    try:
        entries = os.scandir(path)
    except FileNotFoundError:
        return _DayScan({}, 0, 0)
    with entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            hourly = _LIQUIDATION_NAME.fullmatch(entry.name)
            legacy = None if hourly is not None else _LEGACY_LIQUIDATION_NAME.fullmatch(entry.name)
            match = hourly or legacy
            day = _parse_compact(match.group(1)) if match is not None else None
            if day is None:
                malformed += 1
                continue
            present[day] = present.get(day, 0) + (1 if hourly is not None else _HOURS_PER_DAY)
            total_bytes += _entry_size(entry)
    return _DayScan(present, total_bytes, malformed)


def _scan_run_audits(path: Path) -> _DayScan:
    present: dict[date, int] = {}
    runs_by_day: dict[date, list[str]] = {}
    total_bytes = 0
    malformed = 0
    try:
        runs = os.scandir(path)
    except FileNotFoundError:
        return _DayScan({}, 0, 0)
    with runs:
        for run in runs:
            if not run.is_dir(follow_symlinks=False):
                continue
            try:
                audits = os.scandir(Path(run.path) / "audit")
            except FileNotFoundError:
                continue
            with audits:
                for audit in audits:
                    if not audit.is_file(follow_symlinks=False):
                        continue
                    day = _parse_iso_stem(audit.name)
                    if day is None:
                        malformed += 1
                        continue
                    present[day] = present.get(day, 0) + 1
                    total_bytes += _entry_size(audit)
                    runs_by_day.setdefault(day, []).append(run.name)
    for day, run_ids in runs_by_day.items():
        if len(run_ids) > 1:
            logger.debug(
                "[DATA] stage=data_inventory dataset=paper_audit day=%s runs=%s",
                day.isoformat(),
                ",".join(sorted(set(run_ids))),
            )
    return _DayScan(present, total_bytes, malformed)


def _scan_archive_coverage(root: Path, layout: DatasetLayout) -> _DayScan:
    dataset = layout.name.removeprefix("archive_")
    present: dict[date, int] = {}
    ok_symbols: set[str] = set()
    for (symbol, day), row in load_coverage(root / layout.rel_path, dataset).items():
        if row.status != "ok":
            continue
        present[day] = present.get(day, 0) + 1
        ok_symbols.add(symbol)
    total_bytes = _sum_files(root / layout.rel_path / dataset, depth=1)
    return _DayScan(present, total_bytes, 0, ok_symbols)


def _scan_layout(root: Path, layout: DatasetLayout) -> _DayScan:
    path = root / layout.rel_path
    if layout.day_rule == "day_dir":
        return _scan_day_dirs(path, hourly=layout.expected_units_per_day == 24)
    if layout.day_rule == "day_file":
        suffix = ".jsonl.xz" if layout.name.startswith("raw_") else ".json.gz"
        return _scan_day_files(path, suffix=suffix)
    if layout.day_rule == "liquidation_hour_file":
        return _scan_liquidation_hours(path)
    if layout.day_rule == "run_audit_file":
        return _scan_run_audits(path)
    return _scan_archive_coverage(root, layout)


def _calendar_gaps(first: date, last: date, present: dict[date, int]) -> tuple[date, ...]:
    gaps: list[date] = []
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        if day not in present:
            gaps.append(day)
    return tuple(gaps)


def _inventory_row(root: Path, layout: DatasetLayout, *, today: date) -> DatasetInventory:
    scan = _scan_layout(root, layout)
    days = sorted(scan.present)
    first = days[0] if days else None
    last = days[-1] if days else None
    missing = _calendar_gaps(first, last, scan.present) if first is not None and last is not None else ()
    expected = layout.expected_units_per_day
    incomplete = tuple(
        day for day in days if expected is not None and scan.present[day] < expected and not (day == last == today)
    )
    logger.info(
        "[DATA] stage=data_inventory dataset=%s present=%d missing=%d incomplete=%d bytes=%d malformed=%d",
        layout.name,
        len(days),
        len(missing),
        len(incomplete),
        scan.bytes,
        scan.malformed,
    )
    return DatasetInventory(
        name=layout.name,
        replaceable=layout.replaceable,
        first_day=first,
        last_day=last,
        present_days=len(days),
        missing_days=missing,
        incomplete_days=incomplete,
        bytes=scan.bytes,
        symbols=None if scan.ok_symbols is None else len(scan.ok_symbols),
    )


def build_inventory(root: Path, *, today: date) -> list[DatasetInventory]:
    """Scan every INVENTORY_LAYOUTS dataset under root and report day coverage, gaps and size; absent datasets yield empty rows."""
    return [_inventory_row(root, layout, today=today) for layout in INVENTORY_LAYOUTS]


def _iso(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def write_inventory(rows: Sequence[DatasetInventory], path: Path) -> None:
    """Atomically overwrite the derived JSON inventory report at path."""
    payload = [
        {
            "name": row.name,
            "replaceable": row.replaceable,
            "first_day": _iso(row.first_day),
            "last_day": _iso(row.last_day),
            "present_days": row.present_days,
            "missing_days": [_iso(day) for day in row.missing_days],
            "incomplete_days": [_iso(day) for day in row.incomplete_days],
            "bytes": row.bytes,
            "symbols": row.symbols,
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
