from __future__ import annotations

import concurrent.futures
import heapq
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError
from src.common.parquet_io import write_parquet_atomic
from src.market_data.public_archive import catalog, derive, fetch, store

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CollectionPlan:
    dataset: str
    symbol: str
    days: tuple[date, ...]
    slice_hours: tuple[int, int] | None
    pending_days: tuple[date, ...] = ()


@dataclass(frozen=True, slots=True)
class CollectionReport:
    planned_days: int
    ok: int
    failed: int
    absent: int
    pending: int
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class _Ready:
    day: date
    frame: pd.DataFrame
    sliced: pd.DataFrame | None
    sha: str


@dataclass(frozen=True, slots=True)
class _Skipped:
    day: date
    kind: str
    detail: str | None


def _yesterday_utc(now: pd.Timestamp) -> date:
    day: date = (now - timedelta(days=1)).date()
    return day


def _month_starts(start: date, end: date) -> list[date]:
    months: list[date] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        months.append(cursor)
        cursor = date(cursor.year + 1, 1, 1) if cursor.month == 12 else date(cursor.year, cursor.month + 1, 1)
    return months


def _month_prefix(dataset: str, symbol: str, month: date) -> str:
    return (
        f"data/futures/um/daily/{dataset}/{symbol}/"
        f"{symbol}-{dataset}-{month.strftime('%Y-%m')}"
    )


def _symbol_prefix(dataset: str, symbol: str) -> str:
    return f"data/futures/um/daily/{dataset}/{symbol}/{symbol}-{dataset}-"


def _daily_key_day(key: str) -> date | None:
    """Day of a daily archive key; None (with a warning) for non-daily names such as ``-unsort.zip``."""
    try:
        return catalog.key_day(key)
    except DataIntegrityError:
        logger.warning("[DATA] stage=public_archive skipped_non_daily_key key=%s", key)
        return None


class _ListingFailedError(Exception):
    """A bucket listing failed; the whole plan is abandoned (fail closed)."""


def _is_ok(coverage: Mapping[tuple[str, date], store.CoverageRow], symbol: str, day: date) -> bool:
    row = coverage.get((symbol, day))
    return row is not None and row.status == "ok"


def _unmasked_first_day(
    dataset: str, symbol: str, coverage: Mapping[tuple[str, date], store.CoverageRow],
    *, start: date | None, end: date, incremental_window_days: int | None,
) -> date:
    if start is not None:
        return start
    done = [day for (sym, day), row in coverage.items() if sym == symbol and row.status == "ok"]
    if done:
        base = max(done) + timedelta(days=1)
    elif incremental_window_days is not None:
        base = max(catalog.DATASETS[dataset].first_day, end - timedelta(days=incremental_window_days))
    else:
        base = catalog.DATASETS[dataset].first_day
    return max(catalog.HISTORY_START, base)


def _candidate_days(
    dataset: str, symbol: str, coverage: Mapping[tuple[str, date], store.CoverageRow],
    allow_map: Mapping[str, Collection[date]] | None,
    *, start: date | None, end: date, capped_end: date, incremental_window_days: int | None,
) -> list[date]:
    """Days still to collect for a symbol: allowed (or cursor-derived) days not yet ``ok``."""
    if allow_map is not None:
        wanted = set(allow_map.get(symbol, ()))
        floor = start if start is not None else catalog.MASK_POLICIES[dataset].start
        if incremental_window_days is not None:
            floor = max(floor, end - timedelta(days=incremental_window_days))
        pool = [day for day in wanted if floor <= day <= capped_end]
    else:
        first = _unmasked_first_day(
            dataset, symbol, coverage, start=start, end=end, incremental_window_days=incremental_window_days,
        )
        span = (capped_end - first).days + 1
        pool = [first + timedelta(days=offset) for offset in range(max(span, 0))]
    return sorted(day for day in pool if not _is_ok(coverage, symbol, day))


def _list_days(
    downloader: object, dataset: str, symbol: str, months: Sequence[date], *, whole_symbol: bool,
) -> set[date]:
    """Daily keys present in the bucket for the months (or the whole symbol prefix)."""
    prefixes = (
        [_symbol_prefix(dataset, symbol)]
        if whole_symbol
        else [_month_prefix(dataset, symbol, month) for month in months]
    )
    listed: set[date] = set()
    for prefix in prefixes:
        try:
            keys = downloader.list_object_keys(prefix=prefix)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.error("[DATA] stage=public_archive listing_failed detail=%s", _short_detail(exc))
            raise _ListingFailedError from exc
        if keys is None:
            raise _ListingFailedError
        for key in keys:
            if not key.endswith(".zip"):
                continue
            day = _daily_key_day(key)
            if day is not None:
                listed.add(day)
    return listed


def _plan_symbol(
    downloader: object, dataset: str, symbol: str, candidates: list[date], capped_end: date,
    window: tuple[int, int] | None,
) -> CollectionPlan | None:
    months = sorted({date(day.year, day.month, 1) for day in candidates})
    listed = _list_days(downloader, dataset, symbol, months, whole_symbol=len(months) > 3)
    wanted = set(candidates)
    days = tuple(day for day in candidates if day in listed)
    pending = tuple(
        day for offset in range(3)
        if (day := capped_end - timedelta(days=offset)) in wanted and day not in listed
    )
    if not days and not pending:
        return None
    return CollectionPlan(dataset=dataset, symbol=symbol, days=days, slice_hours=window, pending_days=pending)


def plan_collection(
    downloader: object,
    *,
    root: Path,
    datasets: Sequence[str],
    symbols: Mapping[str, Sequence[str]],
    start: date | None,
    end: date,
    slice_hours: tuple[int, int] = (23, 24),
    slice_symbols: Collection[str] = (),
    incremental_window_days: int | None = None,
    allowed: Mapping[str, Mapping[str, Collection[date]]] | None = None,
) -> list[CollectionPlan] | None:
    """List missing bucket days per symbol without downloading anything."""
    for dataset in datasets:
        if dataset not in catalog.DATASETS:
            raise DataIntegrityError(f"unknown archive dataset: {dataset!r}")
    if not (0 <= slice_hours[0] < slice_hours[1] <= 24):
        raise ValueError("invalid slice window")
    if incremental_window_days is not None and incremental_window_days < 0:
        raise ValueError("incremental_window_days must be non-negative")
    capped_end = min(end, _yesterday_utc(pd.Timestamp.now(tz="UTC")))
    slice_set = set(slice_symbols)
    plans: list[CollectionPlan] = []
    try:
        for dataset in datasets:
            coverage = store.load_coverage(root, dataset)
            allow_map = allowed[dataset] if allowed is not None and dataset in allowed else None
            for symbol in sorted(set(symbols.get(dataset, ()))):
                candidates = _candidate_days(
                    dataset, symbol, coverage, allow_map, start=start, end=end,
                    capped_end=capped_end, incremental_window_days=incremental_window_days,
                )
                if not candidates:
                    continue
                window = slice_hours if dataset == "aggTrades" and symbol in slice_set else None
                plan = _plan_symbol(downloader, dataset, symbol, candidates, capped_end, window)
                if plan is not None:
                    plans.append(plan)
    except _ListingFailedError:
        return None
    return plans


def _derive_day(dataset: str, raw: pd.DataFrame, *, symbol: str, day: date) -> pd.DataFrame:
    if dataset == "metrics":
        return derive.derive_metrics(raw, symbol=symbol, day=day)
    if dataset == "bookDepth":
        return derive.derive_book_depth(raw, symbol=symbol, day=day)
    if dataset == "aggTrades":
        return derive.derive_agg_trades(raw, symbol=symbol, day=day)
    raise DataIntegrityError(f"unknown archive dataset: {dataset!r}")


def _merge_day_batch(
    root: Path,
    dataset: str,
    symbol: str,
    batch: list[tuple[date, pd.DataFrame]],
    *,
    kind: str = "derived",
) -> None:
    month = batch[0][0].strftime("%Y%m")
    path = store.partition_path(root, dataset, symbol, month, kind=kind)  # type: ignore[arg-type]
    key = "transact_time_ms" if kind == "slice" else "timestamp"
    for _day, frame in batch:
        if len(frame) and key not in frame.columns:
            raise DataIntegrityError(f"missing sort column: {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(path) if path.exists() else None
    if existing is not None and len(existing) and key not in existing.columns:
        raise DataIntegrityError(f"partition missing sort column: {key}")
    drop = {day for day, _ in batch}
    if existing is not None and len(existing):
        keep = [store._row_day(v, kind) not in drop for v in existing[key].tolist()]
        kept = existing.loc[keep]
    else:
        kept = None
    parts = [frame for _, frame in batch if len(frame)]
    if kept is None or len(kept) == 0:
        merged = pd.concat(parts, ignore_index=True) if parts else (existing.iloc[:0] if existing is not None else batch[0][1].iloc[:0])
    else:
        merged = pd.concat([kept, *parts], ignore_index=True) if parts else kept
    if len(merged) and any(store._row_month(v, kind) != month for v in merged[key].tolist()):
        raise DataIntegrityError("partition row outside partition month")
    merged = merged.sort_values(key, kind="mergesort").reset_index(drop=True)
    if existing is None and not len(merged):
        return
    write_parquet_atomic(merged, path, compression="zstd")


def _heartbeat_path(root: Path) -> Path:
    return Path(root) / "_status" / "collect_heartbeat.json"


def _merge_slice_files(root: Path, dataset: str, symbol: str, batch: list[tuple[date, Path]]) -> None:
    month = batch[0][0].strftime("%Y%m")
    path = store.partition_path(root, dataset, symbol, month, kind="slice")
    path.parent.mkdir(parents=True, exist_ok=True)
    replaced = {day for day, _ in batch}

    def existing_chunks() -> Iterator[tuple[date, pa.Table]]:
        if path.exists():
            for chunk in pq.ParquetFile(path).iter_batches(batch_size=65536):
                frame = chunk.to_pandas()
                days = pd.to_datetime(frame["transact_time_ms"], unit="ms", utc=True).dt.date
                for day, group in frame.groupby(days, sort=True):
                    if day not in replaced:
                        yield day, pa.Table.from_pandas(group, preserve_index=False)

    def new_chunks() -> Iterator[tuple[date, pa.Table]]:
        for day, source in sorted(batch):
            for chunk in pq.ParquetFile(source).iter_batches(batch_size=65536):
                yield day, pa.Table.from_batches([chunk])

    tmp = batch[0][1].parent / "month.parquet"
    writer: pq.ParquetWriter | None = None
    try:
        for _, table in heapq.merge(existing_chunks(), new_chunks(), key=lambda item: item[0]):
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            table = pq.read_table(batch[0][1])
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.close()
        writer = None
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if writer is not None:
            writer.close()


_Outcome = tuple[date, str, "str | None", int]


class _PlanRunner:
    """One ``execute_plan`` run: bounded workers share counters, heartbeat and coverage writes."""

    def __init__(
        self, downloader: object, plans: Sequence[CollectionPlan], *, root: Path,
        heartbeat_s: float, now_fn: Callable[[], pd.Timestamp],
    ) -> None:
        self.downloader = downloader
        self.plans = plans
        self.root = root
        self.heartbeat_s = heartbeat_s
        self.now = now_fn
        self.started = now_fn()
        self.recent_cutoff = _yesterday_utc(self.started) - timedelta(days=2)
        self.total = sum(len(plan.days) for plan in plans)
        self.state = {"done": 0, "ok": 0, "failed": 0, "absent": 0, "pending": sum(len(p.pending_days) for p in plans)}
        self.guard = threading.Lock()
        self.beat_guard = threading.Lock()
        self.last_beat = time.monotonic()

    def _write_beat(self, force: bool) -> None:
        beat_now = time.monotonic()
        if not force and beat_now - self.last_beat < self.heartbeat_s:
            return
        self.last_beat = beat_now
        with self.guard:
            snapshot = dict(self.state)
        stamp = self.now()
        payload = {"done": snapshot["done"], "failed": snapshot["failed"], "total": self.total, "ts": stamp.isoformat()}
        path = _heartbeat_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        logger.info(
            "[DATA] stage=public_archive done=%d/%d ok=%d failed=%d elapsed_s=%.1f",
            snapshot["done"], self.total, snapshot["ok"], snapshot["failed"], (stamp - self.started).total_seconds(),
        )

    def beat(self, force: bool = False) -> None:
        with self.beat_guard:
            self._write_beat(force)

    def bump(self, field: str) -> None:
        with self.guard:
            self.state[field] += 1
            self.state["done"] += 1

    def record(self, plan: CollectionPlan, day: date, status: Literal["ok", "absent", "failed"], *, sha: str | None, rows: int, detail: str | None) -> bool:
        row = store.CoverageRow(
            dataset=plan.dataset, symbol=plan.symbol, day=day, status=status,
            raw_sha256=sha, rows=rows, recorded_at=self.now(), detail=detail,
        )
        try:
            store.append_coverage(self.root, row)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[DATA] stage=public_archive coverage_failed symbol=%s day=%s detail=%s",
                plan.symbol, day, _short_detail(exc),
            )
            return False

    def run_unit(self, plan: CollectionPlan, day: date) -> _Ready | _Skipped:
        try:
            key = catalog.daily_key(plan.dataset, plan.symbol, day)
            fetched = fetch.fetch_verified_zip(self.downloader, key)  # type: ignore[arg-type]
            if fetched is None:
                return _Skipped(day=day, kind="pending" if day >= self.recent_cutoff else "absent", detail=None)
            zip_bytes, sha = fetched
            del fetched
            raw = fetch.read_single_csv(zip_bytes)
            del zip_bytes
            frame = _derive_day(plan.dataset, raw, symbol=plan.symbol, day=day)
            sliced = (
                derive.slice_agg_trades(raw, day=day, start_hour_utc=plan.slice_hours[0], end_hour_utc=plan.slice_hours[1])
                if plan.slice_hours is not None
                else None
            )
            del raw
            return _Ready(day=day, frame=frame, sliced=sliced, sha=sha)
        except Exception as exc:  # noqa: BLE001
            return _Skipped(day=day, kind="failed", detail=_short_detail(exc))

    def gather_month(
        self, plan: CollectionPlan, days: list[date], staging: Path,
    ) -> tuple[list[tuple[date, pd.DataFrame]], list[tuple[date, Path]], list[_Outcome]]:
        derived: list[tuple[date, pd.DataFrame]] = []
        slices: list[tuple[date, Path]] = []
        outcomes: list[_Outcome] = []
        for day in days:
            result = self.run_unit(plan, day)
            if isinstance(result, _Skipped):
                outcomes.append((result.day, result.kind, result.detail, 0))
                self.bump("failed" if result.kind == "failed" else result.kind)
                self.beat()
                continue
            derived.append((result.day, result.frame))
            if result.sliced is not None:
                source = staging / f"{result.day.isoformat()}.parquet"
                try:
                    write_parquet_atomic(result.sliced, source, compression="zstd")
                except Exception as exc:  # noqa: BLE001
                    derived.pop()
                    outcomes.append((result.day, "failed", _short_detail(exc), 0))
                    self.bump("failed")
                    continue
                slices.append((result.day, source))
            outcomes.append((result.day, "ok", result.sha, len(result.frame)))
        return derived, slices, outcomes

    def commit_month(
        self, plan: CollectionPlan, days: list[date], staging: Path,
        derived: list[tuple[date, pd.DataFrame]], slices: list[tuple[date, Path]],
    ) -> str | None:
        """Write the month partitions; None on success, else the failure detail after rollback."""
        month = days[0].strftime("%Y%m")
        backups: dict[Path, Path | None] = {}
        committed: list[Path] = []
        derived_path = store.partition_path(self.root, plan.dataset, plan.symbol, month)
        slice_path = store.partition_path(self.root, plan.dataset, plan.symbol, month, kind="slice")
        try:
            if slices:
                for kind, path in (("derived", derived_path), ("slice", slice_path)):
                    backup = staging / f"{kind}.backup"
                    if path.exists():
                        os.link(path, backup)
                        backups[path] = backup
                    else:
                        backups[path] = None
            if derived:
                _merge_day_batch(self.root, plan.dataset, plan.symbol, derived)
                if slices:
                    committed.append(derived_path)
            if slices:
                _merge_slice_files(self.root, plan.dataset, plan.symbol, slices)
                committed.append(slice_path)
        except Exception as exc:  # noqa: BLE001
            for path in committed:
                original = backups[path]
                if original is not None:
                    original.replace(path)
                else:
                    path.unlink(missing_ok=True)
            return _short_detail(exc)
        return None

    def run_month(self, plan: CollectionPlan, days: list[date], staging: Path) -> None:
        derived, slices, outcomes = self.gather_month(plan, days, staging)
        detail = self.commit_month(plan, days, staging, derived, slices)
        for day, status, info, rows in outcomes:
            if status == "failed":
                self.record(plan, day, "failed", sha=None, rows=0, detail=str(info))
            elif status == "ok" and detail is not None:
                self.record(plan, day, "failed", sha=None, rows=0, detail=detail)
                self.bump("failed")
            elif status == "ok":
                recorded = self.record(plan, day, "ok", sha=str(info), rows=rows, detail=None)
                self.bump("ok" if recorded else "failed")
                self.beat()
        if detail is not None:
            self.beat()

    def run_group(self, plan: CollectionPlan) -> None:
        groups: dict[str, list[date]] = {}
        for day in sorted(set(plan.days)):
            groups.setdefault(day.strftime("%Y%m"), []).append(day)
        for days in groups.values():
            with tempfile.TemporaryDirectory(prefix=".archive-", dir=self.root) as directory:
                self.run_month(plan, days, Path(directory))

    def run_symbol(self, group: list[CollectionPlan]) -> None:
        for plan in group:
            self.run_group(plan)

    def run(self, workers: int) -> CollectionReport:
        stop = threading.Event()

        def periodic() -> None:
            while not stop.wait(max(self.heartbeat_s, 0.01)):
                self.beat(force=True)

        self.beat(force=True)
        heartbeat = threading.Thread(target=periodic, daemon=True)
        heartbeat.start()
        try:
            by_symbol: dict[str, list[CollectionPlan]] = {}
            for plan in self.plans:
                by_symbol.setdefault(plan.symbol, []).append(plan)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(self.run_symbol, by_symbol.values()))
        finally:
            stop.set()
            heartbeat.join()
        self.beat(force=True)
        with self.guard:
            final = dict(self.state)
        return CollectionReport(
            planned_days=self.total, ok=final["ok"], failed=final["failed"], absent=final["absent"],
            pending=final["pending"], elapsed_s=(self.now() - self.started).total_seconds(),
        )


def execute_plan(
    downloader: object,
    plans: Sequence[CollectionPlan],
    *,
    root: Path,
    workers: int = 4,
    heartbeat_s: float = 60.0,
    now_fn: Callable[[], pd.Timestamp] | None = None,
) -> CollectionReport:
    """Fetch, derive, merge, and record every planned day with bounded workers."""
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    runner = _PlanRunner(
        downloader, plans, root=root, heartbeat_s=heartbeat_s,
        now_fn=now_fn or (lambda: pd.Timestamp.now(tz="UTC")),
    )
    return runner.run(workers)


def _short_detail(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:200]
