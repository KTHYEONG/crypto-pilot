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
) -> list[CollectionPlan] | None:
    """List missing bucket days per symbol without downloading anything."""
    for dataset in datasets:
        if dataset not in catalog.DATASETS:
            raise DataIntegrityError(f"unknown archive dataset: {dataset!r}")
    if not (0 <= slice_hours[0] < slice_hours[1] <= 24):
        raise ValueError("invalid slice window")
    capped_end = min(end, _yesterday_utc(pd.Timestamp.now(tz="UTC")))
    slice_set = set(slice_symbols)
    plans: list[CollectionPlan] = []
    for dataset in datasets:
        coverage = store.load_coverage(root, dataset)
        for symbol in sorted(set(symbols.get(dataset, ()))):
            if start is not None:
                first = start
            else:
                done = [day for (sym, day), row in coverage.items() if sym == symbol and row.status == "ok"]
                first = max(done) + timedelta(days=1) if done else catalog.DATASETS[dataset].first_day
            if first > capped_end:
                continue
            listed: set[date] = set()
            for month in _month_starts(first, capped_end):
                try:
                    keys = downloader.list_object_keys(prefix=_month_prefix(dataset, symbol, month))  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001
                    logger.error("[DATA] stage=public_archive listing_failed detail=%s", _short_detail(exc))
                    return None
                if keys is None:
                    return None
                for key in keys:
                    if not key.endswith(".zip"):
                        continue
                    listed.add(catalog.key_day(key))
            days = tuple(
                day
                for day in sorted(listed)
                if first <= day <= capped_end
                and (coverage.get((symbol, day)) is None or coverage[(symbol, day)].status != "ok")
            )
            pending = tuple(
                day for offset in range(3)
                if (day := capped_end - timedelta(days=offset)) >= first
                and day not in listed
                and (coverage.get((symbol, day)) is None or coverage[(symbol, day)].status != "ok")
            )
            if not days and not pending:
                continue
            window = slice_hours if dataset == "aggTrades" and symbol in slice_set else None
            plans.append(CollectionPlan(dataset=dataset, symbol=symbol, days=days, slice_hours=window, pending_days=pending))
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
    now = now_fn or (lambda: pd.Timestamp.now(tz="UTC"))
    started = now()
    recent_cutoff = _yesterday_utc(started) - timedelta(days=2)
    total = sum(len(plan.days) for plan in plans)
    state = {"done": 0, "ok": 0, "failed": 0, "absent": 0, "pending": sum(len(p.pending_days) for p in plans)}
    guard = threading.Lock()
    beat_guard = threading.Lock()
    last_beat = [time.monotonic()]

    def _write_beat(force: bool = False) -> None:
        beat_now = time.monotonic()
        if not force and beat_now - last_beat[0] < heartbeat_s:
            return
        last_beat[0] = beat_now
        with guard:
            snapshot = dict(state)
        stamp = now()
        payload = {
            "done": snapshot["done"],
            "failed": snapshot["failed"],
            "total": total,
            "ts": stamp.isoformat(),
        }
        path = _heartbeat_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        elapsed = (stamp - started).total_seconds()
        logger.info(
            "[DATA] stage=public_archive done=%d/%d ok=%d failed=%d elapsed_s=%.1f",
            snapshot["done"], total, snapshot["ok"], snapshot["failed"], elapsed,
        )

    def _beat(force: bool = False) -> None:
        with beat_guard:
            _write_beat(force)

    def _bump(field: str) -> None:
        with guard:
            state[field] += 1
            state["done"] += 1

    def _record(row: store.CoverageRow) -> bool:
        try:
            store.append_coverage(root, row)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[DATA] stage=public_archive coverage_failed symbol=%s day=%s detail=%s", row.symbol, row.day, _short_detail(exc))
            return False

    def _run_unit(plan: CollectionPlan, day: date) -> _Ready | _Skipped:
        try:
            key = catalog.daily_key(plan.dataset, plan.symbol, day)
            fetched = fetch.fetch_verified_zip(downloader, key)  # type: ignore[arg-type]
            if fetched is None:
                return _Skipped(day=day, kind="pending" if day >= recent_cutoff else "absent", detail=None)
            zip_bytes, sha = fetched
            del fetched
            raw = fetch.read_single_csv(zip_bytes)
            del zip_bytes
            if plan.dataset == "aggTrades" and raw.columns.tolist() == list(range(len(raw.columns))):
                raw.columns = [
                    "agg_trade_id", "price", "quantity", "first_trade_id",
                    "last_trade_id", "transact_time", "is_buyer_maker",
                ]
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

    def _run_month(plan: CollectionPlan, days: list[date], staging: Path) -> None:
            derived_batch: list[tuple[date, pd.DataFrame]] = []
            slice_batch: list[tuple[date, Path]] = []
            outcomes: list[tuple[date, str, str | None, int]] = []
            for day in days:
                result = _run_unit(plan, day)
                if isinstance(result, _Skipped):
                    if result.kind == "failed":
                        outcomes.append((result.day, "failed", result.detail, 0))
                    else:
                        outcomes.append((result.day, result.kind, None, 0))
                    _bump("failed" if result.kind == "failed" else result.kind)
                    _beat()
                    continue
                derived_batch.append((result.day, result.frame))
                if result.sliced is not None:
                    source = staging / f"{result.day.isoformat()}.parquet"
                    try:
                        write_parquet_atomic(result.sliced, source, compression="zstd")
                    except Exception as exc:  # noqa: BLE001
                        derived_batch.pop()
                        outcomes.append((result.day, "failed", _short_detail(exc), 0))
                        _bump("failed")
                        del result
                        continue
                    slice_batch.append((result.day, source))
                outcomes.append((result.day, "ok", result.sha, len(result.frame)))
                del result
            backups: dict[Path, Path | None] = {}
            committed: list[Path] = []
            try:
                if slice_batch:
                    for kind in ("derived", "slice"):
                        path = store.partition_path(root, plan.dataset, plan.symbol, days[0].strftime("%Y%m"), kind=kind)
                        backup = staging / f"{kind}.backup"
                        if path.exists():
                            os.link(path, backup)
                            backups[path] = backup
                        else:
                            backups[path] = None
                if derived_batch:
                    _merge_day_batch(root, plan.dataset, plan.symbol, derived_batch)
                    if slice_batch:
                        committed.append(store.partition_path(root, plan.dataset, plan.symbol, days[0].strftime("%Y%m")))
                if slice_batch:
                    _merge_slice_files(root, plan.dataset, plan.symbol, slice_batch)
                    committed.append(store.partition_path(root, plan.dataset, plan.symbol, days[0].strftime("%Y%m"), kind="slice"))
            except Exception as exc:  # noqa: BLE001
                for path in committed:
                    original = backups[path]
                    if original is not None:
                        original.replace(path)
                    else:
                        path.unlink(missing_ok=True)
                detail = _short_detail(exc)
                for day_out, status, _sha, _rows in outcomes:
                    if status == "failed":
                        _record(store.CoverageRow(
                            dataset=plan.dataset, symbol=plan.symbol, day=day_out,
                            status="failed", raw_sha256=None, rows=0,
                            recorded_at=now(), detail=_sha,
                        ))
                    if status == "ok":
                        _record(store.CoverageRow(
                            dataset=plan.dataset, symbol=plan.symbol, day=day_out,
                            status="failed", raw_sha256=None, rows=0,
                            recorded_at=now(), detail=detail,
                        ))
                        _bump("failed")
                _beat()
                return
            for day_out, status, sha, rows in outcomes:
                if status != "ok":
                    if status == "failed":
                        _record(store.CoverageRow(
                            dataset=plan.dataset, symbol=plan.symbol, day=day_out,
                            status="failed", raw_sha256=None, rows=0,
                            recorded_at=now(), detail=str(sha),
                        ))
                    continue
                recorded = _record(store.CoverageRow(
                    dataset=plan.dataset, symbol=plan.symbol, day=day_out,
                    status="ok", raw_sha256=str(sha), rows=rows,
                    recorded_at=now(), detail=None,
                ))
                _bump("ok" if recorded else "failed")
                _beat()

    def _run_group(plan: CollectionPlan) -> None:
        groups: dict[str, list[date]] = {}
        for day in sorted(set(plan.days)):
            groups.setdefault(day.strftime("%Y%m"), []).append(day)
        for days in groups.values():
            with tempfile.TemporaryDirectory(prefix=".archive-", dir=root) as directory:
                _run_month(plan, days, Path(directory))

    stop = threading.Event()

    def _periodic_beat() -> None:
        while not stop.wait(max(heartbeat_s, 0.01)):
            _beat(force=True)

    _beat(force=True)
    heartbeat = threading.Thread(target=_periodic_beat, daemon=True)
    heartbeat.start()
    try:
        by_symbol: dict[str, list[CollectionPlan]] = {}
        for plan in plans:
            by_symbol.setdefault(plan.symbol, []).append(plan)

        def _run_symbol(group: list[CollectionPlan]) -> None:
            for plan in group:
                _run_group(plan)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run_symbol, by_symbol.values()))
    finally:
        stop.set()
        heartbeat.join()
    _beat(force=True)
    with guard:
        final = dict(state)
    return CollectionReport(
        planned_days=total,
        ok=final["ok"],
        failed=final["failed"],
        absent=final["absent"],
        pending=final["pending"],
        elapsed_s=(now() - started).total_seconds(),
    )


def _short_detail(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:200]
