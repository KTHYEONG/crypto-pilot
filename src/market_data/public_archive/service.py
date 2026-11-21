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
from dataclasses import dataclass, replace
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
    sealed_days: tuple[date, ...] = ()
    raw_only: bool = False


@dataclass(frozen=True, slots=True)
class CollectionReport:
    planned_days: int
    ok: int
    failed: int
    absent: int
    pending: int
    elapsed_s: float
    sealed: int = 0
    stopped_at_deadline: bool = False
    remaining_days: int = 0
    defective: int = 0


@dataclass(frozen=True, slots=True)
class _Ready:
    day: date
    frame: pd.DataFrame | None
    sliced: pd.DataFrame | None
    sha: str
    raw: _RawStaged | None = None


@dataclass(frozen=True, slots=True)
class _Skipped:
    day: date
    kind: str
    detail: str | None
    sha: str | None = None


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
    return f"data/futures/um/daily/{dataset}/{symbol}/{symbol}-{dataset}-{month.strftime('%Y-%m')}"


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


def _is_settled(
    coverage: Mapping[tuple[str, date], store.CoverageRow], symbol: str, day: date, *, retry_defective: bool
) -> bool:
    row = coverage.get((symbol, day))
    return row is not None and (row.status == "ok" or (row.status == "defective" and not retry_defective))


def _unmasked_first_day(
    dataset: str,
    symbol: str,
    coverage: Mapping[tuple[str, date], store.CoverageRow],
    *,
    start: date | None,
    end: date,
    incremental_window_days: int | None,
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
    dataset: str,
    symbol: str,
    coverage: Mapping[tuple[str, date], store.CoverageRow],
    allow_map: Mapping[str, Collection[date]] | None,
    *,
    start: date | None,
    end: date,
    capped_end: date,
    incremental_window_days: int | None,
    retry_defective: bool = False,
) -> list[date]:
    """Days still to collect for a symbol: allowed (or cursor-derived) days not yet settled."""
    if allow_map is not None:
        wanted = set(allow_map.get(symbol, ()))
        floor = start if start is not None else catalog.MASK_POLICIES[dataset].start
        if incremental_window_days is not None:
            floor = max(floor, end - timedelta(days=incremental_window_days))
        pool = [day for day in wanted if floor <= day <= capped_end]
    else:
        first = _unmasked_first_day(
            dataset,
            symbol,
            coverage,
            start=start,
            end=end,
            incremental_window_days=incremental_window_days,
        )
        span = (capped_end - first).days + 1
        pool = [first + timedelta(days=offset) for offset in range(max(span, 0))]
    return sorted(day for day in pool if not _is_settled(coverage, symbol, day, retry_defective=retry_defective))


def _list_days(
    downloader: object,
    dataset: str,
    symbol: str,
    months: Sequence[date],
    *,
    whole_symbol: bool,
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
    downloader: object,
    dataset: str,
    symbol: str,
    candidates: list[date],
    capped_end: date,
    window: tuple[int, int] | None,
    *,
    allow_whole_symbol: bool = True,
) -> CollectionPlan | None:
    months = sorted({date(day.year, day.month, 1) for day in candidates})
    listed = _list_days(downloader, dataset, symbol, months, whole_symbol=allow_whole_symbol and len(months) > 3)
    wanted = set(candidates)
    days = tuple(day for day in candidates if day in listed)
    recent_cutoff = _yesterday_utc(pd.Timestamp.now(tz="UTC")) - timedelta(days=2)
    pending = tuple(
        day
        for offset in range(3)
        if (day := capped_end - timedelta(days=offset)) >= recent_cutoff and day in wanted and day not in listed
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
    retry_defective: bool = False,
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
            sealed = store.sealed_months(root, dataset, coverage)
            allow_map = allowed[dataset] if allowed is not None and dataset in allowed else None
            for symbol in sorted(set(symbols.get(dataset, ()))):
                candidates = _candidate_days(
                    dataset,
                    symbol,
                    coverage,
                    allow_map,
                    start=start,
                    end=end,
                    capped_end=capped_end,
                    incremental_window_days=incremental_window_days,
                    retry_defective=retry_defective,
                )
                if not candidates:
                    continue
                window = slice_hours if dataset == "aggTrades" and symbol in slice_set else None
                sliced = window is not None
                sealed_days: list[date] = []
                open_candidates: list[date] = []
                for day in candidates:
                    month = day.strftime("%Y%m")
                    if (("derived", symbol, month) in sealed) or (sliced and ("slice", symbol, month) in sealed):
                        sealed_days.append(day)
                    else:
                        open_candidates.append(day)
                plan = _plan_symbol(
                    downloader,
                    dataset,
                    symbol,
                    open_candidates,
                    capped_end,
                    window,
                    allow_whole_symbol=not any(
                        sym == symbol and (kind == "derived" or sliced) for kind, sym, _ in sealed
                    ),
                )
                if plan is None:
                    if sealed_days:
                        plans.append(CollectionPlan(dataset, symbol, (), window, sealed_days=tuple(sealed_days)))
                else:
                    if sealed_days:
                        plan = replace(plan, sealed_days=tuple(sealed_days))
                    plans.append(plan)
    except _ListingFailedError:
        return None
    return plans


def _derive_zip(plan: CollectionPlan, day: date, zip_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if plan.dataset == "aggTrades":
        try:
            return derive.derive_agg_trades_streamed(
                fetch.iter_agg_trade_batches(zip_bytes), day=day, slice_hours=plan.slice_hours
            )
        except derive.StreamFallbackError as exc:
            logger.debug("[DATA] stage=public_archive stream_fallback day=%s reason=%s", day, exc)
    raw = fetch.read_single_csv(zip_bytes)
    frame = _derive_day(plan.dataset, raw, symbol=plan.symbol, day=day)
    if plan.slice_hours is None:
        return frame, None
    return frame, derive.slice_agg_trades(
        raw, day=day, start_hour_utc=plan.slice_hours[0], end_hour_utc=plan.slice_hours[1]
    )


def _derive_day(dataset: str, raw: pd.DataFrame, *, symbol: str, day: date) -> pd.DataFrame:
    if dataset == "metrics":
        return derive.derive_metrics(raw, symbol=symbol, day=day)
    if dataset == "bookDepth":
        return derive.derive_book_depth(raw, symbol=symbol, day=day)
    if dataset == "aggTrades":
        return derive.derive_agg_trades(raw, symbol=symbol, day=day)
    raise ValueError(f"unknown archive dataset: {dataset!r}")


def _heartbeat_path(root: Path) -> Path:
    return Path(root) / "_status" / "collect_heartbeat.json"


def _merge_day_files(path: Path, column: str, batch: list[tuple[date, Path]], *, raw: bool = False) -> None:
    """Replace the batch's days inside one month file, streaming; days are keyed by ``column`` (epoch ms)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    replaced = {day for day, _ in batch}

    def existing_chunks() -> Iterator[tuple[date, pa.Table]]:
        if path.exists():
            for chunk in pq.ParquetFile(path).iter_batches(batch_size=65536):
                frame = chunk.to_pandas()
                days = pd.to_datetime(frame[column], unit="ms", utc=True).dt.date
                for day, group in frame.groupby(days, sort=True):
                    if day not in replaced:
                        yield day, pa.Table.from_pandas(group, preserve_index=False)

    def new_chunks() -> Iterator[tuple[date, pa.Table]]:
        for day, source in sorted(batch):
            for chunk in pq.ParquetFile(source).iter_batches(batch_size=65536):
                yield day, pa.Table.from_batches([chunk])

    def options(schema: pa.Schema) -> dict[str, object]:
        return store.raw_agg_writer_options(schema) if raw else {"compression": "zstd"}

    tmp = batch[0][1].parent / f"merge-{path.name}"
    writer: pq.ParquetWriter | None = None
    try:
        for _, table in heapq.merge(existing_chunks(), new_chunks(), key=lambda item: item[0]):
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, **options(table.schema))
            writer.write_table(table)
        if writer is None:
            table = pq.read_table(batch[0][1])
            writer = pq.ParquetWriter(tmp, table.schema, **options(table.schema))
        writer.close()
        writer = None
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if writer is not None:
            writer.close()


def _merge_slice_files(root: Path, dataset: str, symbol: str, batch: list[tuple[date, Path]]) -> None:
    month = batch[0][0].strftime("%Y%m")
    path = store.partition_path(root, dataset, symbol, month, kind="slice")
    _merge_day_files(path, "transact_time_ms", batch)


@dataclass(frozen=True, slots=True)
class _Outcome:
    day: date
    status: Literal["ok", "failed"]
    info: str | None
    rows: int
    raw_sha256_content: str | None = None
    raw_tier: bool = False


@dataclass(frozen=True, slots=True)
class _RawStaged:
    path: Path | None  # None when a day has no trade rows: the raw tier is complete without a file
    report: store.RawWriteReport


def _stage_raw(dataset: str, day: date, zip_bytes: bytes, sha: str, staging: Path) -> _RawStaged | None:
    """Write and verify one day's raw tier inside the month staging dir; nothing is published yet.

    None when the typed trade reader cannot represent the day (the derived path still handles it), so the day
    is stored without a raw tier rather than failing.
    """
    if dataset == "aggTrades":
        path = staging / f"raw-{day.isoformat()}.parquet"
        try:
            report = store.write_raw_agg_trades(
                path, fetch.iter_agg_trade_batches(zip_bytes), source_zip_sha256=sha
            )
        except pa.ArrowInvalid:
            return None
        if report.rows == 0:
            return _RawStaged(path=None, report=report)
        store.verify_raw_agg_trades(path, fetch.iter_agg_trade_batches(zip_bytes))
        return _RawStaged(path=path, report=report)
    csv_bytes = fetch.read_single_csv_bytes(zip_bytes)
    path = staging / f"raw-{day.isoformat()}.csv.zst"
    report = store.write_raw_csv(path, csv_bytes, source_zip_sha256=sha)
    store.verify_raw_csv(path, csv_bytes)
    return _RawStaged(path=path, report=report)


class _RawBudget:
    """Per-UTC-day cap on new raw bytes; the spend persists in one status file across runs."""

    def __init__(self, root: Path, now: pd.Timestamp) -> None:
        self._path = Path(root) / "_status" / "raw_backfill_budget.json"
        self._day = now.strftime("%Y-%m-%d")
        self._cap = int(catalog.RAW_BACKFILL_MAX_GIB_PER_DAY) * (1 << 30)
        self._spent = 0
        self._guard = threading.Lock()
        if self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise DataIntegrityError("malformed raw backfill budget") from exc
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("utc_day"), str)
                or type(payload.get("bytes")) is not int
                or payload["bytes"] < 0
            ):
                raise DataIntegrityError("malformed raw backfill budget")
            self._spent = payload["bytes"] if payload["utc_day"] == self._day else 0

    def exhausted(self) -> bool:
        with self._guard:
            return self._spent >= self._cap

    def spend(self, nbytes: int) -> None:
        with self._guard:
            self._spent += nbytes
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"utc_day": self._day, "bytes": self._spent}, sort_keys=True), encoding="utf-8")
            tmp.replace(self._path)


class _PlanRunner:
    """One ``execute_plan`` run: bounded workers share counters, heartbeat and coverage writes."""

    def __init__(
        self,
        downloader: object,
        plans: Sequence[CollectionPlan],
        *,
        root: Path,
        heartbeat_s: float,
        now_fn: Callable[[], pd.Timestamp],
        deadline: pd.Timestamp | None = None,
    ) -> None:
        if deadline is not None:
            stamp = pd.Timestamp(deadline)
            if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
                raise ValueError("deadline must be timezone-aware UTC")
            deadline = stamp.tz_convert("UTC")
        self.downloader = downloader
        self.plans = plans
        self.root = root
        self.heartbeat_s = heartbeat_s
        self.now = now_fn
        self.deadline, self._stop, self._started = deadline, False, 0
        self.started = now_fn()
        self.recent_cutoff = _yesterday_utc(self.started) - timedelta(days=2)
        self.total = sum(len(plan.days) for plan in plans)
        self.state = {
            "done": 0, "ok": 0, "failed": 0, "absent": 0, "defective": 0,
            "pending": sum(len(p.pending_days) for p in plans),
        }
        self.guard = threading.Lock()
        self.beat_guard = threading.Lock()
        self.last_beat = time.monotonic()
        self._sealed: set[tuple[str, str, str, str]] = set()
        self._coverage: dict[str, Mapping[tuple[str, date], store.CoverageRow]] = {}
        for dataset in {p.dataset for p in plans}:
            cov = store.load_coverage(root, dataset)
            self._coverage[dataset] = cov
            for kind, symbol, month in store.sealed_months(root, dataset, cov):
                self._sealed.add((dataset, kind, symbol, month))
        self._budget = _RawBudget(root, self.started) if any(p.raw_only for p in plans) else None

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
            snapshot["done"],
            self.total,
            snapshot["ok"],
            snapshot["failed"],
            (stamp - self.started).total_seconds(),
        )

    def beat(self, force: bool = False) -> None:
        with self.beat_guard:
            self._write_beat(force)

    def bump(self, field: str) -> None:
        with self.guard:
            self.state[field] += 1
            self.state["done"] += 1

    def _existing_rows(self, plan: CollectionPlan, day: date) -> int:
        row = self._coverage.get(plan.dataset, {}).get((plan.symbol, day))
        return 0 if row is None else row.rows

    def record(
        self,
        plan: CollectionPlan,
        day: date,
        status: Literal["ok", "absent", "failed", "defective"],
        *,
        sha: str | None,
        rows: int,
        detail: str | None,
        raw_tier: bool = False,
        raw_sha256_content: str | None = None,
    ) -> bool:
        row = store.CoverageRow(
            dataset=plan.dataset,
            symbol=plan.symbol,
            day=day,
            status=status,
            raw_sha256=sha,
            rows=rows,
            recorded_at=self.now(),
            detail=detail,
            raw_sha256_content=raw_sha256_content,
            raw_tier=raw_tier,
        )
        try:
            store.append_coverage(self.root, row)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[DATA] stage=public_archive coverage_failed symbol=%s day=%s detail=%s",
                plan.symbol,
                day,
                _short_detail(exc),
            )
            return False

    def run_unit(self, plan: CollectionPlan, day: date, staging: Path) -> _Ready | _Skipped:
        try:
            fetched = fetch.fetch_verified_zip(
                self.downloader, catalog.daily_key(plan.dataset, plan.symbol, day)  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            return _Skipped(day=day, kind="failed", detail=_short_detail(exc))
        if fetched is None:
            return _Skipped(day=day, kind="pending" if day >= self.recent_cutoff else "absent", detail=None)
        zip_bytes, sha = fetched
        del fetched
        if plan.raw_only:
            try:
                raw = _stage_raw(plan.dataset, day, zip_bytes, sha, staging)
            except Exception as exc:  # noqa: BLE001
                return _Skipped(day=day, kind="failed", detail=_short_detail(exc))
            if raw is None:
                return _Skipped(day=day, kind="failed", detail="raw tier unavailable for typed reader")
            return _Ready(day=day, frame=None, sliced=None, sha=sha, raw=raw)
        try:
            frame, sliced = _derive_zip(plan, day, zip_bytes)
        except DataIntegrityError as exc:
            return _Skipped(day=day, kind="defective", detail=_short_detail(exc), sha=sha)
        except Exception as exc:  # noqa: BLE001
            return _Skipped(day=day, kind="failed", detail=_short_detail(exc))
        raw_staged: _RawStaged | None = None
        if plan.dataset in catalog.RAW_TIER_DATASETS:
            try:
                raw_staged = _stage_raw(plan.dataset, day, zip_bytes, sha, staging)
            except Exception as exc:  # noqa: BLE001
                return _Skipped(day=day, kind="failed", detail=_short_detail(exc))
        return _Ready(day=day, frame=frame, sliced=sliced, sha=sha, raw=raw_staged)

    def gather_month(
        self,
        plan: CollectionPlan,
        days: list[date],
        staging: Path,
    ) -> tuple[
        list[tuple[date, pd.DataFrame]],
        list[tuple[date, Path]],
        list[tuple[date, _RawStaged]],
        list[_Outcome],
    ]:
        derived: list[tuple[date, pd.DataFrame]] = []
        slices: list[tuple[date, Path]] = []
        raws: list[tuple[date, _RawStaged]] = []
        outcomes: list[_Outcome] = []
        for index, day in enumerate(days):
            if plan.raw_only and self._budget is not None and self._budget.exhausted():
                with self.guard:
                    self._started -= len(days) - index
                    self._stop = True
                break
            result = self.run_unit(plan, day, staging)
            if isinstance(result, _Skipped):
                kind = result.kind
                if kind == "defective" and not plan.raw_only and not self.record(
                    plan, day, "defective", sha=result.sha, rows=0, detail=result.detail
                ):
                    kind = "failed"
                elif kind == "failed":
                    outcomes.append(_Outcome(day=result.day, status="failed", info=result.detail, rows=0))
                self.bump(kind)
                self.beat()
                continue
            if result.frame is not None:
                derived.append((result.day, result.frame))
                if result.sliced is not None:
                    source = staging / f"{result.day.isoformat()}.parquet"
                    try:
                        write_parquet_atomic(result.sliced, source, compression="zstd")
                    except Exception as exc:  # noqa: BLE001
                        derived.pop()
                        outcomes.append(_Outcome(day=result.day, status="failed", info=_short_detail(exc), rows=0))
                        self.bump("failed")
                        continue
                    slices.append((result.day, source))
            rows = len(result.frame) if result.frame is not None else self._existing_rows(plan, day)
            if result.raw is not None:
                raws.append((result.day, result.raw))
                if self._budget is not None:
                    self._budget.spend(result.raw.report.bytes_written)
            outcomes.append(
                _Outcome(
                    day=result.day,
                    status="ok",
                    info=result.sha,
                    rows=rows,
                    raw_sha256_content=None if result.raw is None else result.raw.report.content_sha256,
                    raw_tier=result.raw is not None,
                )
            )
        return derived, slices, raws, outcomes

    def commit_month(
        self,
        plan: CollectionPlan,
        days: list[date],
        staging: Path,
        derived: list[tuple[date, pd.DataFrame]],
        slices: list[tuple[date, Path]],
        raws: Sequence[tuple[date, _RawStaged]] = (),
    ) -> str | None:
        """Publish derived, slice and raw partitions together; None on success, else the failure detail after rollback."""
        month = days[0].strftime("%Y%m")
        derived_path = store.partition_path(self.root, plan.dataset, plan.symbol, month)
        slice_path = store.partition_path(self.root, plan.dataset, plan.symbol, month, kind="slice")
        staged = [(day, item.path) for day, item in raws if item.path is not None]
        raw_merge: list[tuple[date, Path]] = []
        raw_moves: list[tuple[Path, Path]] = []
        raw_targets: list[Path] = []
        if plan.dataset == "aggTrades":
            if staged:
                raw_agg = store.raw_partition_path(self.root, plan.dataset, plan.symbol, month)
                raw_targets.append(raw_agg)
                raw_merge = staged
        else:
            for day, source in staged:
                target = store.raw_partition_path(self.root, plan.dataset, plan.symbol, day.isoformat())
                raw_targets.append(target)
                raw_moves.append((source, target))
        if derived and not derived_path.exists() and (plan.dataset, "derived", plan.symbol, month) in self._sealed:
            return "sealed_month"
        if slices and not slice_path.exists() and (plan.dataset, "slice", plan.symbol, month) in self._sealed:
            return "sealed_month"
        if any(
            not target.exists() and (plan.dataset, "raw", plan.symbol, month) in self._sealed for target in raw_targets
        ):
            return "sealed_month"
        touched: list[Path] = []
        if derived:
            touched.append(derived_path)
        if slices:
            touched.append(slice_path)
        touched.extend(raw_targets)
        if not touched:
            return None
        backups: dict[Path, Path | None] = {}
        committed: list[Path] = []
        try:
            for index, path in enumerate(touched):
                backups[path] = None
                if path.exists():
                    backup = staging / f"{index}.backup"
                    os.link(path, backup)
                    backups[path] = backup
            if derived:
                store.merge_day_batch(self.root, plan.dataset, plan.symbol, derived)
                committed.append(derived_path)
            if slices:
                _merge_slice_files(self.root, plan.dataset, plan.symbol, slices)
                committed.append(slice_path)
            if raw_merge:
                _merge_day_files(raw_targets[0], "transact_time", raw_merge, raw=True)
                committed.append(raw_targets[0])
            for source, target in raw_moves:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                committed.append(target)
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
        derived, slices, raws, outcomes = self.gather_month(plan, days, staging)
        detail = self.commit_month(plan, days, staging, derived, slices, raws)
        for outcome in outcomes:
            if outcome.status == "failed":
                if not plan.raw_only:
                    self.record(plan, outcome.day, "failed", sha=None, rows=0, detail=str(outcome.info))
            elif detail is not None:
                if not plan.raw_only:
                    self.record(plan, outcome.day, "failed", sha=None, rows=0, detail=detail)
                self.bump("failed")
            else:
                recorded = self.record(
                    plan,
                    outcome.day,
                    "ok",
                    sha=outcome.info,
                    rows=outcome.rows,
                    detail=None,
                    raw_tier=outcome.raw_tier,
                    raw_sha256_content=outcome.raw_sha256_content,
                )
                self.bump("ok" if recorded else "failed")
                self.beat()
        if detail is not None:
            self.beat()

    def run_group(self, plan: CollectionPlan) -> None:
        groups: dict[str, list[date]] = {}
        for day in sorted(set(plan.days)):
            groups.setdefault(day.strftime("%Y%m"), []).append(day)
        for days in groups.values():
            with self.guard:
                if self._stop or (self.deadline is not None and self.now() >= self.deadline):
                    self._stop = True
                    continue
                self._started += len(days)
            with tempfile.TemporaryDirectory(prefix=store.STAGING_PREFIX, dir=self.root) as directory:
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
            started = self._started
            stopped = self._stop
        return CollectionReport(
            planned_days=self.total,
            ok=final["ok"],
            failed=final["failed"],
            absent=final["absent"],
            defective=final["defective"],
            pending=final["pending"],
            elapsed_s=(self.now() - self.started).total_seconds(),
            sealed=sum(len(p.sealed_days) for p in self.plans),
            stopped_at_deadline=stopped,
            remaining_days=max(0, self.total - started),
        )


def execute_plan(
    downloader: object,
    plans: Sequence[CollectionPlan],
    *,
    root: Path,
    workers: int = 4,
    heartbeat_s: float = 60.0,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    deadline: pd.Timestamp | None = None,
) -> CollectionReport:
    """Fetch, derive, merge, and record every planned day with bounded workers."""
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    runner = _PlanRunner(
        downloader,
        plans,
        root=root,
        heartbeat_s=heartbeat_s,
        now_fn=now_fn or (lambda: pd.Timestamp.now(tz="UTC")),
        deadline=deadline,
    )
    return runner.run(workers)


def plan_raw_backfill(
    *, root: Path, dataset: str, symbols: Mapping[str, Collection[date]]
) -> list[CollectionPlan]:
    """Raw-only plans for days already settled as ok whose raw tier is still missing; nothing is downloaded."""
    if dataset not in catalog.RAW_TIER_DATASETS:
        raise DataIntegrityError(f"dataset has no raw tier: {dataset!r}")
    coverage = store.load_coverage(root, dataset)
    plans: list[CollectionPlan] = []
    for symbol in sorted(symbols):
        days = tuple(
            sorted(
                day
                for day in set(symbols[symbol])
                if (row := coverage.get((symbol, day))) is not None and row.status == "ok" and not row.raw_tier
            )
        )
        if days:
            plans.append(CollectionPlan(dataset=dataset, symbol=symbol, days=days, slice_hours=None, raw_only=True))
    return plans


def _short_detail(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:200]
