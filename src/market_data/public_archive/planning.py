from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.public_archive import catalog, store

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


def yesterday_utc(now: pd.Timestamp) -> date:
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
            logger.error("[DATA] stage=public_archive listing_failed detail=%s", short_detail(exc))
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
    recent_cutoff = yesterday_utc(pd.Timestamp.now(tz="UTC")) - timedelta(days=2)
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
    capped_end = min(end, yesterday_utc(pd.Timestamp.now(tz="UTC")))
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


def short_detail(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:200]
