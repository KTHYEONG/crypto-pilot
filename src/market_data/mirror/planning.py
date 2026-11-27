from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from src.market_data.binance.vision import BinanceVisionDownloader, BucketObject
from src.market_data.mirror.datasets import MirrorDataset
from src.market_data.public_archive.catalog import is_perpetual_usdt

logger = logging.getLogger("MirrorPlanning")
_DAILY_SUFFIX = re.compile(r"(\d{4}-\d{2}-\d{2})\.zip$")
_MONTHLY_SUFFIX = re.compile(r"(\d{4}-\d{2})\.zip$")
ALL_PARTITION = "ALL"


@dataclass(frozen=True, slots=True)
class SourceRef:
    key: str
    etag: str
    size: int
    symbol: str
    period: str


@dataclass(frozen=True, slots=True)
class PartitionPlan:
    month: str
    partition: str
    sources: tuple[SourceRef, ...]


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    dataset: str
    listing_ok: bool
    listed_at: str
    partitions: tuple[PartitionPlan, ...]
    vanished_keys: tuple[str, ...]
    latest_period: str | None


def _parse_period(dataset: MirrorDataset, name: str) -> str | None:
    if dataset.cadence == "daily":
        match = _DAILY_SUFFIX.search(name)
        if match is None:
            return None
        try:
            return date.fromisoformat(match.group(1)).isoformat()
        except ValueError:
            return None
    match = _MONTHLY_SUFFIX.search(name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m").strftime("%Y-%m")
    except ValueError:
        return None


def _parse_ref(dataset: MirrorDataset, key: str, size: int, etag: str) -> SourceRef | None:
    relative = key[len(dataset.bucket_root) :]
    symbol = relative.split("/", 1)[0]
    period = _parse_period(dataset, key.rsplit("/", 1)[-1])
    if not symbol or period is None:
        return None
    return SourceRef(key=key, etag=etag, size=size, symbol=sys.intern(symbol), period=sys.intern(period))


def _symbol_prefix(dataset: MirrorDataset, symbol: str) -> str:
    return f"{dataset.bucket_root}{symbol}/{dataset.key_infix}"


def list_dataset(
    downloader: BinanceVisionDownloader,
    dataset: MirrorDataset,
    *,
    known_symbols: Collection[str],
    workers: int,
) -> tuple[bool, tuple[SourceRef, ...]]:
    """Bucket listing of one dataset; ``(False, ())`` when symbol discovery or any symbol listing fails."""
    discovered = [
        s for s in downloader.list_symbols_from_s3_xml_listing(dataset_prefix=dataset.bucket_root) if is_perpetual_usdt(s)
    ]
    if not discovered:
        return False, ()
    symbols = sorted(set(discovered) | set(known_symbols))

    def _list_one(symbol: str) -> list[BucketObject] | None:
        return downloader.list_objects(prefix=_symbol_prefix(dataset, symbol))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        listings = list(pool.map(_list_one, symbols))
    if any(listing is None for listing in listings):
        return False, ()
    refs: list[SourceRef] = []
    for listing in listings:
        assert listing is not None
        for obj in listing:
            if not obj.key.endswith(".zip"):
                continue
            ref = _parse_ref(dataset, obj.key, obj.size, obj.etag)
            if ref is None:
                logger.info("[DATA] ignored unparseable key %s", obj.key)
                continue
            refs.append(ref)
    return True, tuple(sorted(refs, key=lambda ref: (ref.period, ref.key)))


def _first_day_after_month(period: str) -> date:
    year, month = (int(part) for part in period[:7].split("-"))
    return date(year + (month // 12), month % 12 + 1, 1)


def _month_of(dataset: MirrorDataset, period: str) -> str:
    return period[:7] if dataset.cadence == "daily" else period


def plan_dataset(
    dataset: MirrorDataset,
    listed: Sequence[SourceRef],
    *,
    listing_ok: bool,
    mirrored: Mapping[str, frozenset[str]],
    today: date,
    close_grace_days: int,
    listed_at: str,
) -> DatasetPlan:
    """Pending partitions of a listing minus committed and outbox ``(key, etag)`` pairs.

    ``mirrored`` must include outbox sidecars as well as committed ones, so outbox items are not rebuilt.
    """
    if not listing_ok:
        return DatasetPlan(dataset.name, False, listed_at, (), (), None)
    listed_keys = {ref.key for ref in listed}
    vanished = tuple(sorted(key for key in mirrored if key not in listed_keys))
    latest = max((ref.period for ref in listed), default=None)
    pending_by_partition: dict[tuple[str, str], list[SourceRef]] = defaultdict(list)
    for ref in listed:
        if ref.etag in mirrored.get(ref.key, frozenset()):
            continue
        month = _month_of(dataset, ref.period)
        if dataset.cadence == "daily":
            closes_on = _first_day_after_month(month) + timedelta(days=close_grace_days)
            if today < closes_on:
                continue
        partition = ref.symbol if dataset.pack_by == "symbol" else ALL_PARTITION
        pending_by_partition[(month, partition)].append(ref)
    partitions = tuple(
        PartitionPlan(month=month, partition=partition, sources=tuple(refs))
        for (month, partition), refs in sorted(pending_by_partition.items())
    )
    return DatasetPlan(dataset.name, True, listed_at, partitions, vanished, latest)


def write_listing_cache(path: Path, listed_at: str, listed: Sequence[SourceRef]) -> None:
    payload = {
        "listed_at": listed_at,
        "refs": [
            {"key": r.key, "etag": r.etag, "size": r.size, "symbol": r.symbol, "period": r.period} for r in listed
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def read_listing_cache(path: Path, *, now: datetime, max_age_hours: int) -> tuple[str, tuple[SourceRef, ...]] | None:
    """Cached successful listing younger than ``max_age_hours``; None (list again) when absent, stale or unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        listed_at = str(payload["listed_at"])
        listed_time = datetime.fromisoformat(listed_at)
        refs = tuple(
            SourceRef(
                key=str(r["key"]),
                etag=str(r["etag"]),
                size=int(r["size"]),
                symbol=sys.intern(str(r["symbol"])),
                period=sys.intern(str(r["period"])),
            )
            for r in payload["refs"]
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("[DATA] discarded unreadable listing cache %s: %s", path, exc)
        return None
    if now - listed_time > timedelta(hours=max_age_hours):
        return None
    return listed_at, refs
