from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from src.market_data.binance.vision import BinanceVisionDownloader
from src.market_data.mirror import packing
from src.market_data.mirror.config import MirrorConfig
from src.market_data.mirror.datasets import MIRROR_DATASET_SPECS, MirrorDataset
from src.market_data.mirror.index import IndexSummary, iter_index, sidecar_path
from src.market_data.mirror.packing import Capacity, run_partition, utc_now
from src.market_data.mirror.planning import (
    DatasetPlan,
    SourceRef,
    list_dataset,
    plan_dataset,
    read_listing_cache,
    write_listing_cache,
)
from src.market_data.mirror.status import update_status

logger = logging.getLogger("MirrorBuild")
_VANISHED_LOG = "vanished.jsonl"


@dataclass(frozen=True, slots=True)
class BuildReport:
    built_objects: int
    built_sources: int
    verbatim_sources: int
    failed_sources: int
    pending_sources: int
    outbox_bytes: int
    more: bool
    capacity_blocked: str | None
    anomalies: tuple[str, ...]
    listing_failed: tuple[str, ...]


def _recover(config: MirrorConfig) -> None:
    """Delete half-written temp files and outbox objects that have no sidecar; complete outbox pairs stay."""
    for root in (config.work_dir, config.outbox_objects_dir, config.outbox_index_dir):
        if not root.is_dir():
            continue
        for leftover in (*root.rglob("*.tmp"), *root.rglob("*.partial")):
            if leftover.is_file():
                leftover.unlink()
    if not config.outbox_objects_dir.is_dir():
        return
    for obj in config.outbox_objects_dir.rglob("*"):
        if not obj.is_file():
            continue
        object_path = "objects/" + obj.relative_to(config.outbox_objects_dir).as_posix()
        if not sidecar_path(config.outbox_index_dir, object_path).exists():
            logger.warning("[DATA] removed outbox object without sidecar: %s", object_path)
            obj.unlink()


def _outbox_bytes(config: MirrorConfig) -> int:
    if not config.outbox_objects_dir.is_dir():
        return 0
    return sum(obj.stat().st_size for obj in config.outbox_objects_dir.rglob("*") if obj.is_file())


def _listing(
    downloader: BinanceVisionDownloader,
    config: MirrorConfig,
    dataset: MirrorDataset,
    known_symbols: set[str],
    now: datetime,
) -> tuple[bool, tuple[SourceRef, ...], str]:
    cache = config.status_dir / "plan" / f"{dataset.name}.json"
    cached = read_listing_cache(cache, now=now, max_age_hours=config.listing_max_age_hours)
    if cached is not None:
        return True, cached[1], cached[0]
    listed_at = now.isoformat()
    ok, refs = list_dataset(downloader, dataset, known_symbols=known_symbols, workers=config.download_workers)
    if ok:
        write_listing_cache(cache, listed_at, refs)
    return ok, refs, listed_at


def _period_end(dataset: MirrorDataset, period: str) -> date:
    if dataset.cadence == "daily":
        return date.fromisoformat(period)
    year, month = (int(part) for part in period.split("-"))
    following = date(year + (month // 12), month % 12 + 1, 1)
    return following - timedelta(days=1)


def _report_anomalies(
    config: MirrorConfig, plan: DatasetPlan, dataset: MirrorDataset, now: datetime, seen: set[tuple[str, str]]
) -> list[str]:
    anomalies: list[str] = []
    if not plan.listing_ok:
        return anomalies
    fresh = [key for key in plan.vanished_keys if (dataset.name, key) not in seen]
    if fresh:
        config.status_dir.mkdir(parents=True, exist_ok=True)
        with (config.status_dir / _VANISHED_LOG).open("a", encoding="utf-8") as handle:
            for key in fresh:
                handle.write(json.dumps({"dataset": dataset.name, "key": key, "detected_at": now.isoformat()}) + "\n")
                seen.add((dataset.name, key))
        anomalies.append(f"vanished_upstream:{dataset.name}:{len(fresh)}")
    if plan.latest_period is not None:
        threshold = config.stalled_daily_days if dataset.cadence == "daily" else config.stalled_monthly_days
        if (now.date() - _period_end(dataset, plan.latest_period)).days > threshold:
            anomalies.append(f"upstream_stalled:{dataset.name}:{plan.latest_period}")
    return anomalies


def _seen_vanished(config: MirrorConfig) -> set[tuple[str, str]]:
    path = config.status_dir / _VANISHED_LOG
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            seen.add((str(entry["dataset"]), str(entry["key"])))
    return seen


@dataclass(frozen=True, slots=True)
class MirrorPlans:
    plans: dict[str, DatasetPlan]
    listing_failed: tuple[str, ...]
    committed_bytes: int


def plan_datasets(downloader: BinanceVisionDownloader, config: MirrorConfig, *, now: datetime) -> MirrorPlans:
    """Pending work per dataset: bucket listing minus committed and outbox sidecars.

    Datasets are planned one at a time and their index summaries released, so memory follows the largest
    dataset rather than the whole mirror.
    """
    plans: dict[str, DatasetPlan] = {}
    listing_failed: list[str] = []
    committed_bytes = 0
    for name in config.datasets:
        dataset = MIRROR_DATASET_SPECS[name]
        summary = IndexSummary(mirrored={}, symbols=set())
        committed_bytes += summary.add(iter_index(config.index_dir, name))
        summary.add(iter_index(config.outbox_index_dir, name))
        ok, refs, listed_at = _listing(downloader, config, dataset, summary.symbols, now)
        if not ok:
            listing_failed.append(name)
        plans[name] = plan_dataset(
            dataset,
            refs,
            listing_ok=ok,
            mirrored=summary.mirrored,
            today=now.date(),
            close_grace_days=config.close_grace_days,
            listed_at=listed_at,
        )
    return MirrorPlans(plans, tuple(listing_failed), committed_bytes)


def build_chunk(
    downloader: BinanceVisionDownloader,
    config: MirrorConfig,
    *,
    remote_bytes: int,
    drive_free_bytes: int,
    now: datetime,
    deadline: datetime | None,
) -> BuildReport:
    """One bounded build pass: recover, plan, download, pack and publish into the outbox."""
    if remote_bytes < 0 or drive_free_bytes < 0:
        raise ValueError("remote_bytes and drive_free_bytes must be non-negative")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    _recover(config)
    planned = plan_datasets(downloader, config, now=now)
    plans = planned.plans
    capacity = Capacity(
        config,
        remote_bytes=remote_bytes,
        drive_free_bytes=drive_free_bytes,
        committed_bytes=planned.committed_bytes,
        outbox_bytes=_outbox_bytes(config),
    )
    anomalies: list[str] = []
    seen_vanished = _seen_vanished(config)
    for name in config.datasets:
        anomalies.extend(_report_anomalies(config, plans[name], MIRROR_DATASET_SPECS[name], now, seen_vanished))

    pending_by_dataset = {name: sum(len(p.sources) for p in plan.partitions) for name, plan in plans.items()}
    pending_total = sum(pending_by_dataset.values())
    update_status(config.status_dir, lambda data: _stamp_build(data, now, pending_by_dataset))

    cap_bytes = config.outbox_cap_gib * packing.GIB_BYTES
    built_objects = built_sources = verbatim_sources = failed_sources = published = 0
    remaining = dict(pending_by_dataset)
    more = False
    capacity_blocked: str | None = None
    processed = dict.fromkeys(config.datasets, 0)
    verbatim = dict.fromkeys(config.datasets, 0)
    drifted: set[str] = set()
    work = [(name, partition) for name in config.datasets for partition in plans[name].partitions]
    for name, partition in work:
        if name in drifted:
            continue
        if capacity.outbox_bytes >= cap_bytes or (deadline is not None and utc_now() >= deadline):
            more = True
            break
        reason = capacity.reason(0)
        if reason is not None:
            capacity_blocked = reason
            break
        outcome = run_partition(
            downloader,
            config,
            MIRROR_DATASET_SPECS[name],
            partition,
            capacity,
            now=now,
            deadline=deadline,
            drifted=functools.partial(_drifted, config, processed[name], verbatim[name]),
        )
        built_objects += outcome.built_objects
        built_sources += outcome.built_sources
        verbatim_sources += outcome.verbatim_sources
        failed_sources += outcome.failed_sources
        published += outcome.published_sources
        remaining[name] -= outcome.published_sources
        processed[name] += outcome.processed_sources
        verbatim[name] += outcome.verbatim_seen
        if outcome.deadline_hit:
            more = True
            break
        if outcome.capacity_reason is not None:
            capacity_blocked = outcome.capacity_reason
            break
        if outcome.format_drift:
            anomalies.append(f"format_drift:{name}")
            drifted.add(name)

    update_status(config.status_dir, lambda data: _stamp_build(data, now, remaining))
    return BuildReport(
        built_objects=built_objects,
        built_sources=built_sources,
        verbatim_sources=verbatim_sources,
        failed_sources=failed_sources,
        pending_sources=max(0, pending_total - published),
        outbox_bytes=capacity.outbox_bytes,
        more=more,
        capacity_blocked=capacity_blocked,
        anomalies=tuple(anomalies),
        listing_failed=planned.listing_failed,
    )


def _drifted(config: MirrorConfig, prior_processed: int, prior_verbatim: int, processed: int, verbatim: int) -> bool:
    total_verbatim = prior_verbatim + verbatim
    if total_verbatim < config.verbatim_min_sources:
        return False
    return total_verbatim / (prior_processed + processed) > config.verbatim_max_fraction


def _stamp_build(data: dict[str, object], now: datetime, pending: dict[str, int]) -> None:
    data.setdefault("first_build_at", now.isoformat())
    data["last_build_at"] = now.isoformat()
    data["pending_sources"] = pending
