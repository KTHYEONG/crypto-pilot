"""Pipeline data readiness planner: coverage inspection, sync planning, derived cache, retention."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CoverageStatus = Literal[
    "ready",
    "missing",
    "stale_start",
    "stale_end",
    "gap",
    "unavailable_lifecycle",
    "failed_retryable",
    "missing_skipped",
]

# Timeframe -> bar duration in nanoseconds
_TF_DURATION_NS: dict[str, int] = {
    "1m": 60 * 1_000_000_000,
    "1h": 3600 * 1_000_000_000,
    "4h": 14400 * 1_000_000_000,
    "1d": 86400 * 1_000_000_000,
}


def _bar_duration_ns(timeframe: str) -> int:
    return _TF_DURATION_NS.get(timeframe, 3600 * 1_000_000_000)


def _date_to_ns(d: date) -> int:
    return int(pd.Timestamp(d, tz="UTC").value)


def _resolve_lifecycle_date(
    lifecycle_entry: object,
    key: str,
) -> date | None:
    if isinstance(lifecycle_entry, dict):
        v = lifecycle_entry.get(key)
        return v if isinstance(v, date) else None
    return getattr(lifecycle_entry, key, None)


@dataclass(frozen=True, slots=True)
class DataRequirement:
    timeframe: str
    start: date
    end: date
    required: bool
    consumer_phases: tuple[Literal["l0", "l1", "l2", "l3"], ...]
    min_coverage_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class CacheCoverage:
    symbol: str
    timeframe: str
    required_start_ns: int
    required_end_ns: int
    observed_start_ns: int | None
    observed_end_ns: int | None
    coverage_ratio: float
    max_gap_bars: int
    duplicate_count: int
    status: CoverageStatus
    cache_mtime_ns: int | None
    cache_size_bytes: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class SyncTarget:
    symbol: str
    timeframe: str
    start: date
    end: date
    required: bool
    reason: CoverageStatus


@dataclass(frozen=True, slots=True)
class DataSyncPlan:
    requirements: tuple[DataRequirement, ...]
    coverage: tuple[CacheCoverage, ...]
    targets: tuple[SyncTarget, ...]
    required_failures: tuple[CacheCoverage, ...]


@dataclass(frozen=True, slots=True)
class LtfSyncSelection:
    candidate_symbols: tuple[str, ...]
    selected_symbols: tuple[str, ...]
    covered_symbols: frozenset[str]
    targets: tuple[SyncTarget, ...]
    coverage: tuple[CacheCoverage, ...]
    selector_version: str


@dataclass(frozen=True, slots=True)
class DerivedCacheTarget:
    symbol: str
    timeframe: str
    start: date
    end: date
    source_fingerprint: str
    status: Literal["ready", "materialize", "pruned"]


@dataclass(frozen=True, slots=True)
class DerivedCachePlan:
    targets: tuple[DerivedCacheTarget, ...]
    retained_timeframes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    mode: Literal["preserve", "rolling"]
    base_guard_days: int = 92
    ltf_guard_days: int = 31
    rewrite_ratio: float = 0.25
    rewrite_min_bytes: int = 67_108_864
    max_files_per_run: int = 8
    max_reclaim_bytes: int = 1_073_741_824


class DerivedTimeframeStore:
    def load(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> object | None:
        raise NotImplementedError

    def store(
        self,
        *,
        symbol: str,
        timeframe: str,
        data: object,
    ) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}::{timeframe}"


def _make_cache_path(cache_root: Path, symbol: str, timeframe: str) -> Path:
    return cache_root / timeframe / f"{symbol}.parquet"


def _check_manifest_fast_path(
    manifest: Mapping[str, object],
    symbol: str,
    timeframe: str,
    cache_path: Path,
) -> tuple[bool, dict[str, Any] | None]:
    """Fast path: return (True, entry) if manifest matches current file stat."""
    key = _manifest_key(symbol, timeframe)
    entry_raw = manifest.get(key)
    if entry_raw is None or not isinstance(entry_raw, dict):
        return False, None
    entry: dict[str, Any] = entry_raw
    if not cache_path.exists():
        return False, None
    stat = cache_path.stat()
    file_mtime_ns = int(stat.st_mtime_ns)
    file_size = stat.st_size
    if entry.get("mtime_ns") == file_mtime_ns and entry.get("size_bytes") == file_size:
        return True, entry
    return False, entry


def _inspect_cache_file(
    cache_path: Path,
    expected_start_ns: int,
    expected_end_ns: int,
    max_gap_bars: int,
    bar_duration_ns: int,
) -> CacheCoverage:
    """Inspect a single parquet file for coverage metrics."""
    if not cache_path.exists():
        return CacheCoverage(
            symbol=cache_path.stem,
            timeframe=cache_path.parent.name,
            required_start_ns=expected_start_ns,
            required_end_ns=expected_end_ns,
            observed_start_ns=None,
            observed_end_ns=None,
            coverage_ratio=0.0,
            max_gap_bars=0,
            duplicate_count=0,
            status="missing",
            cache_mtime_ns=None,
            cache_size_bytes=None,
            reason="file not found",
        )

    stat = cache_path.stat()
    mtime_ns = int(stat.st_mtime_ns)
    size_bytes = stat.st_size

    ts = pd.read_parquet(cache_path, columns=["timestamp"])
    if ts.empty:
        return CacheCoverage(
            symbol=cache_path.stem,
            timeframe=cache_path.parent.name,
            required_start_ns=expected_start_ns,
            required_end_ns=expected_end_ns,
            observed_start_ns=None,
            observed_end_ns=None,
            coverage_ratio=0.0,
            max_gap_bars=0,
            duplicate_count=0,
            status="missing",
            cache_mtime_ns=mtime_ns,
            cache_size_bytes=size_bytes,
            reason="empty parquet file",
        )

    ts_ns = ts["timestamp"].values.astype(np.int64)
    observed_start = int(ts_ns.min())
    observed_end = int(ts_ns.max())

    # Duplicate count
    _, counts = np.unique(ts_ns, return_counts=True)
    duplicate_count = int((counts > 1).sum())

    # Gap analysis
    sorted_ts = np.sort(ts_ns)
    diffs = np.diff(sorted_ts)
    max_gap_observed = int(diffs.max() // bar_duration_ns) if len(diffs) > 0 else 0

    # Coverage ratio
    total_expected_bars = max(1, (expected_end_ns - expected_start_ns) // bar_duration_ns)
    distinct_bars = len(np.unique(ts_ns))
    coverage_ratio = min(1.0, distinct_bars / total_expected_bars)

    # Determine status
    if duplicate_count > 0:
        status: CoverageStatus = "gap"
        reason = f"duplicate timestamps: {duplicate_count}"
    elif max_gap_observed > max_gap_bars:
        status = "gap"
        reason = f"largest gap {max_gap_observed} bars exceeds max {max_gap_bars}"
    elif observed_start > expected_start_ns:
        status = "stale_start"
        reason = f"first bar {observed_start} > required start {expected_start_ns}"
    elif observed_end < (expected_end_ns - bar_duration_ns):
        status = "stale_end"
        reason = f"last bar {observed_end} < required end {expected_end_ns}"
    elif coverage_ratio < 1.0:
        status = "missing"
        reason = f"coverage ratio {coverage_ratio:.4f} < 1.0"
    else:
        status = "ready"
        reason = ""

    return CacheCoverage(
        symbol=cache_path.stem,
        timeframe=cache_path.parent.name,
        required_start_ns=expected_start_ns,
        required_end_ns=expected_end_ns,
        observed_start_ns=observed_start,
        observed_end_ns=observed_end,
        coverage_ratio=coverage_ratio,
        max_gap_bars=max_gap_observed,
        duplicate_count=duplicate_count,
        status=status,
        cache_mtime_ns=mtime_ns,
        cache_size_bytes=size_bytes,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_pipeline_requirements(
    *,
    phase: Literal["l1", "l2", "l3"],
    layered_window: Any,
) -> tuple[DataRequirement, ...]:
    """Resolve base data requirements for a given phase and layered window."""
    if not hasattr(layered_window, "fetch_start") or not hasattr(layered_window, "holdout_end"):
        raise ValueError("layered_window must have fetch_start and holdout_end")

    fetch_start: date = layered_window.fetch_start
    holdout_end: date = layered_window.holdout_end
    l2_start: date = layered_window.l2_start
    holdout_start: date = layered_window.holdout_start

    phase_map: dict[str, date] = {
        "l1": l2_start,
        "l2": holdout_start,
        "l3": holdout_end,
    }
    required_end = phase_map[phase]

    # 1d has separate end (historical warmup for 1d is longer)
    # Per spec: 1d starts 272 days before fetch_start
    import datetime

    d1_start = fetch_start - datetime.timedelta(days=272)

    return (
        DataRequirement(
            timeframe="1h",
            start=fetch_start,
            end=required_end,
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        ),
        DataRequirement(
            timeframe="4h",
            start=fetch_start,
            end=required_end,
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        ),
        DataRequirement(
            timeframe="1d",
            start=d1_start,
            end=required_end,
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        ),
    )


def inspect_cache_coverage(
    *,
    symbol: str,
    requirement: DataRequirement,
    cache_path: Path,
    onboard_date: date | None,
    delivery_date: date | None,
    max_gap_bars: int,
    manifest: Mapping[str, object],
) -> CacheCoverage:
    """Inspect a single (symbol, timeframe) cache for coverage."""
    effective_start = requirement.start
    effective_end = requirement.end

    if onboard_date is not None and onboard_date > effective_start:
        effective_start = onboard_date
    if delivery_date is not None and delivery_date < effective_end:
        effective_end = delivery_date

    expected_start_ns = _date_to_ns(effective_start)
    expected_end_ns = _date_to_ns(effective_end)
    bar_duration_ns = _bar_duration_ns(requirement.timeframe)

    if effective_start >= effective_end:
        return CacheCoverage(
            symbol=symbol,
            timeframe=requirement.timeframe,
            required_start_ns=expected_start_ns,
            required_end_ns=expected_end_ns,
            observed_start_ns=None,
            observed_end_ns=None,
            coverage_ratio=0.0,
            max_gap_bars=0,
            duplicate_count=0,
            status="unavailable_lifecycle",
            cache_mtime_ns=None,
            cache_size_bytes=None,
            reason=f"empty interval after lifecycle clip: {effective_start} >= {effective_end}",
        )

    file_path = _make_cache_path(cache_path, symbol, requirement.timeframe)
    hit, entry = _check_manifest_fast_path(manifest, symbol, requirement.timeframe, file_path)

    if hit and entry is not None:
        return CacheCoverage(
            symbol=symbol,
            timeframe=requirement.timeframe,
            required_start_ns=expected_start_ns,
            required_end_ns=expected_end_ns,
            observed_start_ns=int(entry["first_bar_ns"]),
            observed_end_ns=int(entry["last_bar_ns"]),
            coverage_ratio=float(entry.get("coverage_ratio", 1.0)),
            max_gap_bars=int(entry.get("max_gap_bars_observed", 0)),
            duplicate_count=int(entry.get("duplicate_count", 0)),
            status="ready",
            cache_mtime_ns=int(entry["mtime_ns"]),
            cache_size_bytes=int(entry["size_bytes"]),
            reason="manifest hit",
        )

    coverage = _inspect_cache_file(
        file_path,
        expected_start_ns,
        expected_end_ns,
        max_gap_bars,
        bar_duration_ns,
    )
    return coverage


def build_base_data_plan(
    *,
    symbols: Sequence[str],
    requirements: Sequence[DataRequirement],
    lifecycle_by_symbol: Mapping[str, object],
    cache_root: Path,
    max_gap_bars: int,
    manifest: Mapping[str, object],
) -> DataSyncPlan:
    """Build a data sync plan for base requirements."""
    coverages: list[CacheCoverage] = []
    targets: list[SyncTarget] = []
    required_failures: list[CacheCoverage] = []

    for symbol in symbols:
        onboard = _resolve_lifecycle_date(lifecycle_by_symbol.get(symbol), "onboard_date")
        delivery = _resolve_lifecycle_date(lifecycle_by_symbol.get(symbol), "delivery_date")

        for req in requirements:
            cov = inspect_cache_coverage(
                symbol=symbol,
                requirement=req,
                cache_path=cache_root,
                onboard_date=onboard,
                delivery_date=delivery,
                max_gap_bars=max_gap_bars,
                manifest=manifest,
            )
            coverages.append(cov)

            if cov.status == "unavailable_lifecycle":
                if req.required:
                    required_failures.append(cov)
                continue

            if cov.status != "ready":
                effective_start_ns = cov.required_start_ns
                effective_end_ns = cov.required_end_ns
                bar_duration = _bar_duration_ns(req.timeframe)
                target = SyncTarget(
                    symbol=symbol,
                    timeframe=req.timeframe,
                    start=pd.Timestamp(effective_start_ns, unit="ns", tz="UTC").date(),
                    end=pd.Timestamp(effective_end_ns + bar_duration, unit="ns", tz="UTC").date(),
                    required=req.required,
                    reason=cov.status,
                )
                targets.append(target)
                if req.required:
                    required_failures.append(cov)

    return DataSyncPlan(
        requirements=tuple(requirements),
        coverage=tuple(coverages),
        targets=tuple(targets),
        required_failures=tuple(required_failures),
    )


def build_ltf_1m_plan(
    *,
    admitted_symbols: Sequence[str],
    universe_priority: Mapping[str, float],
    requirement: DataRequirement,
    lifecycle_by_symbol: Mapping[str, object],
    cache_root: Path,
    max_gap_bars: int,
    min_coverage_ratio: float,
    max_symbols: int,
    manifest: Mapping[str, object],
) -> LtfSyncSelection:
    """Build LTF 1m plan from admitted symbols and deterministic priority."""
    coverages: list[CacheCoverage] = []
    targets: list[SyncTarget] = []
    covered: set[str] = set()
    uncovered: set[str] = set()
    priority_map: dict[str, float] = dict(universe_priority)

    for sym in admitted_symbols:
        onboard = _resolve_lifecycle_date(lifecycle_by_symbol.get(sym), "onboard_date")
        delivery = _resolve_lifecycle_date(lifecycle_by_symbol.get(sym), "delivery_date")
        cov = inspect_cache_coverage(
            symbol=sym,
            requirement=requirement,
            cache_path=cache_root,
            onboard_date=onboard,
            delivery_date=delivery,
            max_gap_bars=max_gap_bars,
            manifest=manifest,
        )
        coverages.append(cov)

        is_ready = cov.status == "ready" and cov.coverage_ratio >= min_coverage_ratio
        if is_ready:
            covered.add(sym)
        else:
            uncovered.add(sym)
            if cov.status != "unavailable_lifecycle":
                target = SyncTarget(
                    symbol=sym,
                    timeframe=requirement.timeframe,
                    start=requirement.start,
                    end=requirement.end,
                    required=False,
                    reason=cov.status,
                )
                targets.append(target)

    covered_sorted = sorted(
        covered,
        key=lambda s: (-priority_map.get(s, 0.0), s),
    )
    uncovered_sorted = sorted(
        uncovered,
        key=lambda s: (-priority_map.get(s, 0.0), s),
    )

    selected = list(covered_sorted)
    budget_remaining = max_symbols - len(selected)
    if budget_remaining > 0:
        selected.extend(uncovered_sorted[:budget_remaining])

    return LtfSyncSelection(
        candidate_symbols=tuple(admitted_symbols),
        selected_symbols=tuple(selected),
        covered_symbols=frozenset(covered),
        targets=tuple(targets),
        coverage=tuple(coverages),
        selector_version="v1",
    )


def build_derived_cache_plan(
    *,
    symbols: Sequence[str],
    requirements: Sequence[DataRequirement],
    enabled_timeframes: Sequence[str],
    cache_root: Path,
    source_manifest: Mapping[str, object],
) -> DerivedCachePlan:
    """Build derived cache plan from source fingerprint comparison."""
    derived_targets: list[DerivedCacheTarget] = []

    for symbol in symbols:
        for tf in enabled_timeframes:
            source_key = f"{symbol}::1h"
            source_entry = source_manifest.get(source_key)
            source_fp = ""
            if isinstance(source_entry, dict):
                source_fp = source_entry.get("fingerprint", "")

            derived_key = f"{symbol}::{tf}"
            derived_path = cache_root / "derived_ohlcv" / tf / f"{symbol}.parquet"
            derived_ready = derived_path.exists()

            if derived_ready and source_fp:
                derived_entry = source_manifest.get(derived_key)
                if isinstance(derived_entry, dict) and derived_entry.get("source_fingerprint") == source_fp:
                    derived_targets.append(
                        DerivedCacheTarget(
                            symbol=symbol,
                            timeframe=tf,
                            start=date(1970, 1, 1),
                            end=date(1970, 1, 1),
                            source_fingerprint=source_fp,
                            status="ready",
                        )
                    )
                    continue

            derived_targets.append(
                DerivedCacheTarget(
                    symbol=symbol,
                    timeframe=tf,
                    start=date(1970, 1, 1),
                    end=date(1970, 1, 1),
                    source_fingerprint=source_fp,
                    status="materialize",
                )
            )

    return DerivedCachePlan(
        targets=tuple(derived_targets),
        retained_timeframes=tuple(enabled_timeframes),
    )


def materialize_derived_cache(
    *,
    plan: DerivedCachePlan,
    cache_root: Path,
    max_workers: int = 1,
) -> DerivedTimeframeStore:
    raise NotImplementedError


def run_retention_compaction(
    *,
    policy: RetentionPolicy,
    requirements: Sequence[DataRequirement],
    ltf_selection: LtfSyncSelection | None,
    cache_root: Path,
) -> tuple[CacheCoverage, ...]:
    raise NotImplementedError
