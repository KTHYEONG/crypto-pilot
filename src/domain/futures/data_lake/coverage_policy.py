from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from src.domain.futures.compound.contracts import MultiscaleAlphaDefinition
from src.domain.futures.data_lake.contracts import DatasetKind, DataSnapshot
from src.domain.futures.data_lake.ingestion import DataCoverageError
from src.domain.futures.universe.contracts import UniverseStateCube

_logger = logging.getLogger(__name__)


class DataCriticality(StrEnum):
    CORE = "core"
    OPTIONAL = "optional"


class RecipeDataStatus(StrEnum):
    ENABLED = "enabled"
    DISABLED_DATA = "disabled_data"


@dataclass(slots=True, frozen=True)
class DatasetRequirement:
    dataset: DatasetKind
    fields: tuple[str, ...]
    criticality: DataCriticality
    start_time_ns: int
    end_time_ns: int
    min_coverage_ratio: float
    max_gap_ns: int
    recipe_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("fields must not be empty")
        if self.end_time_ns <= self.start_time_ns:
            raise ValueError(f"end_time_ns {self.end_time_ns} must be > start_time_ns {self.start_time_ns}")
        if not 0.0 < self.min_coverage_ratio <= 1.0:
            raise ValueError(f"min_coverage_ratio must be in (0, 1], got {self.min_coverage_ratio}")
        if self.max_gap_ns < 0:
            raise ValueError(f"max_gap_ns must be >= 0, got {self.max_gap_ns}")


@dataclass(slots=True, frozen=True)
class DatasetCoverage:
    requirement: DatasetRequirement
    expected_observations: int
    observed_observations: int
    coverage_ratio: float
    max_gap_ns: int
    passed: bool
    reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class RecipeAvailability:
    recipe_id: str
    status: RecipeDataStatus
    reasons: tuple[str, ...]


def evaluate_layered_coverage(
    *,
    snapshot: DataSnapshot,
    universe: UniverseStateCube,
    requirements: tuple[DatasetRequirement, ...],
) -> tuple[DatasetCoverage, ...]:
    results: list[DatasetCoverage] = []
    core_failures: list[str] = []

    ns_per_hour = 3_600_000_000_000
    ns_per_day = 86_400_000_000_000

    known_symbols = set(universe.instrument_ids) if hasattr(universe, "instrument_ids") else set()
    eligible_2d = universe.eligible if hasattr(universe, "eligible") else None

    for req in requirements:
        dataset_paths = [
            p for p in snapshot.partitions
            if p.dataset == req.dataset
        ]

        expected_total = 0
        observed_total = 0
        max_gap = 0
        reasons: list[str] = []

        if eligible_2d is not None and known_symbols:
            sym_list = sorted(known_symbols)
            for sym_idx, symbol in enumerate(sym_list):
                sym_paths = [p for p in dataset_paths if p.symbol == symbol or p.symbol == "__all__"]
                if not sym_paths:
                    expected_total += 1
                    continue

                if eligible_2d.ndim == 2:
                    import numpy as np
                    timestamps = np.arange(req.start_time_ns, req.end_time_ns, ns_per_hour, dtype=np.int64)
                    usable_rows = min(len(timestamps), eligible_2d.shape[0])
                    causal_mask = (
                        (timestamps[:usable_rows] >= req.start_time_ns - ns_per_day)
                        & (timestamps[:usable_rows] < req.end_time_ns + ns_per_day)
                    )
                    causal_hours = max(
                        1,
                        int(np.sum(eligible_2d[:usable_rows, sym_idx][causal_mask])),
                    )
                    expected_total += causal_hours

                for p in sym_paths:
                    if p.row_count is not None:
                        observed_total += int(p.row_count)

                if sym_paths and req.max_gap_ns > 0:
                    for p in sorted(sym_paths, key=lambda x: x.start_time_ms if x.start_time_ms else 0):
                        gap_start = int(p.end_time_ms or 0) * 1_000_000
                        gap_end = int(p.start_time_ms or 0) * 1_000_000
                        gap = gap_start - gap_end if gap_start > gap_end else 0
                        if gap > max_gap:
                            max_gap = gap
        else:
            expected_total = len(dataset_paths) or 1
            observed_total = sum(int(p.row_count or 0) for p in dataset_paths)
            for p in sorted(dataset_paths, key=lambda x: x.start_time_ms if x.start_time_ms else 0):
                if hasattr(p, "end_time_ms") and hasattr(p, "start_time_ms"):
                    gap = (int(p.end_time_ms or 0) - int(p.start_time_ms or 0)) * 1_000_000
                    if gap < 0:
                        gap = 0
                    if gap > max_gap:
                        max_gap = gap

        coverage_ratio = observed_total / max(expected_total, 1)
        if coverage_ratio > 1.0:
            coverage_ratio = 1.0

        coverage_gap_ok = max_gap <= req.max_gap_ns
        coverage_ratio_ok = coverage_ratio >= req.min_coverage_ratio

        passed = coverage_ratio_ok and coverage_gap_ok

        if not coverage_ratio_ok:
            reasons.append(f"coverage_ratio {coverage_ratio:.4f} < min {req.min_coverage_ratio}")
        if not coverage_gap_ok:
            reasons.append(f"max_gap {max_gap} ns > allowed {req.max_gap_ns} ns")

        results.append(DatasetCoverage(
            requirement=req,
            expected_observations=expected_total,
            observed_observations=observed_total,
            coverage_ratio=coverage_ratio,
            max_gap_ns=max_gap,
            passed=passed,
            reasons=tuple(reasons),
        ))

        if not passed and req.criticality == DataCriticality.CORE:
            core_failures.append(
                f"CORE dataset {req.dataset.value} fields={req.fields} "
                f"coverage={coverage_ratio:.4f} gap={max_gap}ns"
            )

    result = tuple(results)

    if core_failures:
        msg = "CORE data coverage failures:\n" + "\n".join(core_failures)
        raise DataCoverageError(msg)

    return result


def resolve_recipe_availability(
    catalog: tuple[MultiscaleAlphaDefinition, ...],
    coverage: tuple[DatasetCoverage, ...],
) -> tuple[RecipeAvailability, ...]:
    core_failures: list[str] = []
    failed_optional_recipe_ids: set[str] = set()

    for cov in coverage:
        if cov.passed:
            continue
        if cov.requirement.criticality == DataCriticality.CORE:
            core_failures.append(cov.requirement.dataset.value)
        else:
            for rid in cov.requirement.recipe_ids:
                failed_optional_recipe_ids.add(rid)

    if core_failures:
        msg = f"CORE field failure detected in datasets: {', '.join(core_failures)}"
        raise DataCoverageError(msg)

    results: list[RecipeAvailability] = []
    for recipe in catalog:
        if recipe.recipe_id in failed_optional_recipe_ids:
            results.append(RecipeAvailability(
                recipe_id=recipe.recipe_id,
                status=RecipeDataStatus.DISABLED_DATA,
                reasons=(f"optional field requirement failed for recipe {recipe.recipe_id}",),
            ))
        else:
            results.append(RecipeAvailability(
                recipe_id=recipe.recipe_id,
                status=RecipeDataStatus.ENABLED,
                reasons=(),
            ))

    return tuple(results)


__all__ = [
    "DataCoverageError",
    "DataCriticality",
    "DatasetCoverage",
    "DatasetRequirement",
    "RecipeAvailability",
    "RecipeDataStatus",
    "evaluate_layered_coverage",
    "resolve_recipe_availability",
]
