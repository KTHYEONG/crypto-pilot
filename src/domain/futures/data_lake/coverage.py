from __future__ import annotations

import logging
from collections.abc import Sequence

from src.domain.futures.compound.contracts import (
    MultiscaleAlphaDefinition,
    StrategyDataCoverage,
    StrategyDataCoverageEntry,
)
from src.domain.futures.data_lake.contracts import DataSnapshot, IngestionPlan

_logger = logging.getLogger(__name__)


class DataCoverageError(RuntimeError):
    ...


def validate_strategy_data_coverage(
    *,
    snapshot: DataSnapshot,
    plan: IngestionPlan,
    catalog: Sequence[MultiscaleAlphaDefinition],
) -> StrategyDataCoverage:
    entries: list[StrategyDataCoverageEntry] = []
    all_ready = True

    for recipe in catalog:
        required = set(recipe.required_fields)
        covered = required.copy()
        missing: list[str] = []

        for req in required:
            found = False
            for manifest in snapshot.partitions:
                if req in str(manifest.dataset):
                    found = True
                    break
            if not found:
                covered.discard(req)
                missing.append(req)

        if missing:
            all_ready = False
            entries.append(StrategyDataCoverageEntry(
                dataset=",".join(missing),
                recipe_id=recipe.recipe_id,
                ratio=0.0,
                max_gap_bars=9999,
                readiness="shadow",
                reason=f"missing_fields:{','.join(missing)}",
            ))
        else:
            entries.append(StrategyDataCoverageEntry(
                dataset=",".join(recipe.required_fields),
                recipe_id=recipe.recipe_id,
                ratio=1.0,
                max_gap_bars=0,
                readiness="ready",
                reason="",
            ))

    for ds in plan.datasets:
        found = any(ds.value in str(m.dataset) for m in snapshot.partitions)
        if not found:
            _logger.warning("dataset %s not found in snapshot partitions", ds.value)

    entry_tuple = tuple(entries)
    _logger.info(
        "strategy data coverage: %d/%d ready",
        sum(1 for e in entry_tuple if e.readiness == "ready"),
        len(entry_tuple),
    )

    return StrategyDataCoverage(
        entries=entry_tuple,
        all_ready=all_ready,
        data_manifest_hash=snapshot.manifest_hash,
    )
