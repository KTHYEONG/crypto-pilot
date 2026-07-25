from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.domain.futures.data_lake.contracts import (
    BinanceDataClient,
    DataCatalog,
    DataSnapshot,
    PreparedBootstrap,
    PreparedQuarterlyData,
    SyncMode,
)
from src.domain.futures.data_lake.coverage_policy import (
    DataCoverageError,
    DataCriticality,
    DatasetRequirement,
    evaluate_layered_coverage,
    resolve_recipe_availability,
)
from src.domain.futures.data_lake.ingestion import (
    build_ingestion_plan,
    sync_futures_data_lake,
)
from src.domain.futures.data_lake.reconciliation import (
    reconcile_local_catalog,
)
from src.domain.futures.data_lake.run_windows import (
    QuarterlyRunWindow,
)


@dataclass(slots=True, frozen=True)
class DataLakeRuntime:
    client: BinanceDataClient
    catalog: DataCatalog

_logger = logging.getLogger(__name__)


def build_data_lake_runtime(config: CompoundRunConfig) -> DataLakeRuntime:
    _logger.info("building data lake runtime from config")
    from src.domain.futures.data_lake.query import BinanceQueryClient, LocalDataCatalog

    read_only = config.sync == SyncMode.LOCAL
    client: BinanceDataClient = BinanceQueryClient()
    catalog: DataCatalog = LocalDataCatalog(
        root=config.data_lake.root, read_only=read_only,
    )
    return DataLakeRuntime(client=client, catalog=catalog)


def prepare_data_snapshot(
    *, config: CompoundRunConfig, runtime: DataLakeRuntime
) -> DataSnapshot:
    _logger.info("preparing data snapshot for reference_date=%s", config.reference_date)
    reference_time_ms: int = 0
    if config.reference_date:
        ref_dt = date.fromisoformat(config.reference_date)
        reference_time_ms = int(ref_dt.strftime("%s")) * 1000
    else:
        reference_time_ms = int(datetime.now(UTC).timestamp() * 1000)

    snapshot = runtime.catalog.load_snapshot(reference_time_ms)

    ingestion_plan = build_ingestion_plan(
        config=config.data_lake,
        reference_date=date.fromtimestamp(reference_time_ms // 1000),
    )

    if runtime.catalog.has_complete_coverage(snapshot=snapshot, plan=ingestion_plan):
        _logger.info("local snapshot complete: %s", snapshot.snapshot_id)
        return snapshot

    if config.sync == SyncMode.LOCAL:
        raise DataCoverageError(
            f"incomplete local snapshot at {reference_time_ms}; "
            "CORE gap unresolved in local mode"
        )

    _logger.info("local snapshot incomplete, syncing from network")
    synced_snapshot = sync_futures_data_lake(
        plan=ingestion_plan,
        client=runtime.client,
        catalog=runtime.catalog,
    )

    if not runtime.catalog.has_complete_coverage(snapshot=synced_snapshot, plan=ingestion_plan):
        raise DataCoverageError(
            f"snapshot still incomplete after sync: {synced_snapshot.snapshot_id}"
        )

    _logger.info("data snapshot ready: %s", synced_snapshot.snapshot_id)
    return synced_snapshot


def prepare_quarterly_bootstrap(
    *,
    config: CompoundRunConfig,
    runtime: DataLakeRuntime,
    window: QuarterlyRunWindow,
) -> PreparedBootstrap:
    _logger.info(
        "quarterly bootstrap: cutoff=%s sync=%s",
        window.cutoff_date, config.sync.value,
    )

    if config.sync == SyncMode.AUTO:
        reconcile_report = reconcile_local_catalog(
            root=config.data_lake.root,
            cutoff_exclusive_ns=window.cutoff_exclusive_ns,
        )
        _logger.info(
            "reconciliation: scanned=%d added=%d quarantined=%d",
            reconcile_report.scanned_files,
            reconcile_report.added_rows,
            len(reconcile_report.quarantined_files),
        )
    else:
        reconcile_report = None

    snapshot = prepare_data_snapshot(config=config, runtime=runtime)

    _logger.info("bootstrap snapshot ready: %s", snapshot.snapshot_id)
    return PreparedBootstrap(
        window=window,
        snapshot=snapshot,
        reconciliation_report=reconcile_report,
    )


def finalize_quarterly_signal_data(
    *,
    config: CompoundRunConfig,
    runtime: DataLakeRuntime,
    bootstrap: PreparedBootstrap,
    universe: object,
    catalog: tuple[object, ...],
) -> PreparedQuarterlyData:
    _logger.info("finalizing signal data for quarterly window")
    snapshot = bootstrap.snapshot

    from src.domain.futures.compound.contracts import MultiscaleAlphaDefinition
    from src.domain.futures.data_lake.contracts import DatasetKind

    window: QuarterlyRunWindow = bootstrap.window  # type: ignore[assignment]

    requirements: list[DatasetRequirement] = [
        DatasetRequirement(
            dataset=DatasetKind.KLINES_1H,
            fields=("open", "high", "low", "close", "quote_volume", "taker_buy_quote"),
            criticality=DataCriticality.CORE,
            start_time_ns=window.acquisition_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            min_coverage_ratio=0.98,
            max_gap_ns=21_600_000_000_000,
            recipe_ids=tuple(r.recipe_id for r in catalog if isinstance(r, MultiscaleAlphaDefinition)),
        ),
        DatasetRequirement(
            dataset=DatasetKind.FUNDING_EVENT,
            fields=("funding_rate",),
            criticality=DataCriticality.CORE,
            start_time_ns=window.l1_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            min_coverage_ratio=0.98,
            max_gap_ns=86_400_000_000_000,
            recipe_ids=(),
        ),
        DatasetRequirement(
            dataset=DatasetKind.PREMIUM_5M,
            fields=("close",),
            criticality=DataCriticality.OPTIONAL,
            start_time_ns=window.acquisition_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            min_coverage_ratio=0.5,
            max_gap_ns=7 * 86_400_000_000_000,
            recipe_ids=("carry_funding_event_h8", "flow_imbalance_15m_h1"),
        ),
        DatasetRequirement(
            dataset=DatasetKind.MARK_1M,
            fields=("close",),
            criticality=DataCriticality.OPTIONAL,
            start_time_ns=window.acquisition_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            min_coverage_ratio=0.5,
            max_gap_ns=7 * 86_400_000_000_000,
            recipe_ids=("basis_reversion_1h_h8",),
        ),
        DatasetRequirement(
            dataset=DatasetKind.INDEX_1M,
            fields=("close",),
            criticality=DataCriticality.OPTIONAL,
            start_time_ns=window.acquisition_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            min_coverage_ratio=0.5,
            max_gap_ns=7 * 86_400_000_000_000,
            recipe_ids=("basis_reversion_1h_h8",),
        ),
        DatasetRequirement(
            dataset=DatasetKind.METRICS_5M,
            fields=("sum_open_interest_value",),
            criticality=DataCriticality.OPTIONAL,
            start_time_ns=window.acquisition_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            min_coverage_ratio=0.5,
            max_gap_ns=7 * 86_400_000_000_000,
            recipe_ids=("flow_oi_confirm_1h_h4",),
        ),
        DatasetRequirement(
            dataset=DatasetKind.KLINES_1M,
            fields=("close",),
            criticality=DataCriticality.OPTIONAL,
            start_time_ns=window.acquisition_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            min_coverage_ratio=0.3,
            max_gap_ns=14 * 86_400_000_000_000,
            recipe_ids=("flow_imbalance_15m_h1", "liquidity_exhaustion_15m_h1"),
        ),
    ]

    from src.domain.futures.universe.contracts import UniverseStateCube

    if isinstance(universe, UniverseStateCube):
        universe_cube = universe
    else:
        candidate = getattr(universe, "state_cube", None)
        if not isinstance(candidate, UniverseStateCube):
            raise TypeError("universe must be UniverseStateCube or expose a valid state_cube")
        universe_cube = candidate
    coverage = evaluate_layered_coverage(
        snapshot=snapshot,
        universe=universe_cube,
        requirements=tuple(requirements),
    )

    recipe_plan = resolve_recipe_availability(
        catalog=tuple(c for c in catalog if isinstance(c, MultiscaleAlphaDefinition)),
        coverage=coverage,
    )

    enabled_fields: set[str] = set()
    mandatory_fields = {"open", "high", "low", "close", "quote_volume", "taker_buy_quote",
                        "funding", "execution_cost_bps", "capacity_usdt"}
    enabled_fields.update(mandatory_fields)

    for rp in recipe_plan:
        if rp.status.value == "enabled":
            recipe = next((c for c in catalog if isinstance(c, MultiscaleAlphaDefinition) and c.recipe_id == rp.recipe_id), None)
            if recipe:
                enabled_fields.update(recipe.required_fields)

    downloaded = 0

    data_report: dict[str, object] = {
        "window": {
            "requested_date": str(window.requested_date),
            "cutoff_date": str(window.cutoff_date),
        },
        "coverage": [
            {
                "dataset": c.requirement.dataset.value,
                "criticality": c.requirement.criticality.value,
                "coverage_ratio": c.coverage_ratio,
                "passed": c.passed,
                "reasons": list(c.reasons),
            }
            for c in coverage
        ],
        "recipe_plan": [
            {"recipe_id": r.recipe_id, "status": r.status.value, "reasons": list(r.reasons)}
            for r in recipe_plan
        ],
        "downloaded_partitions": downloaded,
    }

    report_path = Path("logs/futures/compound/data_sync_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data_report, indent=2))

    return PreparedQuarterlyData(
        window=window,
        snapshot=snapshot,
        field_plan=tuple(sorted(enabled_fields)),
        recipe_plan=recipe_plan,
        reconciliation=bootstrap.reconciliation_report,
        coverage=coverage,
        downloaded_partitions=downloaded,
    )
