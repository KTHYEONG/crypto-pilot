from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.domain.futures.data_lake.contracts import (
    BinanceDataClient,
    DataCatalog,
    DataSnapshot,
)
from src.domain.futures.data_lake.ingestion import (
    DataCoverageError,
    build_ingestion_plan,
    sync_futures_data_lake,
)


@dataclass(slots=True, frozen=True)
class DataLakeRuntime:
    client: BinanceDataClient
    catalog: DataCatalog

_logger = logging.getLogger(__name__)


def build_data_lake_runtime(config: CompoundRunConfig) -> DataLakeRuntime:
    _logger.info("building data lake runtime from config")
    from src.domain.futures.data_lake.query import BinanceQueryClient, LocalDataCatalog

    client: BinanceDataClient = BinanceQueryClient()
    catalog: DataCatalog = LocalDataCatalog(root=config.data_lake.root)
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
        from datetime import UTC, datetime

        reference_time_ms = int(datetime.now(UTC).timestamp() * 1000)

    snapshot = runtime.catalog.load_snapshot(reference_time_ms)

    ingestion_plan = build_ingestion_plan(
        config=config.data_lake,
        reference_date=date.fromtimestamp(reference_time_ms // 1000),
    )

    if runtime.catalog.has_complete_coverage(snapshot=snapshot, plan=ingestion_plan):
        _logger.info("local snapshot complete: %s", snapshot.snapshot_id)
        return snapshot

    if not config.allow_network_sync:
        raise DataCoverageError(
            f"incomplete local snapshot at {reference_time_ms}; "
            "use --allow-network-sync to enable download"
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
