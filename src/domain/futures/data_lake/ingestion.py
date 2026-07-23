from __future__ import annotations

import hashlib
import logging
from datetime import date

from src.domain.futures.data_lake.contracts import (
    BinanceDataClient,
    DataCatalog,
    DataLakeConfig,
    DataSnapshot,
    IngestionPlan,
    PartitionManifest,
)

_logger = logging.getLogger(__name__)


class ChecksumMismatchError(RuntimeError):
    ...


class StorageBudgetError(RuntimeError):
    ...


class DataCoverageError(RuntimeError):
    ...


def build_ingestion_plan(
    *, config: DataLakeConfig, reference_date: date
) -> IngestionPlan:
    if reference_date > date.today():
        msg = f"reference_date {reference_date} cannot be in the future"
        raise ValueError(msg)

    _logger.info("building ingestion plan for %s with market=%s", reference_date, config.market)

    broad_symbols: tuple[str, ...] = ()

    return IngestionPlan(
        reference_date=reference_date,
        broad_symbols=broad_symbols,
        selected_symbols=(),
        datasets=(),
        config=config,
    )


def sync_futures_data_lake(
    *, plan: IngestionPlan, client: BinanceDataClient, catalog: DataCatalog
) -> DataSnapshot:
    if plan.config.hard_cap_gib <= 0:
        msg = f"invalid hard_cap_gib: {plan.config.hard_cap_gib}"
        raise ValueError(msg)

    current_bytes = catalog.total_bytes()
    projected_gib = current_bytes / (1024**3)

    if projected_gib >= plan.config.hard_cap_gib:
        raise StorageBudgetError(
            f"current {projected_gib:.1f} GiB >= hard cap {plan.config.hard_cap_gib} GiB"
        )

    partitions: list[PartitionManifest] = []

    for dataset_kind in plan.datasets:
        for sym in plan.broad_symbols if dataset_kind.value.startswith("klines_1h") else plan.selected_symbols:
            payload = client.download_partition(dataset_kind, sym)
            expected_checksum = client.download_checksum(dataset_kind, sym)
            actual_checksum = hashlib.sha256(payload).hexdigest()

            if actual_checksum != expected_checksum:
                raise ChecksumMismatchError(
                    f"checksum mismatch for {dataset_kind}/{sym}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )

    _logger.info("sync complete: %d partitions, %d bytes", len(partitions), 0)

    return DataSnapshot(
        snapshot_id=f"snapshot-{plan.reference_date.isoformat()}",
        reference_time_ms=int(plan.reference_date.strftime("%s")) * 1000,
        partitions=tuple(partitions),
        manifest_hash="",
        total_bytes=0,
    )



