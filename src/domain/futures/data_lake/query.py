from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.domain.futures.data_lake.contracts import (
    DatasetKind,
    DataSnapshot,
    GridRequest,
    IngestionPlan,
    NativeFeatureGrid,
    PartitionManifest,
)

_logger = logging.getLogger(__name__)


class BinanceQueryClient:
    def __init__(self) -> None:
        self.download_calls = 0

    def download_partition(self, *args: Any, **kwargs: Any) -> bytes:
        self.download_calls += 1
        return b""

    def download_checksum(self, *args: Any, **kwargs: Any) -> str:
        return hashlib.sha256(b"").hexdigest()


class LocalDataCatalog:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._committed: list[PartitionManifest] = []

    def commit_partition(self, manifest: PartitionManifest) -> None:
        self._committed.append(manifest)

    def partition_exists(self, dataset: DatasetKind, symbol: str, start_time_ms: int) -> bool:
        return False

    def total_bytes(self) -> int:
        return 0

    def load_snapshot(self, reference_time_ms: int) -> DataSnapshot:
        return DataSnapshot(
            snapshot_id=f"local-{reference_time_ms}",
            reference_time_ms=reference_time_ms,
            partitions=(),
            manifest_hash="",
            total_bytes=0,
        )

    def has_complete_coverage(self, snapshot: DataSnapshot, plan: IngestionPlan) -> bool:
        return False


def materialize_native_grid(
    *, request: GridRequest, snapshot: DataSnapshot
) -> NativeFeatureGrid:
    if not request.symbols:
        msg = "grid request must specify at least one symbol"
        raise ValueError(msg)
    if not request.fields:
        msg = "grid request must specify at least one field"
        raise ValueError(msg)

    n_bars = 0
    for p in snapshot.partitions:
        if p.dataset.value.startswith("klines"):
            n_bars = max(n_bars, (p.end_time_ms - p.start_time_ms) // 3600000)

    n_bars = max(n_bars, 1)
    n_syms = len(request.symbols)

    fields: dict[str, np.ndarray] = {}
    available: dict[str, np.ndarray] = {}
    for f in request.fields:
        fields[f] = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
        available[f] = np.zeros((n_bars, n_syms), dtype=np.bool_)

    timestamps = np.arange(n_bars, dtype=np.int64) * 3600000000000 + request.start_time_ns

    _logger.info(
        "materialized native grid: %d bars x %d symbols, timeframe=%s, source=%s",
        n_bars, n_syms, request.timeframe, request.source_timeframe,
    )

    return NativeFeatureGrid(
        timestamps_ns=timestamps,
        symbols=request.symbols,
        fields=fields,
        available=available,
        data_manifest_hash=snapshot.manifest_hash,
    )
