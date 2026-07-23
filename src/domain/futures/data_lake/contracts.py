from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.universe.contracts import UniverseStateCube


class DatasetKind(StrEnum):
    EXCHANGE_INFO = "exchange_info"
    KLINES_1H = "klines_1h"
    KLINES_1M = "klines_1m"
    FUNDING_EVENT = "funding_event"
    PREMIUM_5M = "premium_5m"
    MARK_1M = "mark_1m"
    INDEX_1M = "index_1m"
    METRICS_5M = "metrics_5m"
    COST_CALIBRATION = "cost_calibration"
    UNIVERSE_STATE = "universe_state"


@dataclass(slots=True, frozen=True)
class DataLakeConfig:
    root: Path
    soft_cap_gib: int = 48
    hard_cap_gib: int = 64
    max_workers: int = 4
    zstd_level: int = 6
    market: str = "um"
    quote_asset: str = "USDT"

    def __post_init__(self) -> None:
        if not (0 < self.soft_cap_gib < self.hard_cap_gib <= 64):
            msg = f"soft {self.soft_cap_gib} < hard {self.hard_cap_gib} <= 64 required"
            raise ValueError(msg)
        if self.max_workers < 1 or self.max_workers > 4:
            msg = f"max_workers must be 1-4, got {self.max_workers}"
            raise ValueError(msg)
        if self.market != "um":
            msg = f"market must be 'um', got {self.market!r}"
            raise ValueError(msg)
        if self.quote_asset != "USDT":
            msg = f"quote_asset must be 'USDT', got {self.quote_asset!r}"
            raise ValueError(msg)
        if self.zstd_level < 1 or self.zstd_level > 22:
            msg = f"zstd_level must be 1-22, got {self.zstd_level}"
            raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class PartitionManifest:
    dataset: DatasetKind
    symbol: str
    start_time_ms: int
    end_time_ms: int
    row_count: int
    sha256: str
    source: str
    is_final: bool
    path: Path


@dataclass(slots=True, frozen=True)
class UniverseStateRow:
    effective_time_ns: int
    knowledge_time_ns: int
    symbol: str
    eligible: bool
    entry_block: bool
    exit_required: bool
    capacity_usdt: float
    risk_scale: float
    execution_cost_bps: float
    state_reason: str
    universe_config_hash: str
    source_manifest_hash: str


@dataclass(slots=True, frozen=True)
class UniverseStateRequest:
    execution_timestamps_ns: NDArray[np.int64]
    max_axis_symbols: int

    def __post_init__(self) -> None:
        timestamps = self.execution_timestamps_ns
        if timestamps.size == 0:
            raise ValueError("execution_timestamps_ns must not be empty")
        if self.max_axis_symbols < 1:
            raise ValueError("max_axis_symbols must be >= 1")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("execution_timestamps_ns must be strictly increasing")


@dataclass(slots=True, frozen=True)
class LakeUniverse:
    symbols: tuple[str, ...]
    state_cube: UniverseStateCube
    state_hash: str


@dataclass(slots=True, frozen=True)
class DataSnapshot:
    snapshot_id: str
    reference_time_ms: int
    partitions: tuple[PartitionManifest, ...]
    manifest_hash: str
    universe_state_hash: str
    total_bytes: int


@dataclass(slots=True, frozen=True)
class IngestionPlan:
    reference_date: date
    broad_symbols: tuple[str, ...]
    selected_symbols: tuple[str, ...]
    datasets: tuple[DatasetKind, ...]
    config: DataLakeConfig
    start_date: date | None = None


@dataclass(slots=True, frozen=True)
class GridRequest:
    symbols: tuple[str, ...]
    timeframe: str
    source_timeframe: str
    fields: tuple[str, ...]
    start_time_ns: int
    end_time_ns: int


@dataclass(slots=True, frozen=True)
class NativeFeatureGrid:
    timestamps_ns: NDArray[np.int64]
    symbols: tuple[str, ...]
    fields: dict[str, NDArray[np.float64] | NDArray[np.float32]]
    available: dict[str, NDArray[np.bool_]]
    data_manifest_hash: str


class BinanceDataClient(Protocol):
    def download_partition(self, *args: Any, **kwargs: Any) -> bytes: ...
    def download_checksum(self, *args: Any, **kwargs: Any) -> str: ...
    def fetch_exchange_info(self) -> dict[str, Any]: ...


class DataCatalog(Protocol):
    def commit_partition(self, manifest: PartitionManifest) -> None: ...
    def partition_exists(self, dataset: DatasetKind, symbol: str, start_time_ms: int) -> bool: ...
    def total_bytes(self) -> int: ...
    def load_snapshot(self, reference_time_ms: int) -> DataSnapshot: ...
    def has_complete_coverage(self, snapshot: DataSnapshot, plan: IngestionPlan) -> bool: ...
    def compute_universe_state_hash(self, snapshot: DataSnapshot) -> str: ...
