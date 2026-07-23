from __future__ import annotations

import hashlib
import io
from datetime import date
from pathlib import Path

import pytest
import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.data_lake_runtime import (
    DataLakeRuntime,
    build_data_lake_runtime,
    prepare_data_snapshot,
)
from src.domain.futures.data_lake.contracts import (
    DataLakeConfig,
    DataSnapshot,
    IngestionPlan,
    PartitionManifest,
)
from src.domain.futures.data_lake.ingestion import (
    ChecksumMismatchError,
    DataCoverageError,
    StorageBudgetError,
    sync_futures_data_lake,
)
from src.domain.futures.universe.config import PITUniverseConfig


def _snap(
    snapshot_id: str = "s1",
    reference_time_ms: int = 1_000_000,
    partitions: tuple = (),
    manifest_hash: str = "h1",
    total_bytes: int = 0,
) -> DataSnapshot:
    return DataSnapshot(
        snapshot_id=snapshot_id,
        reference_time_ms=reference_time_ms,
        partitions=partitions,
        manifest_hash=manifest_hash,
        universe_state_hash="",
        total_bytes=total_bytes,
    )


class FakeClient:
    def __init__(self) -> None:
        self.download_calls = 0

    def download_partition(self, *args: object, **kwargs: object) -> bytes:
        self.download_calls += 1
        buffer = io.BytesIO()
        pd.DataFrame({"timestamp": [1_783_440_000_000], "close": [100.0]}).to_parquet(buffer, index=False)
        return buffer.getvalue()

    def download_checksum(self, *args: object, **kwargs: object) -> str:
        return hashlib.sha256(self.download_partition()).hexdigest()


class FakeCatalog:
    def __init__(self, snapshot: DataSnapshot, *, complete: bool) -> None:
        self.snapshot = snapshot
        self.complete = complete
        self.committed: list[PartitionManifest] = []
        self._bytes = 0

    def load_snapshot(self, reference_time_ms: int) -> DataSnapshot:
        return self.snapshot

    def has_complete_coverage(self, snapshot: DataSnapshot, plan: IngestionPlan) -> bool:
        return self.complete

    def commit_partition(self, manifest: PartitionManifest) -> None:
        self.committed.append(manifest)

    def partition_exists(self, dataset: object, symbol: str, start_time_ms: int) -> bool:
        return False

    def total_bytes(self) -> int:
        return self._bytes


class TestConfigDefaults:
    def test_config_disables_network_sync_by_default(self) -> None:
        from src.application.futures.runner.compound_config import (
            build_compound_run_config,
        )

        config = build_compound_run_config({"sync": "skip", "seed": 42})
        assert config.allow_network_sync is False
        assert isinstance(config.data_lake, DataLakeConfig)
        assert isinstance(config.universe, PITUniverseConfig)


class TestRuntimeFactory:
    def test_runtime_factory_does_not_download(self) -> None:
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
        )
        runtime = build_data_lake_runtime(config)
        assert isinstance(runtime, DataLakeRuntime)
        assert runtime.client.download_calls == 0


class TestLocalSnapshot:
    def test_complete_local_snapshot_avoids_network(self) -> None:
        snap = _snap(snapshot_id="test-complete")
        runtime = DataLakeRuntime(
            client=FakeClient(),
            catalog=FakeCatalog(snap, complete=True),
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
            allow_network_sync=False,
        )
        result = prepare_data_snapshot(config=config, runtime=runtime)
        assert result.snapshot_id == "test-complete"


class TestCoverageFailure:
    def test_incomplete_cache_without_approval_fails_closed(self) -> None:
        snap = _snap(snapshot_id="test-incomplete")
        runtime = DataLakeRuntime(
            client=FakeClient(),
            catalog=FakeCatalog(snap, complete=False),
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
            allow_network_sync=False,
        )
        with pytest.raises(DataCoverageError):
            prepare_data_snapshot(config=config, runtime=runtime)


class TestApprovedSync:
    def test_approved_sync_revalidates_snapshot(self, tmp_path: Path) -> None:
        snap = _snap(snapshot_id="test-incomplete")
        client = FakeClient()
        catalog = FakeCatalog(snap, complete=False)
        runtime = DataLakeRuntime(client=client, catalog=catalog)
        config = CompoundRunConfig(
            reference_date="2026-07-08", sync="skip", refresh_universe=False,
            allow_network_sync=True,
            data_lake=DataLakeConfig(root=tmp_path / "lake"),
        )
        with pytest.raises(DataCoverageError, match="still incomplete after sync"):
            prepare_data_snapshot(config=config, runtime=runtime)


class TestChecksumFailure:
    def test_checksum_failure_is_not_committed(self) -> None:
        from src.domain.futures.data_lake.contracts import DatasetKind

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=Path("/tmp/lake")),
        )
        cat = FakeCatalog(
            _snap(snapshot_id="s1", reference_time_ms=1),
            complete=False,
        )

        class BadClient(FakeClient):
            def download_partition(self, *args: object, **kwargs: object) -> bytes:
                self.download_calls += 1
                return b"tampered"

            def download_checksum(self, *args: object, **kwargs: object) -> str:
                return hashlib.sha256(b"valid").hexdigest()

        with pytest.raises(ChecksumMismatchError):
            sync_futures_data_lake(plan=plan, client=BadClient(), catalog=cat)
        assert len(cat.committed) == 0


class TestHardCap:
    def test_hard_cap_preserves_canonical_partitions(self) -> None:
        from src.domain.futures.data_lake.contracts import DatasetKind

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=Path("/tmp/lake"), hard_cap_gib=64),
        )

        class FullCatalog(FakeCatalog):
            def total_bytes(self) -> int:
                return 100 * 1024**3

        cat = FullCatalog(
            _snap(snapshot_id="s1", reference_time_ms=1),
            complete=False,
        )
        with pytest.raises(StorageBudgetError):
            sync_futures_data_lake(plan=plan, client=FakeClient(), catalog=cat)
