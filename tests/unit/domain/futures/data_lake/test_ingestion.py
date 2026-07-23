from __future__ import annotations

import hashlib
import io
import threading
from datetime import date
from pathlib import Path

import pytest
import pandas as pd

from src.domain.futures.data_lake.contracts import (
    DataLakeConfig,
    DataSnapshot,
    DatasetKind,
    IngestionPlan,
    PartitionManifest,
)
from src.domain.futures.data_lake.ingestion import (
    ChecksumMismatchError,
    StorageBudgetError,
    build_ingestion_plan,
    sync_futures_data_lake,
)
from src.domain.futures.data_lake.query import LocalDataCatalog


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


class TestIngestionPlan:
    def test_future_date_raises(self) -> None:
        from datetime import date, timedelta

        tomorrow = date.today() + timedelta(days=1)
        with pytest.raises(ValueError, match="cannot be in the future"):
            build_ingestion_plan(
                config=DataLakeConfig(root=Path("/tmp")),  # noqa: S108
                reference_date=tomorrow,
            )

    def test_plan_excludes_non_binance_symbol_candidates(self, tmp_path: Path) -> None:
        source = tmp_path / "data" / "ohlcv" / "1h"
        source.mkdir(parents=True)
        frame = pd.DataFrame(
            {"timestamp": [1_783_440_000_000], "close": [100.0], "quote_volume": [1_000_000.0]}
        )
        frame.to_parquet(source / "BTCUSDT.parquet", index=False)
        frame.to_parquet(source / "币安人生USDT.parquet", index=False)

        plan = build_ingestion_plan(
            config=DataLakeConfig(root=tmp_path / "data" / "lake"),
            reference_date=date(2026, 7, 8),
        )

        assert plan.broad_symbols == ("BTCUSDT",)


class TestChecksumFailure:
    def test_checksum_failure_is_not_committed(self) -> None:
        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=Path("/tmp")),  # noqa: S108
        )

        class BadClient(FakeClient):
            def download_partition(self, *args: object, **kwargs: object) -> bytes:
                self.download_calls += 1
                return b"tampered"

            def download_checksum(self, *args: object, **kwargs: object) -> str:
                return hashlib.sha256(b"valid").hexdigest()

        cat = FakeCatalog(
            DataSnapshot(
                snapshot_id="s1",
                reference_time_ms=1,
                partitions=(),
                manifest_hash="",
                total_bytes=0,
            ),
            complete=False,
        )
        with pytest.raises(ChecksumMismatchError):
            sync_futures_data_lake(plan=plan, client=BadClient(), catalog=cat)
        assert len(cat.committed) == 0


class TestHardCap:
    def test_hard_cap_preserves_canonical_partitions(self) -> None:
        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=Path("/tmp"), hard_cap_gib=64),  # noqa: S108
        )

        class FullCatalog(FakeCatalog):
            def total_bytes(self) -> int:
                return 100 * 1024**3

        cat = FullCatalog(
            DataSnapshot(
                snapshot_id="s1",
                reference_time_ms=1,
                partitions=(),
                manifest_hash="",
                total_bytes=0,
            ),
            complete=False,
        )
        with pytest.raises(StorageBudgetError):
            sync_futures_data_lake(plan=plan, client=FakeClient(), catalog=cat)


class TestLocalCommit:
    def test_high_frequency_datasets_limit_history_to_recent_180_days(self, tmp_path: Path) -> None:
        payload_buffer = io.BytesIO()
        pd.DataFrame({"timestamp": [1_783_440_000_000], "close": [100.0]}).to_parquet(
            payload_buffer, index=False
        )
        payload = payload_buffer.getvalue()

        class RecordingClient:
            def __init__(self) -> None:
                self.start_times: list[int] = []

            def download_partition(
                self, dataset: DatasetKind, symbol: str, start_time_ms: int
            ) -> bytes:
                _ = (dataset, symbol)
                self.start_times.append(start_time_ms)
                return payload

            def download_checksum(self, *args: object, **kwargs: object) -> str:
                _ = (args, kwargs)
                return hashlib.sha256(payload).hexdigest()

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1M,),
            config=DataLakeConfig(root=tmp_path / "lake"),
            start_date=date(2024, 7, 8),
        )
        client = RecordingClient()

        sync_futures_data_lake(plan=plan, client=client, catalog=LocalDataCatalog(plan.config.root))

        assert len(client.start_times) == 7
        assert min(client.start_times) == 1_767_225_600_000

    def test_verified_payload_is_committed_to_durable_catalog(self, tmp_path: Path) -> None:
        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=(),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=tmp_path / "lake"),
            start_date=date(2026, 7, 1),
        )
        catalog = LocalDataCatalog(plan.config.root)
        snapshot = sync_futures_data_lake(plan=plan, client=FakeClient(), catalog=catalog)

        assert len(snapshot.partitions) == 1
        assert snapshot.partitions[0].path.exists()
        assert catalog.partition_exists(
            DatasetKind.KLINES_1H, "BTCUSDT", snapshot.partitions[0].start_time_ms
        )
        repeated = sync_futures_data_lake(plan=plan, client=FakeClient(), catalog=catalog)
        assert len(repeated.partitions) == 1

    def test_empty_month_is_not_committed(self, tmp_path: Path) -> None:
        class EmptyClient(FakeClient):
            def download_partition(self, *args: object, **kwargs: object) -> bytes:
                return b""

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=(),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=tmp_path / "lake"),
            start_date=date(2026, 7, 1),
        )
        snapshot = sync_futures_data_lake(
            plan=plan, client=EmptyClient(), catalog=LocalDataCatalog(plan.config.root)
        )

        assert snapshot.partitions == ()

    def test_invalid_parquet_month_is_quarantined_without_catalog_commit(self, tmp_path: Path) -> None:
        class InvalidParquetClient:
            payload = b"not-a-parquet"

            def download_partition(self, *args: object, **kwargs: object) -> bytes:
                _ = (args, kwargs)
                return self.payload

            def download_checksum(self, *args: object, **kwargs: object) -> str:
                _ = (args, kwargs)
                return hashlib.sha256(self.payload).hexdigest()

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=(),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=tmp_path / "lake"),
            start_date=date(2026, 7, 1),
        )

        snapshot = sync_futures_data_lake(
            plan=plan,
            client=InvalidParquetClient(),
            catalog=LocalDataCatalog(plan.config.root),
        )

        assert snapshot.partitions == ()

    def test_bounded_parallel_downloads_commit_all_partitions(self, tmp_path: Path) -> None:
        payload_buffer = io.BytesIO()
        pd.DataFrame({"timestamp": [1_783_440_000_000], "close": [100.0]}).to_parquet(
            payload_buffer, index=False
        )
        payload = payload_buffer.getvalue()

        class ConcurrentClient:
            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()
                self.barrier = threading.Barrier(4)

            def download_partition(self, *args: object, **kwargs: object) -> bytes:
                _ = (args, kwargs)
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                self.barrier.wait(timeout=3)
                with self.lock:
                    self.active -= 1
                return payload

            def download_checksum(self, *args: object, **kwargs: object) -> str:
                _ = (args, kwargs)
                return hashlib.sha256(payload).hexdigest()

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"),
            selected_symbols=(),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=tmp_path / "lake", max_workers=4),
            start_date=date(2026, 7, 1),
        )
        client = ConcurrentClient()

        snapshot = sync_futures_data_lake(
            plan=plan, client=client, catalog=LocalDataCatalog(plan.config.root)
        )

        assert client.max_active == 4
        assert len(snapshot.partitions) == 4
