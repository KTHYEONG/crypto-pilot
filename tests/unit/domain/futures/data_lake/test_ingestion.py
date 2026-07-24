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
    UniverseStateRequest,
)
from src.domain.futures.data_lake.ingestion import (
    ChecksumMismatchError,
    StorageBudgetError,
    build_ingestion_plan,
    migrate_legacy_universe_state,
    refresh_live_universe_state,
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
        lake_root = tmp_path / "data" / "lake"
        kline_root = lake_root / DatasetKind.KLINES_1H.value
        btc_path = kline_root / "symbol=BTCUSDT" / "year=2026" / "month=07" / "part.parquet"
        btc_path.parent.mkdir(parents=True)
        bad_path = kline_root / "symbol=币安人生USDT" / "year=2026" / "month=07" / "part.parquet"
        bad_path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {"timestamp": [1_783_440_000_000], "close": [100.0], "quote_volume": [1_000_000.0]}
        )
        frame.to_parquet(btc_path, index=False)
        frame.to_parquet(bad_path, index=False)

        plan = build_ingestion_plan(
            config=DataLakeConfig(root=lake_root),
            reference_date=date(2026, 7, 8),
        )

        assert plan.broad_symbols == ("BTCUSDT",)


def test_universe_state_request_rejects_invalid_axes() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="must not be empty"):
        UniverseStateRequest(np.array([], dtype=np.int64), 1)
    with pytest.raises(ValueError, match="must be >= 1"):
        UniverseStateRequest(np.array([1], dtype=np.int64), 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        UniverseStateRequest(np.array([2, 1], dtype=np.int64), 1)


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
                universe_state_hash="",
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
                universe_state_hash="",
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

    def test_metrics_5m_is_not_limited_to_recent_180_days(self, tmp_path: Path) -> None:
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
            datasets=(DatasetKind.METRICS_5M,),
            config=DataLakeConfig(root=tmp_path / "lake"),
            start_date=date(2024, 7, 8),
        )
        client = RecordingClient()

        sync_futures_data_lake(plan=plan, client=client, catalog=LocalDataCatalog(plan.config.root))

        assert len(client.start_times) == 25
        assert min(client.start_times) == 1_719_792_000_000

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


def test_migrate_legacy_universe_state_writes_lake_state_and_exchange_info(
    tmp_path: Path,
) -> None:
    import sqlite3

    root = tmp_path / "lake"
    kline = root / "klines_1h" / "symbol=BTCUSDT" / "year=2026" / "month=07" / "part.parquet"
    kline.parent.mkdir(parents=True)
    pd.DataFrame({"timestamp": [1_783_440_000_000], "close": [100.0]}).to_parquet(kline, index=False)
    ledger = tmp_path / "universe_ledger.db"
    conn = sqlite3.connect(ledger)
    conn.execute(
        "CREATE TABLE ledger (symbol TEXT, date TEXT, knowledge_date TEXT, is_listed INT, "
        "is_trading INT, status TEXT, adv_usdt_median REAL, listing_age_days INT, "
        "taker_fee_bps REAL, contract_type TEXT, quote_asset TEXT)"
    )
    conn.execute(
        "INSERT INTO ledger VALUES ('BTCUSDT', '2026-07-01', '2026-07-02', 1, 1, "
        "'TRADING', 1000000, 100, 5, 'PERPETUAL', 'USDT')"
    )
    conn.commit()
    conn.close()

    catalog = LocalDataCatalog(root)
    state_hash = migrate_legacy_universe_state(source_ledger=ledger, catalog=catalog, root=root)
    snapshot = catalog.load_snapshot(1_800_000_000_000)

    assert state_hash == snapshot.universe_state_hash
    assert {part.dataset for part in snapshot.partitions} == {
        DatasetKind.EXCHANGE_INFO,
        DatasetKind.UNIVERSE_STATE,
    }
    state_partition = next(
        part for part in snapshot.partitions if part.dataset is DatasetKind.UNIVERSE_STATE
    )
    state = pd.read_parquet(state_partition.path)
    assert int(state.loc[0, "effective_time_ns"]) == 1_783_036_800_000_000_000


def test_migrate_legacy_universe_state_rejects_symbols_missing_from_lake(
    tmp_path: Path,
) -> None:
    import sqlite3

    root = tmp_path / "lake"
    kline = root / DatasetKind.KLINES_1H.value / "symbol=BTCUSDT" / "part.parquet"
    kline.parent.mkdir(parents=True)
    pd.DataFrame({"timestamp": [1], "close": [1.0]}).to_parquet(kline, index=False)
    ledger = tmp_path / "universe_ledger.db"
    conn = sqlite3.connect(ledger)
    conn.execute(
        "CREATE TABLE ledger (symbol TEXT, date TEXT, knowledge_date TEXT, is_listed INT, "
        "is_trading INT, status TEXT, adv_usdt_median REAL, listing_age_days INT, "
        "taker_fee_bps REAL, contract_type TEXT, quote_asset TEXT)"
    )
    conn.execute(
        "INSERT INTO ledger VALUES ('ETHUSDT', '2026-07-01', '2026-07-02', 1, 1, "
        "'TRADING', 1000000, 100, 5, 'PERPETUAL', 'USDT')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="no rows for lake symbols"):
        migrate_legacy_universe_state(
            source_ledger=ledger, catalog=LocalDataCatalog(root), root=root,
        )


def test_migrate_legacy_universe_state_requires_lake_klines(tmp_path: Path) -> None:
    import sqlite3

    ledger = tmp_path / "universe_ledger.db"
    conn = sqlite3.connect(ledger)
    conn.execute(
        "CREATE TABLE ledger (symbol TEXT, date TEXT, knowledge_date TEXT, is_listed INT, "
        "is_trading INT, status TEXT, adv_usdt_median REAL, listing_age_days INT, "
        "taker_fee_bps REAL, contract_type TEXT, quote_asset TEXT)"
    )
    conn.execute(
        "INSERT INTO ledger VALUES ('BTCUSDT', '2026-07-01', '2026-07-02', 1, 1, "
        "'TRADING', 1000000, 100, 5, 'PERPETUAL', 'USDT')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="no valid klines_1h"):
        migrate_legacy_universe_state(
            source_ledger=ledger,
            catalog=LocalDataCatalog(tmp_path / "lake"),
            root=tmp_path / "lake",
        )


def test_refresh_live_universe_state_writes_next_causal_day(tmp_path: Path) -> None:
    class ExchangeInfoClient:
        def fetch_exchange_info(self) -> dict[str, object]:
            return {"symbols": [
                {"symbol": "BTCUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                {"symbol": "BAD", "quoteAsset": "BTC", "contractType": "PERPETUAL", "status": "TRADING"},
            ]}

    catalog = LocalDataCatalog(tmp_path)
    hash_value = refresh_live_universe_state(
        root=tmp_path,
        catalog=catalog,
        client=ExchangeInfoClient(),
        knowledge_time_ns=1_783_468_800_000_000_000,
    )
    snapshot = catalog.load_snapshot(1_783_555_200_000)
    state_path = next(part.path for part in snapshot.partitions if part.dataset is DatasetKind.UNIVERSE_STATE)
    state = pd.read_parquet(state_path)
    assert hash_value == snapshot.universe_state_hash
    assert state.symbol.tolist() == ["BTCUSDT"]
    assert int(state.effective_time_ns.iloc[0]) > int(state.knowledge_time_ns.iloc[0])


def test_refresh_live_universe_state_limits_to_lake_symbols(tmp_path: Path) -> None:
    kline = tmp_path / DatasetKind.KLINES_1H.value / "symbol=BTCUSDT" / "part.parquet"
    kline.parent.mkdir(parents=True)
    pd.DataFrame({"timestamp": [1], "close": [1.0]}).to_parquet(kline, index=False)

    class ExchangeInfoClient:
        def fetch_exchange_info(self) -> dict[str, object]:
            return {"symbols": [
                {"symbol": "BTCUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                {"symbol": "ETHUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
            ]}

    catalog = LocalDataCatalog(tmp_path)
    refresh_live_universe_state(
        root=tmp_path, catalog=catalog, client=ExchangeInfoClient(),
        knowledge_time_ns=1_783_468_800_000_000_000,
    )
    state_path = next(
        part.path for part in catalog.load_snapshot(1_783_555_200_000).partitions
        if part.dataset is DatasetKind.UNIVERSE_STATE
    )
    assert pd.read_parquet(state_path)["symbol"].tolist() == ["BTCUSDT"]


def test_refresh_live_universe_state_rejects_empty_exchange_info(tmp_path: Path) -> None:
    class EmptyExchangeInfoClient:
        def fetch_exchange_info(self) -> dict[str, object]:
            return {"symbols": []}

    with pytest.raises(RuntimeError, match="no USDT perpetual"):
        refresh_live_universe_state(
            root=tmp_path,
            catalog=LocalDataCatalog(tmp_path),
            client=EmptyExchangeInfoClient(),
            knowledge_time_ns=1_783_468_800_000_000_000,
        )


def test_refresh_live_universe_state_rejects_invalid_exchange_schema(tmp_path: Path) -> None:
    class InvalidExchangeInfoClient:
        def fetch_exchange_info(self) -> dict[str, object]:
            return {"symbols": "invalid"}

    with pytest.raises(RuntimeError, match="symbols must be a list"):
        refresh_live_universe_state(
            root=tmp_path,
            catalog=LocalDataCatalog(tmp_path),
            client=InvalidExchangeInfoClient(),
            knowledge_time_ns=1_783_468_800_000_000_000,
        )


def test_refresh_live_universe_state_skips_non_mapping_records(tmp_path: Path) -> None:
    class NonMappingExchangeInfoClient:
        def fetch_exchange_info(self) -> dict[str, object]:
            return {"symbols": [None]}

    with pytest.raises(RuntimeError, match="no USDT perpetual"):
        refresh_live_universe_state(
            root=tmp_path,
            catalog=LocalDataCatalog(tmp_path),
            client=NonMappingExchangeInfoClient(),
            knowledge_time_ns=1_783_468_800_000_000_000,
        )

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
