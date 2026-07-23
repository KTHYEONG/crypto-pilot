from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from src.domain.futures.data_lake.contracts import (
    DataSnapshot,
    DatasetKind,
    GridRequest,
    NativeFeatureGrid,
)
from src.domain.futures.data_lake.query import (
    BinanceQueryClient,
    LocalDataCatalog,
    materialize_native_grid,
)


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


class TestMaterializeNativeGrid:
    def test_requires_symbols(self) -> None:
        snap = _snap()
        request = GridRequest(
            symbols=(),
            timeframe="1h",
            source_timeframe="1h",
            fields=("close",),
            start_time_ns=0,
            end_time_ns=3_600_000_000_000,
        )
        with pytest.raises(ValueError, match="at least one symbol"):
            materialize_native_grid(request=request, snapshot=snap)

    def test_requires_fields(self) -> None:
        snap = _snap()
        request = GridRequest(
            symbols=("BTCUSDT",),
            timeframe="1h",
            source_timeframe="1h",
            fields=(),
            start_time_ns=0,
            end_time_ns=3_600_000_000_000,
        )
        with pytest.raises(ValueError, match="at least one field"):
            materialize_native_grid(request=request, snapshot=snap)

    def test_rejects_unsupported_native_timeframe(self) -> None:
        snap = _snap()
        request = GridRequest(
            symbols=("BTCUSDT",), timeframe="5m", source_timeframe="5m", fields=("close",),
            start_time_ns=0, end_time_ns=300_000_000_000,
        )
        with pytest.raises(ValueError, match="unsupported native timeframe"):
            materialize_native_grid(request=request, snapshot=snap)

    def test_rejects_resampling_inside_native_grid(self) -> None:
        snap = _snap()
        request = GridRequest(
            symbols=("BTCUSDT",), timeframe="1h", source_timeframe="5m", fields=("close",),
            start_time_ns=0, end_time_ns=3_600_000_000_000,
        )
        with pytest.raises(ValueError, match="matching request and source"):
            materialize_native_grid(request=request, snapshot=snap)

    def test_returns_native_feature_grid(self) -> None:
        snap = _snap()
        request = GridRequest(
            symbols=("BTCUSDT",),
            timeframe="1h",
            source_timeframe="1h",
            fields=("close",),
            start_time_ns=0,
            end_time_ns=3_600_000_000_000,
        )
        result = materialize_native_grid(request=request, snapshot=snap)
        assert isinstance(result, NativeFeatureGrid)
        assert result.symbols == ("BTCUSDT",)
        assert "close" in result.fields


class TestBinanceQueryClient:
    def test_downloads_partition_from_vision(self, tmp_path: Path, monkeypatch) -> None:
        client = BinanceQueryClient(source_root=tmp_path)
        monkeypatch.setattr(
            client._vision,
            "fetch_klines_archive_monthly",
            lambda *_: pd.DataFrame([[1_783_440_000_000, 100.0, 102.0, 99.0, 101.0]]),
        )
        assert client.download_calls == 0
        payload = client.download_partition(DatasetKind.KLINES_1H, "BTCUSDT", 1_783_440_000_000)
        assert client.download_calls == 1
        assert pd.read_parquet(__import__("io").BytesIO(payload))["close"].tolist() == [101.0]
        assert client.download_checksum(DatasetKind.KLINES_1H, "BTCUSDT", 1_783_440_000_000) == hashlib.sha256(payload).hexdigest()

    def test_uses_vision_only_when_local_symbol_is_absent(self, tmp_path: Path, monkeypatch) -> None:
        client = BinanceQueryClient(source_root=tmp_path)
        monkeypatch.setattr(
            client._vision,
            "fetch_klines_archive_monthly",
            lambda *_: pd.DataFrame([[1_783_440_000_000, 100.0, 102.0, 99.0, 101.0]]),
        )
        payload = client.download_partition(DatasetKind.KLINES_1H, "BTCUSDT", 1_783_440_000_000)
        assert pd.read_parquet(__import__("io").BytesIO(payload))["close"].tolist() == [101.0]

    def test_uses_vision_one_minute_when_no_cache(self, tmp_path: Path, monkeypatch) -> None:
        client = BinanceQueryClient(source_root=tmp_path)
        called: list[tuple[str, str]] = []

        def fetch(symbol: str, interval: str, *_: object) -> pd.DataFrame:
            called.append((symbol, interval))
            return pd.DataFrame([[1_783_440_000_000, 100.0, 102.0, 99.0, 101.0]])

        monkeypatch.setattr(client._vision, "fetch_klines_archive_monthly", fetch)
        payload = client.download_partition(DatasetKind.KLINES_1M, "BTCUSDT", 1_783_440_000_000)
        assert called == [("BTCUSDT", "1m")]
        assert pd.read_parquet(__import__("io").BytesIO(payload))["close"].tolist() == [101.0]

    def test_normalizes_vision_funding(self, tmp_path: Path, monkeypatch) -> None:
        client = BinanceQueryClient(source_root=tmp_path)
        monkeypatch.setattr(
            client._vision,
            "fetch_funding_rate_monthly",
            lambda *_: pd.DataFrame([[1_783_440_000_000, 0.0001]]),
        )
        payload = client.download_partition(DatasetKind.FUNDING_EVENT, "BTCUSDT", 1_783_440_000_000)
        assert pd.read_parquet(__import__("io").BytesIO(payload))["funding_rate"].tolist() == [0.0001]

    def test_empty_dataframe_functions_return_empty(self) -> None:
        assert BinanceQueryClient._normalize_vision_funding(pd.DataFrame()).empty
        assert BinanceQueryClient._normalize_vision_klines(pd.DataFrame()).empty
        assert BinanceQueryClient._normalize_timestamp(pd.DataFrame({"close": [1.0]})).empty
        normalized = BinanceQueryClient._normalize_timestamp(
            pd.DataFrame({"datetime": [pd.Timestamp("2026-07-01", tz="UTC")], "close": [1.0]})
        )
        assert normalized["timestamp"].tolist() == [1782864000000]

    def test_normalizes_missing_quote_volume_from_close_and_volume(self) -> None:
        normalized = BinanceQueryClient._normalize_timestamp(
            pd.DataFrame(
                {
                    "timestamp": [1_783_440_000_000],
                    "close": [101.0],
                    "volume": [2.0],
                    "quote_vol": [np.nan],
                }
            )
        )
        assert normalized["quote_volume"].tolist() == [202.0]


class TestLocalDataCatalog:
    def test_read_only_catalog_uses_canonical_manifest(self, tmp_path: Path) -> None:
        writable = LocalDataCatalog(root=tmp_path)
        writable._connection.close()

        read_only = LocalDataCatalog(root=tmp_path, read_only=True)

        assert read_only.total_bytes() == 0

    def test_default_no_coverage(self, tmp_path: Path) -> None:
        catalog = LocalDataCatalog(root=tmp_path)
        assert not catalog.partition_exists(DatasetKind.KLINES_1H, "BTCUSDT", 1)
        snap = _snap()
        from src.domain.futures.data_lake.contracts import IngestionPlan, DataLakeConfig

        plan = IngestionPlan(
            reference_date=__import__("datetime").date.today(),
            broad_symbols=(),
            selected_symbols=(),
            datasets=(),
            config=DataLakeConfig(root=tmp_path),
        )
        assert catalog.has_complete_coverage(snapshot=snap, plan=plan) is False


def test_materialize_feature_grid_rejects_mismatched_timeframes() -> None:
    from src.domain.futures.data_lake.contracts import GridRequest
    from src.domain.futures.data_lake.query import materialize_feature_grid

    with pytest.raises(ValueError, match="matching request"):
        materialize_feature_grid(
            request=GridRequest(
                symbols=("BTCUSDT",), timeframe="1h", source_timeframe="5m",
                fields=("close",), start_time_ns=0, end_time_ns=3_600_000_000_000,
            ),
            snapshot=_snap(),
            dataset=DatasetKind.KLINES_1H,
        )


def test_materialize_feature_grid_rejects_empty_fields() -> None:
    from src.domain.futures.data_lake.contracts import GridRequest
    from src.domain.futures.data_lake.query import materialize_feature_grid

    with pytest.raises(ValueError, match="at least one field"):
        materialize_feature_grid(
            request=GridRequest(
                symbols=("BTCUSDT",), timeframe="1h", source_timeframe="1h",
                fields=(), start_time_ns=0, end_time_ns=3_600_000_000_000,
            ),
            snapshot=_snap(),
            dataset=DatasetKind.KLINES_1H,
        )


def test_materialize_feature_grid_rejects_empty_symbols() -> None:
    from src.domain.futures.data_lake.contracts import GridRequest
    from src.domain.futures.data_lake.query import materialize_feature_grid

    with pytest.raises(ValueError, match="at least one symbol"):
        materialize_feature_grid(
            request=GridRequest(
                symbols=(), timeframe="1h", source_timeframe="1h",
                fields=("close",), start_time_ns=0, end_time_ns=3_600_000_000_000,
            ),
            snapshot=_snap(),
            dataset=DatasetKind.KLINES_1H,
        )

    def test_lock_falls_back_to_recovery_catalog(self, tmp_path: Path, monkeypatch) -> None:
        import duckdb
        import src.domain.futures.data_lake.query as query_module

        real_connect = duckdb.connect
        calls = 0

        def connect_with_first_lock(path: str, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise duckdb.IOException("locked")
            return real_connect(path, *args, **kwargs)

        monkeypatch.setattr(query_module.duckdb, "connect", connect_with_first_lock)
        catalog = LocalDataCatalog(tmp_path)
        assert catalog._database.name == "catalog_recovered.duckdb"

    def test_persists_manifest_and_materializes_exact_timestamp(self, tmp_path: Path) -> None:
        root = tmp_path / "lake"
        part = root / "klines_1h" / "symbol=BTCUSDT" / "year=2026" / "month=07" / "part.parquet"
        part.parent.mkdir(parents=True)
        pd.DataFrame(
            {"timestamp": [1_783_440_000_000], "open": [99.0], "high": [101.0], "low": [98.0], "close": [100.0], "quote_volume": [12_000_000.0]}
        ).to_parquet(part, index=False)
        manifest = __import__("src.domain.futures.data_lake.contracts", fromlist=["PartitionManifest"]).PartitionManifest(
            dataset=DatasetKind.KLINES_1H, symbol="BTCUSDT", start_time_ms=1_783_440_000_000,
            end_time_ms=1_783_440_000_000, row_count=1, sha256="h", source="cache", is_final=True, path=part,
        )
        catalog = LocalDataCatalog(root)
        catalog.commit_partition(manifest)
        assert catalog.total_bytes() == part.stat().st_size
        snapshot = catalog.load_snapshot(1_783_440_000_000)
        grid = materialize_native_grid(
            request=GridRequest(symbols=("BTCUSDT",), timeframe="1h", source_timeframe="1h", fields=("close", "funding"), start_time_ns=1_783_440_000_000_000_000, end_time_ns=1_783_443_600_000_000_000),
            snapshot=snapshot,
        )
        assert grid.fields["close"].tolist() == [[100.0]]
        assert grid.available["close"].tolist() == [[True]]
        assert np.isnan(grid.fields["funding"][0, 0])
        from datetime import date

        from src.domain.futures.data_lake.contracts import DataLakeConfig, IngestionPlan

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8), broad_symbols=("BTCUSDT",), selected_symbols=(),
            datasets=(DatasetKind.KLINES_1H,), config=DataLakeConfig(root=root),
        )
        assert catalog.has_complete_coverage(snapshot, plan)
