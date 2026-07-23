from __future__ import annotations

import pytest

from src.domain.futures.data_lake.contracts import (
    DataSnapshot,
    GridRequest,
    NativeFeatureGrid,
)
from src.domain.futures.data_lake.query import (
    BinanceQueryClient,
    LocalDataCatalog,
    materialize_native_grid,
)


class TestMaterializeNativeGrid:
    def test_requires_symbols(self) -> None:
        snap = DataSnapshot(
            snapshot_id="s1",
            reference_time_ms=1,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
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
        snap = DataSnapshot(
            snapshot_id="s1",
            reference_time_ms=1,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
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

    def test_returns_native_feature_grid(self) -> None:
        snap = DataSnapshot(
            snapshot_id="s1",
            reference_time_ms=1,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
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
    def test_download_calls_tracked(self) -> None:
        client = BinanceQueryClient()
        assert client.download_calls == 0
        client.download_partition()
        assert client.download_calls == 1


class TestLocalDataCatalog:
    def test_default_no_coverage(self) -> None:
        catalog = LocalDataCatalog(root="/tmp")  # noqa: S108
        snap = DataSnapshot(
            snapshot_id="s1",
            reference_time_ms=1,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
        from src.domain.futures.data_lake.contracts import IngestionPlan, DataLakeConfig

        plan = IngestionPlan(
            reference_date=__import__("datetime").date.today(),
            broad_symbols=(),
            selected_symbols=(),
            datasets=(),
            config=DataLakeConfig(root="/tmp"),  # noqa: S108
        )
        assert catalog.has_complete_coverage(snapshot=snap, plan=plan) is False
