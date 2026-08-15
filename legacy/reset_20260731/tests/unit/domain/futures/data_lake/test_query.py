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
    PartitionManifest,
)
from src.domain.futures.data_lake.query import (
    _load_partition_data,
    BinanceQueryClient,
    LocalDataCatalog,
    materialize_causal_metrics_grid,
    materialize_feature_grid,
    materialize_feature_grid_parallel,
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


def test_partition_loader_normalizes_taker_buy_volume_alias(tmp_path: Path) -> None:
    path = tmp_path / "part.parquet"
    pd.DataFrame({
        "timestamp": [1_704_067_200_000],
        "taker_buy_quote": [float("nan")],
        "taker_buy_quote_volume": [123.45],
    }).to_parquet(path, index=False)

    frame = _load_partition_data(
        [path],
        start_time_ns=1_704_067_200_000_000_000,
        end_time_ns=1_704_070_800_000_000_000,
        fields=("taker_buy_quote",),
    )

    assert frame["taker_buy_quote"].tolist() == [123.45]


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

    def test_normalize_vision_funding_uses_last_column_for_three_column_archive(self) -> None:
        normalized = BinanceQueryClient._normalize_vision_funding(
            pd.DataFrame([[1_783_440_000_000, 4, -0.00033019]])
        )
        assert normalized["funding_rate"].tolist() == [-0.00033019]

    def test_rejects_interval_value_in_two_column_funding(self) -> None:
        from src.domain.futures.compound.contracts import FundingDataIntegrityError

        with pytest.raises(FundingDataIntegrityError, match="exceed"):
            BinanceQueryClient._normalize_vision_funding(
                pd.DataFrame([[1_783_440_000_000, 8.0]])
            )

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


def _write_metrics_parquet(
    root: Path, symbol: str, rows: list[dict[str, float]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    part = root / "metrics_5m" / f"symbol={symbol}" / "year=2026" / "month=07" / "part.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    # real Binance Vision metrics_5m files embed a "symbol" column alongside the Hive-style
    # symbol= partition directory; pyarrow's dataset factory infers a dictionary-typed partition
    # column from the path and can collide with an embedded large_string "symbol" column
    # (ArrowTypeError) unless the reader avoids ds.dataset()-based schema unification.
    for row in rows:
        row.setdefault("symbol", symbol)
    pd.DataFrame(rows).to_parquet(part, index=False)


class TestMaterializeCausalMetricsGrid:
    def test_materialize_causal_metrics_grid_asof_join_uses_available_at(self, tmp_path: Path) -> None:
        one_hour_ms = 3_600_000
        ts_base = 1_783_440_000_000
        rows: list[dict[str, float]] = []
        for i in range(12):
            t = ts_base + i * 300_000
            avail = t + 300_000
            rows.append({
                "timestamp": float(t),
                "available_at": float(avail),
                "top_trader_long_short_ratio": float(1.0 + i * 0.1),
            })
        _write_metrics_parquet(tmp_path, "BTCUSDT", rows)

        second_hour_ns = (ts_base + one_hour_ms) * 1_000_000
        start_ns = second_hour_ns
        end_ns = (ts_base + 2 * one_hour_ms) * 1_000_000
        result = materialize_causal_metrics_grid(
            symbols=("BTCUSDT",),
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            lake_root=tmp_path,
            field="top_trader_long_short_ratio",
        )
        assert isinstance(result, NativeFeatureGrid)
        assert result.symbols == ("BTCUSDT",)
        assert result.timestamps_ns.shape[0] == 1
        val = result.fields["top_trader_long_short_ratio"][0, 0]
        assert np.isfinite(val), f"Expected finite value, got {val}"
        assert result.available["top_trader_long_short_ratio"][0, 0]
        last_obs_idx = 11
        expected_val = 1.0 + last_obs_idx * 0.1
        assert abs(val - expected_val) < 1e-6, (
            f"Expected {expected_val} (last available_at <= grid), got {val}"
        )

    def test_materialize_causal_metrics_grid_tolerance_exceeded_returns_nan(self, tmp_path: Path) -> None:
        one_hour_ms = 3_600_000
        ts_base = 1_783_440_000_000
        rows: list[dict[str, float]] = [
            {"timestamp": float(ts_base), "available_at": float(ts_base + 60_000),
             "top_trader_long_short_ratio": 1.5},
        ]
        later = ts_base + 4 * one_hour_ms
        rows.append({
            "timestamp": float(later), "available_at": float(later + 60_000),
            "top_trader_long_short_ratio": 2.0,
        })
        _write_metrics_parquet(tmp_path, "BTCUSDT", rows)

        start_ns = (ts_base + one_hour_ms) * 1_000_000
        end_ns = (ts_base + 6 * one_hour_ms) * 1_000_000
        result = materialize_causal_metrics_grid(
            symbols=("BTCUSDT",),
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            lake_root=tmp_path,
            field="top_trader_long_short_ratio",
            tolerance_ns=7_200_000_000_000,
        )
        assert result.timestamps_ns.shape[0] == 5
        assert np.isfinite(result.fields["top_trader_long_short_ratio"][0, 0])
        assert result.available["top_trader_long_short_ratio"][0, 0]
        assert np.isnan(result.fields["top_trader_long_short_ratio"][2, 0])
        assert not result.available["top_trader_long_short_ratio"][2, 0]

    def test_rejects_empty_symbols(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="symbols must not be empty"):
            materialize_causal_metrics_grid(
                symbols=(),
                start_time_ns=0,
                end_time_ns=3_600_000_000_000,
                lake_root=tmp_path,
                field="top_trader_long_short_ratio",
            )

    def test_rejects_empty_field(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="field must not be empty"):
            materialize_causal_metrics_grid(
                symbols=("BTCUSDT",),
                start_time_ns=0,
                end_time_ns=3_600_000_000_000,
                lake_root=tmp_path,
                field="",
            )

    def test_grid_entirely_before_first_available_at_returns_all_nan(self, tmp_path: Path) -> None:
        one_hour_ms = 3_600_000
        ts_base = 1_783_440_000_000
        available_at = ts_base + int(2.9 * one_hour_ms)
        rows: list[dict[str, float]] = [
            {"timestamp": float(ts_base), "available_at": float(available_at),
             "top_trader_long_short_ratio": 1.5},
        ]
        _write_metrics_parquet(tmp_path, "BTCUSDT", rows)

        start_ns = ts_base * 1_000_000
        end_ns = (ts_base + 3 * one_hour_ms) * 1_000_000
        result = materialize_causal_metrics_grid(
            symbols=("BTCUSDT",),
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            lake_root=tmp_path,
            field="top_trader_long_short_ratio",
        )
        assert result.timestamps_ns.shape[0] == 3
        assert np.all(np.isnan(result.fields["top_trader_long_short_ratio"]))
        assert not np.any(result.available["top_trader_long_short_ratio"])

    def test_symbol_with_no_partition_directory_returns_all_nan(self, tmp_path: Path) -> None:
        result = materialize_causal_metrics_grid(
            symbols=("NOSUCHSYMBOL",),
            start_time_ns=1_783_440_000_000 * 1_000_000,
            end_time_ns=(1_783_440_000_000 + 3_600_000) * 1_000_000,
            lake_root=tmp_path,
            field="top_trader_long_short_ratio",
        )
        assert result.timestamps_ns.shape[0] == 1
        assert np.all(np.isnan(result.fields["top_trader_long_short_ratio"]))
        assert not np.any(result.available["top_trader_long_short_ratio"])

    def test_parquet_missing_required_columns_is_skipped(self, tmp_path: Path) -> None:
        part = tmp_path / "metrics_5m" / "symbol=BTCUSDT" / "year=2026" / "month=07" / "part.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [1_783_440_000_000.0], "unrelated_col": [1.0]}).to_parquet(part, index=False)

        result = materialize_causal_metrics_grid(
            symbols=("BTCUSDT",),
            start_time_ns=1_783_440_000_000 * 1_000_000,
            end_time_ns=(1_783_440_000_000 + 3_600_000) * 1_000_000,
            lake_root=tmp_path,
            field="top_trader_long_short_ratio",
        )
        assert result.timestamps_ns.shape[0] == 1
        assert np.all(np.isnan(result.fields["top_trader_long_short_ratio"]))
        assert not np.any(result.available["top_trader_long_short_ratio"])

    def test_corrupted_parquet_file_is_skipped_without_raising(self, tmp_path: Path) -> None:
        part = tmp_path / "metrics_5m" / "symbol=BTCUSDT" / "year=2026" / "month=07" / "part.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"not a real parquet file")

        result = materialize_causal_metrics_grid(
            symbols=("BTCUSDT",),
            start_time_ns=1_783_440_000_000 * 1_000_000,
            end_time_ns=(1_783_440_000_000 + 3_600_000) * 1_000_000,
            lake_root=tmp_path,
            field="top_trader_long_short_ratio",
        )
        assert result.timestamps_ns.shape[0] == 1
        assert np.all(np.isnan(result.fields["top_trader_long_short_ratio"]))
        assert not np.any(result.available["top_trader_long_short_ratio"])

    def test_data_entirely_outside_requested_range_returns_all_nan(self, tmp_path: Path) -> None:
        one_hour_ms = 3_600_000
        ts_base = 1_783_440_000_000
        rows: list[dict[str, float]] = [
            {"timestamp": float(ts_base), "available_at": float(ts_base + 60_000),
             "top_trader_long_short_ratio": 1.5},
        ]
        _write_metrics_parquet(tmp_path, "BTCUSDT", rows)

        far_future_start = ts_base + 1000 * one_hour_ms
        start_ns = far_future_start * 1_000_000
        end_ns = (far_future_start + 3 * one_hour_ms) * 1_000_000
        result = materialize_causal_metrics_grid(
            symbols=("BTCUSDT",),
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            lake_root=tmp_path,
            field="top_trader_long_short_ratio",
            tolerance_ns=7_200_000_000_000,
        )
        assert result.timestamps_ns.shape[0] == 3
        assert np.all(np.isnan(result.fields["top_trader_long_short_ratio"]))
        assert not np.any(result.available["top_trader_long_short_ratio"])


class TestNormalizeVisionFunding:
    def test_two_column_preserves_funding_rate(self) -> None:
        df = pd.DataFrame([[1_782_864_000_000, -0.0002]], columns=["timestamp", "funding_rate"])
        result = BinanceQueryClient._normalize_vision_funding(df)
        assert "timestamp" in result.columns
        assert "funding_rate" in result.columns
        assert result["funding_rate"].iloc[0] == -0.0002

    def test_three_column_selects_last_column(self) -> None:
        df = pd.DataFrame([[1_782_864_000_000, 8, 0.0001]], columns=["calc_time", "funding_interval_hours", "last_funding_rate"])
        result = BinanceQueryClient._normalize_vision_funding(df)
        assert "timestamp" in result.columns
        assert "funding_rate" in result.columns
        assert result["funding_rate"].iloc[0] == 0.0001

    def test_three_column_interval_hour_rejected(self) -> None:
        from src.domain.futures.data_lake.reconciliation import validate_funding_rates
        df = pd.DataFrame([[1_782_864_000_000, 8, 8.0]], columns=["calc_time", "funding_interval_hours", "last_funding_rate"])
        from src.domain.futures.compound.contracts import FundingDataIntegrityError, MAX_ABS_FUNDING_RATE
        rate = df.iloc[:, 2].iloc[0]
        assert abs(rate) > MAX_ABS_FUNDING_RATE
        rates = np.array([rate], dtype=np.float64)
        with pytest.raises(FundingDataIntegrityError, match="exceed"):
            validate_funding_rates(rates, source="test")


def test_parallel_feature_grid_correctness(tmp_path: Path) -> None:
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    ts_base_ms = 1_783_440_000_000
    n_hours = 6
    partitions: list[PartitionManifest] = []
    for sym_idx, sym in enumerate(symbols):
        rows = []
        for h in range(n_hours):
            ts_ms = ts_base_ms + h * 3_600_000
            rows.append({
                "timestamp": ts_ms,
                "open": float(100 + sym_idx * 10 + h),
                "high": float(100 + sym_idx * 10 + h + 2),
                "low": float(100 + sym_idx * 10 + h - 1),
                "close": float(100 + sym_idx * 10 + h + 1),
                "quote_volume": float(1_000_000 + sym_idx * 100_000),
            })
        part = tmp_path / f"{sym}.parquet"
        pd.DataFrame(rows).to_parquet(part, index=False)
        partitions.append(PartitionManifest(
            DatasetKind.KLINES_1H, sym,
            ts_base_ms, ts_base_ms + (n_hours - 1) * 3_600_000,
            n_hours, "test", "cache", True, part,
        ))

    snap = DataSnapshot("s1", ts_base_ms + n_hours * 3_600_000,
                         tuple(partitions), "h1", "", 0)
    request = GridRequest(
        symbols=symbols, timeframe="1h", source_timeframe="1h",
        fields=("open", "high", "low", "close", "quote_volume"),
        start_time_ns=ts_base_ms * 1_000_000,
        end_time_ns=(ts_base_ms + n_hours * 3_600_000) * 1_000_000,
    )

    seq_grid = materialize_feature_grid(request=request, snapshot=snap, dataset=DatasetKind.KLINES_1H)
    par_grid = materialize_feature_grid_parallel(request=request, snapshot=snap, dataset=DatasetKind.KLINES_1H)

    assert seq_grid.symbols == par_grid.symbols
    assert np.array_equal(seq_grid.timestamps_ns, par_grid.timestamps_ns)
    for field in request.fields:
        assert field in seq_grid.fields, f"missing {field} in sequential"
        assert field in par_grid.fields, f"missing {field} in parallel"
        np.testing.assert_array_equal(seq_grid.fields[field], par_grid.fields[field],
                                       err_msg=f"field {field} mismatch")
        np.testing.assert_array_equal(seq_grid.available[field], par_grid.available[field],
                                       err_msg=f"available {field} mismatch")


def test_parallel_feature_grid_rejects_empty_symbols() -> None:
    snap = DataSnapshot("s1", 1_000_000, (), "h1", "", 0)
    request = GridRequest(
        symbols=(), timeframe="1h", source_timeframe="1h",
        fields=("close",), start_time_ns=0, end_time_ns=3_600_000_000_000,
    )
    with pytest.raises(ValueError, match="at least one symbol"):
        materialize_feature_grid_parallel(request=request, snapshot=snap, dataset=DatasetKind.KLINES_1H)


def test_parallel_feature_grid_rejects_empty_fields() -> None:
    snap = DataSnapshot("s1", 1_000_000, (), "h1", "", 0)
    request = GridRequest(
        symbols=("BTCUSDT",), timeframe="1h", source_timeframe="1h",
        fields=(), start_time_ns=0, end_time_ns=3_600_000_000_000,
    )
    with pytest.raises(ValueError, match="at least one field"):
        materialize_feature_grid_parallel(request=request, snapshot=snap, dataset=DatasetKind.KLINES_1H)


def test_parallel_feature_grid_rejects_timeframe_mismatch() -> None:
    snap = DataSnapshot("s1", 1_000_000, (), "h1", "", 0)
    request = GridRequest(
        symbols=("BTCUSDT",), timeframe="1h", source_timeframe="5m",
        fields=("close",), start_time_ns=0, end_time_ns=3_600_000_000_000,
    )
    with pytest.raises(ValueError, match="matching request and source"):
        materialize_feature_grid_parallel(request=request, snapshot=snap, dataset=DatasetKind.KLINES_1H)
