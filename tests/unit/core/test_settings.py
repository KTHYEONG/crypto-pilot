from pathlib import Path

import pandas as pd
import pytest

from src.core.settings import FUTURES_DATA_DIR, FuturesStorageLayout


def test_futures_storage_layout_paths() -> None:
    # Scenario 1: Happy Path - path resolutions
    # Test ohlcv path
    ohlcv_path = FuturesStorageLayout.get_ohlcv_path("BTC/USDT", "1m")
    assert ohlcv_path == FUTURES_DATA_DIR / "ohlcv" / "1m" / "BTC_USDT.parquet"

    # Test enriched path
    enriched_path = FuturesStorageLayout.get_enriched_path("BTC/USDT", "1m")
    assert enriched_path == FUTURES_DATA_DIR / "enriched" / "1m" / "BTC_USDT.parquet"

    # Test funding path
    funding_path = FuturesStorageLayout.get_funding_path("BTC/USDT")
    assert funding_path == FUTURES_DATA_DIR / "funding" / "BTC_USDT.parquet"

    # Test metrics path
    metrics_path = FuturesStorageLayout.get_metrics_path("BTC/USDT")
    assert metrics_path == FUTURES_DATA_DIR / "metrics" / "BTC_USDT.parquet"

    # Test metadata path
    metadata_path = FuturesStorageLayout.get_metadata_path("parquet_cache_meta.json")
    assert metadata_path == FUTURES_DATA_DIR / "metadata" / "parquet_cache_meta.json"


def test_futures_storage_layout_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Scenario 2: Auto-Migration - LIMIT-03
    # Use tmp_path for FUTURES_DATA_DIR to avoid side-effects on live data
    mock_futures_dir = tmp_path / "futures"
    mock_futures_dir.mkdir()
    monkeypatch.setattr("src.core.settings.FUTURES_DATA_DIR", mock_futures_dir)

    # 1. Create legacy files
    legacy_ohlcv = mock_futures_dir / "BTC_USDT_1m.parquet"
    legacy_funding = mock_futures_dir / "BTC_USDT_funding.parquet"
    legacy_metrics = mock_futures_dir / "BTC_USDT_metrics.parquet"
    legacy_metadata = mock_futures_dir / "parquet_cache_meta.json"

    # Write dummy data to legacy files
    df = pd.DataFrame({"timestamp": [1700000000000], "open": [30000.0]})
    df.to_parquet(legacy_ohlcv, index=False)
    df.to_parquet(legacy_funding, index=False)
    df.to_parquet(legacy_metrics, index=False)
    legacy_metadata.write_text("{}", encoding="utf-8")

    # 2. Resolve paths using layout and verify migration occurs
    ohlcv_new = FuturesStorageLayout.get_ohlcv_path("BTC/USDT", "1m")
    funding_new = FuturesStorageLayout.get_funding_path("BTC/USDT")
    metrics_new = FuturesStorageLayout.get_metrics_path("BTC/USDT")
    metadata_new = FuturesStorageLayout.get_metadata_path("parquet_cache_meta.json")

    # Assert new paths are correct
    assert ohlcv_new == mock_futures_dir / "ohlcv" / "1m" / "BTC_USDT.parquet"
    assert funding_new == mock_futures_dir / "funding" / "BTC_USDT.parquet"
    assert metrics_new == mock_futures_dir / "metrics" / "BTC_USDT.parquet"
    assert metadata_new == mock_futures_dir / "metadata" / "parquet_cache_meta.json"

    # Assert files are moved to new paths
    assert ohlcv_new.exists()
    assert funding_new.exists()
    assert metrics_new.exists()
    assert metadata_new.exists()

    # Assert legacy files are removed
    assert not legacy_ohlcv.exists()
    assert not legacy_funding.exists()
    assert not legacy_metrics.exists()
    assert not legacy_metadata.exists()


def test_futures_storage_zstd_float32_datetime_reconstruction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup mock FUTURES_DATA_DIR
    mock_futures_dir = tmp_path / "futures"
    mock_futures_dir.mkdir()
    monkeypatch.setattr("src.core.settings.FUTURES_DATA_DIR", mock_futures_dir)
    monkeypatch.setattr("src.domain.futures.backtest.data_loader.FUTURES_DATA_DIR", mock_futures_dir)

    from src.domain.futures.backtest.data_loader import DataCollector

    collector = DataCollector()

    # Create dummy DataFrame with datetime, open, high, low, close as float64
    df_orig = pd.DataFrame(
        {
            "timestamp": [1700000000000],
            "datetime": pd.to_datetime([1700000000000], unit="ms", utc=True),
            "open": [30000.0],
            "high": [31000.0],
            "low": [29000.0],
            "close": [30500.0],
            "volume": [1.5],
        }
    )

    # Save cache (this will internally trigger _save_cache, dropping datetime,
    # downcasting price cols to float32, and saving with zstd)
    collector._save_cache("BTC/USDT", "1m", df_orig)

    # Check the file written to disk
    resolved_path = FuturesStorageLayout.get_ohlcv_path("BTC/USDT", "1m", base_dir=mock_futures_dir)
    assert resolved_path.exists()

    df_disk = pd.read_parquet(resolved_path)
    assert "datetime" not in df_disk.columns
    assert df_disk["open"].dtype == "float32"
    assert df_disk["high"].dtype == "float32"
    assert df_disk["low"].dtype == "float32"
    assert df_disk["close"].dtype == "float32"
    assert df_disk["volume"].dtype == "float64"  # Volume should not be downcasted

    # Load cache (should dynamically reconstruct datetime)
    df_loaded = collector._load_cache("BTC/USDT", "1m")
    assert "datetime" in df_loaded.columns
    assert pd.api.types.is_datetime64_any_dtype(df_loaded["datetime"])
    assert df_loaded["datetime"].iloc[0] == df_orig["datetime"].iloc[0]


def test_list_cached_parquet_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_futures_dir = tmp_path / "futures"
    mock_futures_dir.mkdir()
    monkeypatch.setattr("src.domain.futures.backtest.data_loader.FUTURES_DATA_DIR", mock_futures_dir)

    from src.domain.futures.backtest.data_loader import DataCollector

    collector = DataCollector()

    # 1. Create a file in the new partitioned directory
    p_ohlcv_dir = mock_futures_dir / "ohlcv" / "4h"
    p_ohlcv_dir.mkdir(parents=True)
    (p_ohlcv_dir / "BTC_USDT.parquet").write_text("dummy", encoding="utf-8")

    # 2. Create a file in the legacy flat directory
    (mock_futures_dir / "ETH_USDT_4h.parquet").write_text("dummy", encoding="utf-8")

    # 3. Test list_cached_parquet_symbols
    symbols = collector.list_cached_parquet_symbols("4h")
    assert "BTC/USDT" in symbols
    assert "ETH/USDT" in symbols
    assert len(symbols) == 2
