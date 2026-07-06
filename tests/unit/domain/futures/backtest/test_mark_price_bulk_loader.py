"""Tests for premiumIndex bulk data loader and mark price generator."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.domain.futures.backtest.data_loader import (
    build_mark_price_1m_array,
    fetch_premiumindex_bulk,
)


def test_fetch_premiumindex_bulk_mocked() -> None:
    """Test fetch_premiumindex_bulk shapes and functionality with mock vision download."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = Path(tmp_dir)
        symbol = "BTCUSDT"
        n_bars = 288

        # Mock function that generates unique open_times per date to avoid deduplication
        def get_mock_df(sym: str, dt_obj: Any) -> pd.DataFrame:
            # We base the start timestamp on the target date to ensure uniqueness
            day_ts = int(pd.Timestamp(dt_obj).timestamp() * 1000)
            timestamps = [day_ts + i * 5 * 60 * 1000 for i in range(n_bars)]
            return pd.DataFrame(
                {
                    "open_time": timestamps,
                    "open": ["10000"] * n_bars,
                    "high": ["10100"] * n_bars,
                    "low": ["9900"] * n_bars,
                    "close": ["10050"] * n_bars,
                    "volume": ["100"] * n_bars,
                }
            )

        with patch("src.domain.futures.backtest.data_loader.BinanceVisionDownloader") as mock_downloader_cls:
            mock_downloader = MagicMock()
            mock_downloader.fetch_premiumindex_daily.side_effect = get_mock_df
            mock_downloader_cls.return_value = mock_downloader

            # Fetch for 3 days
            start = date(2020, 9, 13)
            end = date(2020, 9, 15)

            df = fetch_premiumindex_bulk(
                symbol=symbol,
                start_date=start,
                end_date=end,
                cache_dir=cache_path,
            )

            # 3 days * 288 bars = 864 bars
            assert len(mock_downloader.fetch_premiumindex_daily.mock_calls) == 3
            assert not df.empty
            assert len(df) == 864
            assert "close" in df.columns

            # Check cache directories were created and populated
            safe_symbol = symbol.replace("/", "").replace("_", "")
            for i in range(3):
                curr_date = date(2020, 9, 13 + i).strftime("%Y-%m-%d")
                cache_file = cache_path / safe_symbol / f"{curr_date}.parquet"
                assert cache_file.exists()


def test_fetch_premiumindex_bulk_cache_hit() -> None:
    """Test that fetch_premiumindex_bulk hits local parquet cache and avoids vision down."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = Path(tmp_dir)
        symbol = "BTCUSDT"
        safe_symbol = symbol.replace("/", "").replace("_", "")
        n_bars = 288

        # Create distinct data for cached days to avoid open_time duplicates
        ts_1 = [1600000000000 + i * 5 * 60 * 1000 for i in range(n_bars)]
        df_1 = pd.DataFrame(
            {
                "open_time": ts_1,
                "open": ["10000"] * n_bars,
                "high": ["10100"] * n_bars,
                "low": ["9900"] * n_bars,
                "close": ["10050"] * n_bars,
                "volume": ["100"] * n_bars,
            }
        )

        ts_2 = [1600000000000 + 288 * 5 * 60 * 1000 + i * 5 * 60 * 1000 for i in range(n_bars)]
        df_2 = pd.DataFrame(
            {
                "open_time": ts_2,
                "open": ["10000"] * n_bars,
                "high": ["10100"] * n_bars,
                "low": ["9900"] * n_bars,
                "close": ["10050"] * n_bars,
                "volume": ["100"] * n_bars,
            }
        )

        date_dir = cache_path / safe_symbol
        date_dir.mkdir(parents=True, exist_ok=True)
        df_1.to_parquet(date_dir / "2020-09-13.parquet", index=False)
        df_2.to_parquet(date_dir / "2020-09-14.parquet", index=False)

        with patch("src.domain.futures.backtest.data_loader.BinanceVisionDownloader") as mock_downloader_cls:
            mock_downloader = MagicMock()
            mock_downloader_cls.return_value = mock_downloader

            start = date(2020, 9, 13)
            end = date(2020, 9, 14)

            df = fetch_premiumindex_bulk(
                symbol=symbol,
                start_date=start,
                end_date=end,
                cache_dir=cache_path,
            )

            # Vision fetch should be called 0 times since it hits local cache
            assert len(mock_downloader.fetch_premiumindex_daily.mock_calls) == 0
            assert len(df) == 576  # 2 days * 288


def test_build_mark_price_1m_array_shape_and_ffill() -> None:
    """Test build_mark_price_1m_array shape generation and forward fill behavior."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = Path(tmp_dir)

        # 3 minutes = 180,000 ms.
        # df index starts at 0ms and next tick is at 180,000ms (3 minutes)
        df_btc = pd.DataFrame(
            {"close": [100.0, 102.0]},
            index=pd.to_datetime([1600000000000, 1600000180000], unit="ms", utc=True),
        )
        df_eth = pd.DataFrame(
            {"close": [200.0, 201.0]},
            index=pd.to_datetime([1600000000000, 1600000180000], unit="ms", utc=True),
        )

        def mock_bulk(symbol: str, **kwargs: object) -> pd.DataFrame:
            if "BTC" in symbol:
                return df_btc
            if "ETH" in symbol:
                return df_eth
            return pd.DataFrame()

        with patch("src.domain.futures.backtest.data_loader.fetch_premiumindex_bulk", side_effect=mock_bulk):
            # Start: 1600000000000 (0ms)
            # End: 1600000360000 (6 minutes = 360000 ms)
            arr = build_mark_price_1m_array(
                symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                start_ts=1600000000000,
                end_ts=1600000360000,
                cache_dir=cache_path,
            )

            # Shape: [6 bars, 3 symbols]
            assert arr.shape == (6, 3)

            # BTC prices:
            # 0m: 100.0
            # 1m: 100.0
            # 2m: 100.0
            # 3m: 102.0
            # 4m: 102.0
            # 5m: 102.0
            np.testing.assert_array_almost_equal(arr[:, 0], [100.0, 100.0, 100.0, 102.0, 102.0, 102.0])

            # ETH prices:
            # 0m-2m: 200.0
            # 3m-5m: 201.0
            np.testing.assert_array_almost_equal(arr[:, 1], [200.0, 200.0, 200.0, 201.0, 201.0, 201.0])

            # SOL (not found) -> NaN column
            assert np.isnan(arr[:, 2]).all()
