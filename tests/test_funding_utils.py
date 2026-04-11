from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.futures.funding_utils import merge_funding_into_ohlcv


def test_merge_funding_assigns_boundary_event_to_same_bar(tmp_path: Path) -> None:
    ohlcv = pd.DataFrame(
        {
            "timestamp": [1640995200000, 1641009600000, 1641024000000],
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [10.0, 11.0, 12.0],
            "datetime": pd.to_datetime([1640995200000, 1641009600000, 1641024000000], unit="ms"),
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": [1640995200006, 1641024000009],
            "funding_rate": [0.0001, 0.0002],
            "datetime": pd.to_datetime([1640995200006, 1641024000009], unit="ms"),
        }
    )
    funding.to_parquet(tmp_path / "SOL_USDT_funding.parquet", index=False)

    merged = merge_funding_into_ohlcv("SOL/USDT", ohlcv, tmp_path)

    assert merged["funding_event_count"].tolist() == [1, 0, 1]
    assert merged["funding_rate_sum"].tolist() == pytest.approx([0.0001, 0.0, 0.0002])


def test_merge_funding_keeps_legacy_last_known_rate_column(tmp_path: Path) -> None:
    ohlcv = pd.DataFrame(
        {
            "timestamp": [1640995200000, 1641009600000, 1641024000000],
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [10.0, 11.0, 12.0],
            "datetime": pd.to_datetime([1640995200000, 1641009600000, 1641024000000], unit="ms"),
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": [1640995200006],
            "funding_rate": [0.0001],
            "datetime": pd.to_datetime([1640995200006], unit="ms"),
        }
    )
    funding.to_parquet(tmp_path / "SOL_USDT_funding.parquet", index=False)

    merged = merge_funding_into_ohlcv("SOL/USDT", ohlcv, tmp_path)

    assert pd.isna(merged.loc[0, "funding_rate"])
    assert merged.loc[1, "funding_rate"] == 0.0001
    assert merged.loc[2, "funding_rate"] == 0.0001
