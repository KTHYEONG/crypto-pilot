from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CostModel, StrategySpec


@pytest.fixture
def spec() -> StrategySpec:
    return StrategySpec()


@pytest.fixture
def costs() -> CostModel:
    return CostModel()


@pytest.fixture
def bars_flat() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000.0,
    }, index=index)


@pytest.fixture
def bars_ramp() -> pd.DataFrame:
    n = 300
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    opens = np.arange(100.0, 100.0 + n, dtype=np.float64)
    return pd.DataFrame({
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens + 0.5,
        "volume": 1000.0,
    }, index=index)


@pytest.fixture
def bars_stop_gap() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open": [100.0, 102.0, 101.0, 103.0, 95.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "high": [101.0, 103.0, 102.0, 104.0, 96.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "low":  [99.0,  101.0, 100.0, 102.0, 94.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        "close":[101.0, 103.0, 102.0, 104.0, 95.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "volume": 1000.0,
    }, index=index)


@pytest.fixture
def bars_both_touch() -> pd.DataFrame:
    """A bar where low touches stop price AND close triggers channel exit."""
    index = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    o = np.full(300, 100.0)
    h = np.full(300, 101.0)
    l_ = np.full(300, 99.0)
    c = np.full(300, 100.0)
    o[210:] = 90.0
    h[210:] = 91.0
    l_[210:] = 89.0
    c[210:] = 90.0
    l_[210] = 85.0
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=index)


@pytest.fixture
def btc_4h_slice() -> pd.DataFrame:
    from src.data.loader import load_ohlcv_4h
    path = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")
    return load_ohlcv_4h(path, end="2025-12-31")
