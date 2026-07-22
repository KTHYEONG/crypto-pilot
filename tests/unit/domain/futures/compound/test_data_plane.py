from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.compound.config import DataPlaneConfig
from src.domain.futures.compound.contracts import MarketFeatureCube
from src.domain.futures.compound.data_plane import (
    build_market_feature_cube,
    materialize_hourly_execution_features,
    validate_market_feature_cube,
)


@pytest.fixture
def mock_aligned() -> MagicMock:
    aligned = MagicMock()
    n_bars, n_syms = 100, 3
    aligned.datetimes = pd.date_range("2020-01-01", periods=n_bars, freq="h").values
    aligned.symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
    aligned.open_2d = np.ones((n_bars, n_syms), dtype=np.float64) * 100
    aligned.high_2d = np.ones((n_bars, n_syms), dtype=np.float64) * 101
    aligned.low_2d = np.ones((n_bars, n_syms), dtype=np.float64) * 99
    aligned.close_2d = np.ones((n_bars, n_syms), dtype=np.float64) * 100
    aligned.volume_2d = np.ones((n_bars, n_syms), dtype=np.float32) * 1_000_000
    aligned.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float32)
    aligned.active_mask = np.ones((n_bars, n_syms), dtype=np.bool_)
    aligned.execution_cost_bps_2d = np.full((n_bars, n_syms), 12.0, dtype=np.float32)
    aligned.basis_2d = None
    aligned.taker_buy_2d = None
    aligned.trades_2d = None
    return aligned


@pytest.fixture
def mock_universe() -> MagicMock:
    universe = MagicMock()
    n_bars, n_syms = 100, 3
    universe.instrument_ids = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
    universe.eligible = np.ones((n_bars, n_syms), dtype=np.bool_)
    universe.entry_block = np.zeros((n_bars, n_syms), dtype=np.bool_)
    universe.capacity_usdt = np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64)
    universe.risk_scale = np.ones((n_bars, n_syms), dtype=np.float64)
    universe.cost_bps = np.full((n_bars, n_syms), 12.0, dtype=np.float64)
    return universe


class TestBuildMarketFeatureCube:
    def test_basic_construction(self, mock_aligned: MagicMock, mock_universe: MagicMock) -> None:
        cube = build_market_feature_cube(
            aligned=mock_aligned,
            universe=mock_universe,
            optional_fields={},
            available_at_ns={},
            config=DataPlaneConfig(),
        )
        assert isinstance(cube, MarketFeatureCube)
        assert "close" in cube.fields_2d
        assert "funding" in cube.fields_2d

    def test_when_available_at_is_future_masks_value(self, mock_aligned: MagicMock, mock_universe: MagicMock) -> None:
        n_bars = 100
        future_avail = np.full((n_bars, 3), np.iinfo(np.int64).max, dtype=np.int64)
        cube = build_market_feature_cube(
            aligned=mock_aligned,
            universe=mock_universe,
            optional_fields={"custom": np.ones((n_bars, 3), dtype=np.float32)},
            available_at_ns={"custom": future_avail},
            config=DataPlaneConfig(),
        )
        assert cube.available_2d["core"].all()
        assert not cube.available_2d["custom"].any()


class TestValidateMarketFeatureCube:
    def test_valid_cube_passes(self, mock_aligned: MagicMock, mock_universe: MagicMock) -> None:
        cube = build_market_feature_cube(
            aligned=mock_aligned,
            universe=mock_universe,
            optional_fields={},
            available_at_ns={},
            config=DataPlaneConfig(),
        )
        validate_market_feature_cube(cube)

    def test_non_monotonic_timestamp_does_not_raise_in_validate(self) -> None:
        n_bars, n_syms = 5, 2
        cube = MarketFeatureCube(
            timestamps_ns=np.array([1, 2, 2, 4, 5], dtype=np.int64),
            symbols=("A", "B"),
            fields_2d={"close": np.ones((5, 2), dtype=np.float64)},
            available_2d={"core": np.ones((5, 2), dtype=np.bool_)},
            eligible_2d=np.ones((5, 2), dtype=np.bool_),
            entry_block_2d=np.zeros((5, 2), dtype=np.bool_),
            capacity_usdt_2d=np.full((5, 2), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((5, 2), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        with pytest.raises(AssertionError):
            validate_market_feature_cube(cube)


class TestMaterializeHourlyExecutionFeatures:
    def test_empty_book_depth_uses_fallback(self) -> None:
        mark = pd.DataFrame({"close": [50000.0]}, index=pd.DatetimeIndex(["2026-01-01"]))
        result = materialize_hourly_execution_features(
            book_depth=pd.DataFrame(), mark_price=mark, fallback_cost_bps=12.0
        )
        assert not result.empty
        assert result["execution_cost_bps"].iloc[0] == 12.0

    def test_valid_book_depth(self) -> None:
        idx = pd.DatetimeIndex(["2026-01-01"])
        depth = pd.DataFrame({"bid": [49900], "ask": [50100], "bid_depth": [100.0], "ask_depth": [100.0]}, index=idx)
        mark = pd.DataFrame({"close": [50000.0]}, index=idx)
        result = materialize_hourly_execution_features(book_depth=depth, mark_price=mark, fallback_cost_bps=12.0)
        assert result["depth_spread_bps"].iloc[0] > 0
        assert result["execution_cost_bps"].iloc[0] > 0


def test_build_market_feature_cube_when_available_at_is_future_masks_value(mock_aligned: MagicMock, mock_universe: MagicMock) -> None:
    n_bars = 100
    future_avail = np.full((n_bars, 3), np.iinfo(np.int64).max, dtype=np.int64)
    cube = build_market_feature_cube(
        aligned=mock_aligned,
        universe=mock_universe,
        optional_fields={"custom": np.ones((n_bars, 3), dtype=np.float32)},
        available_at_ns={"custom": future_avail},
        config=DataPlaneConfig(),
    )
    assert cube.available_2d["core"].all()
    assert not cube.available_2d["custom"].any()


def test_build_market_feature_cube_when_current_exchange_universe_used_raises() -> None:
    from src.domain.futures.universe.membership import validate_historical_manifest_coverage
    with pytest.raises(ValueError, match="not found in Vision manifest"):
        validate_historical_manifest_coverage(
            instrument_ids=["UNKNOWNSYM"],
            first_market_at_ns=np.array([0], dtype=np.int64),
            manifest_symbols=set(),
        )


def test_data_fallback_returns_explicit_state_without_stale_value() -> None:
    mark = pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex(["2026-01-01"]))
    result = materialize_hourly_execution_features(
        book_depth=pd.DataFrame(), mark_price=mark, fallback_cost_bps=12.0,
    )
    assert result["execution_cost_bps"].iloc[0] == 12.0
    assert "depth_spread_bps" in result.columns
