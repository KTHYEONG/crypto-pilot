from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    AlphaForecastTape,
    AlphaLifecycle,
    ExecutionLedger,
    MarketFeatureCube,
)
from src.domain.futures.compound.data_plane import materialize_hourly_execution_features
from src.domain.futures.compound.simulator import simulate_compound_portfolio


@pytest.fixture
def small_cube() -> MarketFeatureCube:
    n_bars, n_syms = 128, 2
    close = np.column_stack((
        np.linspace(100, 110, n_bars),
        np.linspace(50, 55, n_bars),
    )).astype(np.float64)
    return MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("BTCUSDT", "ETHUSDT"),
        fields_2d={
            "open": close.copy(),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 50_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )


@pytest.fixture
def simple_tape(small_cube: MarketFeatureCube) -> AlphaForecastTape:
    n_bars = small_cube.timestamps_ns.size
    n_syms = len(small_cube.symbols)
    mu = np.zeros((n_bars, n_syms, 2), dtype=np.float32)
    mu[:, 0, 0] = 0.001
    mu[:, 1, 0] = 0.0005
    mu[:, 0, 1] = 0.0015
    mu[:, 1, 1] = 0.0008
    return AlphaForecastTape(
        timestamps_ns=small_cube.timestamps_ns,
        symbols=small_cube.symbols,
        recipe_ids=("trend_h4", "carry_h12"),
        gross_mu_3d=mu,
        forecast_var_3d=np.full((n_bars, n_syms, 2), 1e-8, dtype=np.float32),
        reliability_3d=np.ones((n_bars, n_syms, 2), dtype=np.float32),
        valid_3d=np.ones((n_bars, n_syms, 2), dtype=np.bool_),
        horizon_bars_1d=np.array([4, 12], dtype=np.int16),
        lifecycle_by_recipe=(AlphaLifecycle.ACTIVE, AlphaLifecycle.ACTIVE),
        model_version="v1",
        data_manifest_hash="h1",
        fold_manifest_hash="fh1",
    )


class TestSimulateCompoundPortfolio:
    def test_returns_execution_ledger(self, small_cube: MarketFeatureCube, simple_tape: AlphaForecastTape) -> None:
        config = CompoundEngineConfig()
        ledger = simulate_compound_portfolio(cube=small_cube, tape=simple_tape, config=config)
        assert isinstance(ledger, ExecutionLedger)
        assert len(ledger.net_returns_1d) > 0
        assert ledger.equity_1d[0] == 1.0

    def test_ledger_has_cost_fields(self, small_cube: MarketFeatureCube, simple_tape: AlphaForecastTape) -> None:
        config = CompoundEngineConfig()
        ledger = simulate_compound_portfolio(cube=small_cube, tape=simple_tape, config=config)
        assert len(ledger.fee_returns_1d) > 0
        assert len(ledger.slippage_returns_1d) > 0
        assert len(ledger.impact_returns_1d) > 0


def test_simulator_when_market_data_stale_for_two_bars_forces_exit() -> None:
    n_bars, n_syms = 128, 2
    close = np.column_stack((np.linspace(100, 110, n_bars), np.linspace(50, 55, n_bars))).astype(np.float64)
    cube = MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("BTCUSDT", "ETHUSDT"),
        fields_2d={
            "open": close.copy(), "high": close * 1.001, "low": close * 0.999, "close": close,
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 50_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 25_000_000,
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    mu = np.zeros((n_bars, n_syms, 2), dtype=np.float32)
    mu[:, 0, 0] = 0.001
    tape = AlphaForecastTape(
        timestamps_ns=cube.timestamps_ns, symbols=cube.symbols,
        recipe_ids=("trend_h4", "carry_h12"),
        gross_mu_3d=mu, forecast_var_3d=np.full((n_bars, n_syms, 2), 1e-8, dtype=np.float32),
        reliability_3d=np.ones((n_bars, n_syms, 2), dtype=np.float32),
        valid_3d=np.ones((n_bars, n_syms, 2), dtype=np.bool_),
        horizon_bars_1d=np.array([4, 12], dtype=np.int16),
        lifecycle_by_recipe=(AlphaLifecycle.ACTIVE, AlphaLifecycle.ACTIVE),
        model_version="v1", data_manifest_hash="h1", fold_manifest_hash="fh1",
    )
    config = CompoundEngineConfig()
    ledger = simulate_compound_portfolio(cube=cube, tape=tape, config=config)
    assert isinstance(ledger, ExecutionLedger)


def test_simulator_rejects_invalid_book_depth_and_uses_fallback_cost() -> None:
    mark = pd.DataFrame({"close": [50000.0]}, index=pd.DatetimeIndex(["2026-01-01"]))
    result = materialize_hourly_execution_features(
        book_depth=pd.DataFrame(), mark_price=mark, fallback_cost_bps=12.0,
    )
    assert result["execution_cost_bps"].iloc[0] == 12.0
