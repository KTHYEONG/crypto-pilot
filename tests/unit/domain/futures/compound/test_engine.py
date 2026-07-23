from __future__ import annotations

import numpy as np
import pytest

from src.application.futures.runner.compound_universe import DailyPITUniverse
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.engine import (
    run_compound_engine,
    run_multiscale_compound_engine,
)


@pytest.fixture
def small_cube() -> MarketFeatureCube:
    n_bars, n_syms = 512, 5
    close = np.column_stack(tuple(
        np.linspace(100, 110 + i, n_bars) for i in range(n_syms)
    )).astype(np.float64)
    return MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"),
        fields_2d={
            "open": close.copy(),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 50_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 25_000_000,
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )


class TestRunCompoundEngine:
    def test_returns_compound_engine_result(self, small_cube: MarketFeatureCube) -> None:
        manifest = SealedHoldoutManifest(
            holdout_id="test", start_time_ns=0, end_time_ns=0,
            holdout_days=90, model_version="v1", data_manifest_hash="h1",
        )
        config = CompoundEngineConfig()
        result = run_compound_engine(cube=small_cube, holdout_manifest=manifest, config=config)
        assert isinstance(result, CompoundEngineResult)
        assert result.alpha_tape is not None
        assert result.ledger is not None
        assert result.l2 is not None
        assert result.l3 is not None
        assert isinstance(result.l2.annualized_log_growth, float)


class TestRunMultiscaleCompoundEngine:
    def test_returns_compound_engine_result(self) -> None:
        n = 500
        close = np.column_stack((
            np.linspace(100, 110, n),
            np.linspace(50, 55, n),
        )).astype(np.float64)
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n, dtype=np.int64) * 3_600_000_000_000,
            symbols=("BTCUSDT", "ETHUSDT"),
            fields_2d={
                "open": close.copy(),
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "quote_volume": np.ones((n, 2), dtype=np.float32) * 50_000_000,
                "funding": np.zeros((n, 2), dtype=np.float32),
                "premium": np.zeros((n, 2), dtype=np.float32),
                "taker_buy_quote": np.ones((n, 2), dtype=np.float32) * 25_000_000,
            },
            available_2d={"core": np.ones((n, 2), dtype=np.bool_)},
            eligible_2d=np.ones((n, 2), dtype=np.bool_),
            entry_block_2d=np.zeros((n, 2), dtype=np.bool_),
            exit_required_2d=np.zeros((n, 2), dtype=np.bool_),
            capacity_usdt_2d=np.full((n, 2), 1_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((n, 2), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        universe = DailyPITUniverse(symbols=cube.symbols, decision_dates=())
        manifest = SealedHoldoutManifest(
            holdout_id="test-ms",
            start_time_ns=int(cube.timestamps_ns[-180]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=180,
            model_version="v1",
            data_manifest_hash="h1",
        )
        config = CompoundEngineConfig()
        result = run_multiscale_compound_engine(
            market=cube, universe=universe,
            holdout_manifest=manifest, config=config,
        )
        assert isinstance(result, CompoundEngineResult)
        assert result.ledger is not None
        assert result.l2 is not None
        assert result.l3 is not None
