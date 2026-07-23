from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    DeploymentVerdict,
    L3ValidationResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.engine import run_multiscale_compound_engine
from src.domain.futures.compound.holdout_store import SealedHoldoutStore


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


class TestRunMultiscaleCompoundEngine:
    def test_returns_compound_engine_result(self, tmp_path, small_cube: MarketFeatureCube) -> None:
        n_syms = len(small_cube.symbols)
        universe = type("Universe", (), {
            "symbols": small_cube.symbols, "snapshots": (),
        })()
        store = SealedHoldoutStore(tmp_path / "engine_test.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="engine-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)
        config = CompoundEngineConfig()
        result = run_multiscale_compound_engine(
            market=small_cube,
            universe=universe,
            holdout_store=store,
            holdout_id="engine-test",
            config=config,
        )
        assert isinstance(result, CompoundEngineResult)
        assert result.handoff is not None
        assert result.ledger is not None
        assert result.l2 is not None
        assert result.l3 is not None

    def test_engine_idempotent_on_repeat(self, tmp_path, small_cube: MarketFeatureCube) -> None:
        universe = type("Universe", (), {
            "symbols": small_cube.symbols, "snapshots": (),
        })()
        store = SealedHoldoutStore(tmp_path / "engine_idem.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="idem-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)
        config = CompoundEngineConfig()
        r1 = run_multiscale_compound_engine(
            market=small_cube, universe=universe,
            holdout_store=store, holdout_id="idem-test", config=config,
        )
        r2 = run_multiscale_compound_engine(
            market=small_cube, universe=universe,
            holdout_store=store, holdout_id="idem-test", config=config,
        )
        assert r1.l3 == r2.l3
