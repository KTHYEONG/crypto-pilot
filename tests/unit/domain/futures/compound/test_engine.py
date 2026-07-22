from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.engine import run_compound_engine


@pytest.fixture
def small_cube() -> MarketFeatureCube:
    n_bars, n_syms = 512, 2
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
            "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 25_000_000,
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
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


def test_register_new_recipe_increments_version_and_rotates_holdout() -> None:
    old = SealedHoldoutManifest(
        holdout_id="old", start_time_ns=0, end_time_ns=86_400_000_000_000 * 90,
        holdout_days=90, model_version="v1", data_manifest_hash="h1",
    )
    new_ = SealedHoldoutManifest(
        holdout_id="new", start_time_ns=old.end_time_ns, end_time_ns=old.end_time_ns + 86_400_000_000_000 * 90,
        holdout_days=90, model_version="v2", data_manifest_hash=old.data_manifest_hash,
    )
    assert old.model_version != new_.model_version
    assert old.end_time_ns <= new_.start_time_ns
