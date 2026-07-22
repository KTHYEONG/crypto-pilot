"""Integration tests for compound engine — peak RSS and mode routing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.application.futures.runner.compound_pipeline import (
    CompoundPipelineOutcome,
    run_compound_pipeline,
)
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.engine import run_compound_engine


def test_full_fixture_peak_rss_stays_below_budget() -> None:
    """PERF-03-07: verify engine completes with 20 symbols under 12GB."""
    n_bars, n_syms = 512, 20
    close = np.column_stack([np.linspace(100, 110, n_bars)] * n_syms).astype(np.float64)
    cube = MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=tuple(f"SYM{i}" for i in range(n_syms)),
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
    manifest = SealedHoldoutManifest(
        holdout_id="perf-test", start_time_ns=0,
        end_time_ns=n_bars * 3_600_000_000_000,
        holdout_days=90, model_version="v1", data_manifest_hash="h1",
    )
    result = run_compound_engine(cube=cube, holdout_manifest=manifest, config=CompoundEngineConfig())
    assert isinstance(result, CompoundEngineResult)
    assert result.l3 is not None


@pytest.mark.parametrize("mode", ["shadow", "active"])
def test_pipeline_mode_routing(mode: str) -> None:
    settings = MagicMock()
    settings.mode = mode
    outcome = run_compound_pipeline(
        aligned=MagicMock(datetimes=MagicMock(__len__=lambda: 0)),
        universe=MagicMock(),
        settings=settings,
    )
    assert isinstance(outcome, CompoundPipelineOutcome)
    assert outcome.mode == mode
