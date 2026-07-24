from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import LadderConfig
from src.domain.futures.compound.contracts import LadderStageResult, MarketFeatureCube
from src.domain.futures.compound.ladder import run_experiment_ladder

_NS_PER_HOUR = 3_600_000_000_000


def _make_market(n_bars: int, n_syms: int, symbols: tuple[str, ...]) -> MarketFeatureCube:
    rng = np.random.default_rng(42)
    close = np.cumprod(
        1.0 + rng.normal(0.0002, 0.005, (n_bars, n_syms)),
        axis=0,
    ).astype(np.float32) * 100.0
    return MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * _NS_PER_HOUR,
        symbols=symbols,
        fields_2d={
            "close": close,
            "open": close * 0.9995,
            "high": close * 1.005,
            "low": close * 0.995,
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 1_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            "mark": close.copy(),
            "index": close.copy(),
            "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 500_000,
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 8.0, dtype=np.float32),
        data_manifest_hash="test_hash",
    )


class TestRunExperimentLadder:
    @pytest.fixture
    def small_market(self):
        return _make_market(1024, 3, ("BTCUSDT", "ETHUSDT", "SOLUSDT"))

    def test_all_stages_execute_successfully(self, small_market):
        config = LadderConfig(cost_bps=8.0, n_bootstrap=100)
        results = run_experiment_ladder(small_market, small_market.eligible_2d, config, rng_seed=42)

        assert len(results) == 8
        for r in results:
            assert isinstance(r, LadderStageResult)
            assert r.status in ("ok", "error")
            assert "|" in r.stage_id

    def test_all_ok_stages_have_valid_metrics(self, small_market):
        config = LadderConfig(cost_bps=8.0, n_bootstrap=50)
        results = run_experiment_ladder(small_market, small_market.eligible_2d, config, rng_seed=42)

        ok_results = [r for r in results if r.status == "ok"]
        assert len(ok_results) >= 6

        for r in ok_results:
            assert np.isfinite(r.oos_log_growth)
            assert np.isfinite(r.sharpe)
            assert r.max_drawdown <= 0.0
            assert r.turnover_per_year >= 0.0

    def test_stage_ids_format(self, small_market):
        config = LadderConfig(cost_bps=8.0, n_bootstrap=50)
        results = run_experiment_ladder(small_market, small_market.eligible_2d, config, rng_seed=42)

        expected_prefixes = ["L1-0", "L1-1", "L1-2", "L1-3"]
        expected_l2 = ["L2-0", "L2-1"]
        for r in results:
            parts = r.stage_id.split("|")
            assert len(parts) == 2, f"bad stage_id: {r.stage_id}"
            assert parts[0] in expected_prefixes
            assert parts[1] in expected_l2

    def test_error_resilience(self):
        n_bars, n_syms = 500, 2
        rng = np.random.default_rng(7)
        close = np.cumprod(1.0 + rng.normal(0.0002, 0.005, (n_bars, n_syms)), axis=0).astype(np.float32) * 100.0
        market = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * _NS_PER_HOUR,
            symbols=("A", "B"),
            fields_2d={"close": close},
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 8.0, dtype=np.float32),
            data_manifest_hash="h2",
        )
        config = LadderConfig(cost_bps=8.0, n_bootstrap=50)
        results = run_experiment_ladder(market, market.eligible_2d, config, rng_seed=7)
        assert len(results) == 8
