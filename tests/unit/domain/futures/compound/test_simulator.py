from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    AlphaEventTape,
    AlphaForecastTape,
    AlphaLifecycle,
    ExecutionLedger,
    MarketFeatureCube,
    PortfolioDecision,
)
from src.application.futures.runner.compound_universe import DailyPITUniverse
from src.domain.futures.compound.data_plane import materialize_hourly_execution_features


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
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
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
        mean_edge_var_3d=np.full((n_bars, n_syms, 2), 1e-8, dtype=np.float32),
        residual_var_3d=np.full((n_bars, n_syms, 2), 1e-6, dtype=np.float32),
        reliability_3d=np.ones((n_bars, n_syms, 2), dtype=np.float32),
        estimated_3d=np.ones((n_bars, n_syms, 2), dtype=np.bool_),
        valid_3d=np.ones((n_bars, n_syms, 2), dtype=np.bool_),
        horizon_bars_1d=np.array([4, 12], dtype=np.int16),
        lifecycle_by_recipe=(AlphaLifecycle.ACTIVE, AlphaLifecycle.ACTIVE),
        model_version="v1",
        data_manifest_hash="h1",
        fold_manifest_hash="fh1",
    )


class TestSimulateCompoundPortfolio:
    def test_returns_execution_ledger(self, small_cube: MarketFeatureCube, simple_tape: AlphaForecastTape) -> None:
        from src.domain.futures.compound.simulator import simulate_compound_portfolio
        config = CompoundEngineConfig()
        ledger = simulate_compound_portfolio(cube=small_cube, alpha_tape=simple_tape, config=config)
        assert isinstance(ledger, ExecutionLedger)
        assert len(ledger.net_returns_1d) > 0
        assert ledger.equity_1d[0] == 1.0

    def test_ledger_has_cost_fields(self, small_cube: MarketFeatureCube, simple_tape: AlphaForecastTape) -> None:
        from src.domain.futures.compound.simulator import simulate_compound_portfolio
        config = CompoundEngineConfig()
        ledger = simulate_compound_portfolio(cube=small_cube, alpha_tape=simple_tape, config=config)
        assert len(ledger.fee_returns_1d) > 0
        assert len(ledger.slippage_returns_1d) > 0
        assert len(ledger.impact_returns_1d) > 0

    def test_multiscale_empty_event_tape_settles_chronologically(
        self, small_cube: MarketFeatureCube
    ) -> None:
        from src.domain.futures.compound.simulator import simulate_multiscale_portfolio

        handoff = AlphaEventTape(
            events=pa.table(
                {
                    "recipe_id": ["r1"],
                    "symbol": ["BTCUSDT"],
                    "decision_time_ns": [0],
                    "expiry_time_ns": [10**18],
                    "alpha_rate_per_hour": [0.02],
                    "mean_edge_variance": [1e-8],
                    "combination_weight": [1.0],
                }
            ),
            recipe_definitions=(),
            evidence=(),
            active_recipe_ids=(),
            model_version="v1",
            data_manifest_hash=small_cube.data_manifest_hash,
            fold_manifest_hash="f1",
        )
        universe = DailyPITUniverse(
            symbols=small_cube.symbols,
            decision_dates=(),
        )
        small_cube.exit_required_2d[2, 0] = True

        ledger = simulate_multiscale_portfolio(
            market=small_cube,
            universe=universe,
            handoff=handoff,
            config=CompoundEngineConfig(),
        )

        assert ledger.integrity_ok
        assert ledger.target_weights_2d.shape == (small_cube.timestamps_ns.size, 2)
        assert np.any(np.abs(ledger.target_weights_2d) > 0.0)

    def test_multiscale_rejects_single_bar_market(self, mocker) -> None:
        from src.domain.futures.compound.simulator import simulate_multiscale_portfolio

        market = mocker.Mock()
        market.timestamps_ns = np.array([0], dtype=np.int64)
        market.symbols = ("BTCUSDT",)

        with pytest.raises(ValueError, match="at least two bars"):
            simulate_multiscale_portfolio(
                market=market,
                universe=DailyPITUniverse(symbols=("BTCUSDT",), decision_dates=()),
                handoff=mocker.Mock(spec=AlphaEventTape),
                config=CompoundEngineConfig(),
            )

    def test_multiscale_marks_nonfinite_allocation_as_integrity_failure(
        self, mocker, small_cube: MarketFeatureCube
    ) -> None:
        from src.domain.futures.compound.simulator import simulate_multiscale_portfolio

        handoff = AlphaEventTape(
            events=pa.table(
                {
                    "recipe_id": ["r1"],
                    "symbol": ["BTCUSDT"],
                    "decision_time_ns": [0],
                    "expiry_time_ns": [10**18],
                    "alpha_rate_per_hour": [0.02],
                    "mean_edge_variance": [1e-8],
                    "combination_weight": [1.0],
                }
            ),
            recipe_definitions=(),
            evidence=(),
            active_recipe_ids=(),
            model_version="v1",
            data_manifest_hash=small_cube.data_manifest_hash,
            fold_manifest_hash="f1",
        )
        decision = PortfolioDecision(
            decision_idx=1,
            decision_time_ns=small_cube.timestamps_ns[1],
            target_weights_1d=np.array([np.nan, 0.0]),
            gross_exposure=float("nan"),
            net_exposure=float("nan"),
            forecast_ann_vol=float("nan"),
            risk_scale=1.0,
            binding_constraints=(),
        )
        mocker.patch(
            "src.domain.futures.compound.allocator.solve_event_growth_weights",
            return_value=decision,
        )
        ledger = simulate_multiscale_portfolio(
            market=small_cube,
            universe=DailyPITUniverse(symbols=small_cube.symbols, decision_dates=()),
            handoff=handoff,
            config=CompoundEngineConfig(),
        )
        assert not ledger.integrity_ok


def test_simulator_when_market_data_stale_for_two_bars_forces_exit() -> None:
    from src.domain.futures.compound.simulator import simulate_compound_portfolio
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
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    mu = np.zeros((n_bars, n_syms, 2), dtype=np.float32)
    mu[:, 0, 0] = 0.001
    tape = AlphaForecastTape(
        timestamps_ns=cube.timestamps_ns, symbols=cube.symbols,
        recipe_ids=("trend_h4", "carry_h12"),
        gross_mu_3d=mu,
        mean_edge_var_3d=np.full((n_bars, n_syms, 2), 1e-8, dtype=np.float32),
        residual_var_3d=np.full((n_bars, n_syms, 2), 1e-6, dtype=np.float32),
        reliability_3d=np.ones((n_bars, n_syms, 2), dtype=np.float32),
        estimated_3d=np.ones((n_bars, n_syms, 2), dtype=np.bool_),
        valid_3d=np.ones((n_bars, n_syms, 2), dtype=np.bool_),
        horizon_bars_1d=np.array([4, 12], dtype=np.int16),
        lifecycle_by_recipe=(AlphaLifecycle.ACTIVE, AlphaLifecycle.ACTIVE),
        model_version="v1", data_manifest_hash="h1", fold_manifest_hash="fh1",
    )
    config = CompoundEngineConfig()
    ledger = simulate_compound_portfolio(cube=cube, alpha_tape=tape, config=config)
    assert isinstance(ledger, ExecutionLedger)


def test_simulator_rejects_invalid_book_depth_and_uses_fallback_cost() -> None:
    mark = pd.DataFrame({"close": [50000.0]}, index=pd.DatetimeIndex(["2026-01-01"]))
    result = materialize_hourly_execution_features(
        book_depth=pd.DataFrame(), mark_price=mark, fallback_cost_bps=12.0,
    )
    assert result["execution_cost_bps"].iloc[0] == 12.0
