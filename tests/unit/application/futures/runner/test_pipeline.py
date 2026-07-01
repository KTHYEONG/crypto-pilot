from __future__ import annotations

from typing import Any

import pytest
from pytest_mock import MockerFixture

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import MarketDataBundle, RunnerResult, RunWindow
from src.application.futures.runner.pipeline import run_pipeline


def make_run_config(phase: str = "l3") -> FuturesRunConfig:
    return FuturesRunConfig(
        timeframe="4h", date="2026-05-01", trials=3,
        phase=phase, sync="skip", refresh_universe=False, sync_metrics=False,  # type: ignore[arg-type]
    )

def make_window() -> RunWindow:
    from datetime import date
    return RunWindow(
        fetch_start="2024-01-01", is_start="2024-04-01", oos_start="2025-01-01",
        end_date="2026-05-01", fetch_start_date=date(2024,1,1), is_start_date=date(2024,4,1),
        oos_start_date=date(2025,1,1), end_date_value=date(2026,5,1),
    )

def make_data_bundle() -> MarketDataBundle:
    return MarketDataBundle(
        data_maps={"BTCUSDT": {"4h": object()}},
        oos_data_maps={"BTCUSDT": {"4h": object()}},
        valid_symbols=("BTCUSDT",),
    )


def _track(order: list[str], key: str, val: Any = None) -> Any:
    order.append(key)
    return val


class TestRunPipeline:
    def test_run_pipeline_l3_preserves_orchestration_order(self, mocker: MockerFixture) -> None:
        order: list[str] = []

        mocker.patch("src.application.futures.runner.window.resolve_run_window", return_value=make_window())
        mocker.patch("src.application.futures.runner.window.resolve_layered_window", return_value=None)

        mocker.patch(
            "src.application.futures.runner.sync.ensure_universe_ledger_sync",
            side_effect=lambda *a: _track(order, "sync"),
        )
        mocker.patch(
            "src.application.futures.runner.stages.universe.run_universe_stage",
            side_effect=lambda *args, **kwargs: _track(order, "universe", object()),
        )
        mocker.patch(
            "src.application.futures.runner.sync.ensure_cached_symbol_data",
            side_effect=lambda *args, **kwargs: _track(order, "cache"),
        )
        mocker.patch(
            "src.application.futures.runner.stages.data.run_data_stage",
            side_effect=lambda *args, **kwargs: _track(order, "data", make_data_bundle()),
        )
        mocker.patch(
            "src.application.futures.runner.stages.regime.run_regime_stage",
            side_effect=lambda *a: _track(order, "regime"),
        )
        mocker.patch(
            "src.application.futures.runner.stages.strategy.run_strategy_stage",
            side_effect=lambda *args, **kwargs: _track(order, "strategy"),
        )
        mock_optimize = mocker.patch(
            "src.application.futures.runner.stages.optimize.run_optimization_stage",
            side_effect=lambda *a, **kw: _track(order, "optimize", RunnerResult(0, "l3_done")),
        )

        result = run_pipeline(make_run_config("l3"))

        assert result == RunnerResult(0, "l3_done")
        assert order == ["sync", "universe", "cache", "data", "regime", "strategy", "optimize"]
        mock_optimize.assert_called_once()

    @pytest.mark.parametrize("phase", ["l1", "l2"])
    def test_run_pipeline_l1_l2_skip_optimization(self, mocker: MockerFixture, phase: str) -> None:
        mocker.patch("src.application.futures.runner.window.resolve_run_window", return_value=make_window())
        mocker.patch("src.application.futures.runner.window.resolve_layered_window", return_value=None)
        mocker.patch("src.application.futures.runner.sync.ensure_universe_ledger_sync")
        mocker.patch(
            "src.application.futures.runner.stages.universe.run_universe_stage",
            return_value=object(),
        )
        mocker.patch("src.application.futures.runner.sync.ensure_cached_symbol_data")
        mocker.patch(
            "src.application.futures.runner.stages.data.run_data_stage",
            return_value=make_data_bundle(),
        )
        mocker.patch("src.application.futures.runner.stages.regime.run_regime_stage")
        mock_strategy = mocker.patch("src.application.futures.runner.stages.strategy.run_strategy_stage")
        if phase == "l2":
            mock_strategy.return_value = RunnerResult(0, "tiered_pipeline_l2_completed")

        mock_optimize = mocker.patch("src.application.futures.runner.stages.optimize.run_optimization_stage")

        expected = (
            RunnerResult(0, "l1_mode_done") if phase == "l1"
            else RunnerResult(0, "tiered_pipeline_l2_completed")
        )
        result = run_pipeline(make_run_config(phase))

        assert result == expected
        mock_optimize.assert_not_called()
