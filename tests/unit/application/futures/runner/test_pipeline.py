from __future__ import annotations

from typing import Any

import pytest
from pytest_mock import MockerFixture

from src.application.futures.runner.active_pipeline import RunnerResult as ActiveRunnerResult
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

        mocker.patch("src.application.futures.runner.active_pipeline._resolve_quarterly_window", return_value=make_window())  # noqa: E501
        mocker.patch("src.application.futures.runner.active_pipeline._resolve_layered_window", return_value=None)
        mocker.patch("src.application.futures.runner.active_pipeline._selected_symbols_from_snapshot", return_value=("BTCUSDT",))  # noqa: E501

        mocker.patch(
            "src.application.futures.runner.active_pipeline._ensure_universe_ledger_sync",
            side_effect=lambda *a: _track(order, "sync"),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_universe_stage",
            side_effect=lambda *args, **kwargs: _track(order, "universe", (["BTCUSDT"], [], [], ["BTCUSDT"], object(), [], object())),  # noqa: E501
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._ensure_cached_symbol_data_for_targets",
            side_effect=lambda *args, **kwargs: _track(order, "cache"),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_data_stage",
            side_effect=lambda *args, **kwargs: _track(order, "data", make_data_bundle()),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_regime_evaluation_stage",
            side_effect=lambda *a: _track(order, "regime", None),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_strategy_stage",
            side_effect=lambda *args, **kwargs: _track(order, "strategy"),
        )
        mock_optimize = mocker.patch(
            "src.application.futures.runner.active_pipeline._run_optimization_stage",
            side_effect=lambda *a, **kw: _track(order, "optimize", ActiveRunnerResult(0, "l3_done")),
        )

        result = run_pipeline(make_run_config("l3"))

        assert result == RunnerResult(0, "l3_done")
        assert order == ["sync", "universe", "cache", "data", "regime", "strategy", "optimize"]
        mock_optimize.assert_called_once()

    @pytest.mark.parametrize("phase", ["l1", "l2"])
    def test_run_pipeline_l1_l2_skip_optimization(self, mocker: MockerFixture, phase: str) -> None:
        mocker.patch("src.application.futures.runner.active_pipeline._resolve_quarterly_window", return_value=make_window())  # noqa: E501
        mocker.patch("src.application.futures.runner.active_pipeline._resolve_layered_window", return_value=None)
        mocker.patch("src.application.futures.runner.active_pipeline._selected_symbols_from_snapshot", return_value=("BTCUSDT",))  # noqa: E501
        mocker.patch("src.application.futures.runner.active_pipeline._ensure_universe_ledger_sync")
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_universe_stage",
            return_value=(["BTCUSDT"], [], [], ["BTCUSDT"], object(), [], object()),
        )
        mocker.patch("src.application.futures.runner.active_pipeline._ensure_cached_symbol_data_for_targets")
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_data_stage",
            return_value=make_data_bundle(),
        )
        mocker.patch("src.application.futures.runner.active_pipeline._run_regime_evaluation_stage", return_value=None)
        mock_strategy = mocker.patch("src.application.futures.runner.active_pipeline._run_strategy_stage")
        if phase == "l2":
            mock_strategy.return_value = ActiveRunnerResult(0, "tiered_pipeline_l2_completed")

        mock_optimize = mocker.patch("src.application.futures.runner.active_pipeline._run_optimization_stage")

        expected = (
            RunnerResult(0, "l1_mode_done") if phase == "l1"
            else RunnerResult(0, "tiered_pipeline_l2_completed")
        )
        result = run_pipeline(make_run_config(phase))

        assert result == expected
        mock_optimize.assert_not_called()
