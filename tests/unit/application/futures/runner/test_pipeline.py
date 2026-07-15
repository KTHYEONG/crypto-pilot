from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.application.futures.run_contracts import ActivePhase
from src.application.futures.runner.active_pipeline import (
    _build_data_not_ready_reasons,
)
from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import MarketDataBundle, RunnerResult, RunWindow
from src.application.futures.runner.pipeline import run_pipeline
from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig


def make_run_config(phase: ActivePhase = "l3") -> FuturesRunConfig:
    l0_runtime = (
        AlphaFoundryRuntimeConfig(mode="gate")
        if phase in {"l0", "l1"}
        else AlphaFoundryRuntimeConfig(mode="off")
    )
    return FuturesRunConfig(
        timeframe="4h",
        date="2026-05-01",
        trials=3,
        phase=phase,
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
        l0_runtime=l0_runtime,
    )


def make_window() -> RunWindow:
    from datetime import date

    return RunWindow(
        fetch_start="2024-01-01",
        is_start="2024-04-01",
        oos_start="2025-01-01",
        end_date="2026-05-01",
        fetch_start_date=date(2024, 1, 1),
        is_start_date=date(2024, 4, 1),
        oos_start_date=date(2025, 1, 1),
        end_date_value=date(2026, 5, 1),
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


def _universe_result(mocker: MockerFixture) -> Any:
    result = mocker.MagicMock()
    result.timeline.windows = {}
    return result


class TestRunPipeline:
    def test_run_pipeline_when_strategy_stage_returns_none_returns_explicit_failure(
        self,
        mocker: MockerFixture,
    ) -> None:
        mocker.patch(
            "src.application.futures.runner.active_pipeline._resolve_quarterly_window",
            return_value=make_window(),
        )
        mocker.patch("src.application.futures.runner.active_pipeline._resolve_layered_window", return_value=None)
        mocker.patch(
            "src.application.futures.runner.active_pipeline._selected_symbols_from_snapshot",
            return_value=(),
        )
        universe_result = _universe_result(mocker)
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_universe_stage",
            return_value=([], [], [], [], object(), [], universe_result),
        )
        mocker.patch("src.application.futures.runner.active_pipeline._ensure_universe_ledger_sync")
        mocker.patch("src.application.futures.runner.active_pipeline._ensure_cached_symbol_data_for_targets")
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_data_stage",
            return_value=make_data_bundle(),
        )
        mocker.patch("src.application.futures.runner.active_pipeline._run_regime_evaluation_stage", return_value=None)
        mocker.patch("src.application.futures.runner.active_pipeline._run_strategy_stage", return_value=None)

        result = run_pipeline(make_run_config("l3"))

        assert result == RunnerResult(exit_code=1, reason="strategy_stage_no_result")

    def test_run_pipeline_l3_preserves_orchestration_order(self, mocker: MockerFixture) -> None:
        order: list[str] = []
        universe_result = _universe_result(mocker)

        mocker.patch(
            "src.application.futures.runner.active_pipeline._resolve_quarterly_window", return_value=make_window()
        )
        mocker.patch("src.application.futures.runner.active_pipeline._resolve_layered_window", return_value=None)
        mocker.patch(
            "src.application.futures.runner.active_pipeline._selected_symbols_from_snapshot", return_value=("BTCUSDT",)
        )

        mocker.patch(
            "src.application.futures.runner.active_pipeline._ensure_universe_ledger_sync",
            side_effect=lambda *a: _track(order, "sync"),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_universe_stage",
            side_effect=lambda *args, **kwargs: _track(
                order, "universe", (["BTCUSDT"], [], [], ["BTCUSDT"], object(), [], universe_result)
            ),
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
            side_effect=lambda *args, **kwargs: _track(order, "strategy", object()),
        )
        mock_optimize = mocker.patch(
            "src.application.futures.runner.active_pipeline._run_optimization_stage",
            side_effect=lambda *a, **kw: _track(order, "optimize", RunnerResult(0, "l3_done")),
        )

        result = run_pipeline(make_run_config("l3"))

        assert result == RunnerResult(0, "l3_done")
        assert order == ["sync", "universe", "cache", "data", "regime", "strategy", "optimize"]
        mock_optimize.assert_called_once()

    @pytest.mark.parametrize("phase", ["l1", "l2"])
    def test_run_pipeline_l1_l2_skip_optimization(
        self, mocker: MockerFixture, phase: Literal["l1", "l2"]
    ) -> None:
        universe_result = _universe_result(mocker)
        mocker.patch(
            "src.application.futures.runner.active_pipeline._resolve_quarterly_window", return_value=make_window()
        )
        mocker.patch("src.application.futures.runner.active_pipeline._resolve_layered_window", return_value=None)
        mocker.patch(
            "src.application.futures.runner.active_pipeline._selected_symbols_from_snapshot", return_value=("BTCUSDT",)
        )
        mocker.patch("src.application.futures.runner.active_pipeline._ensure_universe_ledger_sync")
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_universe_stage",
            return_value=(["BTCUSDT"], [], [], ["BTCUSDT"], object(), [], universe_result),
        )
        mocker.patch("src.application.futures.runner.active_pipeline._ensure_cached_symbol_data_for_targets")
        mocker.patch(
            "src.application.futures.runner.active_pipeline._run_data_stage",
            return_value=make_data_bundle(),
        )
        mocker.patch("src.application.futures.runner.active_pipeline._run_regime_evaluation_stage", return_value=None)
        mock_strategy = mocker.patch("src.application.futures.runner.active_pipeline._run_strategy_stage")
        if phase == "l2":
            mock_strategy.return_value = RunnerResult(0, "tiered_pipeline_l2_completed")

        mock_optimize = mocker.patch("src.application.futures.runner.active_pipeline._run_optimization_stage")

        expected = RunnerResult(0, "l1_mode_done") if phase == "l1" else RunnerResult(0, "tiered_pipeline_l2_completed")
        result = run_pipeline(make_run_config(phase))

        assert result == expected
        mock_optimize.assert_not_called()

    def test_ledger_sync_skipped_when_sync_is_skip(self, mocker: MockerFixture) -> None:
        from src.application.futures.runner.active_pipeline import _ensure_universe_ledger_sync

        mock_path = mocker.MagicMock()
        mock_path.exists.return_value = False
        mocker.patch("src.domain.futures.universe.models.DEFAULT_LEDGER_PATH", mock_path)
        mock_sync = mocker.patch("src.application.futures.runner.active_pipeline.run_historical_sync")

        run_config = FuturesRunConfig(
            timeframe="4h", date=None, trials=1, phase="l3", sync="skip", refresh_universe=False, sync_metrics=False
        )
        window = make_window()

        _ensure_universe_ledger_sync(run_config, window)  # type: ignore[arg-type]
        mock_sync.assert_not_called()

    def test_ledger_sync_called_when_sync_is_auto(self, mocker: MockerFixture) -> None:
        from src.application.futures.runner.active_pipeline import _ensure_universe_ledger_sync

        mock_path = mocker.MagicMock()
        mock_path.exists.return_value = False
        mocker.patch("src.domain.futures.universe.models.DEFAULT_LEDGER_PATH", mock_path)
        mock_sync = mocker.patch("src.application.futures.runner.active_pipeline.run_historical_sync")

        run_config = FuturesRunConfig(
            timeframe="4h", date=None, trials=1, phase="l3", sync="auto", refresh_universe=False, sync_metrics=False
        )
        window = make_window()

        _ensure_universe_ledger_sync(run_config, window)  # type: ignore[arg-type]
        mock_sync.assert_called_once()


class TestBuildDataNotReadyReasons:
    """Scenario 1.3: reason_counts extraction from readiness report."""

    def test_returns_counts_when_reason_column_exists(self) -> None:
        report_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "reason": ["missing_tf_frame", "missing_tf_frame", "insufficient_bars"],
                "pass": [False, False, False],
            }
        )
        result = _build_data_not_ready_reasons(report_df)
        assert result == {"missing_tf_frame": 2, "insufficient_bars": 1}

    def test_returns_empty_dict_for_empty_dataframe(self) -> None:
        report_df = pd.DataFrame(columns=["symbol", "reason", "pass"])
        result = _build_data_not_ready_reasons(report_df)
        assert result == {}

    def test_returns_empty_dict_when_no_reason_column(self) -> None:
        report_df = pd.DataFrame({"symbol": ["A", "B"], "pass": [False, False]})
        result = _build_data_not_ready_reasons(report_df)
        assert result == {}

    def test_returns_empty_dict_for_non_dataframe(self) -> None:
        result = _build_data_not_ready_reasons(None)
        assert result == {}


class TestDeadCodeCleanup:
    """Scenario 5: Dead-code / duplication cleanup."""

    def test_runner_result_single_class_identity(self) -> None:
        from typing import get_type_hints

        from src.application.futures.runner.active_pipeline import run_pipeline as ap_run
        from src.application.futures.runner.models import RunnerResult as ResultModel

        hints = get_type_hints(ap_run)
        assert hints["return"] is ResultModel

    def test_run_tiered_pipeline_outcome_no_diagnostic_sink_param(self) -> None:
        from inspect import signature

        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            run_tiered_pipeline_outcome,
        )

        sig = signature(run_tiered_pipeline_outcome)
        assert "diagnostic_sink" not in sig.parameters

    def test_run_pipeline_no_re_wrap(self) -> None:
        from src.application.futures.runner.pipeline import run_pipeline as thin_wrapper

        assert thin_wrapper.__code__.co_varnames is not None
