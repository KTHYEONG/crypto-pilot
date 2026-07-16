from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.application.futures.runner.models import RunnerResult
from src.domain.futures.strategy.tiered_workflow.cross_tf_diagnostics import STAGE_ORDER


@dataclass
class _FakeCandidate:
    l1_tfs: tuple[str, ...]


@dataclass
class _FakeConfig:
    candidate: _FakeCandidate


@dataclass
class _FakeFold:
    fit_start: int
    fit_end: int
    oos_start: int
    oos_end: int


@dataclass
class _FakeRoute:
    timeframe: str
    selected_recipe_ids: tuple[str, ...]
    allocated_budget_units: float


@dataclass
class _FakeManifest:
    routes: tuple[_FakeRoute, ...]


@dataclass
class _FakeBridgeOutput:
    labeled_events_by_tf: dict[str, pd.DataFrame]
    l0_delivery_manifest: _FakeManifest


def _wire_closure_exercising_mocks(mocker: MockerFixture) -> None:
    """Make the mocked run_pipeline() actually call into every monkeypatched
    module attribute (build_override/phase3_capture/multi_override/
    bridge_override/select_capture/per_tf_capture) so their bodies execute
    instead of staying dead code under mock.patch of run_pipeline alone."""
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.strategy_service.build_candidate_strategy_config",
        return_value=_FakeConfig(candidate=_FakeCandidate(l1_tfs=("4h",))),
    )
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.active_pipeline.build_candidate_strategy_config",
        return_value=_FakeConfig(candidate=_FakeCandidate(l1_tfs=("4h",))),
    )
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.bridge_helpers.run_alpha_foundry_l0_gate_multi_tf",
        return_value={},
    )
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.bridge_helpers._run_phase3_sequential",
        return_value={},
    )

    def _fake_original_bridge(**kwargs: Any) -> _FakeBridgeOutput:
        from src.domain.futures.strategy_runtime import bridge

        cast(Any, bridge)._build_single_tf_panels(tf_i="4h")
        return _FakeBridgeOutput(
            labeled_events_by_tf={"4h": pd.DataFrame({"a": [1]})},
            l0_delivery_manifest=_FakeManifest(
                routes=(_FakeRoute(timeframe="4h", selected_recipe_ids=("r1",), allocated_budget_units=1.0),)
            ),
        )

    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.strategy_service.run_candidate_strategy_for_universe",
        side_effect=_fake_original_bridge,
    )
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.bridge._build_single_tf_panels",
        return_value=(None, None, []),
    )
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.select_l1_delivery_events",
        return_value=pd.DataFrame({"a": [1]}),
    )
    mock_per_tf_result = mocker.MagicMock()
    mock_per_tf_result.event_grid_audit = mocker.MagicMock()
    mock_per_tf_result.event_grid_audit.n_dropped = 3
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.run_per_tf_l1",
        return_value=mock_per_tf_result,
    )

    def _fake_run_pipeline(*args: Any, **kwargs: Any) -> RunnerResult:
        from src.application.futures.optimization import strategy_service
        from src.domain.futures.alpha_foundry import bridge_helpers
        from src.domain.futures.strategy.tiered_workflow import pipeline as tiered_pipeline

        cast(Any, strategy_service).build_candidate_strategy_config(
            strategy_cfg=None, opt_config=None, timeframe="4h"
        )
        cast(Any, bridge_helpers).run_alpha_foundry_l0_gate_multi_tf()
        cast(Any, bridge_helpers)._run_phase3_sequential(evidence_by_tf={})
        cast(Any, strategy_service).run_candidate_strategy_for_universe()
        cast(Any, tiered_pipeline).select_l1_delivery_events(tf="4h")
        cast(Any, tiered_pipeline).run_per_tf_l1(tf="4h", outer_folds=(_FakeFold(0, 4, 4, 6),))
        return RunnerResult(exit_code=0, reason="l1_mode_done")

    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.active_pipeline.run_pipeline",
        side_effect=_fake_run_pipeline,
    )
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.build_effective_run_config",
        return_value=cast(Any, object()),
    )


def _default_mocks(mocker: MockerFixture) -> None:
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.active_pipeline.run_pipeline",
        return_value=RunnerResult(exit_code=0, reason="l1_mode_done"),
    )
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.build_effective_run_config",
        return_value=cast(Any, object()),
    )
    mock_per_tf = mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.run_per_tf_l1")
    mock_per_tf_result = mocker.MagicMock()
    mock_per_tf_result.event_grid_audit = mocker.MagicMock()
    mock_per_tf_result.event_grid_audit.n_dropped = 0
    mock_per_tf.return_value = mock_per_tf_result
    mocker.patch(
        "src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.select_l1_delivery_events",
        return_value=pd.DataFrame(),
    )


class TestRunOnce:
    """Scenario 1 (Happy Path): run_once returns RunnerResult and populates trace."""

    def test_run_once_persists_runner_result_and_all_ten_stages(self, mocker: MockerFixture) -> None:
        _default_mocks(mocker)
        from src.domain.futures.strategy.run_l1_cross_tf_replay import run_once

        trace: dict[str, dict[str, dict[str, object]]] = {stage: {} for stage in STAGE_ORDER}
        result = run_once(label="control", tfs=("2h",), ablate_1h_fusion=False, trace=trace)

        assert result == RunnerResult(exit_code=0, reason="l1_mode_done")
        assert trace.get("runner_result") == {"exit_code": 0, "reason": "l1_mode_done"}
        assert set(STAGE_ORDER).issubset(trace.keys())

    def test_run_once_exercises_all_stage_capture_closures(self, mocker: MockerFixture) -> None:
        """[LIMIT-01] Verify build_override/phase3_capture/multi_override/bridge_override/
        select_capture/per_tf_capture actually populate trace with real captured values,
        not just the pre-seeded empty-dict keys."""
        _wire_closure_exercising_mocks(mocker)
        from src.domain.futures.strategy.run_l1_cross_tf_replay import run_once

        trace: dict[str, dict[str, dict[str, object]]] = {stage: {} for stage in STAGE_ORDER}
        result = run_once(label="control", tfs=("4h",), ablate_1h_fusion=False, trace=trace)

        assert result == RunnerResult(exit_code=0, reason="l1_mode_done")
        assert trace["native_panels"]["4h"]["count"] == 0
        assert trace["native_labeled_events"]["4h"]["count"] == 1
        assert trace["manifest_route"]["4h"]["count"] == 1
        assert trace["l1_delivery_events"]["4h"]["count"] == 1
        assert trace["outer_folds"]["4h"] == {
            "count": 1,
            "digest": mocker.ANY,
        }
        assert trace["terminal_event_audit"]["4h"]["count"] == 3
        assert isinstance(trace["l1_result"]["4h"]["gate_passed"], bool)

    def test_run_once_with_layer1_blocked(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.active_pipeline.run_pipeline",
            return_value=RunnerResult(exit_code=1, reason="layer1_blocked"),
        )
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.build_effective_run_config",
            return_value=cast(Any, object()),
        )
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.run_per_tf_l1")
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.select_l1_delivery_events")

        from src.domain.futures.strategy.run_l1_cross_tf_replay import run_once

        trace: dict[str, dict[str, dict[str, object]]] = {stage: {} for stage in STAGE_ORDER}
        result = run_once(label="control", tfs=("2h",), ablate_1h_fusion=False, trace=trace)

        assert result == RunnerResult(exit_code=1, reason="layer1_blocked")
        assert trace.get("runner_result") == {"exit_code": 1, "reason": "layer1_blocked"}

    def test_run_once_with_l0_gate_no_delivery(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.active_pipeline.run_pipeline",
            return_value=RunnerResult(exit_code=1, reason="l0_gate_no_delivery"),
        )
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.build_effective_run_config",
            return_value=cast(Any, object()),
        )
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.run_per_tf_l1")
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.select_l1_delivery_events")

        from src.domain.futures.strategy.run_l1_cross_tf_replay import run_once

        trace: dict[str, dict[str, dict[str, object]]] = {stage: {} for stage in STAGE_ORDER}
        result = run_once(label="control", tfs=("2h",), ablate_1h_fusion=False, trace=trace)

        assert result == RunnerResult(exit_code=1, reason="l0_gate_no_delivery")
        assert trace.get("runner_result") == {"exit_code": 1, "reason": "l0_gate_no_delivery"}


class TestMain:
    """Scenario 1 (exit code) and Scenario 3 (RunnerResult fidelity)."""

    def test_main_returns_zero_when_runner_result_exit_code_zero(self, mocker: MockerFixture, tmp_path: Path) -> None:
        _default_mocks(mocker)
        mocker.patch("sys.argv", ["prog", "control"])
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.Path",
            side_effect=lambda *a: tmp_path / "/".join(a) if a else Path(*a),
        )
        from src.domain.futures.strategy.run_l1_cross_tf_replay import main

        exit_code = main()
        assert exit_code == 0

    def test_main_returns_one_when_runner_result_exit_code_one(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.active_pipeline.run_pipeline",
            return_value=RunnerResult(exit_code=1, reason="layer1_blocked"),
        )
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.build_effective_run_config",
            return_value=cast(Any, object()),
        )
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.run_per_tf_l1")
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.select_l1_delivery_events")
        mocker.patch("sys.argv", ["prog", "control"])

        from src.domain.futures.strategy.run_l1_cross_tf_replay import main

        exit_code = main()
        assert exit_code == 1

    def test_main_returns_two_for_bad_label(self, mocker: MockerFixture) -> None:
        mocker.patch("sys.argv", ["prog", "invalid_label"])

        from src.domain.futures.strategy.run_l1_cross_tf_replay import main

        exit_code = main()
        assert exit_code == 2

    def test_main_persists_partial_trace_on_exception(self, mocker: MockerFixture, tmp_path: Path) -> None:
        mock_run_once = mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.run_once")
        mock_run_once.side_effect = ValueError("test crash at tf 2 of 3")
        mocker.patch("sys.argv", ["prog", "control"])
        from src.domain.futures.strategy.run_l1_cross_tf_replay import main

        with pytest.raises(ValueError, match="test crash at tf 2 of 3"):
            main()

    def test_main_returns_one_for_layer1_blocked(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.active_pipeline.run_pipeline",
            return_value=RunnerResult(exit_code=1, reason="layer1_blocked"),
        )
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_replay.build_effective_run_config",
            return_value=cast(Any, object()),
        )
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.run_per_tf_l1")
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_replay.tiered_pipeline.select_l1_delivery_events")
        mocker.patch("sys.argv", ["prog", "control"])

        from src.domain.futures.strategy.run_l1_cross_tf_replay import main

        exit_code = main()
        assert exit_code == 1
