from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pytest

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.optimization.optimizer import MLPhaseDContext
from src.domain.futures.optimization.workflow import PhaseBPlan, run_phased_optimization_skeleton


@dataclass
class _DummyStudy:
    study_name: str


def _base_ctx(run_id: str, **kwargs: Any) -> MLPhaseDContext:
    return MLPhaseDContext(data_maps={}, symbols=[], tf="4h", run_id=run_id, **kwargs)


def test_phase_runner_study_naming_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_loop(**kwargs: Any) -> _DummyStudy:
        calls.append(kwargs)
        return _DummyStudy(study_name=kwargs["study_name"])

    monkeypatch.setattr("src.domain.futures.optimization.workflow.run_optimization_loop", _fake_loop)

    bundle = run_phased_optimization_skeleton(
        base_ctx=_base_ctx("r1"),
        base_study_name="base_study",
        storage_url="sqlite:///tmp.db",
        storage=optuna.storages.InMemoryStorage(),
        n_trials=3,
        seed=42,
        resume=True,
        n_workers=1,
        enqueue_seeds=[{"K_LONG": 3}],
    )

    assert [c["study_name"] for c in calls] == [
        "base_study_phase_a1",
        "base_study_phase_a2",
        "base_study_phase_b",
    ]
    assert calls[0]["directions"] == ("maximize",)
    assert calls[1]["directions"] == ("maximize", "minimize")
    assert calls[2]["directions"] == ("maximize",)
    assert callable(calls[0]["objective_fn"])
    assert callable(calls[1]["objective_fn"])
    assert callable(calls[2]["objective_fn"])
    assert bundle.study_names["phase_b"] == "base_study_phase_b"
    assert calls[0]["base_ctx"].coordinate_phase == "phase_a1"
    assert calls[1]["base_ctx"].coordinate_phase == "phase_a2"
    assert calls[2]["base_ctx"].coordinate_phase == "phase_b"


def test_phase_runner_uses_phase_specific_trial_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_loop(**kwargs: Any) -> _DummyStudy:
        calls.append(kwargs)
        return _DummyStudy(study_name=kwargs["study_name"])

    monkeypatch.setattr("src.domain.futures.optimization.workflow.run_optimization_loop", _fake_loop)

    run_phased_optimization_skeleton(
        base_ctx=_base_ctx("r3"),
        base_study_name="base_study",
        storage_url="sqlite:///tmp.db",
        storage=optuna.storages.InMemoryStorage(),
        n_trials=999,
        n_trials_a1=150,
        n_trials_a2=100,
        n_trials_b=300,
        seed=42,
        resume=True,
        n_workers=1,
        enqueue_seeds=None,
    )

    assert calls[0]["n_trials"] == 150
    assert calls[1]["n_trials"] == 100
    assert calls[2]["n_trials"] == 300


def test_phase_b_receives_enqueue_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_loop(**kwargs: Any) -> _DummyStudy:
        calls.append(kwargs)
        return _DummyStudy(study_name=kwargs["study_name"])

    monkeypatch.setattr("src.domain.futures.optimization.workflow.run_optimization_loop", _fake_loop)

    seeds = [{"K_LONG": 2, "TARGET_ANN_VOL": 0.12}]
    run_phased_optimization_skeleton(
        base_ctx=_base_ctx("r2"),
        base_study_name="joint",
        storage_url="sqlite:///tmp.db",
        storage=optuna.storages.InMemoryStorage(),
        n_trials=2,
        seed=7,
        resume=False,
        n_workers=1,
        enqueue_seeds=seeds,
    )

    phase_b_call = calls[2]
    assert phase_b_call["study_name"] == "joint_phase_b"
    assert phase_b_call["enqueue_params"] == seeds


def test_phase_runner_propagates_frozen_and_shrunk_to_phase_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class _DummyTrial:
        def __init__(
            self,
            number: int,
            value: float,
            params: dict[str, Any],
            attrs: dict[str, Any] | None = None,
        ) -> None:
            self.number = number
            self.value = value
            self.params = params
            self.user_attrs = attrs or {}
            self.state = __import__("optuna").trial.TrialState.COMPLETE

    class _DummyStudyWithTrials(_DummyStudy):
        def __init__(self, study_name: str, trials: list[Any]):
            super().__init__(study_name=study_name)
            self._trials = trials

        def get_trials(self, deepcopy: bool = False) -> list[Any]:
            return self._trials

        def set_user_attr(self, _k: str, _v: Any) -> None:
            return None

    def _fake_loop(**kwargs: Any) -> _DummyStudyWithTrials:
        calls.append(kwargs)
        sname = kwargs["study_name"]
        if sname.endswith("_phase_a1"):
            return _DummyStudyWithTrials(
                sname,
                [
                    _DummyTrial(
                        1,
                        1.0,
                        {
                            "BETA_REGIME_BEAR": 0.4,
                            "BETA_REGIME_CHOP": 0.2,
                            "K_LONG": 3,
                            "K_SHORT": 1,
                            "REBALANCE_BARS": 6,
                            "EV_HURDLE_BPS": 14.0,
                        },
                    )
                ],
            )
        if sname.endswith("_phase_a2"):
            return _DummyStudyWithTrials(
                sname,
                [
                    _DummyTrial(
                        2,
                        0.0,
                        {
                            "PORTFOLIO_KAPPA": 0.21,
                            "TARGET_ANN_VOL": 0.15,
                            "MAX_EXPOSURE": 1.1,
                            "MAX_EXPOSURE_PER_COIN": 0.2,
                        },
                        {"sortino_lcb": 1.9},
                    )
                ],
            )
        return _DummyStudyWithTrials(sname, [])

    monkeypatch.setattr("src.domain.futures.optimization.workflow.run_optimization_loop", _fake_loop)
    monkeypatch.setattr(
        "src.domain.futures.optimization.workflow.build_phase_b_plan",
        lambda *_a, **_k: PhaseBPlan(
            fixed_params={"K_SHORT": 2},
            shrunk_ranges={"TARGET_ANN_VOL": (0.1, 0.2)},
            seed_combos=[{"K_LONG": 3}],
            importance_report={},
        ),
    )

    run_phased_optimization_skeleton(
        base_ctx=_base_ctx("r4"),
        base_study_name="joint",
        storage_url="sqlite:///tmp.db",
        storage=optuna.storages.InMemoryStorage(),
        n_trials=2,
        seed=7,
        resume=False,
        n_workers=1,
        enqueue_seeds=None,
    )
    phase_b_call = calls[2]
    ctx_b = phase_b_call["base_ctx"]
    assert ctx_b.coordinate_frozen_params["K_LONG"] == 3
    assert ctx_b.coordinate_frozen_params["PORTFOLIO_KAPPA"] == 0.21
    assert ctx_b.coordinate_frozen_params["K_SHORT"] == 2
    assert ctx_b.phase_ranges["TARGET_ANN_VOL"] == (0.1, 0.2)


def test_phase_runner_inherits_base_phase_ranges_for_all_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class _DummyStudyWithTrials(_DummyStudy):
        def __init__(self, study_name: str):
            super().__init__(study_name=study_name)
            self._trials: list[Any] = []

        def get_trials(self, deepcopy: bool = False) -> list[Any]:
            return self._trials

        def set_user_attr(self, _k: str, _v: Any) -> None:
            return None

    def _fake_loop(**kwargs: Any) -> _DummyStudyWithTrials:
        calls.append(kwargs)
        return _DummyStudyWithTrials(kwargs["study_name"])

    monkeypatch.setattr("src.domain.futures.optimization.workflow.run_optimization_loop", _fake_loop)
    monkeypatch.setattr(
        "src.domain.futures.optimization.workflow.build_phase_b_plan",
        lambda *_a, **_k: PhaseBPlan(
            fixed_params={},
            shrunk_ranges={"TARGET_ANN_VOL": (0.1, 0.2)},
            seed_combos=[],
            importance_report={},
        ),
    )

    run_phased_optimization_skeleton(
        base_ctx=_base_ctx(
            "r5",
            strategy_mode=True,
            phase_ranges={
                "BETA_ALPHA": (4.0, 8.0),
                "EV_HURDLE_BPS": (1.0, 3.0),
                "REBALANCE_BARS": (4, 8),
            },
        ),
        base_study_name="joint",
        storage_url="sqlite:///tmp.db",
        storage=optuna.storages.InMemoryStorage(),
        n_trials=1,
        seed=7,
        resume=False,
        n_workers=1,
        enqueue_seeds=None,
    )

    for idx in range(3):
        ctx = calls[idx]["base_ctx"]
        assert ctx.phase_ranges["BETA_ALPHA"] == (4.0, 8.0)
        assert ctx.phase_ranges["EV_HURDLE_BPS"] == (1.0, 3.0)
        assert ctx.phase_ranges["REBALANCE_BARS"] == (4, 8)
    assert calls[2]["base_ctx"].phase_ranges["TARGET_ANN_VOL"] == (0.1, 0.2)
