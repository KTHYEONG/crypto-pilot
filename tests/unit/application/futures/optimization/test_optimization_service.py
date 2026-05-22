from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import optuna
from optuna.trial import TrialState

from src.application.futures.optimization import optimization_service
from src.domain.futures.optimization.phase_runner import PhaseBundle


def _build_request() -> optimization_service.OptimizationRequest:
    return optimization_service.OptimizationRequest(
        data_maps={},
        symbols=["BTCUSDT"],
        tf="4h",
        fetch_start="2025-01-01",
        is_start="2025-06-01",
        end_date="2026-01-01",
        run_id="run-1",
        study_name="study-1",
        storage_url="sqlite:///logs/test.db",
        storage=cast(optuna.storages.RDBStorage, object()),
        total_trials=7,
        ml_n_jobs=3,
        seed=11,
        resume=True,
        strategy_mode=True,
        n_workers_b=1,
    )


def _build_phase_bundle(study_b: Any) -> PhaseBundle:
    return PhaseBundle(
        study_a1=cast(optuna.Study, object()),
        study_a2=cast(optuna.Study, object()),
        study_b=cast(optuna.Study, study_b),
        study_names={"phase_a1": "a1", "phase_a2": "a2", "phase_b": "b"},
    )


def test_prepare_optimization_context_calls_precompute(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    def _fake_precompute(ctx: Any) -> None:
        called["ctx"] = ctx

    monkeypatch.setattr(
        optimization_service,
        "precompute_ml_optimization_context",
        _fake_precompute,
    )
    request = _build_request()

    ctx = optimization_service.prepare_optimization_context(request)

    assert called["ctx"] is ctx
    assert ctx.run_id == "run-1"
    assert ctx.effective_total_trials == 7
    assert ctx.strategy_mode is True


def test_execute_phase_skeleton_wires_budget_and_workers(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    fake_bundle = _build_phase_bundle(study_b=SimpleNamespace())

    def _fake_run_phase(**kwargs: Any) -> PhaseBundle:
        captured.update(kwargs)
        return fake_bundle

    monkeypatch.setattr(
        optimization_service,
        "precompute_ml_optimization_context",
        lambda _ctx: None,
    )
    monkeypatch.setattr(
        optimization_service,
        "run_v43_phase_optimization_skeleton",
        _fake_run_phase,
    )
    request = _build_request()
    base_ctx = optimization_service.prepare_optimization_context(request)

    result = optimization_service.execute_phase_skeleton(request, base_ctx=base_ctx)

    assert result is fake_bundle
    assert captured["n_trials"] == 7
    assert captured["n_workers"] == 3
    assert captured["n_workers_b"] == 1
    assert captured["target_seeds"] == [11]


def test_extract_best_trial_filters_complete_then_selects(monkeypatch: Any) -> None:
    selected = SimpleNamespace(state=TrialState.COMPLETE, value=2.0, user_attrs={}, params={})
    complete_other = SimpleNamespace(state=TrialState.COMPLETE, value=1.0, user_attrs={}, params={})
    failed = SimpleNamespace(state=TrialState.FAIL, value=None, user_attrs={}, params={})
    captured: dict[str, Any] = {}

    class _Study:
        best_trial = complete_other

        def get_trials(self, deepcopy: bool = False) -> list[Any]:
            assert deepcopy is False
            return [failed, complete_other, selected]

    def _fake_selector(trials: list[Any]) -> Any:
        captured["trials"] = trials
        return selected

    monkeypatch.setattr(
        optimization_service,
        "select_best_trial_by_holdout_log_ret",
        _fake_selector,
    )

    best = optimization_service.extract_best_trial(cast(optuna.Study, _Study()))

    assert best is selected
    assert captured["trials"] == [complete_other, selected]


def test_run_optimization_orchestrates_service_contract(monkeypatch: Any) -> None:
    request = _build_request()
    ctx = optimization_service.MLPhaseDContext(data_maps={}, symbols=[], tf="4h")
    study_b = SimpleNamespace()
    bundle = _build_phase_bundle(study_b=study_b)
    best_trial = SimpleNamespace(state=TrialState.COMPLETE, value=1.0, user_attrs={}, params={})

    monkeypatch.setattr(optimization_service, "prepare_optimization_context", lambda _: ctx)
    monkeypatch.setattr(
        optimization_service,
        "execute_phase_skeleton",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(optimization_service, "extract_best_trial", lambda _study: best_trial)

    result = optimization_service.run_optimization(request)

    assert result.base_ctx is ctx
    assert result.phase_bundle is bundle
    assert result.study_ml is study_b
    assert result.best_trial is best_trial


def test_run_final_evaluation_wraps_domain_call(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    params_called: dict[str, Any] = {}
    best_trial = SimpleNamespace(params={"K_LONG": 2}, user_attrs={})

    def _fake_build(params: dict[str, Any], tf: str) -> dict[str, Any]:
        params_called["params"] = params
        params_called["tf"] = tf
        return {"TIMEFRAME": tf, "K_LONG": 2}

    def _fake_final_eval(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(optimization_service, "build_ml_phase_d_params", _fake_build)
    monkeypatch.setattr(optimization_service, "run_final_oos_evaluation", _fake_final_eval)

    req = optimization_service.FinalEvaluationRequest(
        tf="4h",
        project_root="/home/kth/my_coin_traider",
        study_ml=cast(optuna.Study, object()),
        run_id="run-1",
        ml_ctx=optimization_service.MLPhaseDContext(data_maps={}, symbols=[], tf="4h"),
        n_ml_trials=9,
        target_seeds=[11],
        selected_ops_profile="active",
        pbo_gate=0.5,
        dsr_gate=0.2,
        pbo_obs=0.1,
        dsr_obs=0.1,
        best_trial=cast(optuna.trial.FrozenTrial, best_trial),
        champ_stab_cv=0.0,
        stab_tmp_layer3_awf_fail=False,
        cv_max=0.3,
    )
    optimization_service.run_final_evaluation(req)

    assert params_called["params"] == {"K_LONG": 2}
    assert params_called["tf"] == "4h"
    assert captured["args"].tf == "4h"
    assert captured["params"] == {"TIMEFRAME": "4h", "K_LONG": 2}
