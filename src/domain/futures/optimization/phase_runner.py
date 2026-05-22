from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace
from typing import Any

import optuna
from optuna.trial import TrialState

from config.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.optimization.optimizer import MLPhaseDContext
from src.domain.futures.optimization.phase_c_robustness import (
    evaluate_phase_c_robustness,
)
from src.domain.futures.optimization.phase_importance import PhaseBPlan, build_phase_b_plan
from src.domain.futures.optimization.phase_objectives import build_phase_objective_specs
from src.domain.futures.optimization.phase_samplers import (
    build_phase_a1_pruner,
    build_phase_a1_sampler,
    build_phase_a2_sampler,
    build_phase_b_sampler,
    build_phase_study_name,
)
from src.domain.futures.optimization.run_tracker import run_optimization_loop

_PHASE_OBJECTIVES = build_phase_objective_specs()


@dataclass
class PhaseBundle:
    study_a1: optuna.Study
    study_a2: optuna.Study
    study_b: optuna.Study
    study_names: dict[str, str]
    phase_b_plan: PhaseBPlan | None = None
    phase_c_diagnostics: dict[str, Any] | None = None


def _tag_phase_trials(study: optuna.Study, *, phase: str, run_id: str | None) -> None:
    try:
        storage = study._storage
        study_id = study._study_id
    except Exception:
        return
    for tr in study.get_trials(deepcopy=False):
        if tr.state != TrialState.COMPLETE:
            continue
        try:
            trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, tr.number)
            storage.set_trial_user_attr(trial_id, "phase", phase)
            if run_id:
                storage.set_trial_user_attr(trial_id, "run_id", run_id)
        except Exception:
            continue


def _ctx_for_phase(
    base_ctx: MLPhaseDContext,
    *,
    phase: str,
    frozen_params: dict[str, Any] | None = None,
    phase_ranges: dict[str, tuple[Any, Any]] | None = None,
) -> MLPhaseDContext:
    inherited_frozen = dict(getattr(base_ctx, "coordinate_frozen_params", None) or {})
    inherited_frozen.update(dict(frozen_params or {}))
    inherited_ranges = dict(getattr(base_ctx, "phase_ranges", None) or {})
    inherited_ranges.update(dict(phase_ranges or {}))
    if is_dataclass(base_ctx):
        return replace(
            base_ctx,
            coordinate_phase=phase,
            coordinate_frozen_params=inherited_frozen,
            phase_ranges=inherited_ranges,
        )
    payload = dict(getattr(base_ctx, "__dict__", {}))
    payload["coordinate_phase"] = phase
    payload["coordinate_frozen_params"] = inherited_frozen
    payload["phase_ranges"] = inherited_ranges
    return SimpleNamespace(**payload)


def _best_complete_trial(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    if not hasattr(study, "get_trials"):
        return None
    completed = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    if not completed:
        return None
    return max(completed, key=lambda t: float(t.value) if t.value is not None else -1e18)


def _best_a2_trial_for_risk(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    if not hasattr(study, "get_trials"):
        return None
    completed = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    if not completed:
        return None
    with_sortino = [t for t in completed if t.user_attrs.get("sortino_lcb") is not None]
    if with_sortino:
        return max(with_sortino, key=lambda t: float(t.user_attrs.get("sortino_lcb", -1e18)))
    return completed[0]


def _core_subset(params: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: params[k] for k in keys if k in params}


def run_phase_a1(
    *,
    base_ctx: MLPhaseDContext,
    base_study_name: str,
    storage_url: str,
    storage: optuna.storages.RDBStorage,
    n_trials: int,
    seed: int,
    resume: bool,
    n_workers: int,
) -> optuna.Study:
    spec = _PHASE_OBJECTIVES["phase_a1"]
    ctx_a1 = _ctx_for_phase(base_ctx, phase="phase_a1")
    study = run_optimization_loop(
        base_ctx=ctx_a1,
        study_name=build_phase_study_name(base_study_name, "phase_a1"),
        storage_url=storage_url,
        storage=storage,
        n_trials=n_trials,
        seed=seed,
        resume=resume,
        n_workers=n_workers,
        sampler=build_phase_a1_sampler(seed),
        pruner=build_phase_a1_pruner(),
        objective_fn=spec.objective,
        directions=spec.directions,
        phase_label="Phase 1/3: Signal Quality (A1)",
    )
    _tag_phase_trials(study, phase="phase_a1", run_id=getattr(base_ctx, "run_id", None))
    return study


def run_phase_a2(
    *,
    base_ctx: MLPhaseDContext,
    base_study_name: str,
    storage_url: str,
    storage: optuna.storages.RDBStorage,
    n_trials: int,
    seed: int,
    resume: bool,
    n_workers: int,
    frozen_signal_params: dict[str, Any] | None = None,
) -> optuna.Study:
    spec = _PHASE_OBJECTIVES["phase_a2"]
    ctx_a2 = _ctx_for_phase(
        base_ctx,
        phase="phase_a2",
        frozen_params=frozen_signal_params,
    )
    study = run_optimization_loop(
        base_ctx=ctx_a2,
        study_name=build_phase_study_name(base_study_name, "phase_a2"),
        storage_url=storage_url,
        storage=storage,
        n_trials=n_trials,
        seed=seed,
        resume=resume,
        n_workers=n_workers,
        sampler=build_phase_a2_sampler(seed),
        pruner=build_phase_a1_pruner(),
        objective_fn=spec.objective,
        directions=spec.directions,
        phase_label="Phase 2/3: Risk Management (A2)",
    )
    _tag_phase_trials(study, phase="phase_a2", run_id=getattr(base_ctx, "run_id", None))
    return study


def run_phase_b(
    *,
    base_ctx: MLPhaseDContext,
    base_study_name: str,
    storage_url: str,
    storage: optuna.storages.RDBStorage,
    n_trials: int,
    seed: int,
    resume: bool,
    n_workers: int,
    enqueue_seeds: list[dict[str, Any]] | None = None,
    frozen_params: dict[str, Any] | None = None,
    phase_ranges: dict[str, tuple[Any, Any]] | None = None,
) -> optuna.Study:
    spec = _PHASE_OBJECTIVES["phase_b"]
    ctx_b = _ctx_for_phase(
        base_ctx,
        phase="phase_b",
        frozen_params=frozen_params,
        phase_ranges=phase_ranges,
    )
    study = run_optimization_loop(
        base_ctx=ctx_b,
        study_name=build_phase_study_name(base_study_name, "phase_b"),
        storage_url=storage_url,
        storage=storage,
        n_trials=n_trials,
        seed=seed,
        resume=resume,
        n_workers=n_workers,
        sampler=build_phase_b_sampler(seed),
        pruner=build_phase_a1_pruner(),
        enqueue_params=enqueue_seeds,
        objective_fn=spec.objective,
        directions=spec.directions,
        phase_label="Phase 3/3: Portfolio Robustness (B)",
    )
    _tag_phase_trials(study, phase="phase_b", run_id=getattr(base_ctx, "run_id", None))
    return study


def run_v43_phase_optimization_skeleton(
    *,
    base_ctx: MLPhaseDContext,
    base_study_name: str,
    storage_url: str,
    storage: optuna.storages.RDBStorage,
    n_trials: int,
    seed: int,
    resume: bool,
    n_workers: int,
    n_workers_a1: int | None = None,
    n_workers_a2: int | None = None,
    n_workers_b: int | None = None,
    enqueue_seeds: list[dict[str, Any]] | None = None,
    target_seeds: list[int] | None = None,
    n_trials_a1: int | None = None,
    n_trials_a2: int | None = None,
    n_trials_b: int | None = None,
) -> PhaseBundle:
    phase_a1_workers = int(n_workers_a1) if n_workers_a1 is not None else int(n_workers)
    phase_a2_workers = int(n_workers_a2) if n_workers_a2 is not None else int(n_workers)
    phase_b_workers = int(n_workers_b) if n_workers_b is not None else int(n_workers)
    phase_a1_trials = int(
        n_trials_a1
        if n_trials_a1 is not None
        else OPT_FUTURES_CONFIG.get("FUTURES_V43_PHASE_A1_TRIALS", 150)
    )
    phase_a2_trials = int(
        n_trials_a2
        if n_trials_a2 is not None
        else OPT_FUTURES_CONFIG.get("FUTURES_V43_PHASE_A2_TRIALS", 100)
    )
    phase_b_trials = int(
        n_trials_b
        if n_trials_b is not None
        else OPT_FUTURES_CONFIG.get("FUTURES_V43_PHASE_B_TRIALS", 300)
    )
    if n_trials > 0 and n_trials_a1 is None and n_trials_a2 is None and n_trials_b is None:
        # Backward compatibility for callers that still expect shared budget.
        phase_a1_trials = int(n_trials)
        phase_a2_trials = int(n_trials)
        phase_b_trials = int(n_trials)

    study_a1 = run_phase_a1(
        base_ctx=base_ctx,
        base_study_name=base_study_name,
        storage_url=storage_url,
        storage=storage,
        n_trials=phase_a1_trials,
        seed=seed,
        resume=resume,
        n_workers=phase_a1_workers,
    )
    best_a1_trial_for_a2 = _best_complete_trial(study_a1)
    frozen_signal_for_a2 = _core_subset(
        (best_a1_trial_for_a2.params if best_a1_trial_for_a2 is not None else {}),
        (
            "BETA_ALPHA",
            "BETA_REGIME_BEAR",
            "BETA_REGIME_CHOP",
            "K_LONG",
            "K_SHORT",
            "REBALANCE_BARS",
            "EV_HURDLE_BPS",
        ),
    )
    study_a2 = run_phase_a2(
        base_ctx=base_ctx,
        base_study_name=base_study_name,
        storage_url=storage_url,
        storage=storage,
        n_trials=phase_a2_trials,
        seed=seed,
        resume=resume,
        n_workers=phase_a2_workers,
        frozen_signal_params=frozen_signal_for_a2,
    )
    try:
        phase_b_plan = build_phase_b_plan(study_a1, study_a2)
    except Exception:
        phase_b_plan = None
    phase_b_seeds = enqueue_seeds
    if phase_b_seeds is None and phase_b_plan is not None:
        phase_b_seeds = phase_b_plan.seed_combos
    best_a1_trial = _best_complete_trial(study_a1)
    best_signal = _core_subset(
        (best_a1_trial.params if best_a1_trial is not None else {}),
        (
            "BETA_ALPHA",
            "BETA_REGIME_BEAR",
            "BETA_REGIME_CHOP",
            "K_LONG",
            "K_SHORT",
            "REBALANCE_BARS",
            "EV_HURDLE_BPS",
        ),
    )
    best_a2_trial = _best_a2_trial_for_risk(study_a2)
    best_risk = _core_subset(
        (best_a2_trial.params if best_a2_trial is not None else {}),
        (
            "PORTFOLIO_KAPPA",
            "TARGET_ANN_VOL",
            "MAX_EXPOSURE",
            "MAX_EXPOSURE_PER_COIN",
        ),
    )
    phase_b_frozen: dict[str, Any] = {}
    phase_b_frozen.update(best_signal)
    phase_b_frozen.update(best_risk)
    if phase_b_plan is not None:
        phase_b_frozen.update(dict(phase_b_plan.fixed_params))
    study_b = run_phase_b(
        base_ctx=base_ctx,
        base_study_name=base_study_name,
        storage_url=storage_url,
        storage=storage,
        n_trials=phase_b_trials,
        seed=seed,
        resume=resume,
        n_workers=phase_b_workers,
        enqueue_seeds=phase_b_seeds,
        frozen_params=phase_b_frozen,
        phase_ranges=(phase_b_plan.shrunk_ranges if phase_b_plan is not None else None),
    )
    if phase_b_plan is not None:
        try:
            study_b.set_user_attr("phase_b_plan", phase_b_plan.importance_report)
        except Exception:
            pass
    phase_c_diag = evaluate_phase_c_robustness(
        study_b=study_b,
        target_seeds=target_seeds or [seed],
        top_k=5,
    )
    return PhaseBundle(
        study_a1=study_a1,
        study_a2=study_a2,
        study_b=study_b,
        study_names={
            "phase_a1": study_a1.study_name,
            "phase_a2": study_a2.study_name,
            "phase_b": study_b.study_name,
        },
        phase_b_plan=phase_b_plan,
        phase_c_diagnostics=phase_c_diag,
    )
