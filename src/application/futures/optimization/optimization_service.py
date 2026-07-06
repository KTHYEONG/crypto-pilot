from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import StrategyConfig

import optuna
from optuna.trial import FrozenTrial, TrialState

from src.domain.futures.optimization.final_evaluator import run_final_oos_evaluation
from src.domain.futures.optimization.observability.trial_observability import (
    classify_no_valid_candidates,
)
from src.domain.futures.optimization.optimizer import (
    MLPhaseDContext,
    build_ml_phase_d_params,
    precompute_ml_optimization_context,
    select_best_trial_by_holdout_log_ret,
)
from src.domain.futures.optimization.workflow import (
    PhaseBundle,
    run_phased_optimization_skeleton,
)

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class OptimizationRequest:
    """Optimization orchestration contract for active runner."""

    data_maps: dict[str, dict[str, Any]]
    symbols: list[str]
    tf: str
    fetch_start: str
    is_start: str
    end_date: str
    run_id: str
    study_name: str
    storage_url: str
    storage: optuna.storages.BaseStorage
    total_trials: int
    ml_n_jobs: int
    seed: int = 42
    resume: bool = False
    strategy_mode: bool = True
    # strategy_cfg: strategy/alpha 실행에서 AWF leg refit에 사용
    strategy_cfg: StrategyConfig | None = None
    n_workers_b: int = 1
    enqueue_seeds: list[dict[str, Any]] | None = None
    target_seeds: list[int] | None = None
    n_trials_a1: int | None = None
    n_trials_a2: int | None = None
    n_trials_b: int | None = None


@dataclass(slots=True, frozen=True)
class OptimizationResult:
    """Output contract for optimization phase execution."""

    base_ctx: MLPhaseDContext
    phase_bundle: PhaseBundle
    study_ml: optuna.Study | None
    best_trial: FrozenTrial | None


@dataclass(slots=True, frozen=True)
class FinalEvaluationRequest:
    """Final OOS evaluation contract."""

    tf: str
    project_root: str
    study_ml: optuna.Study
    run_id: str
    ml_ctx: MLPhaseDContext
    n_ml_trials: int
    target_seeds: list[int]
    selected_ops_profile: str
    pbo_gate: float
    dsr_gate: float
    pbo_obs: float
    dsr_obs: float
    best_trial: FrozenTrial
    champ_stab_cv: float | None
    stab_tmp_layer3_awf_fail: bool
    cv_max: float
    phase_c_diagnostics: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    ensemble_results: list[Any] = field(default_factory=list)
    oos_data_maps: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_maps: dict[str, dict[str, Any]] = field(default_factory=dict)
    valid_symbols: list[str] = field(default_factory=list)
    champion_awf_diag: dict[str, Any] = field(default_factory=dict)
    ai_telemetry_payloads: list[dict[str, Any]] = field(default_factory=list)
    selection_summary: dict[str, Any] = field(default_factory=dict)
    run_summary_extras: dict[str, Any] = field(default_factory=dict)


def prepare_optimization_context(request: OptimizationRequest) -> MLPhaseDContext:
    """Build and precompute ML Phase-D context once per run."""
    base_ctx = MLPhaseDContext(
        data_maps=request.data_maps,
        symbols=request.symbols,
        tf=request.tf,
        seed=request.seed,
        effective_total_trials=int(request.total_trials),
        ml_pipeline_fetch_start=request.fetch_start,
        ml_pipeline_end=request.end_date,
        ml_pipeline_is_start=request.is_start,
        ml_pipeline_workers=request.ml_n_jobs,
        run_id=request.run_id,
        strategy_mode=request.strategy_mode,
        strategy_cfg=request.strategy_cfg,
    )
    precompute_ml_optimization_context(base_ctx)
    return base_ctx


def execute_phase_skeleton(
    request: OptimizationRequest,
    *,
    base_ctx: MLPhaseDContext,
) -> PhaseBundle:
    """Run phased optimization skeleton."""
    return run_phased_optimization_skeleton(
        base_ctx=base_ctx,
        base_study_name=request.study_name,
        storage_url=request.storage_url,
        storage=request.storage,
        n_trials=int(request.total_trials),
        n_trials_a1=request.n_trials_a1,
        n_trials_a2=request.n_trials_a2,
        n_trials_b=request.n_trials_b,
        seed=request.seed,
        resume=request.resume,
        n_workers=max(1, int(request.ml_n_jobs)),
        n_workers_b=max(1, int(request.n_workers_b)),
        enqueue_seeds=request.enqueue_seeds,
        target_seeds=request.target_seeds or [request.seed],
    )


def extract_best_trial(study: optuna.Study | None) -> FrozenTrial | None:
    """Select best completed trial with holdout-aware heuristic."""
    if study is None:
        return None
    completed = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    if not completed:
        return None
    try:
        return select_best_trial_by_holdout_log_ret(completed)
    except Exception as exc:
        _logger.debug("holdout selector failed; fallback to study.best_trial: %s", exc)
    try:
        return study.best_trial
    except Exception as exc:
        _logger.debug("study.best_trial unavailable; fallback to max(value): %s", exc)
        return max(
            completed,
            key=lambda trial: float(trial.value) if trial.value is not None else -1e18,
        )


def _log_phase_run_summary(study: optuna.Study, phase: str) -> None:
    """Log [RUN-SUMMARY] for a completed phase study."""
    all_trials = study.get_trials(deepcopy=False)
    completed = [t for t in all_trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in all_trials if t.state == TrialState.PRUNED]
    failed = [t for t in all_trials if t.state == TrialState.FAIL]
    verdict = classify_no_valid_candidates(
        selection_summary=None,
        completed_trials=completed,
        pruned_trials=pruned,
    )
    prune_reasons: dict[str, int] = {}
    for tr in pruned:
        reason = str((tr.user_attrs or {}).get("obs_reason", "unknown_pruned")).strip()
        if not reason:
            reason = "unknown_pruned"
        prune_reasons[reason] = int(prune_reasons.get(reason, 0)) + 1
    top_reasons = sorted(prune_reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
    reasons_str = ", ".join(f"{k}:{v}" for k, v in top_reasons) if top_reasons else "-"
    _logger.info(
        "[RUN-SUMMARY] phase=%s trials=%d complete=%d pruned=%d failed=%d verdict=%s prune_reasons={%s}",
        phase,
        len(all_trials),
        len(completed),
        len(pruned),
        len(failed),
        verdict,
        reasons_str,
    )


def run_optimization(request: OptimizationRequest) -> OptimizationResult:
    """Execute context preparation + phased optimization + best-trial selection."""
    base_ctx = prepare_optimization_context(request)
    phase_bundle = execute_phase_skeleton(request, base_ctx=base_ctx)
    _log_phase_run_summary(phase_bundle.study_a1, "phase_a1")
    _log_phase_run_summary(phase_bundle.study_a2, "phase_a2")
    _log_phase_run_summary(phase_bundle.study_b, "phase_b")
    study_ml = phase_bundle.study_b
    best_trial = extract_best_trial(study_ml)
    return OptimizationResult(
        base_ctx=base_ctx,
        phase_bundle=phase_bundle,
        study_ml=study_ml,
        best_trial=best_trial,
    )


def run_final_evaluation(request: FinalEvaluationRequest) -> None:
    """Thin wrapper around final evaluator with active runner contract."""
    params = (
        dict(request.params)
        if request.params is not None
        else build_ml_phase_d_params(dict(request.best_trial.params), request.tf)
    )
    run_final_oos_evaluation(
        ensemble_results=request.ensemble_results,
        oos_data_maps=request.oos_data_maps,
        data_maps=request.data_maps,
        valid_symbols=request.valid_symbols,
        champion_awf_diag=request.champion_awf_diag,
        args=argparse.Namespace(tf=request.tf),
        project_root=request.project_root,
        study_ml=request.study_ml,
        run_id=request.run_id,
        ai_telemetry_payloads=request.ai_telemetry_payloads,
        selection_summary=request.selection_summary,
        run_summary_extras=request.run_summary_extras,
        ml_ctx=request.ml_ctx,
        n_ml_trials=int(request.n_ml_trials),
        target_seeds=request.target_seeds,
        selected_ops_profile=request.selected_ops_profile,
        pbo_gate=float(request.pbo_gate),
        dsr_gate=float(request.dsr_gate),
        pbo_obs=float(request.pbo_obs),
        dsr_obs=float(request.dsr_obs),
        best_trial=request.best_trial,
        params=params,
        champ_stab_cv=request.champ_stab_cv,
        stab_tmp_layer3_awf_fail=bool(request.stab_tmp_layer3_awf_fail),
        cv_max=float(request.cv_max),
        phase_c_diagnostics=request.phase_c_diagnostics,
    )
