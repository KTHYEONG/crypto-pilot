"""Phase D: CAWF-R ML cross-sectional rank portfolio optimization (Optuna TPE).

Thin wrapper exposing modularized submodules for backward compatibility.
"""

from __future__ import annotations

from src.domain.futures.optimization.common import (
    _trial_diag_sampled,
    _weight_stage_diag,
)
from src.domain.futures.optimization.ml_context import (
    MLPhaseDContext,
    _base_engine_params,
    compute_multi_alignment_info,
    precompute_ml_optimization_context,
)
from src.domain.futures.optimization.objectives import (
    _build_strategy_compose_diag,
    _compose_strategy_scores_inplace,
    _evaluate_awf_phase_d_aggregate,
    _run_portfolio_numba_block,
    objective_ml_phase_d,
    select_best_trial_by_holdout_log_ret,
    topsis_select_best,
)
from src.domain.futures.optimization.samplers import (
    _suggest_ml_joint_nsga2,
    build_ml_phase_d_params,
)

__all__ = [
    "MLPhaseDContext",
    "_base_engine_params",
    "_build_strategy_compose_diag",
    "_compose_strategy_scores_inplace",
    "_evaluate_awf_phase_d_aggregate",
    "_run_portfolio_numba_block",
    "_suggest_ml_joint_nsga2",
    "_trial_diag_sampled",
    "_weight_stage_diag",
    "build_ml_phase_d_params",
    "compute_multi_alignment_info",
    "objective_ml_phase_d",
    "precompute_ml_optimization_context",
    "select_best_trial_by_holdout_log_ret",
    "topsis_select_best",
]

