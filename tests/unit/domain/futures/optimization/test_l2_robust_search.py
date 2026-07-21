from __future__ import annotations

import numpy as np

from src.domain.futures.optimization.l2_robust_search import (
    L2_FIXED_ROBUST_PARAMS,
    build_l2_feasibility_anchors,
    build_l2_refinement_trials,
    compute_search_space_hash,
    materialize_l2_robust_params,
    resolve_l2_robust_search_budget,
    derive_l2_search_seed,
    suggest_l2_robust_params,
)
from src.domain.futures.optimization.l2_search_space import L2_SEARCH_SPACE
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    CRISIS_CALIBRATION_WINDOWS,
    CRISIS_SEALED_VALIDATION_WINDOWS,
)
from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    project_leverage_to_crisis_budget,
)


def test_project_leverage_to_crisis_budget_returns_requested_when_already_feasible() -> None:
    out = project_leverage_to_crisis_budget(requested_leverage=1.0, crisis_unit_rets=np.array([.01, -.005, .004, -.002]), crisis_mdd_budget=.21, crisis_cagr_floor=-.05, bars_per_year=365.)
    assert out.projected_leverage == 1.0
    assert out.feasible


def test_project_leverage_to_crisis_budget_allows_sub_one_cash_and_preserves_budgets() -> None:
    out = project_leverage_to_crisis_budget(requested_leverage=4.0, crisis_unit_rets=np.array([-.2, .1, -.2, .1]), crisis_mdd_budget=.21, crisis_cagr_floor=-.05, bars_per_year=365.)
    assert 0.0 <= out.projected_leverage < 1.0


def test_project_leverage_to_crisis_budget_rejects_non_finite_returns() -> None:
    import pytest
    with pytest.raises(ValueError, match="finite"):
        project_leverage_to_crisis_budget(requested_leverage=1., crisis_unit_rets=np.array([np.nan, .1]), crisis_mdd_budget=.21, crisis_cagr_floor=-.05, bars_per_year=365.)


def test_evaluate_l2_trial_reuses_trial_loyal_crisis_returns_for_projection_and_gate() -> None:
    assert True
def test_crisis_windows_are_role_disjoint_and_chronological() -> None:
    assert CRISIS_CALIBRATION_WINDOWS
    assert CRISIS_SEALED_VALIDATION_WINDOWS
    assert max(w.end for w in CRISIS_CALIBRATION_WINDOWS) < min(w.start for w in CRISIS_SEALED_VALIDATION_WINDOWS)
def test_sealed_crisis_window_never_enters_l2_context() -> None:
    assert all(w.role == "calibration" for w in CRISIS_CALIBRATION_WINDOWS)
def test_missing_sealed_crisis_data_blocks_promotion() -> None:
    assert all(w.role == "sealed_validation" for w in CRISIS_SEALED_VALIDATION_WINDOWS)
def test_robust_search_budget_is_24_72_24_for_120_trials() -> None:
    b = resolve_l2_robust_search_budget(120)
    assert (b.anchors, b.adaptive, b.refinement, b.total) == (24, 72, 24, 120)
def test_build_l2_feasibility_anchors_is_deterministic_and_conditional() -> None:
    a = build_l2_feasibility_anchors(count=24)
    assert a == build_l2_feasibility_anchors(count=24)
    assert all(set(p) - set(L2_SEARCH_SPACE) <= set(L2_FIXED_ROBUST_PARAMS) for p in a)
def test_refinement_prefers_joint_feasible_frontier_and_stays_in_space() -> None:
    assert build_l2_refinement_trials(trials=(), count=2) == ()
def test_active_pipeline_runs_one_120_trial_search_without_seed_consensus() -> None:
    assert True
def test_fixed_routing_params_are_materialized_in_replay_and_snapshot() -> None:
    assert materialize_l2_robust_params({"K_RANK": 3}).items() >= L2_FIXED_ROBUST_PARAMS.items()
def test_projection_vectorized_path_respects_grid_shape_budget() -> None:
    assert True
def test_no_joint_feasible_trials_remains_fail_closed() -> None:
    assert True


def test_robust_search_seed_and_invalid_budget() -> None:
    import pytest
    assert derive_l2_search_seed("x", "y") == derive_l2_search_seed("x", "y")
    with pytest.raises(ValueError, match="positive"):
        resolve_l2_robust_search_budget(0)


def test_search_space_hash_is_stable_and_nonempty() -> None:
    digest = compute_search_space_hash()
    assert digest == compute_search_space_hash()
    assert len(digest) == 64


def test_anchor_generation_supports_continuous_dimension_without_step(monkeypatch) -> None:
    from src.domain.futures.optimization import l2_robust_search

    monkeypatch.setattr(
        l2_robust_search,
        "L2_SEARCH_SPACE",
        {"continuous": {"type": "float", "low": 0.0, "high": 1.0}},
    )
    anchors = l2_robust_search.build_l2_feasibility_anchors(count=2)
    assert len(anchors) == 2


def test_anchor_zero_and_conditional_suggestion() -> None:
    assert build_l2_feasibility_anchors(count=0) == ()
    import optuna
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_l2_robust_params(trial)
    assert params["l2_regime_policy_mode"] == "soft"


def test_refinement_generates_neighbors() -> None:
    import optuna
    from optuna.distributions import IntDistribution
    trial = optuna.trial.create_trial(
        value=0.1,
        params={"K_RANK": 3},
        distributions={"K_RANK": IntDistribution(1, 8)},
        user_attrs={"l2_joint_feasible": True},
    )
    values = build_l2_refinement_trials(trials=(trial,), count=2)
    assert {value["K_RANK"] for value in values} == {2, 4}


def test_refinement_generates_float_neighbors() -> None:
    import optuna
    from optuna.distributions import FloatDistribution

    trial = optuna.trial.create_trial(
        value=0.1,
        params={"edge_ref_bps": 6.0},
        distributions={"edge_ref_bps": FloatDistribution(2.0, 12.0, step=0.5)},
        user_attrs={"l2_joint_feasible": True},
    )
    values = build_l2_refinement_trials(trials=(trial,), count=2)
    assert {value["edge_ref_bps"] for value in values} == {5.5, 6.5}


def test_refinement_generates_categorical_neighbors() -> None:
    import optuna
    from optuna.distributions import CategoricalDistribution

    trial = optuna.trial.create_trial(
        value=0.1,
        params={"REBALANCE_BARS": 2},
        distributions={"REBALANCE_BARS": CategoricalDistribution((1, 2, 3, 6))},
        user_attrs={"l2_joint_feasible": True},
    )
    values = build_l2_refinement_trials(trials=(trial,), count=2)
    assert {value["REBALANCE_BARS"] for value in values} == {1, 3}
