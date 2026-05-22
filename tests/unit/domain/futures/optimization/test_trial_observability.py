from __future__ import annotations

from optuna.distributions import FloatDistribution
from optuna.trial import TrialState, create_trial

from src.domain.futures.optimization.trial_observability import (
    build_compact_trial_summary,
    classify_no_valid_candidates,
)


def test_classify_no_valid_candidates_gate_reject_all() -> None:
    trial = create_trial(
        params={"x": 0.1},
        distributions={"x": FloatDistribution(0.0, 1.0)},
        value=0.0,
        user_attrs={"phase": "phase_b"},
        state=TrialState.COMPLETE,
    )
    reason = classify_no_valid_candidates(
        selection_summary={"selection_reject_reason_count": {"mdd_hard": 3}},
        completed_trials=[trial],
    )
    assert reason == "gate_reject_all"


def test_classify_no_valid_candidates_zero_alpha_components() -> None:
    trial = create_trial(
        params={"x": 0.2},
        distributions={"x": FloatDistribution(0.0, 1.0)},
        value=0.2,
        user_attrs={"avg_trades": 0.0, "awf_mu_log": -10.0},
        state=TrialState.COMPLETE,
    )
    reason = classify_no_valid_candidates(
        selection_summary={},
        completed_trials=[trial],
    )
    assert reason == "zero_alpha_components"


def test_build_compact_trial_summary_includes_reason() -> None:
    trial = create_trial(
        params={"KELLY_FRACTION": 0.25},
        distributions={"KELLY_FRACTION": FloatDistribution(0.0, 1.0)},
        value=0.1,
        user_attrs={"obs_reason": "trial_should_prune", "phase": "phase_b"},
        state=TrialState.PRUNED,
    )
    line = build_compact_trial_summary(trial, elapsed_sec=1.5)
    assert "reason=trial_should_prune" in line
    assert "status=pruned" in line


def test_build_compact_trial_summary_uses_strategy_tuning_keys() -> None:
    trial = create_trial(
        params={"BETA_ALPHA": 4.0, "EV_HURDLE_BPS": 2.0},
        distributions={
            "BETA_ALPHA": FloatDistribution(1.0, 10.0),
            "EV_HURDLE_BPS": FloatDistribution(0.0, 20.0),
        },
        value=0.3,
        user_attrs={"phase": "phase_b"},
        state=TrialState.COMPLETE,
    )
    line = build_compact_trial_summary(trial, elapsed_sec=0.3)
    assert "BETA_ALPHA=4.0" in line
    assert "EV_HURDLE_BPS=2.0" in line


def test_classify_no_valid_candidates_all_pruned_zero_trades() -> None:
    pruned_trial = create_trial(
        params={"x": 0.3},
        distributions={"x": FloatDistribution(0.0, 1.0)},
        value=None,
        user_attrs={"obs_reason": "zero_trades_first_leg"},
        state=TrialState.PRUNED,
    )
    reason = classify_no_valid_candidates(
        selection_summary={},
        completed_trials=[],
        pruned_trials=[pruned_trial],
    )
    assert reason == "all_pruned_zero_trades_first_leg"
