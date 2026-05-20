from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from optuna.trial import TrialState

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.optimization.candidate_selector import select_and_rank_candidates
from src.domain.futures.optimization.phase_samplers import (
    phase_a1_constraints,
    phase_a2_constraints,
    phase_b_constraints,
)


@dataclass
class _FakeTrial:
    number: int
    value: float | None
    state: TrialState
    user_attrs: dict[str, Any]
    params: dict[str, Any]


@dataclass
class _FakeStudy:
    study_name: str
    trials: list[_FakeTrial]

    def get_trials(self, deepcopy: bool = False):
        return list(self.trials)


def test_phase_constraints_missing_metrics_are_violations() -> None:
    t = _FakeTrial(
        number=1,
        value=1.0,
        state=TrialState.COMPLETE,
        user_attrs={},
        params={},
    )
    assert all(v > 0 for v in phase_a1_constraints(t))
    assert all(v > 0 for v in phase_a2_constraints(t))
    assert all(v > 0 for v in phase_b_constraints(t))


def test_phase_constraints_thresholds_and_proxy_flags() -> None:
    t = _FakeTrial(
        number=2,
        value=1.0,
        state=TrialState.COMPLETE,
        user_attrs={
            "oos_ic": 0.02,
            "short_side_ic": 0.011,
            "n_trades": 40.0,
            "active_month_ratio": 0.60,
            "turnover_cost_ratio": 0.10,
            "sortino": 2.0,  # proxy source for sortino_lcb
            "calmar": 1.7,   # proxy source for calmar_lcb
            "awf_worst_mdd_pct": 18.0,  # proxy source for mdd_ucb
            "ev_cost": 3.5,  # proxy source for ev_cost_ratio
            "funding_drag": 0.10,  # proxy source for funding_drag_ratio
            "cagr": 35.0,  # proxy source for cagr_lcb
            "mdd_duration": 120.0,
            "cvar": 20.0,  # must pass: <= 1.3 * 18 = 23.4
            "minority": 0.2,  # proxy source for minority_side_ratio
        },
        params={},
    )
    assert all(v <= 0 for v in phase_a1_constraints(t))
    assert all(v <= 0 for v in phase_a2_constraints(t))
    assert all(v <= 0 for v in phase_b_constraints(t))
    assert t.user_attrs.get("sortino_lcb_proxy_used") == 1
    assert t.user_attrs.get("calmar_lcb_proxy_used") == 1
    assert t.user_attrs.get("mdd_ucb_proxy_used") == 1
    assert t.user_attrs.get("ev_cost_ratio_proxy_used") == 1
    assert t.user_attrs.get("funding_drag_ratio_proxy_used") == 1
    assert t.user_attrs.get("cagr_lcb_proxy_used") == 1
    assert t.user_attrs.get("minority_side_ratio_proxy_used") == 1


def test_phase_constraints_direct_metrics_no_proxy_flags() -> None:
    t = _FakeTrial(
        number=3,
        value=1.0,
        state=TrialState.COMPLETE,
        user_attrs={
            "oos_ic": 0.02,
            "short_side_ic": 0.011,
            "n_trades": 50.0,
            "active_month_ratio": 0.70,
            "turnover_cost_ratio": 0.10,
            "sortino_lcb": 2.1,
            "calmar_lcb": 1.8,
            "mdd_ucb": 18.0,
            "ev_cost_ratio": 3.5,
            "funding_drag_ratio": 0.10,
            "cagr_lcb": 33.0,
            "mdd_duration": 120.0,
            "cvar": 20.0,  # <= 1.3 * 18 = 23.4
            "minority_side_ratio": 0.2,
        },
        params={},
    )
    assert all(v <= 0 for v in phase_a1_constraints(t))
    assert all(v <= 0 for v in phase_a2_constraints(t))
    assert all(v <= 0 for v in phase_b_constraints(t))
    assert "sortino_lcb_proxy_used" not in t.user_attrs
    assert "calmar_lcb_proxy_used" not in t.user_attrs
    assert "mdd_ucb_proxy_used" not in t.user_attrs
    assert "ev_cost_ratio_proxy_used" not in t.user_attrs
    assert "funding_drag_ratio_proxy_used" not in t.user_attrs
    assert "cagr_lcb_proxy_used" not in t.user_attrs
    assert "minority_side_ratio_proxy_used" not in t.user_attrs


def test_phase_b_cvar_mdd_unit_alignment() -> None:
    t = _FakeTrial(
        number=4,
        value=1.0,
        state=TrialState.COMPLETE,
        user_attrs={
            "ev_cost_ratio": 3.1,
            "funding_drag_ratio": 0.10,
            "cagr_lcb": 31.0,
            "sortino_lcb": 1.9,
            "calmar_lcb": 1.6,
            "mdd_ucb": 0.18,  # fraction
            "mdd_duration": 100.0,
            "cvar": 22.0,  # percent
            "minority_side_ratio": 0.2,
        },
        params={},
    )
    vals = phase_b_constraints(t)
    assert vals[7] <= 0  # cvar <= 1.3 * mdd after unit alignment


def test_v43_candidate_selector_filters_to_phase_b_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.domain.futures.optimization.candidate_selector.replay_robust_awf_for_trial_params",
        lambda _ctx, _params: (0.0, {"ok": True}),
    )
    phase_a = _FakeTrial(
        number=10,
        value=2.0,
        state=TrialState.COMPLETE,
        user_attrs={
            "phase": "phase_a2",
            "awf_worst_mdd_pct": 10.0,
            "awf_trade_count_mean": 10.0,
        },
        params={"K_LONG": 2},
    )
    phase_b = _FakeTrial(
        number=11,
        value=1.0,
        state=TrialState.COMPLETE,
        user_attrs={
            "phase": "phase_b",
            "awf_worst_mdd_pct": 10.0,
            "awf_trade_count_mean": 10.0,
        },
        params={"K_LONG": 3},
    )
    study = _FakeStudy(study_name="demo_phase_b", trials=[phase_a, phase_b])

    best, summary = select_and_rank_candidates(
        study_ml=study, base_ctx=SimpleNamespace(), cfg={}
    )

    assert best["trial"].number == 11
    assert summary["selected_trial_number"] == 11
