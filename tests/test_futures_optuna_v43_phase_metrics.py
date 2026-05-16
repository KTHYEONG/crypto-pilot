from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.optimization.phase_metrics import lcb, ucb
from src.domain.futures.optimization.phase_objectives import (
    objective_phase_a1_signal_lcb,
    objective_phase_a2_sortino_mdd,
    objective_phase_b_calmar_lcb,
)


class _DummyTrial:
    def __init__(self) -> None:
        self.user_attrs: dict[str, object] = {}
        self.reported_steps: list[tuple[float, int]] = []
        self.prune_at_step: int | None = None

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value

    def report(self, value: float, step: int) -> None:
        self.reported_steps.append((value, step))

    def should_prune(self) -> bool:
        if self.prune_at_step is None:
            return False
        if not self.reported_steps:
            return False
        return self.reported_steps[-1][1] >= self.prune_at_step


def test_lcb_ucb_basic() -> None:
    vals = [1.0, 2.0, 3.0]
    assert lcb(vals, k=1.0) < 2.0
    assert ucb(vals, k=1.0) > 2.0


def test_phase_objectives_write_lcb_ucb_attrs(monkeypatch) -> None:
    trial = _DummyTrial()
    ctx = SimpleNamespace()

    def _fake_base_objective(tr, _ctx):
        tr.set_user_attr("IS_DSR", 1.5)
        tr.set_user_attr("IS_MDD", 12.0)
        tr.set_user_attr("IS_RET_PCT", 24.0)
        tr.set_user_attr("n_trades", 80.0)
        tr.set_user_attr("awf_leg_log_tw", [0.20, 0.30, 0.10])
        tr.set_user_attr("fold_mdd_values", [10.0, 12.0, 11.0])
        tr.set_user_attr("turnover_cost_ratio", 0.50)
        tr.set_user_attr("turnover_ref", 0.35)
        tr.set_user_attr("lambda_turnover", 1.0)
        tr.set_user_attr("target_trades", 100.0)
        tr.set_user_attr("ev_cost_ratio", 2.0)
        tr.set_user_attr("minority_side_ratio", 0.6)
        tr.set_user_attr("funding_drag_ratio", 0.12)
        tr.set_user_attr("funding_drag_basis", "funding_fee_abs_over_gross_pnl_abs")
        tr.set_user_attr("mdd_duration", 75.0)
        tr.set_user_attr("cvar", 9.5)
        return 0.0

    monkeypatch.setattr(
        "src.domain.futures.optimization.phase_objectives.objective_ml_phase_d",
        _fake_base_objective,
    )

    a1 = objective_phase_a1_signal_lcb(trial, ctx)
    assert isinstance(a1, float)
    assert "signal_score_lcb" in trial.user_attrs
    assert trial.user_attrs["phase"] == "phase_a1"
    expected_net_expectancy = lcb([0.20, 0.30, 0.10], k=1.0)
    expected_activity = math.sqrt(min(80.0 / 100.0, 1.0))
    expected_penalty = max(0.0, 0.50 - 0.35) * 1.0
    assert a1 == expected_net_expectancy * expected_activity - expected_penalty
    assert len(trial.reported_steps) == 3

    a2 = objective_phase_a2_sortino_mdd(trial, ctx)
    assert isinstance(a2, tuple) and len(a2) == 2
    assert "sortino_lcb" in trial.user_attrs
    assert "mdd_ucb" in trial.user_attrs
    assert a2[0] == lcb([0.20, 0.30, 0.10], k=1.0)
    assert a2[1] == ucb([10.0, 12.0, 11.0], k=1.0)

    b = objective_phase_b_calmar_lcb(trial, ctx)
    assert isinstance(b, float)
    assert "calmar_lcb" in trial.user_attrs
    assert "cagr_lcb" in trial.user_attrs
    assert "fold_metric_values" in trial.user_attrs
    assert b == lcb([0.20, 0.30, 0.10], k=1.0) / ucb([10.0, 12.0, 11.0], k=1.0)

    required_keys = {
        "net_expectancy_lcb",
        "n_trades",
        "active_month_ratio",
        "turnover_cost_ratio",
        "ev_cost_ratio",
        "funding_drag_ratio",
        "cagr_lcb",
        "sortino_lcb",
        "mdd_ucb",
        "calmar_lcb",
        "mdd_duration",
        "cvar",
        "minority_side_ratio",
        "funding_drag_basis",
    }
    assert required_keys.issubset(trial.user_attrs.keys())
    assert trial.user_attrs["funding_drag_ratio"] == 0.12
    assert trial.user_attrs["mdd_duration"] == 75.0
    assert trial.user_attrs["cvar"] == 9.5


def test_phase_objective_pruning_path(monkeypatch) -> None:
    import optuna

    trial = _DummyTrial()
    trial.prune_at_step = 1
    ctx = SimpleNamespace()

    def _fake_base_objective(tr, _ctx):
        tr.set_user_attr("awf_leg_log_tw", [0.20, 0.10, -0.05])
        tr.set_user_attr("n_trades", 10.0)
        tr.set_user_attr("target_trades", 10.0)
        tr.set_user_attr("turnover_cost_ratio", 0.0)
        return 0.0

    monkeypatch.setattr(
        "src.domain.futures.optimization.phase_objectives.objective_ml_phase_d",
        _fake_base_objective,
    )

    try:
        objective_phase_a1_signal_lcb(trial, ctx)
        raised = False
    except optuna.TrialPruned:
        raised = True

    assert raised is True
    assert len(trial.reported_steps) >= 2
