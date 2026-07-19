from __future__ import annotations

from unittest.mock import MagicMock

from src.domain.futures.optimization.workflow import _build_l2_user_attrs


def _make_mock_evaluation() -> MagicMock:
    ev = MagicMock()
    ev.objective_value = 0.05
    ev.cagr_hybrid = 0.12
    ev.cagr_baseline = 0.08
    ev.growth_lcb_hybrid = 0.06
    ev.growth_lcb_baseline = 0.04
    ev.sharpe_hac_hybrid = 1.5
    ev.sharpe_hac_baseline = 1.2
    ev.psr_hybrid = 0.85
    ev.mdd_hybrid = 0.15
    ev.cvar_95_hybrid = 0.05
    ev.fold_pass_ratio = 0.7
    ev.break_even_pass_pct = 0.6
    ev.average_gross_exposure = 1.2
    ev.cap_saturation_ratio = 0.1
    ev.total_cost_bps = 15.0
    ev.sortino_hybrid = 1.8
    ev.risk_utilization = 0.75
    ev.recent_fold_sharpe = 1.3
    ev.recent_fold_cagr = 0.10
    ev.latest_to_median_cagr = 1.0
    ev.deploy_leverage = 2.5
    ev.deployment_objective_bonus = 0.01
    ev.worst_fold_sharpe = 0.5
    ev.trade_count = 100
    ev.recent_fold_passed = True
    ev.deploy_binding = "mdd"
    ev.constraint_values = [0.0, -0.1]
    ev.active_block_count = 10
    ev.growth_lcb_deployed = 0.15
    ev.crisis_constraints_measured = True
    ev.block_log_growth_signature = (0.02, 0.03)
    block_mock = MagicMock()
    block_mock.log_growth_hybrid = 0.02
    ev.block_metrics = [block_mock]
    gate = MagicMock()
    gate.promotion_constraint_values = [0.0, -0.05]
    gate.promotion_passed = True
    gate.promotion_blocker = ""
    from src.domain.futures.strategy.tiered_workflow.l2_gate import Layer2ConstraintVector
    cv = Layer2ConstraintVector(
        deployment=-1.0, support_leak=0.0, mdd=-0.1, cvar_95=-0.05,
        fold=-0.2, recent_fold=-1.0, active_blocks=-3.0, friction=-0.1,
        trades=-2.0, crisis_mdd=-1.0, cagr=-0.3, sharpe_uplift=-0.2,
        crisis_cagr=-1.0, crisis_measured=True,
    )
    ev.gate = gate
    return ev


EXPECTED_KEYS: set[str] = {
    "l2_objective_value",
    "cagr_hybrid",
    "cagr_baseline",
    "growth_lcb_hybrid",
    "growth_lcb_baseline",
    "sharpe_hac_hybrid",
    "sharpe_hac_baseline",
    "psr_hybrid",
    "mdd_hybrid",
    "cvar_95_hybrid",
    "fold_pass_ratio",
    "break_even_pass_pct",
    "average_gross_exposure",
    "cap_saturation_ratio",
    "total_cost_bps",
    "sortino_hybrid",
    "risk_utilization",
    "recent_fold_sharpe",
    "recent_fold_cagr",
    "latest_to_median_cagr",
    "deploy_leverage",
    "deployment_objective_bonus",
    "worst_fold_sharpe",
    "trade_count",
    "recent_fold_passed",
    "deploy_binding",
    "l2_constraint_values",
    "l2_optuna_constraint_values",
    "l2_promotion_constraint_values",
    "l2_promotion_passed",
    "l2_promotion_blocker",
    "l2_block_log_growth_signature",
    "active_block_count",
    "growth_lcb_deployed",
    "l2_crisis_measured",
    "l2_joint_feasible",
    "l2_constraint_map",
}


def test_build_l2_user_attrs_matches_legacy_keys() -> None:
    ev = _make_mock_evaluation()
    attrs = _build_l2_user_attrs(ev)
    for key in EXPECTED_KEYS:
        assert key in attrs, f"Missing key: {key}"
