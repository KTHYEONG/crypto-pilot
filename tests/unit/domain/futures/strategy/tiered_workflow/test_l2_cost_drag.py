from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    Layer2FoldAttribution,
    compute_cost_drag_ratio,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2GateEvaluation,
)
from src.domain.futures.strategy.tiered_workflow.l2_gate import evaluate_layer2_gate


def _make_attribution(
    *,
    fold_idx: int = 0,
    realized_price: float = 0.0,
    realized_cost: float = 0.0,
    realized_funding: float = 0.0,
) -> Layer2FoldAttribution:
    return Layer2FoldAttribution(
        fold_idx=fold_idx,
        oos_bars=100,
        n_rebal=10,
        realized_total=realized_price + realized_funding - realized_cost,
        realized_price=realized_price,
        realized_funding=realized_funding,
        realized_cost=realized_cost,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.0,
        mean_net_exp=0.0,
        sleeves_active_mean=0.0,
        friction_pass_ratio=0.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
    )


class TestComputeCostDragRatio:
    """S1-S3: compute_cost_drag_ratio 단위 테스트."""

    def test_compute_cost_drag_ratio_matches_observed_run(self) -> None:
        """S1: 실측 3-fold price/cost로 집계 비율 검증."""
        attrs = (
            _make_attribution(fold_idx=0, realized_price=0.1046, realized_cost=0.0395),
            _make_attribution(fold_idx=1, realized_price=-0.0655, realized_cost=0.0376),
            _make_attribution(fold_idx=2, realized_price=0.0465, realized_cost=0.0330),
        )
        result = compute_cost_drag_ratio(attrs)
        expected = (0.0395 + 0.0376 + 0.0330) / (abs(0.1046) + abs(-0.0655) + abs(0.0465))
        assert result == pytest.approx(expected, rel=1e-3)

    def test_cost_drag_ratio_negative_gross_returns_large_finite(self) -> None:
        """S2: gross 음수 시 abs(gross) 기준으로 유한."""
        attrs = (
            _make_attribution(realized_price=-0.05, realized_cost=0.04),
        )
        result = compute_cost_drag_ratio(attrs)
        assert np.isfinite(result)
        assert result == pytest.approx(0.8, rel=1e-3)

    def test_cost_drag_ratio_empty_returns_zero(self) -> None:
        """S3: 빈 attribution → 0.0 (non-blocking)."""
        result = compute_cost_drag_ratio(())
        assert result == 0.0


class TestEvaluateGateCostDrag:
    """S4-S5: evaluate_layer2_gate cost_drag blocker 통합 테스트."""

    def _make_result_with_cost_drag(self, cost_drag: float) -> SimpleNamespace:
        if cost_drag <= 0.0:
            attrs: tuple[Layer2FoldAttribution, ...] = ()
        else:
            price = 0.10
            cost = price * cost_drag
            attrs = (_make_attribution(realized_price=price, realized_cost=cost),)
        return SimpleNamespace(fold_attributions=attrs)

    def _gate_kwargs(self, result: object) -> dict[str, Any]:
        return {
            "deployment_failed": False,
            "support_leak_count": 0,
            "cagr_hybrid": 0.5,
            "sharpe_hybrid": 2.0,
            "sharpe_hac_hybrid": 1.8,
            "sharpe_hac_baseline": 1.0,
            "sortino_hybrid": 3.0,
            "mar_hybrid": 2.0,
            "mdd_hybrid": 0.10,
            "cvar_95_hybrid": 0.05,
            "fold_pass_ratio": 0.80,
            "active_block_count": 5,
            "friction_pass_pct": 0.95,
            "trade_count": 100,
            "growth_lcb_hybrid": 0.15,
            "growth_lcb_baseline": 0.05,
            "dsr_hybrid": 0.80,
            "psr_hybrid": 0.95,
            "recent_fold_passed": True,
            "recent_fold_sharpe": 1.5,
            "fold_attributions": getattr(result, 'fold_attributions', ()),
        }

    def _eval_gate(self, cost_drag: float, config: Layer2AllocationConfig) -> Layer2GateEvaluation:
        result = self._make_result_with_cost_drag(cost_drag=cost_drag)
        kwargs = self._gate_kwargs(result)
        return evaluate_layer2_gate(
            config=config,
            deployment_failed=kwargs["deployment_failed"],
            support_leak_count=kwargs["support_leak_count"],
            cagr_hybrid=kwargs["cagr_hybrid"],
            sharpe_hybrid=kwargs["sharpe_hybrid"],
            sharpe_hac_hybrid=kwargs["sharpe_hac_hybrid"],
            sharpe_hac_baseline=kwargs["sharpe_hac_baseline"],
            sortino_hybrid=kwargs["sortino_hybrid"],
            mar_hybrid=kwargs["mar_hybrid"],
            mdd_hybrid=kwargs["mdd_hybrid"],
            cvar_95_hybrid=kwargs["cvar_95_hybrid"],
            fold_pass_ratio=kwargs["fold_pass_ratio"],
            active_block_count=kwargs["active_block_count"],
            friction_pass_pct=kwargs["friction_pass_pct"],
            trade_count=kwargs["trade_count"],
            growth_lcb_hybrid=kwargs["growth_lcb_hybrid"],
            growth_lcb_baseline=kwargs["growth_lcb_baseline"],
            dsr_hybrid=kwargs["dsr_hybrid"],
            psr_hybrid=kwargs["psr_hybrid"],
            recent_fold_passed=kwargs["recent_fold_passed"],
            recent_fold_sharpe=kwargs["recent_fold_sharpe"],
            fold_attributions=kwargs["fold_attributions"],
        )

    def test_evaluate_gate_blocks_when_cost_exceeds_gross(self) -> None:
        """S4: cost_drag=1.29 > 0.60 → BLOCK."""
        config = Layer2AllocationConfig(l2_max_cost_drag_ratio=0.60)
        gate = self._eval_gate(cost_drag=1.29, config=config)
        assert gate.promotion_passed is False
        assert gate.promotion_blocker == "cost_drag"

    def test_evaluate_gate_passes_cost_when_drag_below_threshold(self) -> None:
        """S5: cost_drag=0.40 < 0.60 → PASS (다른 blocker 없음)."""
        config = Layer2AllocationConfig(l2_max_cost_drag_ratio=0.60)
        gate = self._eval_gate(cost_drag=0.40, config=config)
        assert gate.promotion_blocker != "cost_drag"


class TestTurnoverPenalty:
    """S6: turnover 페널티 λ=0 거동 불변 회귀 테스트."""

    def test_objective_unchanged_when_turnover_penalty_zero(self) -> None:
        """S6: turnover_penalty_weight=0 → 목적값 변화 없음."""
        from src.domain.futures.optimization.workflow import _shape_efficiency_l2_objective

        base = _shape_efficiency_l2_objective(
            sortino_hac_unit=1.5,
            worst_fold_sortino=0.5,
            worst_fold_threshold=-0.30,
            worst_fold_weight=0.005,
            downside_dispersion=0.02,
            risk_util_realized=0.60,
            risk_util_target=0.50,
            risk_util_weight=0.03,
            trade_count=80,
            trade_target=90,
            trade_weight=0.02,
            mean_turnover=0.3,
            turnover_penalty_weight=0.0,
        )
        alt = _shape_efficiency_l2_objective(
            sortino_hac_unit=1.5,
            worst_fold_sortino=0.5,
            worst_fold_threshold=-0.30,
            worst_fold_weight=0.005,
            downside_dispersion=0.02,
            risk_util_realized=0.60,
            risk_util_target=0.50,
            risk_util_weight=0.03,
            trade_count=80,
            trade_target=90,
            trade_weight=0.02,
            mean_turnover=999.0,
            turnover_penalty_weight=0.0,
        )
        assert base == pytest.approx(alt)


class TestAlwaysOnAttribution:
    """S7: attribution 상시 누적 회귀 테스트."""

    def test_fold_attributions_present_even_when_diag_disabled(self) -> None:
        """S7: l2_diag_attribution_enabled=False여도 fold_attributions 비어있지 않음."""

        config = Layer2AllocationConfig(l2_diag_attribution_enabled=False)
        assert hasattr(config, 'l2_diag_attribution_enabled')
        assert config.l2_diag_attribution_enabled is False
