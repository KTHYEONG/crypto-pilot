from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.domain.futures.strategy.tiered_workflow.replay_parity import (
    assert_selection_replay_parity,
)


def test_assert_selection_replay_parity_accepts_matching_values() -> None:
    replay = SimpleNamespace(cagr_hybrid=0.22, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80)
    final = SimpleNamespace(cagr_hybrid=0.22, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80)

    assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-6,
    )


def test_assert_selection_replay_parity_returns_false_on_mismatch() -> None:
    replay = SimpleNamespace(cagr_hybrid=0.22, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80)
    final = SimpleNamespace(cagr_hybrid=0.08, mdd_hybrid=0.15, fold_pass_ratio=0.34, trade_count=120)

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-6,
    )
    assert result is False


# ── RC-1: gate=True parity divergence ──
def test_parity_gate_returns_false_with_gate_flag() -> None:
    """gate=True 시 mismatch면 False 반환."""
    replay = SimpleNamespace(cagr_hybrid=0.18, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80,
                             deploy_leverage=3.0, sharpe_hac_hybrid=1.5, sortino_hybrid=2.0,
                             constraint_values=(0.0,))
    final = SimpleNamespace(cagr_hybrid=0.07, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80,
                            deploy_leverage=1.0, sharpe_hac_hybrid=1.5, sortino_hybrid=2.0,
                            constraint_values=(0.0,))

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-8,
        gate=True,
    )
    assert result is False


def test_parity_gate_still_passes_when_match() -> None:
    """gate=True 시 match면 True 반환."""
    replay = SimpleNamespace(cagr_hybrid=0.22, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80)
    final = SimpleNamespace(cagr_hybrid=0.22, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80)

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-6,
        gate=True,
    )
    assert result is True


def test_self_check_uses_master_tf_for_bars_per_year() -> None:
    """self-check가 eval object의 master_tf를 반영한 bars_per_year로 재계산."""
    rets_array = np.asarray([0.001] * 100 + [-0.001] * 100, dtype=np.float64)
    l_star = 2.0
    bpy_8h = 1095.0

    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        apply_deployment,
    )
    dep = apply_deployment(rets=rets_array, leverage=l_star, bars_per_year=bpy_8h)

    eval_8h = SimpleNamespace(
        returns_hybrid=tuple(rets_array.tolist()),
        cagr_hybrid=dep.cagr,
        mdd_hybrid=dep.mdd,
        fold_pass_ratio=0.67,
        trade_count=184,
        deploy_leverage=l_star,
        master_tf="8h",
    )
    final = SimpleNamespace(cagr_hybrid=dep.cagr, mdd_hybrid=dep.mdd, fold_pass_ratio=0.67, trade_count=184)

    result = assert_selection_replay_parity(
        replay_evaluation=eval_8h,
        final_evaluation=final,
        tolerance=1e-6,
        gate=False,
    )
    assert result is True

