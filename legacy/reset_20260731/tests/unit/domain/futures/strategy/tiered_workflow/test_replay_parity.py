from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from src.domain.futures.strategy.tiered_workflow.replay_parity import (
    _resolve_bars_per_year,
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
    replay = SimpleNamespace(
        cagr_hybrid=0.18,
        mdd_hybrid=0.09,
        fold_pass_ratio=0.67,
        trade_count=80,
        deploy_leverage=3.0,
        sharpe_hac_hybrid=1.5,
        sortino_hybrid=2.0,
        constraint_values=(0.0,),
    )
    final = SimpleNamespace(
        cagr_hybrid=0.07,
        mdd_hybrid=0.09,
        fold_pass_ratio=0.67,
        trade_count=80,
        deploy_leverage=1.0,
        sharpe_hac_hybrid=1.5,
        sortino_hybrid=2.0,
        constraint_values=(0.0,),
    )

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


def test_resolve_bars_per_year_uses_master_tf_field() -> None:
    """Scenario 1: _resolve_bars_per_year returns correct bpy from master_tf."""
    obj = SimpleNamespace(master_tf="8h")
    assert _resolve_bars_per_year(obj) == 1095.0


def test_resolve_bars_per_year_returns_none_when_master_tf_missing() -> None:
    """Scenario 2: master_tf 없는 객체 → None 반환."""
    obj = SimpleNamespace()
    assert _resolve_bars_per_year(obj) is None

    obj_blank = SimpleNamespace(master_tf="")
    assert _resolve_bars_per_year(obj_blank) is None


def test_parity_selfcheck_no_false_positive_with_8h_tf() -> None:
    """Scenario 3: 8h 챔피언 self-check가 WARNING 없이 통과 (bpy 인플레이션 재현 방지)."""
    import numpy as np

    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        apply_deployment,
    )

    rets_array = np.asarray([0.001] * 100 + [-0.001] * 100, dtype=np.float64)
    l_star = 2.0
    bpy_8h = 1095.0
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
    final = SimpleNamespace(
        returns_hybrid=tuple(rets_array.tolist()),
        cagr_hybrid=dep.cagr,
        mdd_hybrid=dep.mdd,
        fold_pass_ratio=0.67,
        trade_count=184,
        deploy_leverage=l_star,
        master_tf="8h",
    )

    result = assert_selection_replay_parity(
        replay_evaluation=eval_8h,
        final_evaluation=final,
        tolerance=1e-6,
        gate=False,
    )
    assert result is True


def test_parity_gate_no_spurious_warning_on_8h_champion(
    mocker: Any,
    caplog: Any,
) -> None:
    """Scenario 4: gate=True 호출 패턴 — WARNING 미발생 검증."""
    from src.domain.futures.strategy.tiered_workflow import replay_parity as _rp
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        apply_deployment,
    )

    rets_array = np.asarray([0.001] * 100 + [-0.001] * 100, dtype=np.float64)
    l_star = 2.0
    bpy_8h = 1095.0
    dep = apply_deployment(rets=rets_array, leverage=l_star, bars_per_year=bpy_8h)

    replay = SimpleNamespace(
        returns_hybrid=tuple(rets_array.tolist()),
        cagr_hybrid=dep.cagr,
        mdd_hybrid=dep.mdd,
        fold_pass_ratio=0.67,
        trade_count=184,
        deploy_leverage=l_star,
        sharpe_hac_hybrid=1.5,
        sortino_hybrid=2.0,
        constraint_values=(0.0,),
        master_tf="8h",
    )
    final = SimpleNamespace(
        returns_hybrid=tuple(rets_array.tolist()),
        cagr_hybrid=dep.cagr,
        mdd_hybrid=dep.mdd,
        fold_pass_ratio=0.67,
        trade_count=184,
        deploy_leverage=l_star,
        sharpe_hac_hybrid=1.5,
        sortino_hybrid=2.0,
        constraint_values=(0.0,),
        master_tf="8h",
    )

    spy = mocker.spy(_rp._logger, "warning")

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-6,
        gate=True,
    )
    assert result is True
    spy.assert_not_called()


def test_parity_selfcheck_skip_when_master_tf_missing() -> None:
    """bars_per_year None 시 self-check skip (line 73 coverage)."""
    rets_tuple = tuple(float(v) for v in [0.001] * 5)

    replay_no_tf = SimpleNamespace(
        returns_hybrid=rets_tuple,
        cagr_hybrid=0.1,
        mdd_hybrid=0.05,
        fold_pass_ratio=0.67,
        trade_count=10,
        deploy_leverage=1.0,
    )
    final_no_tf = SimpleNamespace(
        cagr_hybrid=0.1,
        mdd_hybrid=0.05,
        fold_pass_ratio=0.67,
        trade_count=10,
    )

    result = assert_selection_replay_parity(
        replay_evaluation=replay_no_tf,
        final_evaluation=final_no_tf,
        tolerance=1e-6,
        gate=False,
    )
    assert result is True


def test_parity_selfcheck_missing_metric_on_one_side() -> None:
    """한쪽에 metric 속성 누락 시 WARNING 없이 skip (line 46-47 coverage)."""
    replay = SimpleNamespace(
        cagr_hybrid=0.1,
        mdd_hybrid=0.05,
        fold_pass_ratio=0.67,
        trade_count=10,
        master_tf="4h",
    )
    final = SimpleNamespace(
        mdd_hybrid=0.05,
        fold_pass_ratio=0.67,
        trade_count=10,
        master_tf="4h",
    )

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-6,
        gate=False,
    )
    assert result is True
