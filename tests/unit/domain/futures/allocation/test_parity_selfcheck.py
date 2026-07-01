from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from src.domain.futures.allocation.parity import (
    _resolve_bars_per_year,
    assert_selection_replay_parity,
)


def test_resolve_bars_per_year_uses_master_tf_field() -> None:
    obj = SimpleNamespace(master_tf="8h")
    assert _resolve_bars_per_year(obj) == 1095.0


def test_resolve_bars_per_year_returns_none_when_master_tf_missing() -> None:
    obj = SimpleNamespace()
    assert _resolve_bars_per_year(obj) is None

    obj_blank = SimpleNamespace(master_tf="")
    assert _resolve_bars_per_year(obj_blank) is None


def test_parity_selfcheck_no_false_positive_with_8h_tf() -> None:
    from src.domain.futures.allocation.deployment import apply_deployment

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
) -> None:
    from src.domain.futures.allocation import parity as _pkg
    from src.domain.futures.allocation.deployment import apply_deployment

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

    spy = mocker.spy(_pkg._logger, "warning")

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-6,
        gate=True,
    )
    assert result is True
    spy.assert_not_called()


def test_allocation_parity_skip_when_master_tf_missing() -> None:
    """bars_per_year None 시 self-check skip."""
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


def test_allocation_parity_missing_metric_on_one_side() -> None:
    """한쪽에 metric 속성 누락 시 skip."""
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


def test_allocation_parity_returns_false_on_mismatch() -> None:
    """metric 불일치 시 False 반환 (line 53 coverage)."""
    replay = SimpleNamespace(
        cagr_hybrid=0.22,
        mdd_hybrid=0.09,
        fold_pass_ratio=0.67,
        trade_count=80,
        master_tf="4h",
    )
    final = SimpleNamespace(
        cagr_hybrid=0.08,
        mdd_hybrid=0.15,
        fold_pass_ratio=0.34,
        trade_count=120,
        master_tf="4h",
    )

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-6,
        gate=False,
    )
    assert result is False


def test_allocation_parity_skip_when_rets_none() -> None:
    """returns_hybrid 없으면 continue (line 68 coverage)."""
    replay = SimpleNamespace(
        cagr_hybrid=0.1,
        mdd_hybrid=0.05,
        fold_pass_ratio=0.67,
        trade_count=10,
        master_tf="4h",
    )
    final = SimpleNamespace(
        cagr_hybrid=0.1,
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
