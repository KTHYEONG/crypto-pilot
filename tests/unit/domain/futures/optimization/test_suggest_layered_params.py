"""Tests for TieredContext, suggest_layered_params, and objective_l1_ic.

Covers:
- TI10: L1 suggest 결과 키집합 검증
- TI11: L2 suggest 결과 키집합 검증
- TI11b: fixed 키 주입 시 suggest_* 미호출 확인
- TI13: objective_l1_ic 가 run_l2_awf (Sharpe) 미호출 확인
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import optuna
import pytest
from optuna.trial import FrozenTrial

from src.domain.futures.optimization.workflow import (
    TieredContext,
    _deployment_shaped_l2_objective,
    layer2_constraints_from_trial,
    objective_l1_ic,
    objective_l2_growth,
    suggest_layered_params,
)

# ---------------------------------------------------------------------------
# TI10: L1 suggest 결과 키 == L1_ALPHA_SPACE 키집합
# ---------------------------------------------------------------------------


def test_suggest_layered_params_l1_keys_only_l1_space() -> None:
    """L1 suggest 결과 키 == L1_ALPHA_SPACE 키집합, L2 키 미포함."""
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE, L2_ALLOC_SPACE

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    result = suggest_layered_params(trial, "L1")

    assert set(result.keys()) == set(L1_ALPHA_SPACE.keys())
    assert not (set(result.keys()) & set(L2_ALLOC_SPACE.keys()))


# ---------------------------------------------------------------------------
# TI11: L2 suggest 결과 키 == L2_ALLOC_SPACE 키집합
# ---------------------------------------------------------------------------


def test_suggest_layered_params_l2_keys_only_l2_space() -> None:
    """L2 suggest 결과 키 == L2_ALLOC_SPACE 키집합, L1 키 미포함."""
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE, L2_ALLOC_SPACE

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    result = suggest_layered_params(trial, "L2")

    assert set(result.keys()) == set(L2_ALLOC_SPACE.keys())
    assert not (set(result.keys()) & set(L1_ALPHA_SPACE.keys()))


def test_deployment_shaped_objective_prefers_better_deployment_at_same_growth() -> None:
    low = _deployment_shaped_l2_objective(
        growth_lcb=0.20,
        risk_utilization=0.10,
        trade_count=40,
        risk_util_target=0.35,
        risk_util_weight=0.03,
        trade_target=90,
        trade_weight=0.02,
    )
    high = _deployment_shaped_l2_objective(
        growth_lcb=0.20,
        risk_utilization=0.30,
        trade_count=85,
        risk_util_target=0.35,
        risk_util_weight=0.03,
        trade_target=90,
        trade_weight=0.02,
    )

    assert high > low


def test_deployment_shaped_objective_keeps_growth_primary() -> None:
    higher_growth = _deployment_shaped_l2_objective(
        growth_lcb=0.18,
        risk_utilization=0.20,
        trade_count=50,
        risk_util_target=0.35,
        risk_util_weight=0.03,
        trade_target=90,
        trade_weight=0.02,
    )
    lower_growth = _deployment_shaped_l2_objective(
        growth_lcb=0.17,
        risk_utilization=0.20,
        trade_count=50,
        risk_util_target=0.35,
        risk_util_weight=0.03,
        trade_target=90,
        trade_weight=0.02,
    )

    assert higher_growth > lower_growth


# ---------------------------------------------------------------------------
# TI11b: fixed 키는 suggest_* 미호출, 값 직접 주입
# ---------------------------------------------------------------------------


def test_suggest_layered_params_fixed_skips_suggest() -> None:
    """fixed 키는 trial.suggest_* 미호출, 값 그대로 주입."""
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    first_key = next(iter(L1_ALPHA_SPACE))
    fixed_val = 99

    result = suggest_layered_params(trial, "L1", fixed={first_key: fixed_val})

    assert result[first_key] == fixed_val


# ---------------------------------------------------------------------------
# TI13: objective_l1_ic 가 run_l2_awf(Sharpe) 미호출 (mock spy)
# ---------------------------------------------------------------------------


def test_objective_l1_ic_does_not_call_sharpe() -> None:
    """objective_l1_ic 는 run_l2_awf(Sharpe) 를 호출하지 않는다."""
    _l1_result = SimpleNamespace(
        pooled_ic=0.04,
        pooled_tstat=2.1,
        gate_passed=True,
        signals_per_fold=(),
        oos_stacked={},
        breadth=0.5,
        valid_coverage=0.9,
        fold_pass_ratio=0.7,
        n_valid=5,
        n_total=10,
    )

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.run_l2_awf"
        ) as mock_l2,
        patch(
            "src.domain.futures.strategy.tiered_workflow.run_l1_swf",
            return_value=_l1_result,
        ),
        patch(
            "src.domain.futures.strategy.walk_forward.build_l1_swf_folds",
            return_value=(MagicMock(),),
        ),
        patch(
            "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
            return_value=(5, 2),
        ),
        patch("numpy.searchsorted", return_value=100),
    ):
        import datetime
        ctx = MagicMock()
        ctx.aligned.datetimes.__len__ = MagicMock(return_value=200)
        ctx.window.l1_start = datetime.date(2023, 1, 1)
        ctx.window.l2_start = datetime.date(2025, 1, 1)
        ctx.fixed_l1_params = None

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        val = objective_l1_ic(trial, ctx)

    assert val == pytest.approx(0.04)
    mock_l2.assert_not_called()   # Sharpe 미참조 핵심 검증


def test_objective_l2_growth_sets_constraint_attrs() -> None:
    evaluation = SimpleNamespace(
        objective_value=0.12,
        constraint_values=(-1.0, -0.1, 0.0, -0.2, -0.3, 0.0, -0.05, -0.01, -1.0),
        cagr_hybrid=0.25,
        cagr_baseline=0.10,
        growth_lcb_hybrid=0.12,
        growth_lcb_baseline=0.03,
        sharpe_hac_hybrid=1.1,
        sharpe_hac_baseline=0.8,
        psr_hybrid=0.9,
        mdd_hybrid=0.08,
        cvar_95_hybrid=0.02,
        fold_pass_ratio=0.75,
        break_even_pass_pct=0.8,
        average_gross_exposure=0.4,
        cap_saturation_ratio=0.0,
        total_cost_bps=12.0,
        block_metrics=(SimpleNamespace(log_growth_hybrid=0.02),),
    )

    with (
        patch(
            "src.domain.futures.optimization.workflow._resolve_l2_signal_batch_and_folds",
            return_value=("signal_batch", (MagicMock(),)),
        ),
        patch(
            "src.domain.futures.optimization.workflow.evaluate_l2_trial",
            return_value=evaluation,
    ),
):
        ctx = TieredContext(
            labeled_events=MagicMock(),
            aligned=MagicMock(),
            cfg=MagicMock(),
            window=MagicMock(),
            caps=MagicMock(),
            tf="4h",
            fixed_l1_params={"signal_batch": object()},
        )
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        value = objective_l2_growth(trial, ctx)

    assert value == pytest.approx(0.12)
    assert trial.user_attrs["l2_constraint_values"] == list(evaluation.constraint_values)
    assert trial.user_attrs["l2_block_log_growth_signature"] == [0.02]


def test_layer2_constraints_from_trial_reads_saved_values() -> None:
    """C3: DSR-in-loop 제거 후 12-tuple로 패딩(짧은 saved values는 1.0/infeasible로 패딩)."""
    trial = cast(FrozenTrial, SimpleNamespace(user_attrs={"l2_constraint_values": [0, -1, 2.5]}))

    constraints = layer2_constraints_from_trial(trial)

    assert constraints == (0.0, -1.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
