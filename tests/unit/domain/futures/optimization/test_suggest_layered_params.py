"""Tests for TieredContext, suggest_layered_params, and objective_l1_ic.

Covers:
- TI10: L1 suggest 결과 키집합 검증
- TI11: L2 suggest 결과 키집합 검증
- TI11b: fixed 키 주입 시 suggest_* 미호출 확인
- TI13: objective_l1_ic 가 run_l2_awf (Sharpe) 미호출 확인
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import optuna
import pytest
from optuna.trial import FrozenTrial

from src.domain.futures.optimization.workflow import (
    TieredContext,
    _deployment_shaped_l2_objective,
    evaluate_l2_trial,
    layer2_constraints_from_trial,
    objective_l1_ic,
    objective_l2_growth,
    suggest_layered_params,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
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


def test_deployment_shaped_objective_penalizes_downside_and_instability() -> None:
    low = _deployment_shaped_l2_objective(
        growth_lcb=0.20,
        block_log_growth=(-0.08, 0.01, 0.00),
        worst_fold_sharpe=0.30,
        worst_fold_threshold=0.50,
        worst_fold_weight=0.20,
    )
    high = _deployment_shaped_l2_objective(
        growth_lcb=0.20,
        block_log_growth=(0.01, 0.02, 0.015),
        worst_fold_sharpe=0.80,
        worst_fold_threshold=0.50,
        worst_fold_weight=0.20,
    )

    assert high > low


def test_deployment_shaped_objective_keeps_growth_primary() -> None:
    higher_growth = _deployment_shaped_l2_objective(
        growth_lcb=0.18,
        block_log_growth=(0.01, 0.02),
        worst_fold_sharpe=0.80,
        worst_fold_threshold=0.50,
        worst_fold_weight=0.20,
    )
    lower_growth = _deployment_shaped_l2_objective(
        growth_lcb=0.17,
        block_log_growth=(0.01, 0.02),
        worst_fold_sharpe=0.80,
        worst_fold_threshold=0.50,
        worst_fold_weight=0.20,
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
            "src.domain.futures.strategy.tiered_workflow.pipeline.run_l1_swf",
            return_value=_l1_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf"
        ) as mock_l2,
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


def test_objective_l2_growth_sets_constraint_attrs(caplog: pytest.LogCaptureFixture) -> None:
    gate = SimpleNamespace(
        optuna_constraint_values=(-1.0, -0.1, 0.0, -0.2, -0.3, 0.0, -0.05, -0.01),
        promotion_constraint_values=(-1.0,) * 14,
        promotion_passed=False,
        promotion_blocker="cagr",
    )
    evaluation = SimpleNamespace(
        objective_value=0.12,
        constraint_values=gate.optuna_constraint_values,
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
        gate=gate,
    )

    with (
        patch(
            "src.domain.futures.optimization.workflow._resolve_l2_signal_batch_and_folds",
            return_value=(MagicMock(events=[]), (MagicMock(),)),
        ),
        patch(
            "src.domain.futures.optimization.workflow.evaluate_l2_trial",
            return_value=evaluation,
        ),
        caplog.at_level(logging.DEBUG, logger="src.domain.futures.optimization.workflow"),
    ):
        ctx = TieredContext(
            labeled_events=MagicMock(),
            aligned=MagicMock(),
            cfg=MagicMock(),
            window=MagicMock(),
            caps=MagicMock(),
            tf="4h",
            fixed_l1_params={"signal_batch": object()},
            l2_sim_cache=MagicMock(),
        )
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        value = objective_l2_growth(trial, ctx)

    assert value == pytest.approx(0.12)
    assert trial.user_attrs["l2_constraint_values"] == list(evaluation.constraint_values)
    assert trial.user_attrs["l2_optuna_constraint_values"] == list(evaluation.constraint_values)
    assert trial.user_attrs["l2_promotion_constraint_values"] == list(gate.promotion_constraint_values)
    assert trial.user_attrs["l2_promotion_passed"] is False
    assert trial.user_attrs["l2_promotion_blocker"] == "cagr"
    assert trial.user_attrs["l2_block_log_growth_signature"] == [0.02]
    assert any("[perf-optuna] Trial" in r.message for r in caplog.records)


def test_evaluate_l2_trial_threads_new_deployable_metrics_into_gate() -> None:
    fake_sim = SimpleNamespace(
        rets_hybrid=[0.02, 0.03, 0.01],
        rets_baseline=[0.01, 0.01, 0.0],
        rets_baseline_ew=[0.01, 0.01, 0.01],
        fit_rets_hybrid=(0.01, 0.02),
        all_turnovers=[0.2, 0.3, 0.1],
        all_gross_exposures=[0.4, 0.5, 0.6],
        all_net_exposures=[0.1, 0.2, 0.1],
        friction_pass_total=2,
        signal_total=3,
        support_leak_count=0,
        total_cost_hybrid=0.01,
        cap_saturation_count=1,
        rebalance_count=3,
        trade_count=12,
        fold_rets_hybrid=([0.02, 0.01], [0.03, -0.01], [0.01, 0.02]),
        block_rets_hybrid=([0.02, 0.01], [0.03, -0.01], [0.01, 0.02]),
        block_rets_baseline=([0.0, 0.0], [0.01, 0.0], [0.0, 0.01]),
        fold_attributions=(
            SimpleNamespace(realized_cost=0.1, realized_price=0.3),
            SimpleNamespace(realized_cost=0.05, realized_price=0.25),
            SimpleNamespace(realized_cost=0.02, realized_price=0.18),
        ),
        fold_selected_symbols=(("BTC",), ("ETH",), ("SOL",)),
        policy_effect_by_fold=(),
    )
    fake_diag = SimpleNamespace(
        fold_pass_ratio=0.67,
        recent_fold_passed=True,
        recent_fold_sharpe=0.8,
        recent_fold_cagr=0.04,
        recent_fold_mdd=0.02,
        latest_to_median_cagr=0.01,
        fold_deployed_cagrs=(0.02, -0.03, 0.01),
        fold_selected_symbols=(("BTC",), ("ETH",), ("SOL",)),
    )
    fake_deployment = SimpleNamespace(cagr=0.12, mdd=0.08, cvar_95=0.03)

    gate_mock = SimpleNamespace(
        optuna_constraint_values=(-1.0,) * 18,
        promotion_constraint_values=(-1.0,) * 18,
        promotion_passed=False,
        promotion_blocker="cagr",
    )

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation",
            return_value=fake_sim,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.risk_deployment.apply_deployment",
            return_value=fake_deployment,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage",
            return_value=(2.0, "mdd", 0.0),
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics",
            return_value=fake_diag,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate",
            return_value=gate_mock,
        ) as mock_gate,
    ):
        evaluation = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=MagicMock(),
            aligned=MagicMock(),
            awf_folds=(MagicMock(),),
            config=Layer2AllocationConfig(),
            caps=MagicMock(),
            tf="4h",
        )

    assert evaluation.worst_fold_cagr == pytest.approx(-0.03)
    assert evaluation.positive_block_delta_ratio == pytest.approx(1.0)
    assert evaluation.gate is gate_mock
    assert evaluation.constraint_values == gate_mock.optuna_constraint_values
    assert mock_gate.call_args is not None
    assert mock_gate.call_args.kwargs["worst_fold_cagr"] == pytest.approx(-0.03)
    assert mock_gate.call_args.kwargs["positive_block_delta_ratio"] == pytest.approx(1.0)


def test_evaluate_l2_trial_uses_universe_audit_warning_for_entry_spike_penalty() -> None:
    fake_sim = SimpleNamespace(
        rets_hybrid=[0.02, 0.03, 0.01],
        rets_baseline=[0.01, 0.01, 0.0],
        rets_baseline_ew=[0.01, 0.01, 0.01],
        fit_rets_hybrid=(0.01, 0.02),
        all_turnovers=[0.2],
        all_gross_exposures=[0.4],
        all_net_exposures=[0.1],
        friction_pass_total=1,
        signal_total=1,
        support_leak_count=0,
        total_cost_hybrid=0.0,
        cap_saturation_count=0,
        rebalance_count=1,
        trade_count=3,
        fold_rets_hybrid=([0.02, 0.01],),
        block_rets_hybrid=([0.02, 0.01],),
        block_rets_baseline=([0.0, 0.0],),
        fold_attributions=(SimpleNamespace(realized_cost=0.0, realized_price=0.2),),
        fold_selected_symbols=(("BTC",),),
        policy_effect_by_fold=(),
    )
    fake_diag = SimpleNamespace(
        fold_pass_ratio=1.0,
        recent_fold_passed=True,
        recent_fold_sharpe=0.8,
        recent_fold_cagr=0.04,
        recent_fold_mdd=0.02,
        latest_to_median_cagr=0.01,
        fold_deployed_cagrs=(0.02,),
        fold_selected_symbols=(("BTC",),),
    )
    fake_deployment = SimpleNamespace(cagr=0.12, mdd=0.08, cvar_95=0.03)
    signal_batch = MagicMock()
    signal_batch.start_idx = 10
    signal_batch.end_idx = 20

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation",
            return_value=fake_sim,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.risk_deployment.apply_deployment",
            return_value=fake_deployment,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage",
            return_value=(2.0, "mdd", 0.0),
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics",
            return_value=fake_diag,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate",
            return_value=SimpleNamespace(
                optuna_constraint_values=(-1.0,) * 18,
                promotion_constraint_values=(-1.0,) * 18,
                promotion_passed=True,
                promotion_blocker="",
            ),
        ),
        patch(
            "src.domain.futures.optimization.workflow.build_layer_universe_audit",
            return_value=SimpleNamespace(warnings=("entry_block_spike",)),
        ) as mock_audit,
    ):
        evaluation = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=signal_batch,
            aligned=MagicMock(),
            awf_folds=(MagicMock(),),
            config=Layer2AllocationConfig(l2_entry_spike_penalty_weight=0.05),
            caps=MagicMock(),
            tf="4h",
        )

    assert mock_audit.call_args is not None
    assert evaluation.entry_spike_penalty == pytest.approx(0.05)


def test_layer2_constraints_from_trial_reads_saved_values() -> None:
    """Optuna safety constraints는 8-tuple로 패딩된다."""
    trial = cast(
        FrozenTrial,
        SimpleNamespace(user_attrs={"l2_optuna_constraint_values": [0, -1, 2.5]}),
    )

    constraints = layer2_constraints_from_trial(trial)

    assert constraints == (0.0, -1.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
