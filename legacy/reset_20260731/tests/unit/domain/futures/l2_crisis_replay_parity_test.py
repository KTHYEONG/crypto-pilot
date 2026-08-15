from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial

from src.domain.futures.optimization.workflow import evaluate_l2_trial_cached


# ── Scenario 1: evaluate_l2_trial_cached crisis_rets memo partitioning ──


def test_evaluate_l2_trial_cached_forwards_crisis_rets_and_partitions_memo(
    mocker: Any,
) -> None:
    crisis_a = np.asarray([0.01, -0.02], dtype=np.float64)
    crisis_b = np.asarray([0.01, -0.03], dtype=np.float64)
    memo: dict[tuple[Any, ...], Any] = {}
    evaluate = mocker.patch(
        "src.domain.futures.optimization.workflow.evaluate_l2_trial",
        side_effect=(object(), object()),
    )

    common = {
        "cache": object(),
        "signal_batch": object(),
        "aligned": object(),
        "awf_folds": (),
        "config": SimpleNamespace(),
        "caps": object(),
        "tf": "4h",
        "_memo": memo,
    }
    first = evaluate_l2_trial_cached(**common, crisis_rets=crisis_a)
    second = evaluate_l2_trial_cached(**common, crisis_rets=crisis_a)
    third = evaluate_l2_trial_cached(**common, crisis_rets=crisis_b)

    assert first is second
    assert third is not first
    assert evaluate.call_count == 2
    assert evaluate.call_args_list[0].kwargs["crisis_rets"] is crisis_a
    assert evaluate.call_args_list[1].kwargs["crisis_rets"] is crisis_b


# ── Scenario 2: select_layer2_champion order independence with crisis-calibrated replay ──
# ── Scenario 3: select_layer2_champion fails closed on parity or crisis context error ──


def _make_trial(
    number: int,
    value: float,
    cagr: float = 0.1,
    growth_lcb: float = 0.05,
    mdd: float = 0.2,
    leverage: float = 1.2,
    fold_pass: float = 0.8,
    trade_count: int = 100,
    constraints: tuple[float, ...] | None = None,
) -> optuna.trial.FrozenTrial:
    cv = constraints or (-1.0,) * 14
    return create_trial(
        params={"K_RANK": 1.0 + number, "l2_replay_max_fallbacks": 24.0},
        distributions={"K_RANK": FloatDistribution(1, 10), "l2_replay_max_fallbacks": FloatDistribution(1, 50)},
        values=[value],
        user_attrs={
            "l2_promotion_passed": True,
            "promotion_passed": True,
            "growth_lcb_hybrid": growth_lcb,
            "cagr_hybrid": cagr,
            "mdd_hybrid": mdd,
            "deploy_leverage": leverage,
            "fold_pass_ratio": fold_pass,
            "trade_count": trade_count,
            "l2_block_log_growth_signature": [float(value)],
            "sharpe_hac_hybrid": 1.0,
            "l2_constraint_values": list(cv),
            "l2_optuna_constraint_values": list(cv),
        },
        state=optuna.trial.TrialState.COMPLETE,
    )


def _make_evaluation(
    cagr: float = 0.1,
    growth_lcb: float = 0.05,
    mdd: float = 0.2,
    leverage: float = 1.2,
    fold_pass: float = 0.8,
    trade_count: int = 100,
    constraint_values: tuple[float, ...] | None = None,
    returns_hybrid: tuple[float, ...] = (0.001,),
) -> Any:
    cv = constraint_values or (-1.0,) * 14
    return SimpleNamespace(
        cagr_hybrid=cagr,
        cagr_baseline=0.05,
        growth_lcb_hybrid=growth_lcb,
        growth_lcb_baseline=0.03,
        mdd_hybrid=mdd,
        deploy_leverage=leverage,
        fold_pass_ratio=fold_pass,
        trade_count=trade_count,
        constraint_values=cv,
        returns_hybrid=returns_hybrid,
        objective_value=cagr,
        sharpe_hac_hybrid=1.0,
        sharpe_hac_baseline=0.0,
        sortino_hybrid=0.5,
        break_even_pass_pct=1.0,
        average_gross_exposure=0.5,
        cap_saturation_ratio=0.0,
        total_cost_bps=10.0,
        block_metrics=(),
        risk_utilization=0.5,
        deployment_objective_bonus=0.0,
        worst_fold_sharpe=0.0,
        psr_hybrid=1.0,
        cvar_95_hybrid=0.1,
        sharpe_hybrid=1.0,
        gate=SimpleNamespace(
            optuna_constraint_values=cv,
            promotion_constraint_values=(-1.0,) * 4,
            promotion_passed=True,
            promotion_blocker="",
        ),
    )


def test_select_layer2_champion_is_order_independent_with_crisis_calibrated_replay(
    mocker: Any,
) -> None:
    from src.domain.futures.optimization.workflow import CrisisReplayBudget
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    study = optuna.create_study(direction="maximize")
    trials = []
    for idx, value, growth_lcb in ((0, 0.30, 0.15), (1, 0.20, 0.10), (2, 0.10, 0.05)):
        trial = _make_trial(number=idx, value=value, cagr=value, growth_lcb=growth_lcb)
        trial.number = idx
        trials.append(trial)
    for trial in trials:
        study.add_trial(trial)

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=3.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._build_layer2_replay_frontier",
        return_value=trials,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.compute_crisis_replay_budget",
        return_value=CrisisReplayBudget(
            mdd_hybrid=0.10,
            mdd_budget=0.21,
            cagr_hybrid=-0.01,
            cagr_floor=-0.05,
        ),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_layer2_gate",
        return_value=SimpleNamespace(
            optuna_constraint_values=(-1.0,) * 13,
            promotion_constraint_values=(-1.0,) * 4,
            promotion_passed=True,
            promotion_blocker="",
        ),
    )

    def _evaluate(*, config: Any, **_: Any) -> Any:
        value = 0.4 - 0.1 * float(config.k_rank)
        return _make_evaluation(cagr=value, growth_lcb=value / 2.0)

    evaluate = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        side_effect=_evaluate,
    )
    crisis_rets = np.asarray([0.01, -0.02], dtype=np.float64)
    result = select_layer2_champion(
        study=study,
        tf="4h",
        signal_batch=mocker.MagicMock(start_idx=0, end_idx=100),
        aligned=mocker.MagicMock(),
        awf_folds=(),
        caps=mocker.MagicMock(),
        prebuilt_cache=mocker.MagicMock(),
        crisis_rets=crisis_rets,
        crisis_replay_ctx=SimpleNamespace(),
    )

    assert result.blocker_reason == ""
    assert result.best_trial_number == 0
    assert evaluate.call_count == 3
    assert all(call.kwargs["crisis_rets"] is crisis_rets for call in evaluate.call_args_list)


@pytest.mark.parametrize(
    ("crisis_rets", "crisis_ctx", "expected_reason"),
    [
        (np.asarray([0.0, 0.0], dtype=np.float64), None, "crisis_context_mismatch"),
        (None, SimpleNamespace(), "crisis_context_mismatch"),
    ],
)
def test_select_layer2_champion_fails_closed_on_parity_or_crisis_context_error(
    crisis_rets: np.ndarray[Any, np.dtype[np.float64]] | None,
    crisis_ctx: object | None,
    expected_reason: str,
    mocker: Any,
) -> None:
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=5.0,
    )

    study = optuna.create_study(direction="maximize")
    trial = _make_trial(number=0, value=0.1, cagr=0.1)
    study.add_trial(trial)

    result = select_layer2_champion(
        study=study,
        tf="4h",
        signal_batch=mocker.MagicMock(start_idx=0, end_idx=100),
        aligned=mocker.MagicMock(
            datetimes=np.array(["2020-01-01"], dtype="datetime64[ns]"),
            close_2d=np.ones((1, 1)),
            symbols=("BTCUSDT",),
        ),
        awf_folds=(),
        caps=mocker.MagicMock(),
        crisis_rets=crisis_rets,
        crisis_replay_ctx=crisis_ctx,
    )
    assert result.blocker_reason == expected_reason
    assert result.blocker_reason != ""


def test_select_layer2_champion_rejects_on_parity_mismatch(mocker: Any) -> None:
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=5.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._build_layer2_replay_frontier",
        side_effect=lambda trials, fallback_limit: list(trials),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        return_value=_make_evaluation(cagr=0.2, growth_lcb=0.1, mdd=0.3, leverage=1.5, fold_pass=0.9),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_layer2_gate",
        return_value=SimpleNamespace(
            optuna_constraint_values=(-1.0,) * 13,
            promotion_constraint_values=(-1.0,) * 4,
            promotion_passed=True,
            promotion_blocker="",
        ),
    )
    crisis_rets = np.asarray([0.01, -0.02], dtype=np.float64)

    study = optuna.create_study(direction="maximize")
    trial = _make_trial(number=0, value=0.1, cagr=0.1)
    study.add_trial(trial)

    result = select_layer2_champion(
        study=study,
        tf="4h",
        signal_batch=mocker.MagicMock(start_idx=0, end_idx=100),
        aligned=mocker.MagicMock(
            datetimes=np.array(["2020-01-01"], dtype="datetime64[ns]"),
            close_2d=np.ones((1, 1)),
            symbols=("BTCUSDT",),
        ),
        awf_folds=(),
        caps=mocker.MagicMock(),
        crisis_rets=crisis_rets,
        crisis_replay_ctx=SimpleNamespace(cache=object(), signal_batch=object(), aligned=object(), awf_folds=()),
    )
    assert result.blocker_reason == "replay_parity_divergence"
    assert result.blocker_reason != ""


def test_select_layer2_champion_rejects_on_crisis_unavailable(mocker: Any) -> None:
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=5.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._build_layer2_replay_frontier",
        side_effect=lambda trials, fallback_limit: list(trials),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        return_value=_make_evaluation(cagr=0.1, growth_lcb=0.05, mdd=0.2, leverage=1.2, fold_pass=0.8),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_layer2_gate",
        return_value=SimpleNamespace(
            optuna_constraint_values=(-1.0,) * 13,
            promotion_constraint_values=(-1.0,) * 4,
            promotion_passed=True,
            promotion_blocker="",
        ),
    )
    crisis_rets = np.asarray([0.01, -0.02], dtype=np.float64)

    study = optuna.create_study(direction="maximize")
    trial = _make_trial(number=0, value=0.1, cagr=0.1)
    study.add_trial(trial)

    result = select_layer2_champion(
        study=study,
        tf="4h",
        signal_batch=mocker.MagicMock(start_idx=0, end_idx=100),
        aligned=mocker.MagicMock(
            datetimes=np.array(["2020-01-01"], dtype="datetime64[ns]"),
            close_2d=np.ones((1, 1)),
            symbols=("BTCUSDT",),
        ),
        awf_folds=(),
        caps=mocker.MagicMock(),
        crisis_rets=crisis_rets,
        crisis_replay_ctx=SimpleNamespace(cache=object(), signal_batch=object(), aligned=object(), awf_folds=()),
    )
    assert result.blocker_reason == "crisis_replay_unavailable"
    assert result.blocker_reason != ""


# ── Scenario 4: active pipeline stop before final run when selection is blocked ──


def test_active_pipeline_stops_before_final_run_when_replay_selection_is_blocked(
    mocker: Any,
) -> None:
    from src.application.futures.runner.active_pipeline import _run_strategy_stage

    blocked = SimpleNamespace(
        blocker_reason="crisis_cagr",
        best_params={"K_RANK": 3},
        best_evaluation=SimpleNamespace(growth_lcb_hybrid=0.05),
    )
    from src.domain.futures.strategy.config import CandidateStrategyConfig, StrategyConfig

    _mock_candidate = CandidateStrategyConfig(timeframe="4h")
    _mock_candidate = dataclasses.replace(_mock_candidate, l1_tfs=("1h", "4h"), signal_only=False, seed=42)
    _mock_scfg = StrategyConfig(candidate=_mock_candidate)

    _mock_l1_result = SimpleNamespace(gate_passed=True, blocker_reason="", deploy_leverage=1.0)

    mocker.patch(
        "src.application.futures.runner.active_pipeline._run_tiered_l2_study",
        return_value=blocked,
    )
    final_run = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_tiered_pipeline",
        return_value=(_mock_l1_result, SimpleNamespace(gate_passed=True, blocker_reason=""), None),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline.OPT_FUTURES_CONFIG",
        {"USE_CS_RANK_ENGINE": True, "L2_META_FEAS": "0"},
        create=True,
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._resolve_layered_window",
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._resolve_base_symbol_scope",
        return_value=["BTCUSDT"],
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._resolve_tradeable_scope",
        return_value=SimpleNamespace(dropped_by_reason={}, admitted=["BTCUSDT"]),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline.build_candidate_strategy_config",
        return_value=_mock_scfg,
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline.run_active_strategy_output_bridge",
        return_value=SimpleNamespace(
            alpha_panel=None,
            rule_report={"selected_total": 0},
            labeled_unfiltered=[],
            l0_delivery_manifest=None,
        ),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._tiered_labeled_events",
        return_value=[],
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._has_l1_delivery_candidates",
        return_value=True,
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline.pick_strategy_data_maps",
        return_value={},
    )
    mocker.patch(
        "src.application.futures.runner.tiered_handoff.consume_candidate_output_for_tiered",
        return_value=SimpleNamespace(
            aligned=SimpleNamespace(
                datetimes=np.array(["2020-01-01"], dtype="datetime64[ns]"),
                close_2d=np.ones((1, 1)),
                symbols=("BTCUSDT",),
            ),
            labeled_events=[],
            aligned_by_tf={},
            labeled_events_by_tf={},
            l0_delivery_manifest=None,
        ),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._resolve_l2_master_tf_from_prior",
        return_value="4h",
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._build_l2_signal_batch",
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
    )
    crisis_replay_ctx = SimpleNamespace(cache=object(), signal_batch=object(), aligned=object(), awf_folds=())
    compute_crisis_rets = mocker.patch(
        "src.application.futures.runner.active_pipeline.compute_crisis_unit_returns",
        return_value=np.asarray([0.0, 0.0], dtype=np.float64),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._load_crisis_replay_context",
        return_value=crisis_replay_ctx,
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._compute_c2_macro",
        return_value=(12.0, 0.0),
    )

    run_config = SimpleNamespace(
        timeframe="4h",
        date="2020-06-01",
        phase="l2",
        trials=50,
        seed=42,
    )
    window = SimpleNamespace(
        fetch_start="2020-01-01",
        is_start="2020-01-01",
        end_date="2020-06-01",
        l2_start="2020-03-01",
        holdout_start="2020-05-01",
        holdout_end="2020-06-01",
    )
    data_stage = SimpleNamespace(
        valid_symbols=["BTCUSDT"],
        data_maps={"BTCUSDT": object()},
        oos_data_maps={"BTCUSDT": object()},
        effective_l0_evidence_end=None,
    )

    from src.application.futures.runner.models import RunnerResult

    result = _run_strategy_stage(run_config, window, data_stage)

    assert isinstance(result, RunnerResult)
    assert result.exit_code == 1
    assert result.reason == "seed_consensus_blocked:0/3"
    assert compute_crisis_rets.call_args.kwargs["crisis_replay_ctx"] is crisis_replay_ctx
    # The shared pipeline function was called once for L1; no second L2 call occurs.
    assert final_run.call_count == 1
