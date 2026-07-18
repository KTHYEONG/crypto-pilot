from __future__ import annotations

import optuna
from optuna.trial import create_trial
from optuna.distributions import FloatDistribution


def test_select_layer2_champion_user_attrs_pre_filter_uses_top_three_when_gate_passed(mocker):
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    mock_cache = mocker.MagicMock()
    mock_cache.vol_matrix_2d = None
    mock_cache.regime_policy_by_fold = ()

    study = optuna.create_study(direction="maximize")
    for i in range(5):
        val = 0.1 * (5 - i)
        constraint_val = -1.0 if i < 3 else 1.0
        trial = create_trial(
            params={"K_RANK": 1.0 + i},
            distributions={"K_RANK": FloatDistribution(1, 10)},
            values=[val],
            user_attrs={
                "l2_promotion_passed": i < 3,
                "growth_lcb_hybrid": float(val),
                "l2_block_log_growth_signature": [float(val)],
                "sharpe_hac_hybrid": 1.0,
                "l2_constraint_values": [constraint_val] * 10,
                "l2_optuna_constraint_values": [constraint_val] * 10,
            },
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(trial)

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._build_layer2_replay_frontier",
        return_value=list(study.trials),
    )

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_layer2_gate",
        return_value=mocker.MagicMock(
            optuna_constraint_values=(-1.0,),
            promotion_constraint_values=(-1.0,) * 4,
            promotion_passed=True,
        ),
    )

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=5.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        return_value=mocker.MagicMock(
            cagr_hybrid=0.1,
            growth_lcb_hybrid=0.05,
            mdd_hybrid=0.2,
            returns_hybrid=[0.001],
            constraint_values=(-1.0,) * 10,
            objective_value=0.05,
            sharpe_hac_hybrid=1.0,
            sharpe_hac_baseline=0.0,
            sortino_hybrid=0.5,
            fold_pass_ratio=1.0,
            break_even_pass_pct=1.0,
            average_gross_exposure=0.5,
            cap_saturation_ratio=0.0,
            total_cost_bps=10.0,
            block_metrics=(),
            trade_count=10,
            risk_utilization=0.5,
            deployment_objective_bonus=0.0,
            worst_fold_sharpe=0.0,
            gate=mocker.MagicMock(
                optuna_constraint_values=(-1.0,) * 10,
                promotion_constraint_values=(-1.0,) * 4,
                promotion_passed=True,
            ),
        ),
    )

    mock_aligned = mocker.MagicMock(datetimes=[], close_2d=None, symbols=())
    mock_signal_batch = mocker.MagicMock(start_idx=0, end_idx=10)

    result = select_layer2_champion(
        study=study,
        tf="4h",
        signal_batch=mock_signal_batch,
        aligned=mock_aligned,
        awf_folds=(),
        caps=mocker.MagicMock(),
        prebuilt_cache=mock_cache,
    )

    assert result is not None
    assert result.completed_trials == 5
    assert result.feasible_trials >= 3
