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
                "l2_constraint_values": [constraint_val] * 13,
                "l2_optuna_constraint_values": [constraint_val] * 13,
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
            optuna_constraint_values=(-1.0,) * 13,
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
            constraint_values=(-1.0,) * 13,
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
                optuna_constraint_values=(-1.0,) * 13,
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


def test_select_layer2_champion_replays_all_gate_passed_candidates_beyond_top_three(mocker):
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    mock_cache = mocker.MagicMock()
    mock_cache.vol_matrix_2d = None
    mock_cache.regime_policy_by_fold = ()

    study = optuna.create_study(direction="maximize")
    for i in range(7):
        val = 0.1 * (7 - i)
        trial = create_trial(
            params={"K_RANK": 1.0 + i, "l2_replay_max_fallbacks": 24},
            distributions={"K_RANK": FloatDistribution(1, 10), "l2_replay_max_fallbacks": FloatDistribution(1, 48)},
            values=[val],
            user_attrs={
                "l2_promotion_passed": True,
                "growth_lcb_hybrid": float(val),
                "l2_block_log_growth_signature": [float(val)],
                "sharpe_hac_hybrid": 1.0,
                "l2_constraint_values": [-1.0] * 13,
                "l2_optuna_constraint_values": [-1.0] * 13,
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
            optuna_constraint_values=(-1.0,) * 13,
            promotion_constraint_values=(-1.0,) * 4,
            promotion_passed=True,
        ),
    )

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=7.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        return_value=mocker.MagicMock(
            cagr_hybrid=0.1,
            growth_lcb_hybrid=0.05,
            mdd_hybrid=0.2,
            returns_hybrid=[0.001],
            constraint_values=(-1.0,) * 13,
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
                optuna_constraint_values=(-1.0,) * 13,
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
    assert result.completed_trials == 7


def test_select_layer2_champion_logs_replay_flip_warning(mocker, caplog):
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    import logging
    caplog.set_level(logging.WARNING)

    mock_cache = mocker.MagicMock()
    mock_cache.vol_matrix_2d = None
    mock_cache.regime_policy_by_fold = ()

    study = optuna.create_study(direction="maximize")
    trial = create_trial(
        params={"K_RANK": 1.0, "l2_replay_max_fallbacks": 24},
        distributions={"K_RANK": FloatDistribution(1, 10), "l2_replay_max_fallbacks": FloatDistribution(1, 48)},
        values=[0.1],
        user_attrs={
            "l2_promotion_passed": True,
            "cagr_hybrid": 0.2,
            "mdd_hybrid": 0.1,
            "growth_lcb_hybrid": 0.05,
            "l2_block_log_growth_signature": [0.05],
            "sharpe_hac_hybrid": 1.0,
            "l2_constraint_values": [-1.0] * 13,
            "l2_optuna_constraint_values": [-1.0] * 13,
        },
        state=optuna.trial.TrialState.COMPLETE,
    )
    study.add_trial(trial)

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._build_layer2_replay_frontier",
        return_value=list(study.trials),
    )

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=1.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        return_value=mocker.MagicMock(
            cagr_hybrid=0.05,
            growth_lcb_hybrid=0.02,
            mdd_hybrid=0.15,
            returns_hybrid=[0.001],
            constraint_values=(-1.0,) * 13,
            objective_value=0.05,
            sharpe_hac_hybrid=0.5,
            sharpe_hac_baseline=0.0,
            sortino_hybrid=0.2,
            fold_pass_ratio=0.5,
            break_even_pass_pct=0.5,
            average_gross_exposure=0.5,
            cap_saturation_ratio=0.0,
            total_cost_bps=10.0,
            block_metrics=(),
            trade_count=5,
            risk_utilization=0.5,
            deployment_objective_bonus=0.0,
            worst_fold_sharpe=0.0,
            gate=mocker.MagicMock(
                optuna_constraint_values=(0.0,) * 13,
                promotion_constraint_values=(0.0,) * 4,
                promotion_passed=False,
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
    assert any("event=replay_parity_violation" in r.message for r in caplog.records)
    assert result.blocker_reason != ""


def test_select_layer2_champion_rejects_candidate_with_crisis_mdd_over_budget(mocker):
    """[S1] crisis MDD가 예산 초과(0.45>>0.21) → candidate가 passed_candidates에서 제외됨."""
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    mock_cache = mocker.MagicMock()
    mock_cache.vol_matrix_2d = None
    mock_cache.regime_policy_by_fold = ()

    study = optuna.create_study(direction="maximize")
    trial = create_trial(
        params={"K_RANK": 1.0, "l2_replay_max_fallbacks": 24},
        distributions={"K_RANK": FloatDistribution(1, 10), "l2_replay_max_fallbacks": FloatDistribution(1, 48)},
        values=[0.1],
        user_attrs={
            "l2_promotion_passed": True,
            "cagr_hybrid": 0.2,
            "mdd_hybrid": 0.1,
            "growth_lcb_hybrid": 0.05,
            "l2_block_log_growth_signature": [0.05],
            "sharpe_hac_hybrid": 1.0,
            "l2_constraint_values": [-1.0] * 13,
            "l2_optuna_constraint_values": [-1.0] * 13,
        },
        state=optuna.trial.TrialState.COMPLETE,
    )
    study.add_trial(trial)

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._build_layer2_replay_frontier",
        return_value=list(study.trials),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=1.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        return_value=mocker.MagicMock(
            cagr_hybrid=0.05,
            growth_lcb_hybrid=0.02,
            mdd_hybrid=0.15,
            returns_hybrid=[0.001],
            constraint_values=(-1.0,) * 13,
            objective_value=0.05,
            sharpe_hac_hybrid=0.5,
            sharpe_hac_baseline=0.0,
            sortino_hybrid=0.2,
            fold_pass_ratio=0.5,
            break_even_pass_pct=0.5,
            average_gross_exposure=0.5,
            cap_saturation_ratio=0.0,
            total_cost_bps=10.0,
            block_metrics=(),
            trade_count=5,
            risk_utilization=0.5,
            deployment_objective_bonus=0.0,
            worst_fold_sharpe=0.0,
            deploy_leverage=1.0,
            gate=mocker.MagicMock(
                optuna_constraint_values=(-1.0,) * 13,
                promotion_constraint_values=(-1.0,) * 4,
                promotion_passed=True,
            ),
        ),
    )
    from src.domain.futures.optimization.workflow import CrisisReplayBudget
    # crisis MDD >> budget → crisis constraint violation
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.compute_crisis_replay_budget",
        return_value=CrisisReplayBudget(mdd_hybrid=0.45, mdd_budget=0.21, cagr_hybrid=-0.10, cagr_floor=-0.05),
    )
    mock_gate = mocker.MagicMock(
        optuna_constraint_values=(-1.0,) * 9 + (0.24, -1.0, -1.0, -1.0),  # 9th slot = crisis violation
        promotion_constraint_values=(-1.0,) * 4,
        promotion_passed=False,
        promotion_blocker="crisis_mdd_over_budget",
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_layer2_gate",
        return_value=mock_gate,
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
        crisis_rets=mocker.MagicMock(),
        crisis_replay_ctx=mocker.MagicMock(),
    )

    assert result is not None
    assert result.blocker_reason != ""


def test_select_layer2_champion_crisis_none_preserves_legacy_behavior(mocker):
    """[S2] crisis_rets=None, crisis_replay_ctx=None → gate 호출 시 crisis 인자가 None으로 전달되어 자동 통과."""
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    mock_cache = mocker.MagicMock()
    mock_cache.vol_matrix_2d = None
    mock_cache.regime_policy_by_fold = ()

    study = optuna.create_study(direction="maximize")
    trial = create_trial(
        params={"K_RANK": 1.0, "l2_replay_max_fallbacks": 24},
        distributions={"K_RANK": FloatDistribution(1, 10), "l2_replay_max_fallbacks": FloatDistribution(1, 48)},
        values=[0.1],
        user_attrs={
            "l2_promotion_passed": True,
            "cagr_hybrid": 0.2,
            "mdd_hybrid": 0.1,
            "growth_lcb_hybrid": 0.05,
            "l2_block_log_growth_signature": [0.05],
            "sharpe_hac_hybrid": 1.0,
            "l2_constraint_values": [-1.0] * 13,
            "l2_optuna_constraint_values": [-1.0] * 13,
        },
        state=optuna.trial.TrialState.COMPLETE,
    )
    study.add_trial(trial)

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._build_layer2_replay_frontier",
        return_value=list(study.trials),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=1.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        return_value=mocker.MagicMock(
            cagr_hybrid=0.05,
            growth_lcb_hybrid=0.02,
            mdd_hybrid=0.15,
            returns_hybrid=[0.001],
            constraint_values=(-1.0,) * 13,
            objective_value=0.05,
            sharpe_hac_hybrid=0.5,
            sharpe_hac_baseline=0.0,
            sortino_hybrid=0.2,
            fold_pass_ratio=0.5,
            break_even_pass_pct=0.5,
            average_gross_exposure=0.5,
            cap_saturation_ratio=0.0,
            total_cost_bps=10.0,
            block_metrics=(),
            trade_count=5,
            risk_utilization=0.5,
            deployment_objective_bonus=0.0,
            worst_fold_sharpe=0.0,
            deploy_leverage=1.0,
            gate=mocker.MagicMock(
                optuna_constraint_values=(-1.0,) * 13,
                promotion_constraint_values=(-1.0,) * 4,
                promotion_passed=True,
            ),
        ),
    )
    # evaluate_layer2_gate spy: crisis 인자가 None으로 전달되는지 확인
    mock_gate = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_layer2_gate",
        return_value=mocker.MagicMock(
            optuna_constraint_values=(-1.0,) * 13,
            promotion_constraint_values=(-1.0,) * 4,
            promotion_passed=True,
        ),
    )

    mock_aligned = mocker.MagicMock(datetimes=[], close_2d=None, symbols=())
    mock_signal_batch = mocker.MagicMock(start_idx=0, end_idx=10)

    # crisis 파라미터 미지정 (기본값 None)
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
    # evaluate_layer2_gate가 crisis_mdd_hybrid=None, crisis_mdd_budget=None으로 호출되었는지 확인
    _, gate_kwargs = mock_gate.call_args
    assert gate_kwargs.get("crisis_mdd_hybrid") is None
    assert gate_kwargs.get("crisis_mdd_budget") is None


def test_select_layer2_champion_rejects_candidate_below_crisis_cagr_floor(mocker):
    """[S4] crisis MDD는 통과하지만 crisis CAGR이 l2_min_crisis_cagr 미만인 trial → champion 탈락."""
    from src.domain.futures.optimization.workflow import CrisisReplayBudget
    from src.domain.futures.strategy.tiered_workflow.selection import select_layer2_champion

    mock_cache = mocker.MagicMock()
    mock_cache.vol_matrix_2d = None
    mock_cache.regime_policy_by_fold = ()

    study = optuna.create_study(direction="maximize")
    trial = create_trial(
        params={"K_RANK": 1.0, "l2_replay_max_fallbacks": 24},
        distributions={"K_RANK": FloatDistribution(1, 10), "l2_replay_max_fallbacks": FloatDistribution(1, 48)},
        values=[0.1],
        user_attrs={
            "l2_promotion_passed": True,
            "cagr_hybrid": 0.2,
            "mdd_hybrid": 0.1,
            "growth_lcb_hybrid": 0.05,
            "l2_block_log_growth_signature": [0.05],
            "sharpe_hac_hybrid": 1.0,
            "l2_constraint_values": [-1.0] * 13,
            "l2_optuna_constraint_values": [-1.0] * 13,
        },
        state=optuna.trial.TrialState.COMPLETE,
    )
    study.add_trial(trial)

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._build_layer2_replay_frontier",
        return_value=list(study.trials),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.calc_n_trials_eff_entropy",
        return_value=1.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial_cached",
        return_value=mocker.MagicMock(
            cagr_hybrid=0.05,
            growth_lcb_hybrid=0.02,
            mdd_hybrid=0.15,
            returns_hybrid=[0.001],
            constraint_values=(-1.0,) * 13,
            objective_value=0.05,
            sharpe_hac_hybrid=0.5,
            sharpe_hac_baseline=0.0,
            sortino_hybrid=0.2,
            fold_pass_ratio=0.5,
            break_even_pass_pct=0.5,
            average_gross_exposure=0.5,
            cap_saturation_ratio=0.0,
            total_cost_bps=10.0,
            block_metrics=(),
            trade_count=5,
            risk_utilization=0.5,
            deployment_objective_bonus=0.0,
            worst_fold_sharpe=0.0,
            deploy_leverage=1.0,
            gate=mocker.MagicMock(
                optuna_constraint_values=(-1.0,) * 13,
                promotion_constraint_values=(-1.0,) * 4,
                promotion_passed=True,
            ),
        ),
    )
    # crisis replay: MDD는 통과(0.15 < 0.21) but CAGR은 미달(-0.15 < -0.05)
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.compute_crisis_replay_budget",
        return_value=CrisisReplayBudget(mdd_hybrid=0.15, mdd_budget=0.21, cagr_hybrid=-0.15, cagr_floor=-0.05),
    )
    # gate: optuna_constraints[9]=MDD OK(<=0), [12]=CAGR FAIL(>0)
    mock_gate = mocker.MagicMock(
        optuna_constraint_values=(-1.0,) * 9 + (-1.0, -1.0, -1.0, 0.10),  # 12th slot = CAGR violation
        promotion_constraint_values=(-1.0,) * 4,
        promotion_passed=False,
        promotion_blocker="crisis_cagr_below_floor",
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.evaluate_layer2_gate",
        return_value=mock_gate,
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
        crisis_rets=mocker.MagicMock(),
        crisis_replay_ctx=mocker.MagicMock(),
    )

    assert result is not None
    assert result.blocker_reason != ""
