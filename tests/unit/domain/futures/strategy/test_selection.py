# tests/unit/domain/futures/strategy/test_selection.py
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest

from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2TrialEvaluation,
)
from src.domain.futures.strategy.tiered_workflow.selection import (
    _layer2_experiment_key,
    select_layer2_champion,
)


class TestLayer2Selection(unittest.TestCase):
    """select_layer2_champion 모듈에 대한 단위 테스트."""

    def setUp(self) -> None:
        self.tf = "4h"
        self.signal_batch = MagicMock()
        self.signal_batch.events = [MagicMock()]
        self.aligned = MagicMock()
        self.aligned.datetimes = np.array(["2026-06-15T00:00:00"], dtype="datetime64[ns]")
        self.awf_folds = (MagicMock(),)
        self.caps = MagicMock()

        # Mock window
        self.window = MagicMock()
        self.window.l2_start = MagicMock()
        self.window.l2_start.isoformat.return_value = "2026-06-15"
        self.window.holdout_start = MagicMock()
        self.window.holdout_start.isoformat.return_value = "2026-09-15"

    def test_experiment_key_generation(self) -> None:
        """_layer2_experiment_key의 고유 해시 기반 키 생성이 일관적인지 검증."""
        key1 = _layer2_experiment_key(
            tf=self.tf,
            window=self.window,
            signal_batch=self.signal_batch,
            search_space_version="v2",
        )
        key2 = _layer2_experiment_key(
            tf=self.tf,
            window=self.window,
            signal_batch=self.signal_batch,
            search_space_version="v2",
        )
        assert key1 == key2
        assert key1.startswith("l2_study_4h_")

    def test_select_champion_no_complete_trials(self) -> None:
        """완료된 trial이 없을 때 blocker_reason='no_complete_trials' 반환 검증."""
        study = MagicMock(spec=optuna.Study)
        study.trials = []

        res = select_layer2_champion(
            study=study,
            tf=self.tf,
            signal_batch=self.signal_batch,
            aligned=self.aligned,
            awf_folds=self.awf_folds,
            caps=self.caps,
        )
        assert res.blocker_reason == "no_complete_trials"
        assert res.best_trial_number is None

    @patch("src.domain.futures.strategy.tiered_workflow.selection.layer2_constraints_from_trial")
    def test_select_champion_no_feasible_trials(self, mock_constraints: MagicMock) -> None:
        """완료된 trial은 있으나 제약을 모두 위반한 경우 blocker_reason='no_feasible_trials' 반환 검증."""
        study = MagicMock(spec=optuna.Study)
        trial = MagicMock(spec=optuna.trial.FrozenTrial)
        trial.state = optuna.trial.TrialState.COMPLETE
        trial.value = 0.05
        trial.number = 0
        trial.user_attrs = {
            "l2_block_log_growth_signature": [0.01] * 11,
            "sharpe_hac_hybrid": 1.2,
        }
        trial.params = {"K_RANK": 3}
        study.trials = [trial]

        # 제약 위반: 1.0 (양수는 위반)
        mock_constraints.return_value = (1.0, 0.0)

        res = select_layer2_champion(
            study=study,
            tf=self.tf,
            signal_batch=self.signal_batch,
            aligned=self.aligned,
            awf_folds=self.awf_folds,
            caps=self.caps,
        )
        assert res.blocker_reason == "no_feasible_trials"
        assert res.best_trial_number == 0
        assert res.best_evaluation is None

    @patch("src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.selection.layer2_constraints_from_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.selection._deflated_sharpe_probability")
    def test_select_champion_success(
        self,
        mock_dsp: MagicMock,
        mock_constraints: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """objective(growth_lcb) 최상위 feasible trial이 검증을 통과해 챔피언으로 선정되는지 테스트."""
        study = MagicMock(spec=optuna.Study)
        trial = MagicMock(spec=optuna.trial.FrozenTrial)
        trial.state = optuna.trial.TrialState.COMPLETE
        trial.value = 0.15
        trial.number = 1
        trial.user_attrs = {
            "l2_block_log_growth_signature": [0.02] * 11,
            "sharpe_hac_hybrid": 2.1,
            "cagr_hybrid": 0.25,
            "growth_lcb_hybrid": 0.15,
            "mdd_hybrid": 0.05,
        }
        trial.params = {"K_RANK": 4}
        study.trials = [trial]

        # 제약 충족: 모든 값이 <= 0.0
        mock_constraints.return_value = (-1.0,) * 8

        # evaluate_l2_trial 반환 설정
        eval_mock = Layer2TrialEvaluation(
            objective_value=0.15,
            constraint_values=(-1.0,) * 8,
            cagr_hybrid=0.35,
            cagr_baseline=0.10,
            growth_lcb_hybrid=0.15,
            growth_lcb_baseline=0.08,
            sharpe_hac_hybrid=2.1,
            sharpe_hac_baseline=1.0,
            psr_hybrid=0.99,
            mdd_hybrid=0.05,
            cvar_95_hybrid=0.02,
            fold_pass_ratio=1.0,
            break_even_pass_pct=1.0,
            sortino_hybrid=1.5,
            trade_count=120,
            average_gross_exposure=1.0,
            cap_saturation_ratio=0.1,
            total_cost_bps=20.0,
            block_metrics=(MagicMock(), MagicMock(), MagicMock()),
            returns_hybrid=(0.01, 0.02),
            returns_baseline=(0.005, 0.01),
            sharpe_hybrid=2.3,
            sharpe_hac_baseline_ew=1.0,
        )
        mock_evaluate.return_value = eval_mock

        # DSR은 diagnostic으로만 첨부됨 (게이트 미적용)
        mock_dsp.return_value = 0.97

        res = select_layer2_champion(
            study=study,
            tf=self.tf,
            signal_batch=self.signal_batch,
            aligned=self.aligned,
            awf_folds=self.awf_folds,
            caps=self.caps,
        )

        assert res.blocker_reason == ""
        assert res.best_trial_number == 1
        assert res.dsr == 0.97
        assert res.best_evaluation is not None

    @patch("src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.selection.layer2_constraints_from_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.selection._deflated_sharpe_probability")
    def test_select_champion_low_dsr_does_not_block_promotion(
        self,
        mock_dsp: MagicMock,
        mock_constraints: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """DSR이 낮아도 dsr_floor는 더 이상 promotion을 차단하지 않는다 (diagnostic only).

        D4 변경: dsr_floor BLOCKER 제거, PSR gate 신설.
        DSR=0.30(낮음) + PSR=0.95(통과) → promotion 성공.
        """
        study = MagicMock(spec=optuna.Study)
        trial = MagicMock(spec=optuna.trial.FrozenTrial)
        trial.state = optuna.trial.TrialState.COMPLETE
        trial.value = 0.15
        trial.number = 2
        trial.user_attrs = {
            "l2_block_log_growth_signature": [0.02] * 11,
            "sharpe_hac_hybrid": 1.5,
            "cagr_hybrid": 0.35,
            "growth_lcb_hybrid": 0.10,
            "mdd_hybrid": 0.08,
        }
        trial.params = {"K_RANK": 2}
        study.trials = [trial]

        mock_constraints.return_value = (-1.0,) * 8

        eval_mock = Layer2TrialEvaluation(
            objective_value=0.10,
            constraint_values=(-1.0,) * 8,
            cagr_hybrid=0.35,
            cagr_baseline=0.10,
            growth_lcb_hybrid=0.10,
            growth_lcb_baseline=0.08,
            sharpe_hac_hybrid=1.5,
            sharpe_hac_baseline=1.0,
            psr_hybrid=0.95,
            mdd_hybrid=0.08,
            cvar_95_hybrid=0.03,
            fold_pass_ratio=1.0,
            break_even_pass_pct=1.0,
            sortino_hybrid=1.5,
            trade_count=120,
            average_gross_exposure=1.0,
            cap_saturation_ratio=0.1,
            total_cost_bps=20.0,
            block_metrics=(MagicMock(), MagicMock(), MagicMock()),
            returns_hybrid=(0.01, 0.015),
            returns_baseline=(0.005, 0.01),
            sharpe_hybrid=1.8,
            sharpe_hac_baseline_ew=1.0,
        )
        mock_evaluate.return_value = eval_mock

        mock_dsp.return_value = 0.30

        res = select_layer2_champion(
            study=study,
            tf=self.tf,
            signal_batch=self.signal_batch,
            aligned=self.aligned,
            awf_folds=self.awf_folds,
            caps=self.caps,
        )

        # DSR=0.30이지만 dsr_floor는 BLOCKER에서 제거됨 → 차단 없음
        assert res.blocker_reason == ""
        assert res.best_trial_number == 2
        assert res.dsr == pytest.approx(0.3)
        assert res.best_evaluation is not None

    @patch("src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.selection.layer2_constraints_from_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.selection._deflated_sharpe_probability")
    def test_select_champion_uses_replay_feasible_fallback_after_mismatch(
        self,
        mock_dsp: MagicMock,
        mock_constraints: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        study = MagicMock(spec=optuna.Study)
        trial_1 = MagicMock(spec=optuna.trial.FrozenTrial)
        trial_1.state = optuna.trial.TrialState.COMPLETE
        trial_1.value = 0.20
        trial_1.number = 10
        trial_1.user_attrs = {
            "l2_block_log_growth_signature": [0.03] * 8,
            "sharpe_hac_hybrid": 2.0,
            "cagr_hybrid": 0.31,
            "growth_lcb_hybrid": 0.20,
            "mdd_hybrid": 0.05,
        }
        trial_1.params = {"K_RANK": 4, "l2_replay_max_fallbacks": 3}

        trial_2 = MagicMock(spec=optuna.trial.FrozenTrial)
        trial_2.state = optuna.trial.TrialState.COMPLETE
        trial_2.value = 0.18
        trial_2.number = 11
        trial_2.user_attrs = {
            "l2_block_log_growth_signature": [0.025] * 8,
            "sharpe_hac_hybrid": 1.9,
            "cagr_hybrid": 0.28,
            "growth_lcb_hybrid": 0.18,
            "mdd_hybrid": 0.06,
        }
        trial_2.params = {"K_RANK": 3, "l2_replay_max_fallbacks": 3}
        study.trials = [trial_1, trial_2]
        mock_constraints.return_value = (-1.0,) * 8

        replay_infeasible = Layer2TrialEvaluation(
            objective_value=0.17,
            constraint_values=(1.0,) + (-1.0,) * 7,
            cagr_hybrid=0.34,
            cagr_baseline=0.10,
            growth_lcb_hybrid=0.17,
            growth_lcb_baseline=0.08,
            sharpe_hac_hybrid=1.8,
            sharpe_hac_baseline=1.0,
            psr_hybrid=0.98,
            mdd_hybrid=0.04,
            cvar_95_hybrid=0.02,
            fold_pass_ratio=1.0,
            break_even_pass_pct=1.0,
            sortino_hybrid=1.5,
            trade_count=120,
            average_gross_exposure=1.0,
            cap_saturation_ratio=0.1,
            total_cost_bps=20.0,
            block_metrics=(MagicMock(), MagicMock(), MagicMock()),
            returns_hybrid=(0.01, 0.02),
            returns_baseline=(0.005, 0.01),
            sharpe_hybrid=2.1,
            sharpe_hac_baseline_ew=1.0,
        )
        replay_feasible = Layer2TrialEvaluation(
            objective_value=0.18,
            constraint_values=(-1.0,) * 8,
            cagr_hybrid=0.35,
            cagr_baseline=0.10,
            growth_lcb_hybrid=0.18,
            growth_lcb_baseline=0.08,
            sharpe_hac_hybrid=1.9,
            sharpe_hac_baseline=1.0,
            psr_hybrid=0.98,
            mdd_hybrid=0.06,
            cvar_95_hybrid=0.02,
            fold_pass_ratio=1.0,
            break_even_pass_pct=1.0,
            sortino_hybrid=1.5,
            trade_count=120,
            average_gross_exposure=1.0,
            cap_saturation_ratio=0.1,
            total_cost_bps=20.0,
            block_metrics=(MagicMock(), MagicMock(), MagicMock()),
            returns_hybrid=(0.01, 0.02),
            returns_baseline=(0.005, 0.01),
            sharpe_hybrid=2.1,
            sharpe_hac_baseline_ew=1.0,
        )
        mock_evaluate.side_effect = [replay_infeasible, replay_feasible]
        mock_dsp.return_value = 0.88

        res = select_layer2_champion(
            study=study,
            tf=self.tf,
            signal_batch=self.signal_batch,
            aligned=self.aligned,
            awf_folds=self.awf_folds,
            caps=self.caps,
        )

        assert res.blocker_reason == ""
        assert res.best_trial_number == 11
        assert res.best_evaluation is replay_feasible

    @patch("src.domain.futures.strategy.tiered_workflow.selection.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.selection.layer2_constraints_from_trial")
    def test_select_champion_blocks_when_all_replays_infeasible(
        self,
        mock_constraints: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        study = MagicMock(spec=optuna.Study)
        trial = MagicMock(spec=optuna.trial.FrozenTrial)
        trial.state = optuna.trial.TrialState.COMPLETE
        trial.value = 0.21
        trial.number = 12
        trial.user_attrs = {
            "l2_block_log_growth_signature": [0.03] * 8,
            "sharpe_hac_hybrid": 2.0,
            "cagr_hybrid": 0.31,
            "growth_lcb_hybrid": 0.20,
            "mdd_hybrid": 0.05,
        }
        trial.params = {"K_RANK": 4, "l2_replay_max_fallbacks": 1}
        study.trials = [trial]
        mock_constraints.return_value = (-1.0,) * 8
        mock_evaluate.return_value = Layer2TrialEvaluation(
            objective_value=0.17,
            constraint_values=(1.0,) + (-1.0,) * 7,
            cagr_hybrid=0.34,
            cagr_baseline=0.10,
            growth_lcb_hybrid=0.17,
            growth_lcb_baseline=0.08,
            sharpe_hac_hybrid=1.8,
            sharpe_hac_baseline=1.0,
            psr_hybrid=0.98,
            mdd_hybrid=0.04,
            cvar_95_hybrid=0.02,
            fold_pass_ratio=1.0,
            break_even_pass_pct=1.0,
            sortino_hybrid=1.5,
            trade_count=120,
            average_gross_exposure=1.0,
            cap_saturation_ratio=0.1,
            total_cost_bps=20.0,
            block_metrics=(MagicMock(), MagicMock(), MagicMock()),
            returns_hybrid=(0.01, 0.02),
            returns_baseline=(0.005, 0.01),
            sharpe_hybrid=2.1,
            sharpe_hac_baseline_ew=1.0,
        )

        res = select_layer2_champion(
            study=study,
            tf=self.tf,
            signal_batch=self.signal_batch,
            aligned=self.aligned,
            awf_folds=self.awf_folds,
            caps=self.caps,
        )

        assert res.blocker_reason == "no_deployment"
        assert res.best_trial_number == 12
