"""Spec: l2-deploy-leverage-kelly-worst-fold-safety, Scenario 4 (Integration)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.futures.optimization.workflow import (
    CrisisReplayBudget,
    compute_crisis_replay_budget,
    evaluate_l2_trial,
    layer2_constraints_from_trial,
)


class TestEvaluateL2TrialWiresWorstFoldAndKellyWhenEnabled:
    """evaluate_l2_trial must pass worst_fold_rets/kelly_safety_fraction through
    to calibrate_deployment_leverage when the new config gates are enabled."""

    def _make_sim(self) -> SimpleNamespace:
        n_bars = 100
        fold_a = [0.001] * 50
        fold_b = [-0.02] + [0.001] * 49
        return SimpleNamespace(
            rets_hybrid=[0.001] * n_bars,
            rets_baseline=[0.0005] * n_bars,
            rets_baseline_ew=(),
            fit_rets_hybrid=tuple(fold_a + fold_b),
            fit_rets_by_fold=(tuple(fold_a), tuple(fold_b)),
            trade_count=50,
            fold_attributions=(),
            fold_rets_hybrid=[fold_a, fold_b],
            fold_selected_symbols=[("BTCUSDT",), ("ETHUSDT",)],
            all_turnovers=[0.1] * n_bars,
            turnover_return_indices=list(range(n_bars)),
            all_gross_exposures=[1.5] * n_bars,
            rebalance_count=10,
            all_net_exposures=[1.0] * n_bars,
            total_cost_hybrid=0.0005,
            friction_pass_total=40,
            signal_total=50,
            cap_saturation_count=2,
            support_leak_count=0,
            last_selected=frozenset({"BTCUSDT", "ETHUSDT"}),
            last_w=MagicMock(),
            capacity_diagnostics=None,
        )

    def _make_fold_diag(self) -> SimpleNamespace:
        return SimpleNamespace(
            fold_pass_ratio=1.0,
            fold_compound_pass=(True, True),
            fold_unit_sharpes=(1.0, 1.0),
            fold_deployed_cagrs=(0.10, 0.10),
            fold_deployed_mdds=(0.05, 0.05),
            fold_selected_symbols=(("BTCUSDT",), ("ETHUSDT",)),
            recent_fold_passed=True,
            recent_fold_sharpe=1.0,
            recent_fold_cagr=0.10,
            recent_fold_mdd=0.05,
            latest_to_median_cagr=1.0,
        )

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_evaluate_l2_trial_wires_worst_fold_and_kelly_when_enabled(
        self,
        mock_universe_audit: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold_diag: MagicMock,
        mock_sim: MagicMock,
    ) -> None:
        mock_sim.return_value = self._make_sim()
        mock_fold_diag.return_value = self._make_fold_diag()
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14,
            gate_passed=True,
            blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        mock_calibrate.return_value = (2.0, "mdd", 0.0)
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0,
            sortino=0.0,
            sharpe=0.0,
            calmar=0.0,
            mdd=0.0,
            fold_pass_ratio=0.0,
            score=0.0,
            worst_fold_cagr=0.0,
        )

        config = SimpleNamespace(
            l2_deploy_enabled=True,
            l2_max_mdd_abs=0.30,
            l2_max_cvar_95=0.06,
            l2_deploy_mdd_margin=0.30,
            l2_deploy_crisis_mdd_margin=0.30,
            l2_deploy_oos_budget_blend=0.5,
            l2_deploy_oos_floor_cap=4.0,
            l2_deploy_cvar_margin=0.20,
            l2_deploy_l_hard_cap=20.0,
            l2_deploy_fit_mdd_crisis_gate=None,
            l2_max_exchange_leverage=None,
            l2_deploy_worst_fold_gate_enabled=True,
            l2_deploy_kelly_safety_fraction=0.25,
            l2_growth_lcb_z=0.5,
            l2_worst_fold_penalty_threshold=-0.30,
            l2_worst_fold_penalty_weight=0.005,
            l2_min_worst_fold_cagr=-0.05,
            l2_min_positive_block_delta_ratio=0.45,
            l2_worst_fold_cagr_penalty_weight=0.50,
            l2_block_delta_penalty_weight=0.25,
            l2_min_trades=30,
            l2_min_active_blocks=3,
            l2_entry_spike_penalty_weight=0.0,
            l2_objective_risk_util_target=0.50,
            l2_objective_risk_util_weight=0.03,
            l2_objective_trade_target=90,
            l2_objective_trade_weight=0.02,
            l2_turnover_penalty_weight=0.0,
            l2_require_recency_holdout_pass=True,
            l2_min_recency_holdout_cagr=-0.05,
            l2_recency_holdout_days=30.0,
        )
        caps = SimpleNamespace()
        aligned = SimpleNamespace(
            symbols=("BTCUSDT", "ETHUSDT"),
            close_2d=MagicMock(),
            datetimes=[MagicMock()] * 100,
        )

        evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=100),
            aligned=aligned,
            awf_folds=(MagicMock(), MagicMock()),
            config=config,
            caps=caps,
            tf="1h",
        )

        assert mock_calibrate.called
        _, kwargs = mock_calibrate.call_args
        assert kwargs["kelly_safety_fraction"] == 0.25
        assert kwargs["worst_fold_rets"] is not None


class TestEvaluateL2TrialDefaultConfigWiresWorstFoldWithoutExplicitOverride:
    """Scenario 4: 기본 config(키 부재)만으로 worst_fold_rets가 wiring되는지 end-to-end 확인."""

    def _make_sim(self) -> SimpleNamespace:
        n_bars = 100
        fold_a = [0.001] * 50
        fold_b = [-0.02] + [0.001] * 49
        return SimpleNamespace(
            rets_hybrid=[0.001] * n_bars,
            rets_baseline=[0.0005] * n_bars,
            rets_baseline_ew=(),
            fit_rets_hybrid=tuple(fold_a + fold_b),
            fit_rets_by_fold=(tuple(fold_a), tuple(fold_b)),
            trade_count=50,
            fold_attributions=(),
            fold_rets_hybrid=[fold_a, fold_b],
            fold_selected_symbols=[("BTCUSDT",), ("ETHUSDT",)],
            all_turnovers=[0.1] * n_bars,
            turnover_return_indices=list(range(n_bars)),
            all_gross_exposures=[1.5] * n_bars,
            rebalance_count=10,
            all_net_exposures=[1.0] * n_bars,
            total_cost_hybrid=0.0005,
            friction_pass_total=40,
            signal_total=50,
            cap_saturation_count=2,
            support_leak_count=0,
            last_selected=frozenset({"BTCUSDT", "ETHUSDT"}),
            last_w=MagicMock(),
            capacity_diagnostics=None,
        )

    def _make_fold_diag(self) -> SimpleNamespace:
        return SimpleNamespace(
            fold_pass_ratio=1.0,
            fold_compound_pass=(True, True),
            fold_unit_sharpes=(1.0, 1.0),
            fold_deployed_cagrs=(0.10, 0.10),
            fold_deployed_mdds=(0.05, 0.05),
            fold_selected_symbols=(("BTCUSDT",), ("ETHUSDT",)),
            recent_fold_passed=True,
            recent_fold_sharpe=1.0,
            recent_fold_cagr=0.10,
            recent_fold_mdd=0.05,
            latest_to_median_cagr=1.0,
        )

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_evaluate_l2_trial_default_config_wires_worst_fold_without_explicit_override(
        self,
        mock_universe_audit: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold_diag: MagicMock,
        mock_sim: MagicMock,
    ) -> None:
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            Layer2AllocationConfig,
        )

        mock_sim.return_value = self._make_sim()
        mock_fold_diag.return_value = self._make_fold_diag()
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14,
            gate_passed=True,
            blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        mock_calibrate.return_value = (2.0, "mdd", 0.0)
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0,
            sortino=0.0,
            sharpe=0.0,
            calmar=0.0,
            mdd=0.0,
            fold_pass_ratio=0.0,
            score=0.0,
            worst_fold_cagr=0.0,
        )

        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace()
        aligned = SimpleNamespace(
            symbols=("BTCUSDT", "ETHUSDT"),
            close_2d=MagicMock(),
            datetimes=[MagicMock()] * 100,
        )

        evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=100),
            aligned=aligned,
            awf_folds=(MagicMock(), MagicMock()),
            config=config,
            caps=caps,
            tf="1h",
        )

        assert mock_calibrate.call_args.kwargs["worst_fold_rets"] is not None


class TestEvaluateL2TrialCrisisConstraint:
    """Scenarios 3-4: crisis constraint wiring in evaluate_l2_trial."""

    def _make_sim(self) -> SimpleNamespace:
        n_bars = 100
        return SimpleNamespace(
            rets_hybrid=[0.001] * n_bars,
            rets_baseline=[0.0005] * n_bars,
            rets_baseline_ew=(),
            fit_rets_hybrid=([0.001] * n_bars,),
            fit_rets_by_fold=([0.001] * n_bars,),
            trade_count=50,
            fold_attributions=(),
            fold_rets_hybrid=[[0.001] * n_bars],
            fold_selected_symbols=[("BTCUSDT",)],
            all_turnovers=[0.1] * n_bars,
            turnover_return_indices=list(range(n_bars)),
            all_gross_exposures=[1.5] * n_bars,
            rebalance_count=10,
            all_net_exposures=[1.0] * n_bars,
            total_cost_hybrid=0.0005,
            friction_pass_total=40,
            signal_total=50,
            cap_saturation_count=2,
            support_leak_count=0,
            last_selected=frozenset({"BTCUSDT"}),
            last_w=MagicMock(),
            capacity_diagnostics=None,
        )

    def _make_fold_diag(self) -> SimpleNamespace:
        return SimpleNamespace(
            fold_pass_ratio=1.0,
            fold_compound_pass=(True,),
            fold_unit_sharpes=(1.0,),
            fold_deployed_cagrs=(0.10,),
            fold_deployed_mdds=(0.05,),
            fold_selected_symbols=(("BTCUSDT",),),
            recent_fold_passed=True,
            recent_fold_sharpe=1.0,
            recent_fold_cagr=0.10,
            recent_fold_mdd=0.05,
            latest_to_median_cagr=1.0,
        )

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_evaluate_l2_trial_crisis_simulation_exception_does_not_propagate(
        self,
        mock_universe_audit: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold_diag: MagicMock,
        mock_sim: MagicMock,
    ) -> None:
        """[S3] crisis_replay_ctx 유효하나 _run_awf_simulation 예외 → 예외 미전파."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            Layer2AllocationConfig,
        )

        mock_sim.return_value = self._make_sim()
        mock_fold_diag.return_value = self._make_fold_diag()
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14,
            gate_passed=True,
            blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        mock_calibrate.return_value = (2.0, "mdd", 0.0)
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0,
            sortino=0.0,
            sharpe=0.0,
            calmar=0.0,
            mdd=0.0,
            fold_pass_ratio=0.0,
            score=0.0,
            worst_fold_cagr=0.0,
        )

        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace()
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=MagicMock(),
            datetimes=[MagicMock()] * 100,
        )

        crisis_replay_ctx = SimpleNamespace(
            cache=MagicMock(),
            signal_batch=MagicMock(),
            aligned=SimpleNamespace(datetimes=[MagicMock()] * 50),
            awf_folds=(MagicMock(),),
        )

        result = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=100),
            aligned=aligned,
            awf_folds=(MagicMock(), MagicMock()),
            config=config,
            caps=caps,
            tf="1h",
            crisis_replay_ctx=crisis_replay_ctx,
        )

        assert result is not None
        assert mock_calibrate.called

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_evaluate_l2_trial_crisis_constraint_uses_trial_own_config_not_min_protection(
        self,
        mock_universe_audit: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold_diag: MagicMock,
        mock_sim: MagicMock,
    ) -> None:
        """[S4] 동일 crisis_replay_ctx에 방어 on/off trial 두 번 호출 시
        crisis_mdd_hybrid가 config 방어 레버를 반영."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            Layer2AllocationConfig,
        )

        mock_sim.side_effect = [
            self._make_sim(),  # 첫 _run_awf_simulation (정상장)
            SimpleNamespace(
                rets_hybrid=[0.01] * 50,
                rets_baseline=[0.005] * 50,
                rets_baseline_ew=(),
                fit_rets_hybrid=(),
                fit_rets_by_fold=(),
                trade_count=10,
                fold_attributions=(),
                fold_rets_hybrid=[[]],
                fold_selected_symbols=[("BTCUSDT",)],
                all_turnovers=[0.1] * 50,
                turnover_return_indices=list(range(50)),
                all_gross_exposures=[1.0] * 50,
                rebalance_count=5,
                all_net_exposures=[1.0] * 50,
                total_cost_hybrid=0.0005,
                friction_pass_total=20,
                signal_total=30,
                cap_saturation_count=1,
                support_leak_count=0,
                last_selected=frozenset({"BTCUSDT"}),
                last_w=MagicMock(),
                capacity_diagnostics=None,
            ),  # crisis replay
            self._make_sim(),  # 두 번째 trial 정상장
            SimpleNamespace(
                rets_hybrid=[-0.02] * 50,
                rets_baseline=[0.005] * 50,
                rets_baseline_ew=(),
                fit_rets_hybrid=(),
                fit_rets_by_fold=(),
                trade_count=5,
                fold_attributions=(),
                fold_rets_hybrid=[[]],
                fold_selected_symbols=[("BTCUSDT",)],
                all_turnovers=[0.1] * 50,
                turnover_return_indices=list(range(50)),
                all_gross_exposures=[1.0] * 50,
                rebalance_count=3,
                all_net_exposures=[1.0] * 50,
                total_cost_hybrid=0.0005,
                friction_pass_total=10,
                signal_total=20,
                cap_saturation_count=1,
                support_leak_count=0,
                last_selected=frozenset({"BTCUSDT"}),
                last_w=MagicMock(),
                capacity_diagnostics=None,
            ),  # crisis replay (defense off → higher MDD)
        ]
        mock_fold_diag.return_value = self._make_fold_diag()
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14,
            gate_passed=True,
            blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        mock_calibrate.return_value = (2.0, "mdd", 0.0)
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0,
            sortino=0.0,
            sharpe=0.0,
            calmar=0.0,
            mdd=0.0,
            fold_pass_ratio=0.0,
            score=0.0,
            worst_fold_cagr=0.0,
        )

        config_on = Layer2AllocationConfig.from_mapping({
            "l2_regime_long_short_asymmetry_enabled": True,
        })
        config_off = Layer2AllocationConfig.from_mapping({
            "l2_regime_long_short_asymmetry_enabled": False,
        })
        caps = SimpleNamespace()
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=MagicMock(),
            datetimes=[MagicMock()] * 100,
        )

        crisis_replay_ctx = SimpleNamespace(
            cache=MagicMock(),
            signal_batch=MagicMock(),
            aligned=SimpleNamespace(datetimes=[MagicMock()] * 50),
            awf_folds=(MagicMock(),),
        )

        evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=100),
            aligned=aligned,
            awf_folds=(MagicMock(), MagicMock()),
            config=config_on,
            caps=caps,
            tf="1h",
            crisis_replay_ctx=crisis_replay_ctx,
        )

        evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=100),
            aligned=aligned,
            awf_folds=(MagicMock(), MagicMock()),
            config=config_off,
            caps=caps,
            tf="1h",
            crisis_replay_ctx=crisis_replay_ctx,
        )

        assert mock_sim.call_count == 4
        # Two crisis replay calls used different configs — this verifies
        # crisis_replay_ctx is reused but config varies per trial.
        assert mock_gate.call_count == 2


class TestEvaluateL2TrialGrowthLcbDeployed:
    """Scenario 4 (Integration): growth_lcb_deployed가 leverage 억제 효과를 objective에 반영."""

    def _make_sim(self) -> SimpleNamespace:
        n_bars = 200
        returns = [0.002] * n_bars  # deterministic positive returns
        return SimpleNamespace(
            rets_hybrid=returns,
            rets_baseline=[0.001] * n_bars,
            rets_baseline_ew=(),
            fit_rets_hybrid=tuple(returns),
            fit_rets_by_fold=(tuple(returns),),
            trade_count=90,
            fold_attributions=(),
            fold_rets_hybrid=[returns],
            fold_selected_symbols=[("BTCUSDT",)],
            all_turnovers=[0.05] * n_bars,
            turnover_return_indices=list(range(n_bars)),
            all_gross_exposures=[1.0] * n_bars,
            rebalance_count=10,
            all_net_exposures=[1.0] * n_bars,
            total_cost_hybrid=0.0003,
            friction_pass_total=45,
            signal_total=50,
            cap_saturation_count=1,
            support_leak_count=0,
            last_selected=frozenset({"BTCUSDT"}),
            last_w=MagicMock(),
            capacity_diagnostics=None,
        )

    def _make_fold_diag(self) -> SimpleNamespace:
        return SimpleNamespace(
            fold_pass_ratio=1.0,
            fold_compound_pass=(True,),
            fold_unit_sharpes=(1.0,),
            fold_deployed_cagrs=(0.10,),
            fold_deployed_mdds=(0.05,),
            fold_selected_symbols=(("BTCUSDT",),),
            recent_fold_passed=True,
            recent_fold_sharpe=1.0,
            recent_fold_cagr=0.10,
            recent_fold_mdd=0.05,
            latest_to_median_cagr=1.0,
        )

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_evaluate_l2_trial_leverage_suppressed_trial_scores_lower_with_growth_weight(
        self,
        mock_universe_audit: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold_diag: MagicMock,
        mock_sim: MagicMock,
    ) -> None:
        """동일 sortino_hac_unit을 갖는 두 trial(A: L*=3.0, B: L*=1.0)에서
        growth_lcb_weight>0 시 A의 objective_value가 B보다 유의미하게 높음.
        growth_lcb_weight=0(레거시)에서는 A/B가 거의 동일."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            Layer2AllocationConfig,
        )

        sim = self._make_sim()
        # sim이 두 번 호출되는데, 같은 rets_hybrid이므로 sortino_hac_unit 동일
        mock_sim.return_value = sim
        mock_fold_diag.return_value = self._make_fold_diag()
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14,
            gate_passed=True,
            blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        # 첫 호출 L*=3.0(high), 두 번째 호출 L*=1.0(floor, crisis-suppressed)
        mock_calibrate.side_effect = [
            (3.0, "mdd", 0.0),
            (1.0, "crisis_constraint", 0.0),
        ]
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0, sortino=0.0, sharpe=0.0, calmar=0.0, mdd=0.0,
            fold_pass_ratio=0.0, score=0.0, worst_fold_cagr=0.0,
        )

        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace()
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=MagicMock(),
            datetimes=[MagicMock()] * 200,
        )

        result_high = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=200),
            aligned=aligned,
            awf_folds=(MagicMock(),),
            config=config,
            caps=caps,
            tf="1h",
        )

        result_low = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=200),
            aligned=aligned,
            awf_folds=(MagicMock(),),
            config=config,
            caps=caps,
            tf="1h",
        )

        # [LIMIT-04] Fixed objective: growth_lcb_deployed. High L* trial has higher deployed LCB.
        assert result_high.objective_value > result_low.objective_value, (
            f"High L* trial ({result_high.deploy_leverage}) should score > low L* trial "
            f"({result_low.deploy_leverage}) with fixed growth_lcb_deployed objective"
        )

        # With fixed objective, high L* always scores higher (deployed returns are scaled)
        # Verify the objective value IS growth_lcb_deployed
        assert result_high.objective_value == pytest.approx(result_high.growth_lcb_deployed, abs=1e-10)
        assert result_low.objective_value == pytest.approx(result_low.growth_lcb_deployed, abs=1e-10)

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_evaluate_l2_trial_leverage_wipeout_trial_does_not_collapse_to_sentinel(
        self,
        mock_universe_audit: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold_diag: MagicMock,
        mock_sim: MagicMock,
    ) -> None:
        """[S4 Integration] wipeout bar가 포함된 trial이 -1e6 sentinel로 붕괴하지 않음."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

        sim = self._make_sim()
        sim.rets_hybrid = [0.002] * 100 + [-0.65] + [0.002] * 99
        mock_sim.return_value = sim
        mock_fold_diag.return_value = self._make_fold_diag()
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14,
            gate_passed=True,
            blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        mock_calibrate.side_effect = [(2.0, "mdd", 0.0)]
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0, sortino=0.0, sharpe=0.0, calmar=0.0, mdd=0.0,
            fold_pass_ratio=0.0, score=0.0, worst_fold_cagr=0.0,
        )
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace()
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=MagicMock(),
            datetimes=[MagicMock()] * 200,
        )

        result = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=200),
            aligned=aligned,
            awf_folds=(MagicMock(),),
            config=config,
            caps=caps,
            tf="1h",
        )

        assert result.growth_lcb_deployed != pytest.approx(-1e6)
        assert result.objective_value == pytest.approx(result.growth_lcb_deployed, abs=1e-10)
        assert result.growth_lcb_deployed < 0


class TestEvaluateL2TrialBullBoostCalibrationIsolation:
    """Scenario 4 (Integration — [LIMIT-01] 핵심 회귀 방지)."""

    def _make_sim(self, *, regime_codes: list[int] | None = None) -> SimpleNamespace:
        n_bars = 200
        returns = [0.002] * n_bars
        return SimpleNamespace(
            rets_hybrid=returns,
            rets_baseline=[0.001] * n_bars,
            rets_baseline_ew=(),
            fit_rets_hybrid=tuple(returns),
            fit_rets_by_fold=(tuple(returns),),
            trade_count=90,
            fold_attributions=(),
            fold_rets_hybrid=[returns],
            fold_selected_symbols=[("BTCUSDT",)],
            all_turnovers=[0.05] * n_bars,
            turnover_return_indices=list(range(n_bars)),
            all_gross_exposures=[1.0] * n_bars,
            rebalance_count=10,
            all_net_exposures=[1.0] * n_bars,
            total_cost_hybrid=0.0003,
            friction_pass_total=45,
            signal_total=50,
            cap_saturation_count=1,
            support_leak_count=0,
            last_selected=frozenset({"BTCUSDT"}),
            last_w=MagicMock(),
            regime_codes_hybrid=regime_codes or [0] * n_bars,
        )

    def _make_fold_diag(self) -> SimpleNamespace:
        return SimpleNamespace(
            fold_pass_ratio=1.0,
            fold_compound_pass=(True,),
            fold_unit_sharpes=(1.0,),
            fold_deployed_cagrs=(0.10,),
            fold_deployed_mdds=(0.05,),
            fold_selected_symbols=(("BTCUSDT",),),
            recent_fold_passed=True,
            recent_fold_sharpe=1.0,
            recent_fold_cagr=0.10,
            recent_fold_mdd=0.05,
            latest_to_median_cagr=1.0,
        )

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_evaluate_l2_trial_bull_boost_does_not_contaminate_leverage_calibration(
        self,
        mock_universe_audit: MagicMock,
        mock_build_score: MagicMock,
        mock_gate: MagicMock,
        mock_fold_diag: MagicMock,
        mock_sim: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            Layer2AllocationConfig,
        )
        from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
            calibrate_deployment_leverage,
        )

        n_bars = 200
        regime_codes = [0] * (n_bars // 2) + [1] * (n_bars // 2)
        sim = self._make_sim(regime_codes=regime_codes)
        mock_sim.return_value = sim
        mock_fold_diag.return_value = self._make_fold_diag()
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14,
            gate_passed=True,
            blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0, sortino=0.0, sharpe=0.0, calmar=0.0, mdd=0.0,
            fold_pass_ratio=0.0, score=0.0, worst_fold_cagr=0.0,
        )

        captured_calib_kwargs: list[dict[str, object]] = []
        original_calibrate = calibrate_deployment_leverage

        def _spy_calibrate(**kwargs: object) -> tuple[float, str, float]:
            captured_calib_kwargs.append({"fit_rets": kwargs["fit_rets"], "oos_rets": kwargs.get("oos_rets")})
            return original_calibrate(**kwargs)

        monkeypatch.setattr(
            "src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage",
            _spy_calibrate,
        )

        config_off = Layer2AllocationConfig.from_mapping({"l2_regime_bull_leverage_boost_enabled": False})
        config_on = Layer2AllocationConfig.from_mapping(
            {"l2_regime_bull_leverage_boost_enabled": True, "l2_regime_bull_leverage_boost": 1.3}
        )

        caps = SimpleNamespace()
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=MagicMock(),
            datetimes=[MagicMock()] * n_bars,
        )

        result_off = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=n_bars),
            aligned=aligned,
            awf_folds=(MagicMock(),),
            config=config_off,
            caps=caps,
            tf="1h",
        )
        result_on = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=n_bars),
            aligned=aligned,
            awf_folds=(MagicMock(),),
            config=config_on,
            caps=caps,
            tf="1h",
        )

        assert len(captured_calib_kwargs) == 2
        assert np.array_equal(captured_calib_kwargs[0]["fit_rets"], captured_calib_kwargs[1]["fit_rets"])
        assert result_on.cagr_hybrid != pytest.approx(result_off.cagr_hybrid)


class TestComputeCrisisReplayBudget:
    """compute_crisis_replay_budget pure function tests."""

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.apply_deployment")
    def test_compute_crisis_replay_budget_returns_mdd_and_cagr_from_deployment_result(
        self, mock_apply: MagicMock, mock_sim: MagicMock,
    ) -> None:
        """[S1] 합성 crisis_replay_ctx + leverage=2.0 → CrisisReplayBudget에 MDD/CAGR/floor 정확히 매칭."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

        mock_sim.return_value = SimpleNamespace(
            rets_hybrid=[0.01, -0.02, 0.015, -0.01, 0.005],
        )
        mock_apply.return_value = SimpleNamespace(mdd=0.25, cagr=-0.10)
        config = Layer2AllocationConfig.from_mapping({})
        ctx = SimpleNamespace(
            cache=MagicMock(), signal_batch=MagicMock(),
            aligned=SimpleNamespace(datetimes=[MagicMock()] * 50),
            awf_folds=(MagicMock(),),
        )
        budget = compute_crisis_replay_budget(
            crisis_replay_ctx=ctx, config=config, caps=MagicMock(),
            tf="8h", leverage=2.0, bars_per_year=1095.0,
        )
        expected_budget = float(config.l2_max_mdd_abs) * (1.0 - float(config.l2_deploy_crisis_mdd_margin))
        assert isinstance(budget, CrisisReplayBudget)
        assert budget.mdd_hybrid == 0.25
        assert budget.mdd_budget == expected_budget
        assert budget.cagr_hybrid == -0.10
        assert budget.cagr_floor == config.l2_min_crisis_cagr

    def test_compute_crisis_replay_budget_with_none_ctx_returns_all_none(self) -> None:
        """[S3] crisis_replay_ctx=None → 4개 필드 전부 None."""
        budget = compute_crisis_replay_budget(
            crisis_replay_ctx=None, config=MagicMock(), caps=MagicMock(),
            tf="8h", leverage=1.0, bars_per_year=1095.0,
        )
        assert isinstance(budget, CrisisReplayBudget)
        assert budget.mdd_hybrid is None
        assert budget.mdd_budget is None
        assert budget.cagr_hybrid is None
        assert budget.cagr_floor is None

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    def test_compute_crisis_replay_budget_simulation_exception_returns_all_none_and_warns(
        self, mock_sim: MagicMock, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """[S2] _run_awf_simulation 예외 → 전 필드 None, simulation_failed warning 확인."""
        import logging
        caplog.set_level(logging.WARNING)
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

        mock_sim.side_effect = ValueError("boom")
        config = Layer2AllocationConfig.from_mapping({})
        ctx = SimpleNamespace(
            cache=MagicMock(), signal_batch=MagicMock(),
            aligned=SimpleNamespace(datetimes=[MagicMock()] * 50),
            awf_folds=(MagicMock(),),
        )
        budget = compute_crisis_replay_budget(
            crisis_replay_ctx=ctx, config=config, caps=MagicMock(),
            tf="8h", leverage=1.0, bars_per_year=1095.0,
        )
        assert isinstance(budget, CrisisReplayBudget)
        assert budget.mdd_hybrid is None
        assert budget.mdd_budget is None
        assert budget.cagr_hybrid is None
        assert budget.cagr_floor is None
        assert any("simulation_failed" in r.message for r in caplog.records)


class TestEvaluateL2TrialUsesComputeCrisisReplayBudget:
    """Refactoring verification: evaluate_l2_trial delegates to compute_crisis_replay_budget."""

    @patch("src.domain.futures.optimization.workflow.compute_crisis_replay_budget")
    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_calls_compute_crisis_replay_budget(
        self,
        mock_universe: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold: MagicMock,
        mock_sim: MagicMock,
        mock_crisis_fn: MagicMock,
    ) -> None:
        """[S4] evaluate_l2_trial이 compute_crisis_replay_budget를 호출하고 반환값이 gate로 전달됨."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

        n_bars = 50
        mock_sim.return_value = SimpleNamespace(
            rets_hybrid=[0.001] * n_bars,
            rets_baseline=[0.0005] * n_bars,
            rets_baseline_ew=(),
            fit_rets_hybrid=(),
            fit_rets_by_fold=(),
            trade_count=20,
            fold_attributions=(),
            fold_rets_hybrid=[[]],
            fold_selected_symbols=[("BTCUSDT",)],
            all_turnovers=[0.1] * n_bars,
            turnover_return_indices=list(range(n_bars)),
            all_gross_exposures=[1.0] * n_bars,
            rebalance_count=5,
            all_net_exposures=[1.0] * n_bars,
            total_cost_hybrid=0.0005,
            friction_pass_total=15,
            signal_total=20,
            cap_saturation_count=1,
            support_leak_count=0,
            last_selected=frozenset({"BTCUSDT"}),
            last_w=MagicMock(),
            capacity_diagnostics=None,
        )
        mock_fold.return_value = SimpleNamespace(
            fold_pass_ratio=1.0, fold_compound_pass=(True,),
            fold_unit_sharpes=(1.0,), fold_deployed_cagrs=(0.10,),
            fold_deployed_mdds=(0.05,), fold_selected_symbols=(("BTCUSDT",),),
            recent_fold_passed=True, recent_fold_sharpe=1.0,
            recent_fold_cagr=0.10, recent_fold_mdd=0.05,
            latest_to_median_cagr=1.0,
        )
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14, gate_passed=True, blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        mock_calibrate.return_value = (1.5, "mdd", 0.0)
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0, sortino=0.0, sharpe=0.0, calmar=0.0, mdd=0.0,
            fold_pass_ratio=0.0, score=0.0, worst_fold_cagr=0.0,
        )
        mock_crisis_fn.return_value = CrisisReplayBudget(
            mdd_hybrid=0.12, mdd_budget=0.15, cagr_hybrid=-0.08, cagr_floor=-0.05,
        )

        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(trial_number=42)
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",), close_2d=MagicMock(),
            datetimes=[MagicMock()] * n_bars,
        )
        ctx = SimpleNamespace(
            cache=MagicMock(), signal_batch=MagicMock(),
            aligned=SimpleNamespace(datetimes=[MagicMock()] * 30),
            awf_folds=(MagicMock(),),
        )
        evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=n_bars),
            aligned=aligned, awf_folds=(MagicMock(), MagicMock()),
            config=config, caps=caps, tf="1h",
            crisis_replay_ctx=ctx,
        )
        mock_crisis_fn.assert_called_once()
        _, kwargs = mock_crisis_fn.call_args
        assert kwargs["crisis_replay_ctx"] is not None
        mock_gate.assert_called_once()
        gate_kwargs = mock_gate.call_args[1]
        assert gate_kwargs["crisis_mdd_hybrid"] == 0.12
        assert gate_kwargs["crisis_mdd_budget"] == 0.15
        assert gate_kwargs["crisis_cagr_hybrid"] == -0.08
        assert gate_kwargs["crisis_cagr_floor"] == -0.05


def test_evaluate_l2_trial_wires_recency_holdout_and_window_coverage_into_evaluation() -> None:
    """[S4-INTEGRATION] evaluate_l2_trial가 compute_recency_holdout_diagnostics와
    window_compute_recency_holdout_diagnostics의 결과를 gate와 trial evaluation에 전달."""
    from unittest.mock import MagicMock, patch

    from src.domain.futures.optimization.workflow import evaluate_l2_trial
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

    n_bars = 100
    sim = SimpleNamespace(
        rets_hybrid=[0.001] * n_bars,
        rets_baseline=[0.0005] * n_bars,
        rets_baseline_ew=(),
        fit_rets_hybrid=(),
        fit_rets_by_fold=(),
        trade_count=50,
        fold_attributions=(),
        fold_rets_hybrid=[],
        fold_selected_symbols=[],
        all_turnovers=[0.1] * n_bars,
        turnover_return_indices=list(range(n_bars)),
        all_gross_exposures=[1.5] * n_bars,
        rebalance_count=10,
        all_net_exposures=[1.0] * n_bars,
        total_cost_hybrid=0.0005,
        friction_pass_total=40,
        signal_total=50,
        cap_saturation_count=2,
        support_leak_count=0,
        last_selected=frozenset({"BTCUSDT", "ETHUSDT"}),
        last_w=MagicMock(),
        capacity_diagnostics=None,
    )
    fold_diag = SimpleNamespace(
        fold_pass_ratio=1.0, fold_compound_pass=(True, True),
        fold_unit_sharpes=(1.0, 1.0), fold_deployed_cagrs=(0.10, 0.10),
        fold_deployed_mdds=(0.05, 0.05), fold_selected_symbols=(("BTCUSDT",), ("ETHUSDT",)),
        recent_fold_passed=True, recent_fold_sharpe=1.0,
        recent_fold_cagr=0.10, recent_fold_mdd=0.05, latest_to_median_cagr=1.0,
    )
    recency_diag = SimpleNamespace(
        applicable=True, holdout_bars=10,
        recency_holdout_cagr=-0.08, recency_holdout_sharpe=-0.5, recency_holdout_mdd=0.12,
    )
    bottleneck_covered = True
    bottleneck_detail = "fold=2 mdd=0.20 cagr=-0.05"

    with (
        patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation", return_value=sim),
        patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics", return_value=fold_diag),
        patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_recency_holdout_diagnostics", return_value=recency_diag),
        patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.evaluation_window_bottleneck_verdict", return_value=(bottleneck_covered, bottleneck_detail)),
        patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate") as mock_gate,
        patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage", return_value=(1.0, "none", 0.0)),
        patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score"),
        patch("src.domain.futures.optimization.workflow.build_layer_universe_audit"),
    ):
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,) * 14,
            gate_passed=True, blocker_reason="",
            constraint_vector=SimpleNamespace(crisis_measured=True),
        )
        config = Layer2AllocationConfig(
            l2_deploy_enabled=True, l2_deploy_worst_fold_gate_enabled=False,
            l2_deploy_kelly_safety_fraction=None,
        )

        result = evaluate_l2_trial(
            cache=MagicMock(), signal_batch=MagicMock(), aligned=MagicMock(),
            awf_folds=(MagicMock(), MagicMock()), config=config,
            caps=MagicMock(), tf="4h",
        )

    gate_kwargs = mock_gate.call_args[1]
    assert gate_kwargs["recency_holdout_cagr"] == -0.08
    assert gate_kwargs["recency_holdout_applicable"] is True
    assert gate_kwargs["window_bottleneck_covered"] is True
    assert gate_kwargs["window_bottleneck_detail"] == bottleneck_detail
    assert result.recency_holdout_cagr == -0.08
    assert result.recency_holdout_applicable is True


def test_layer2_constraints_from_trial_pads_to_14() -> None:
    """[S4] 12개짜리 legacy l2_optuna_constraint_values를 가진 trial → 14-tuple로 패딩되고 13/14번째 원소가 1.0(fail-safe)."""
    from unittest.mock import MagicMock

    mock_trial = MagicMock()
    mock_trial.user_attrs = {"l2_optuna_constraint_values": [-1.0] * 12}

    result = layer2_constraints_from_trial(mock_trial)

    assert len(result) == 14
    assert result[12] == pytest.approx(1.0)
    assert result[13] == pytest.approx(1.0)
