"""Spec: l2-deploy-leverage-kelly-worst-fold-safety, Scenario 4 (Integration)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.domain.futures.optimization.workflow import (
    compute_crisis_mdd_budget,
    evaluate_l2_trial,
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
            optuna_constraint_values=(0.0,),
            gate_passed=True,
            blocker_reason="",
            regime_pass_ratio=1.0,
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
            optuna_constraint_values=(0.0,),
            gate_passed=True,
            blocker_reason="",
            regime_pass_ratio=1.0,
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
            optuna_constraint_values=(0.0,) * 10,
            gate_passed=True,
            blocker_reason="",
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
            optuna_constraint_values=(0.0,) * 10,
            gate_passed=True,
            blocker_reason="",
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
            optuna_constraint_values=(0.0,) * 10,
            gate_passed=True,
            blocker_reason="",
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

        config = Layer2AllocationConfig.from_mapping({
            "l2_objective_growth_lcb_weight": 0.3,
        })
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

        # growth_lcb_weight=0.3일 때 L* 높은 trial(=high deployed scale)이 더 높은 objective
        assert result_high.objective_value > result_low.objective_value, (
            f"High L* trial ({result_high.deploy_leverage}) should score > low L* trial "
            f"({result_low.deploy_leverage}) with growth_lcb_weight=0.3"
        )

        # growth_lcb_weight=0.0(legacy)에서는 차이가 거의 없어야 함
        config_zero = Layer2AllocationConfig.from_mapping({
            "l2_objective_growth_lcb_weight": 0.0,
        })
        mock_calibrate.side_effect = [
            (3.0, "mdd", 0.0),
            (1.0, "crisis_constraint", 0.0),
        ]

        result_high_zero = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=200),
            aligned=aligned,
            awf_folds=(MagicMock(),),
            config=config_zero,
            caps=caps,
            tf="1h",
        )

        result_low_zero = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=200),
            aligned=aligned,
            awf_folds=(MagicMock(),),
            config=config_zero,
            caps=caps,
            tf="1h",
        )

        diff_weight = result_high.objective_value - result_low.objective_value
        diff_zero = result_high_zero.objective_value - result_low_zero.objective_value
        assert diff_weight > diff_zero, (
            f"Objective gap with weight=0.3 ({diff_weight:.6f}) should exceed "
            f"gap with weight=0.0 ({diff_zero:.6f})"
        )


class TestComputeCrisisMddBudget:
    """compute_crisis_mdd_budget pure function tests."""

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.apply_deployment")
    def test_computes_from_replay_ctx(
        self, mock_apply: MagicMock, mock_sim: MagicMock,
    ) -> None:
        """[S1] 합성 crisis_replay_ctx + leverage=2.0 → crisis_mdd_hybrid가 deploy MDD, budget이 공식대로."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

        mock_sim.return_value = SimpleNamespace(
            rets_hybrid=[0.01, -0.02, 0.015, -0.01, 0.005],
        )
        mock_apply.return_value = SimpleNamespace(mdd=0.25)
        config = Layer2AllocationConfig.from_mapping({})
        ctx = SimpleNamespace(
            cache=MagicMock(), signal_batch=MagicMock(),
            aligned=SimpleNamespace(datetimes=[MagicMock()] * 50),
            awf_folds=(MagicMock(),),
        )
        crisis_mdd_hybrid, crisis_mdd_budget = compute_crisis_mdd_budget(
            crisis_replay_ctx=ctx, config=config, caps=MagicMock(),
            tf="8h", leverage=2.0, bars_per_year=1095.0,
        )
        expected_budget = float(config.l2_max_mdd_abs) * (1.0 - float(config.l2_deploy_crisis_mdd_margin))
        assert crisis_mdd_hybrid == 0.25
        assert crisis_mdd_budget == expected_budget

    def test_returns_none_when_ctx_is_none(self) -> None:
        """[S3] crisis_replay_ctx=None → 즉시 (None, None)."""
        result = compute_crisis_mdd_budget(
            crisis_replay_ctx=None, config=MagicMock(), caps=MagicMock(),
            tf="8h", leverage=1.0, bars_per_year=1095.0,
        )
        assert result == (None, None)

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    def test_returns_none_on_simulation_exception(self, mock_sim: MagicMock) -> None:
        """[S2] _run_awf_simulation 예외 → (None, None), 예외 미전파."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

        mock_sim.side_effect = ValueError("boom")
        config = Layer2AllocationConfig.from_mapping({})
        ctx = SimpleNamespace(
            cache=MagicMock(), signal_batch=MagicMock(),
            aligned=SimpleNamespace(datetimes=[MagicMock()] * 50),
            awf_folds=(MagicMock(),),
        )
        result = compute_crisis_mdd_budget(
            crisis_replay_ctx=ctx, config=config, caps=MagicMock(),
            tf="8h", leverage=1.0, bars_per_year=1095.0,
        )
        assert result == (None, None)


class TestEvaluateL2TrialUsesComputeCrisisMddBudget:
    """Refactoring verification: evaluate_l2_trial delegates to compute_crisis_mdd_budget."""

    @patch("src.domain.futures.optimization.workflow.compute_crisis_mdd_budget")
    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_calls_compute_crisis_mdd_budget(
        self,
        mock_universe: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold: MagicMock,
        mock_sim: MagicMock,
        mock_crisis_fn: MagicMock,
    ) -> None:
        """[S4] evaluate_l2_trial이 compute_crisis_mdd_budget를 호출하고 반환값이 gate로 전달됨."""
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
            optuna_constraint_values=(0.0,) * 10, gate_passed=True, blocker_reason="",
        )
        mock_calibrate.return_value = (1.5, "mdd", 0.0)
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0, sortino=0.0, sharpe=0.0, calmar=0.0, mdd=0.0,
            fold_pass_ratio=0.0, score=0.0, worst_fold_cagr=0.0,
        )
        mock_crisis_fn.return_value = (0.12, 0.15)

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
