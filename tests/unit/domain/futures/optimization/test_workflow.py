"""Spec: l2-deploy-leverage-kelly-worst-fold-safety, Scenario 4 (Integration)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.domain.futures.optimization.workflow import evaluate_l2_trial


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
