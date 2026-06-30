"""Spec: futures-l2-reversal-economic-replay, Scenario 3."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.domain.futures.optimization.workflow import evaluate_l2_trial


class TestEvaluateL2TrialFoldOutputs:
    """Scenario 3: Evaluation exposes fold MDD and attribution."""

    def _make_sim(
        self,
        *,
        n_folds: int,
        fold_cagrs: tuple[float | None, ...] = (0.10, -0.02),
        fold_mdds: tuple[float | None, ...] = (0.05, 0.08),
    ) -> SimpleNamespace:
        n_bars = 100
        attribs: list[SimpleNamespace] = []
        fold_rets: list[list[float]] = []
        fold_syms: list[tuple[str, ...]] = []
        for i in range(n_folds):
            attribs.append(
                SimpleNamespace(
                    fold_idx=i,
                    oos_bars=n_bars // n_folds,
                    n_rebal=10,
                    realized_total=0.05,
                    realized_price=0.03,
                    realized_funding=0.01,
                    realized_cost=0.005,
                    expected_net=0.04,
                    alpha_gap=0.01,
                    mean_gross_exp=2.0,
                    mean_net_exp=1.5,
                    sleeves_active_mean=8.0,
                    friction_pass_ratio=0.9,
                    throttle_mult_mean=1.0,
                    dropped_below_cost=0,
                    netting_events=5,
                    realized_price_low_er=0.0,
                    trend_efficiency_corr=0.0,
                    mean_trend_efficiency=0.0,
                    risk_off_bars=3 if i == 0 else 0,
                    risk_off_realized_price=0.01 if i == 0 else 0.0,
                    risk_on_realized_price=0.02,
                )
            )
            fold_rets.append([0.001] * (n_bars // n_folds))
            fold_syms.append(("BTCUSDT",))
        return SimpleNamespace(
            rets_hybrid=[0.001] * n_bars,
            rets_baseline=[0.0005] * n_bars,
            rets_baseline_ew=(),
            fit_rets_hybrid=[],
            trade_count=50,
            fold_attributions=tuple(attribs),
            fold_rets_hybrid=fold_rets,
            fold_selected_symbols=fold_syms,
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

    def _make_fold_diag(
        self,
        *,
        fold_cagrs: tuple[float | None, ...] = (0.10, -0.02),
        fold_mdds: tuple[float | None, ...] = (0.05, 0.08),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            fold_pass_ratio=0.5,
            fold_compound_pass=(True, False),
            fold_unit_sharpes=(1.5, -0.3),
            fold_deployed_cagrs=fold_cagrs,
            fold_deployed_mdds=fold_mdds,
            fold_selected_symbols=(("BTCUSDT",), ("ETHUSDT",)),
            recent_fold_passed=False,
            recent_fold_sharpe=-0.3,
            recent_fold_cagr=-0.02,
            recent_fold_mdd=0.08,
            latest_to_median_cagr=0.5,
        )

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.optimization.workflow.build_layer2_deployable_score")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_evaluate_l2_trial_carries_fold_mdds_and_attributions(
        self,
        mock_universe_audit: MagicMock,
        mock_build_score: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_fold_diag: MagicMock,
        mock_sim: MagicMock,
    ) -> None:
        mock_sim.return_value = self._make_sim(n_folds=2)
        mock_fold_diag.return_value = self._make_fold_diag()
        mock_gate.return_value = SimpleNamespace(
            optuna_constraint_values=(0.0,),
            gate_passed=True,
            blocker_reason="",
            regime_pass_ratio=1.0,
        )
        mock_calibrate.return_value = (1.0, "none", 0.0)
        mock_build_score.return_value = SimpleNamespace(
            cagr=0.0, sortino=0.0, sharpe=0.0, calmar=0.0, mdd=0.0,
            fold_pass_ratio=0.0, score=0.0, worst_fold_cagr=0.0,
        )

        config = SimpleNamespace(
            l2_deploy_enabled=False,
            l2_max_mdd_abs=0.30,
            l2_max_cvar_95=0.06,
            l2_deploy_mdd_margin=0.30,
            l2_deploy_cvar_margin=0.20,
            l2_deploy_l_hard_cap=20.0,
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
            l2_max_exchange_leverage=None,
        )
        caps = SimpleNamespace()
        aligned = SimpleNamespace(
            symbols=("BTCUSDT", "ETHUSDT"),
            close_2d=MagicMock(),
            datetimes=[MagicMock()] * 100,
        )
        evaluation = evaluate_l2_trial(
            cache=MagicMock(),
            signal_batch=SimpleNamespace(start_idx=0, end_idx=100),
            aligned=aligned,
            awf_folds=(MagicMock(), MagicMock()),
            config=config,
            caps=caps,
            tf="1h",
        )
        assert evaluation.fold_deployed_mdds == (0.05, 0.08)
        assert len(evaluation.fold_attributions) == 2
