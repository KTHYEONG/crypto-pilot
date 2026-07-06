import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.pipeline import run_l2_awf, run_tiered_pipeline


def test_run_l2_awf_cache_timing_log(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that build_l2_simulation_cache timing is logged at DEBUG/PERF level."""
    caplog.set_level(logging.DEBUG)

    signal_batch = MagicMock()
    aligned = MagicMock()
    aligned.symbols = ("BTC",)
    aligned.close_2d = np.ones((10, 1), dtype=float)
    aligned.datetimes = np.array([f"2024-01-{i:02d}" for i in range(1, 11)], dtype="datetime64[ns]")

    from src.domain.futures.strategy.walk_forward import WFFold

    awf_folds = (WFFold(fit_start=0, fit_end=5, cal_start=0, cal_end=5, oos_start=5, oos_end=10),)
    config = Layer2AllocationConfig()
    caps = PortfolioCaps()

    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2GateEvaluation, Layer2TrialEvaluation

    with (
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.evaluate_l2_trial") as mock_eval,
        patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache") as mock_cache,
    ):
        mock_cache.return_value = MagicMock()
        mock_eval.return_value = Layer2TrialEvaluation(
            objective_value=0.0,
            constraint_values=(),
            cagr_hybrid=0.0,
            cagr_baseline=0.0,
            growth_lcb_hybrid=0.0,
            growth_lcb_baseline=0.0,
            sharpe_hac_hybrid=0.0,
            sharpe_hac_baseline=0.0,
            psr_hybrid=0.0,
            mdd_hybrid=0.0,
            cvar_95_hybrid=0.0,
            fold_pass_ratio=0.0,
            break_even_pass_pct=0.0,
            average_gross_exposure=0.0,
            cap_saturation_ratio=0.0,
            total_cost_bps=0.0,
            block_metrics=(),
            returns_hybrid=tuple([0.0] * 10),
            returns_baseline=tuple([0.0] * 10),
            rets_baseline_ew=tuple([0.0] * 10),
            last_selected_symbols=("BTC",),
            last_weights=(1.0,),
            all_turnovers=tuple([0.0] * 10),
            rebalance_count=10,
            all_net_exposures=tuple([0.0] * 10),
            gate=Layer2GateEvaluation(
                optuna_constraint_values=(),
                promotion_passed=False,
                promotion_blocker="test",
                promotion_constraint_values=(),
            ),
        )

        run_l2_awf(
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=config,
            caps=caps,
            tf="4h",
            verbose=False,
        )

    # Validate timing log presence
    assert any("[PERF] l2_build_sim_cache took" in rec.message for rec in caplog.records)
    assert any("[MEM]" in rec.message and "rss=" in rec.message for rec in caplog.records)


def test_run_l2_awf_threads_deployable_metrics_into_gate() -> None:
    signal_batch = MagicMock()
    aligned = MagicMock()
    aligned.symbols = ("BTC", "ETH")
    aligned.close_2d = np.ones((12, 2), dtype=float)
    aligned.datetimes = np.array([f"2024-01-{i:02d}" for i in range(1, 13)], dtype="datetime64[ns]")
    from src.domain.futures.strategy.walk_forward import WFFold

    awf_folds = (
        WFFold(fit_start=0, fit_end=4, cal_start=0, cal_end=4, oos_start=4, oos_end=8),
        WFFold(fit_start=4, fit_end=8, cal_start=4, cal_end=8, oos_start=8, oos_end=12),
    )

    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2DeployableScore,
        Layer2GateEvaluation,
        Layer2TrialEvaluation,
    )

    gate_mock = Layer2GateEvaluation(
        optuna_constraint_values=(-1.0,) * 18,
        promotion_passed=True,
        promotion_blocker="",
        promotion_constraint_values=(-1.0,) * 18,
    )
    fake_eval = Layer2TrialEvaluation(
        objective_value=1.0,
        constraint_values=(-1.0,) * 18,
        cagr_hybrid=-0.01,
        cagr_baseline=0.0,
        growth_lcb_hybrid=0.0,
        growth_lcb_baseline=0.0,
        sharpe_hac_hybrid=0.0,
        sharpe_hac_baseline=0.0,
        psr_hybrid=0.0,
        mdd_hybrid=0.08,
        cvar_95_hybrid=0.04,
        fold_pass_ratio=0.5,
        break_even_pass_pct=0.75,
        average_gross_exposure=0.5,
        cap_saturation_ratio=0.25,
        total_cost_bps=100.0,
        block_metrics=(),
        returns_hybrid=(0.02, 0.03, 0.01, 0.02),
        returns_baseline=(0.01, 0.0, 0.0, 0.01),
        sharpe_hybrid=1.0,
        sortino_hybrid=1.5,
        trade_count=12,
        risk_utilization=0.5,
        deployment_objective_bonus=0.0,
        worst_fold_sharpe=0.5,
        gate=gate_mock,
        fit_returns_hybrid=(0.01, 0.02),
        deploy_leverage=2.0,
        deploy_binding="mdd",
        recent_fold_passed=True,
        recent_fold_sharpe=1.0,
        recent_fold_cagr=-0.01,
        recent_fold_mdd=0.08,
        latest_to_median_cagr=1.0,
        fold_deployed_cagrs=(-0.01,),
        fold_selected_symbols=(("BTC",),),
        worst_fold_cagr=-0.01,
        positive_block_delta_ratio=1.0,
        bucket_reliability_mean=0.5,
        entry_spike_penalty=0.0,
        deployable_score=Layer2DeployableScore(
            cagr=-0.01,
            sortino=1.0,
            sharpe=1.0,
            calmar=-0.125,
            mdd=0.08,
            fold_pass_ratio=0.5,
            score=0.4,
            worst_fold_cagr=-0.01,
            positive_block_delta_ratio=1.0,
            cost_drag=0.01,
            bucket_reliability_mean=0.5,
            entry_spike_penalty=0.0,
        ),
        last_selected_symbols=("BTC",),
        last_weights=(1.0,),
        all_turnovers=(0.1, 0.2, 0.1, 0.3),
        rebalance_count=4,
        all_net_exposures=(0.2, 0.3, 0.1, 0.2),
        rets_baseline_ew=(0.01, 0.01, 0.0, 0.01),
    )

    with (
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.evaluate_l2_trial", return_value=fake_eval),
        patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache") as mock_cache,
    ):
        mock_cache.return_value = MagicMock()
        result = run_l2_awf(
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=Layer2AllocationConfig(),
            caps=PortfolioCaps(),
            tf="4h",
            verbose=False,
        )

    assert result.gate_passed is True


def test_run_tiered_pipeline_l2_timing_and_mem_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that predict_layer1_signals timing and RSS are logged during run_tiered_pipeline L2 phase."""
    caplog.set_level(logging.DEBUG)

    labeled_events = MagicMock()
    aligned = MagicMock()
    aligned.datetimes = np.array([f"2024-01-{i:02d}" for i in range(1, 11)], dtype="datetime64[ns]")
    aligned.symbols = ("BTC",)

    window = MagicMock()
    window.l2_start = "2024-01-01"
    window.holdout_start = "2024-01-05"
    window.end_date_value = "2024-01-10"

    cfg = MagicMock()

    l1_result = MagicMock()
    l1_result.gate_passed = True
    l1_result.inference_artifact = MagicMock()
    l1_result.symbol_lifecycle = None

    l2_params = {"l2_deploy_leverage": 1.0}
    caps = PortfolioCaps()

    with (
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.predict_layer1_signals") as mock_pred,
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.build_l2_simulation_folds") as mock_folds,
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf") as mock_l2,
    ):
        mock_pred.return_value = MagicMock()
        mock_folds.return_value = (MagicMock(),)
        mock_l2.return_value = MagicMock()

        run_tiered_pipeline(
            labeled_events=labeled_events,
            aligned=aligned,
            window=window,
            cfg=cfg,
            l1_params={},
            l2_params=l2_params,
            caps=caps,
            target_phase="l2",
            l1_result_override=l1_result,
            verbose=False,
        )

    # Validate timing log presence and format
    messages = [rec.message for rec in caplog.records]
    assert any("[PERF] predict_layer1_signals(L2)" in m for m in messages)
    assert any("[PERF] run_tiered_pipeline_l2_total took" in m for m in messages)
    assert any("[MEM] stage=l2_entry rss=" in m for m in messages)
    assert any("[MEM] stage=l2_awf_complete rss=" in m for m in messages)


def test_l2_awf_fold_build_logged_on_empty_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that awf_fold_build log is present even when build_l2_simulation_folds returns empty fallback."""
    caplog.set_level(logging.DEBUG)

    labeled_events = MagicMock()
    aligned = MagicMock()
    aligned.datetimes = np.array([f"2024-01-{i:02d}" for i in range(1, 11)], dtype="datetime64[ns]")
    aligned.symbols = ("BTC",)

    window = MagicMock()
    window.l2_start = "2024-01-01"
    window.holdout_start = "2024-01-05"
    window.end_date_value = "2024-01-10"

    cfg = MagicMock()

    l1_result = MagicMock()
    l1_result.gate_passed = True
    l1_result.inference_artifact = MagicMock()
    l1_result.symbol_lifecycle = None

    l2_params = {"l2_deploy_leverage": 1.0}
    caps = PortfolioCaps()

    with (
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.predict_layer1_signals") as mock_pred,
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.build_l2_simulation_folds") as mock_folds,
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf") as mock_l2,
    ):
        mock_pred.return_value = MagicMock()
        mock_folds.return_value = ()  # Force fallback
        mock_l2.return_value = MagicMock()

        run_tiered_pipeline(
            labeled_events=labeled_events,
            aligned=aligned,
            window=window,
            cfg=cfg,
            l1_params={},
            l2_params=l2_params,
            caps=caps,
            target_phase="l2",
            l1_result_override=l1_result,
            verbose=False,
        )

    messages = [rec.message for rec in caplog.records]
    assert any("[L2] awf_fold_build took=" in m for m in messages)
    assert any("[L2] AWF window: L2_start_bar=" in m for m in messages)


def test_l2_gate_evaluate_timing_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Verify evaluate_layer2_gate logs timing and evaluation result."""
    caplog.set_level(logging.DEBUG)

    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
    from src.domain.futures.strategy.tiered_workflow.l2_gate import evaluate_layer2_gate

    evaluate_layer2_gate(
        deployment_failed=False,
        support_leak_count=0,
        cagr_hybrid=0.35,
        sharpe_hybrid=1.8,
        sharpe_hac_hybrid=1.7,
        sharpe_hac_baseline=1.2,
        sortino_hybrid=2.0,
        mar_hybrid=1.2,
        mdd_hybrid=0.15,
        cvar_95_hybrid=0.04,
        fold_pass_ratio=0.7,
        active_block_count=10,
        friction_pass_pct=0.8,
        trade_count=100,
        growth_lcb_hybrid=0.1,
        growth_lcb_baseline=0.05,
        dsr_hybrid=0.85,
        config=Layer2AllocationConfig(),
    )

    messages = [rec.message for rec in caplog.records]
    assert any("[L2-GATE] evaluate took=" in m for m in messages)
    assert any("passed=" in m for m in messages)
