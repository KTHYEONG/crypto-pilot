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

    with patch("src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation") as mock_sim, \
         patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache") as mock_cache:

        mock_cache.return_value = MagicMock()
        mock_sim_result = MagicMock()
        mock_sim_result.rets_hybrid = [0.0] * 10
        mock_sim_result.rets_baseline_ew = [0.0] * 10
        mock_sim_result.rets_baseline = [0.0] * 10
        mock_sim_result.fit_rets_hybrid = [0.0] * 10
        mock_sim_result.all_turnovers = [0.0] * 10
        mock_sim_result.all_gross_exposures = [0.0] * 10
        mock_sim_result.all_net_exposures = [0.0] * 10
        mock_sim_result.cap_saturation_count = 0
        mock_sim_result.rebalance_count = 10
        mock_sim_result.friction_pass_total = 10
        mock_sim_result.signal_total = 20
        mock_sim_result.block_rets_hybrid = [[0.0] * 10]
        mock_sim_result.block_rets_baseline = [[0.0] * 10]
        mock_sim_result.fold_rets_hybrid = [[0.0] * 10]
        mock_sim_result.trade_count = 5
        mock_sim_result.support_leak_count = 0
        mock_sim.return_value = mock_sim_result

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

    fake_sim = MagicMock()
    fake_sim.rets_hybrid = [0.02, 0.03, 0.01, 0.02]
    fake_sim.rets_baseline_ew = [0.01, 0.01, 0.0, 0.01]
    fake_sim.rets_baseline = [0.01, 0.0, 0.0, 0.01]
    fake_sim.fit_rets_hybrid = [0.01, 0.02]
    fake_sim.all_turnovers = [0.1, 0.2, 0.1, 0.3]
    fake_sim.all_gross_exposures = [0.4, 0.5, 0.6, 0.5]
    fake_sim.all_net_exposures = [0.2, 0.3, 0.1, 0.2]
    fake_sim.cap_saturation_count = 1
    fake_sim.rebalance_count = 4
    fake_sim.friction_pass_total = 3
    fake_sim.signal_total = 4
    fake_sim.support_leak_count = 0
    fake_sim.total_cost_hybrid = 0.01
    fake_sim.block_rets_hybrid = [[0.02, 0.01], [0.01, 0.02]]
    fake_sim.block_rets_baseline = [[0.0, 0.0], [0.0, 0.01]]
    fake_sim.fold_rets_hybrid = [[0.02, 0.01], [0.01, 0.02]]
    fake_sim.trade_count = 12
    fake_sim.last_selected = frozenset({"BTC"})
    fake_sim.last_w = np.array([1.0, 0.0], dtype=float)
    fake_sim.fold_attributions = (
        MagicMock(realized_cost=0.1, realized_price=0.3),
        MagicMock(realized_cost=0.08, realized_price=0.2),
    )

    fake_deployment = MagicMock(cagr=-0.01, mdd=0.08)
    gate_mock = MagicMock(
        optuna_constraint_values=(-1.0,) * 18,
        promotion_passed=True,
        promotion_blocker="",
        promotion_constraint_values=(-1.0,) * 18,
    )

    with (
        patch("src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation", return_value=fake_sim),
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.apply_deployment", return_value=fake_deployment),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.calibrate_deployment_leverage",
            return_value=(2.0, "mdd", 0.0),
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.evaluate_layer2_gate",
            return_value=gate_mock,
        ) as mock_gate,
    ):
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
    assert mock_gate.call_args is not None
    assert mock_gate.call_args.kwargs["worst_fold_cagr"] == pytest.approx(-0.01)
    assert mock_gate.call_args.kwargs["positive_block_delta_ratio"] == pytest.approx(1.0)


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

    with patch("src.domain.futures.strategy.tiered_workflow.pipeline.predict_layer1_signals") as mock_pred, \
         patch("src.domain.futures.strategy.tiered_workflow.pipeline.build_l2_simulation_folds") as mock_folds, \
         patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf") as mock_l2:

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

    with patch("src.domain.futures.strategy.tiered_workflow.pipeline.predict_layer1_signals") as mock_pred, \
         patch("src.domain.futures.strategy.tiered_workflow.pipeline.build_l2_simulation_folds") as mock_folds, \
         patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf") as mock_l2:

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
