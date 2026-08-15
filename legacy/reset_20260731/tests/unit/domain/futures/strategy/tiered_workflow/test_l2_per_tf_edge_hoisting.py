from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.strategy.tiered_workflow.awf_sim import compute_per_tf_fit_edge
from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache


def _minimal_cache(n_bars: int = 20, n_sym: int = 2, n_sleeve: int = 2) -> L2SimulationCache:
    return L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, n_sym)),
        tradeable_mask_2d=np.ones((n_bars, n_sym), dtype=bool),
        hurdle_2d=np.zeros((n_bars, n_sym)),
        funding_2d=np.zeros((n_bars, n_sym)),
        beta_1d=np.zeros(n_sym),
        expected_gross_bps_2d=np.zeros((n_bars, n_sleeve)),
        expected_net_bps_2d=np.zeros((n_bars, n_sleeve)),
        holding_bars_2d=np.ones((n_bars, n_sleeve)),
        side_2d=np.ones((n_bars, n_sleeve)),
        quality_weight_2d=np.ones((n_bars, n_sleeve)),
        signal_mask_2d=np.ones((n_bars, n_sleeve), dtype=bool),
        sleeve_to_sym=np.zeros(n_sleeve, dtype=int),
        sleeve_keys=(),
    )


def test_l2_simulation_cache_per_tf_edge_by_fold_defaults_empty() -> None:
    cache = _minimal_cache()
    assert cache.per_tf_edge_by_fold == ()


def test_run_tiered_l2_study_precomputes_per_tf_edge_once_per_fold(mocker: Any) -> None:
    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    cfg = CandidateStrategyConfig(wf_n_folds=4)
    n_folds = 4

    mocker.patch(
        "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
        return_value=(
            SimpleNamespace(fit_start=3000, fit_end=3500, cal_start=3500, cal_end=3700, oos_start=3700, oos_end=5000),
            SimpleNamespace(fit_start=3500, fit_end=4000, cal_start=4000, cal_end=4200, oos_start=4200, oos_end=5500),
            SimpleNamespace(fit_start=4000, fit_end=4500, cal_start=4500, cal_end=4700, oos_start=4700, oos_end=6000),
            SimpleNamespace(fit_start=4500, fit_end=5000, cal_start=5000, cal_end=5200, oos_start=5200, oos_end=6500),
        ),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
        return_value=mocker.MagicMock(warnings=()),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.optimization.workflow.objective_l2_growth",
        return_value=0.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
        return_value=mocker.MagicMock(best_params={}, best_trial_number=0, completed_trials=0),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._get_rss_mb",
        return_value=100.0,
    )
    import dataclasses

    mocker.patch.object(dataclasses, "replace", side_effect=lambda obj, **kw: obj)
    mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 6}, clear=False)
    mocker.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=32.0 * (1024.0 ** 3)))

    mock_executor = mocker.MagicMock()
    mock_executor.__enter__.return_value = mock_executor
    mock_future = mocker.MagicMock()
    mock_future.result.return_value = (0.1, {}, 0.01)
    mock_executor.submit.return_value = mock_future
    mocker.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_executor)

    mock_study = mocker.MagicMock()
    mock_study.trials = []
    mock_study.ask.side_effect = [mocker.MagicMock(number=i) for i in range(6)]
    mock_study._stop_flag = False
    mocker.patch(
        "src.application.futures.runner.active_pipeline.get_or_create_study",
        return_value=mock_study,
    )

    spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
        wraps=lambda **kw: {},
    )

    window = SimpleNamespace(holdout_start=date(2025, 6, 1), l2_start=date(2024, 6, 1))
    aligned = SimpleNamespace(
        symbols=("BTCUSDT",),
        close_2d=mocker.MagicMock(),
        datetimes=pd.date_range("2024-01-01", periods=15000, freq="h"),
    )
    caps = SimpleNamespace(trial_number=0)
    signal_batch = mocker.MagicMock()
    signal_batch.start_idx = 0
    signal_batch.end_idx = 500
    signal_batch.registry_version = "v1"
    signal_batch.model_version = "v1"
    signal_batch.events = ()

    _run_tiered_l2_study(
        signal_batch=signal_batch,
        aligned=aligned,
        cfg=cfg,
        window=window,
        caps=caps,
        tf="1h",
        n_trials=6,
        seed=42,
        l2_sim_cache=mocker.MagicMock(),
        l2_wf_n_folds=None,
    )

    assert spy.call_count == n_folds


def test_run_awf_simulation_falls_back_to_recompute_when_cache_empty(mocker: Any) -> None:
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    from src.domain.futures.strategy.candidate_contracts import (
        ValidatedSignalBatch,
        ValidatedSignalEvent,
    )
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
        _run_awf_simulation,
        build_l2_simulation_cache,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
    from src.domain.futures.strategy.walk_forward import WFFold

    n_bars = 60
    n_sym = 2
    close = np.ones((n_bars, n_sym), dtype=np.float64) * 100.0
    aligned = mocker.MagicMock(spec=AlignedMarketData)
    aligned.symbols = ("BTC", "ETH")
    aligned.close_2d = close
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    aligned.funding_2d = np.zeros((n_bars, n_sym), dtype=np.float64)
    aligned.active_mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned.warm_mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned.entry_block_mask = np.zeros((n_bars, n_sym), dtype=bool)
    aligned.kill_mask = np.zeros((n_bars, n_sym), dtype=bool)
    aligned.execution_cost_bps_2d = np.full((n_bars, n_sym), 4.0, dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(n_sym, dtype=np.float64)

    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    signal_batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=0, decision_time=datetimes[0], symbol="BTC",
                strategy_id="trend:fast", activation_context="all", side=1,
                expected_net_bps=0.0, expected_gross_bps=20.0,
                q10_net_bps=0.0, q10_gross_bps=10.0, q90_net_bps=0.0, q90_gross_bps=30.0,
                expected_holding_bars=1, registry_version="test", model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=0, decision_time=datetimes[0], symbol="ETH",
                strategy_id="trend:fast", activation_context="all", side=-1,
                expected_net_bps=0.0, expected_gross_bps=5.0,
                q10_net_bps=0.0, q10_gross_bps=2.0, q90_net_bps=0.0, q90_gross_bps=8.0,
                expected_holding_bars=1, registry_version="test", model_version="test",
            ),
        ),
        start_idx=1, end_idx=3, symbols=("BTC", "ETH"),
        registry_version="test", model_version="test",
    )

    config = Layer2AllocationConfig(
        k_rank=2, rank_buffer=0, kelly_fraction=0.5,
        no_trade_band=0.0, rebalance_bars=1,
        l2_tf_inclusion_enabled=True, l2_tf_inclusion_min_edge=0.0,
    )
    awf_folds = (WFFold(fit_start=0, fit_end=10, cal_start=10, cal_end=10, oos_start=10, oos_end=20),)
    caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)
    cache = build_l2_simulation_cache(aligned, signal_batch, "4h")
    assert cache.per_tf_edge_by_fold == ()

    spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
        wraps=compute_per_tf_fit_edge,
    )

    _run_awf_simulation(
        cache=cache, signal_batch=signal_batch, aligned=aligned,
        awf_folds=awf_folds, config=config, caps=caps, tf="4h",
    )

    assert spy.call_count >= 1


def test_included_tfs_by_fold_identical_between_cached_and_fallback_paths(mocker: Any) -> None:
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    from src.domain.futures.strategy.candidate_contracts import (
        ValidatedSignalBatch,
        ValidatedSignalEvent,
    )
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
        _run_awf_simulation,
        build_l2_simulation_cache,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
    from src.domain.futures.strategy.walk_forward import WFFold

    n_bars = 100
    n_sym = 2
    close = np.ones((n_bars, n_sym), dtype=np.float64) * 100.0
    aligned = mocker.MagicMock(spec=AlignedMarketData)
    aligned.symbols = ("BTC", "ETH")
    aligned.close_2d = close
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    aligned.funding_2d = np.zeros((n_bars, n_sym), dtype=np.float64)
    aligned.active_mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned.warm_mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned.entry_block_mask = np.zeros((n_bars, n_sym), dtype=bool)
    aligned.kill_mask = np.zeros((n_bars, n_sym), dtype=bool)
    aligned.execution_cost_bps_2d = np.full((n_bars, n_sym), 4.0, dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(n_sym, dtype=np.float64)

    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    batch_events = (
        ValidatedSignalEvent(
            decision_idx=0, decision_time=datetimes[0], symbol="BTC",
            strategy_id="trend:fast", activation_context="all", side=1,
            expected_net_bps=0.0, expected_gross_bps=20.0,
            q10_net_bps=0.0, q10_gross_bps=10.0, q90_net_bps=0.0, q90_gross_bps=30.0,
            expected_holding_bars=1, registry_version="test", model_version="test",
        ),
        ValidatedSignalEvent(
            decision_idx=0, decision_time=datetimes[0], symbol="ETH",
            strategy_id="trend:fast", activation_context="all", side=-1,
            expected_net_bps=0.0, expected_gross_bps=5.0,
            q10_net_bps=0.0, q10_gross_bps=2.0, q90_net_bps=0.0, q90_gross_bps=8.0,
            expected_holding_bars=1, registry_version="test", model_version="test",
        ),
    )
    signal_batch = ValidatedSignalBatch(
        events=batch_events, start_idx=1, end_idx=3, symbols=("BTC", "ETH"),
        registry_version="test", model_version="test",
    )

    config = Layer2AllocationConfig(
        k_rank=2, rank_buffer=0, kelly_fraction=0.5,
        no_trade_band=0.0, rebalance_bars=1,
        l2_tf_inclusion_enabled=True, l2_tf_inclusion_min_edge=0.0,
    )
    awf_folds = (
        WFFold(fit_start=0, fit_end=10, cal_start=10, cal_end=10, oos_start=10, oos_end=30),
        WFFold(fit_start=5, fit_end=15, cal_start=15, cal_end=15, oos_start=15, oos_end=35),
    )
    caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

    from dataclasses import replace

    base_cache = build_l2_simulation_cache(aligned, signal_batch, "4h")
    cached_cache = replace(
        base_cache,
        per_tf_edge_by_fold=tuple(
            compute_per_tf_fit_edge(
                cache=base_cache, aligned=aligned,
                fit_start=int(f.fit_start), fit_end=int(f.oos_start),
            )
            for f in awf_folds
        ),
    )

    _run_awf_simulation(
        cache=cached_cache, signal_batch=signal_batch, aligned=aligned,
        awf_folds=awf_folds, config=config, caps=caps, tf="4h",
    )
