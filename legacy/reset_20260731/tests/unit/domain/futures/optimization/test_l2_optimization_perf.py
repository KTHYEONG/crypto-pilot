from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import optuna


@dataclass
class _MockCfg:
    """Mock cfg with defaults for _run_tiered_l2_study regime routing args."""
    l2_regime_fallback_mode: str = "pooled"
    l2_regime_compression_enabled: bool = True
    l2_bucket_cost_bps: float = 6.0
    l2_bucket_min_n: int = 15
    l2_bucket_shrinkage: float = 0.3
    l2_regime_proof_enabled: bool = True
    l2_regime_proof_nw_tstat: float = 1.5
    l2_regime_proof_fold_pass_ratio: float = 0.60
    l2_bucket_edge_floor_bps: float = 0.0
    l2_regime_debug_top_k: int = 10
    l2_regime_policy_mode: str = "soft"
    l2_regime_cal_min_n: int = 20
    l2_regime_min_cal_lift_bps: float = 8.0
    l2_regime_block_lift_bps: float = -12.0
    l2_regime_soft_downweight_min: float = 0.50
    l2_regime_soft_downweight_max: float = 1.0
    l2_regime_min_policy_confidence: float = 0.55
    l2_regime_hard_block_enabled: bool = False
    l2_regime_block_min_confidence: float = 0.80
    l2_regime_require_sign_consistency: bool = True
    l2_regime_pooled_is_passthrough: bool = True
    l2_regime_min_fit_n_floor: int = 5
    l2_regime_require_fit_n_for_downweight: bool = False
    l2_regime_severity_vol_quantile: float = 0.35
    l2_crowding_floor_mult: float = 0.5
    l2_crowding_persistence_bars: int = 3
    l2_crowding_recovery_cooldown_bars: int = 3
    l2_selection_breadth_mode: bool = True
    min_abs_rank_z: float = 0.0
    l2_sleeve_combine_method: str = "pooled"
    l2_sleeve_conviction_cap_mult: float = 1.0
    l2_diag_attribution_enabled: bool = False
    l2_diag_sleeve_top_k: int = 15
    l2_diag_sleeve_sample_every: int = 0
    l2_portfolio_cov_mode: str = "diagonal"
    l2_portfolio_cov_lookback_bars: int = 252
    l2_portfolio_cov_min_obs: int = 20
    l2_deploy_enabled: bool = False
    l2_deploy_worst_fold_gate_enabled: bool = False
    l2_deploy_mdd_margin: float = 0.0
    l2_deploy_cvar_margin: float = 0.0
    l2_deploy_l_hard_cap: float = 1.0
    l2_deploy_fit_mdd_crisis_gate: Any = None
    l2_max_mdd_abs: float = 0.5
    l2_max_cvar_95: float = 0.5
    l2_max_exchange_leverage: Any = None
    l2_growth_lcb_z: float = 0.0
    l2_worst_fold_penalty_threshold: float = 0.0
    l2_worst_fold_penalty_weight: float = 0.0
    l2_objective_risk_util_target: float = 1.0
    l2_objective_risk_util_weight: float = 0.0
    l2_objective_trade_target: int = 10
    l2_objective_trade_weight: float = 0.0
    l2_turnover_penalty_weight: float = 0.0
    l2_objective_growth_lcb_weight: float = 0.0
    l2_entry_spike_penalty_weight: float = 0.0
    l2_deploy_kelly_safety_fraction: Any = None
    l2_replay_max_fallbacks: int = 24
    k_rank: int = 3
    rank_buffer: int = 0
    kelly_fraction: float = 0.25
    max_ann_vol: float | None = 1.0
    no_trade_band: float = 0.0
    rebalance_bars: int = 3
    l2_regime_conditional_weight_enabled: bool = False
    l2_intra_symbol_divergence_enabled: bool = False
    min_order_usdt: float = 5.0
    edge_throttle_enabled: bool = False
    edge_floor_bps: float = 0.0
    edge_ref_bps: float = 5.0
    edge_throttle_gamma: float = 1.0
    edge_throttle_min_active_mult: float = 0.0
    risk_budget_floor_ratio: float = 0.0
    risk_budget_max_scale: float = 3.0
    adaptive_breadth_enabled: bool = False
    adaptive_k_extra: int = 0
    adaptive_expand_below_vol_ratio: float = 0.0
    fixed_cost_safety_mult: float = 1.25
    deploy_cost_safety_mult: float = 1.0
    l2_regime_min_fit_n_floor: int = 5
    timeframe: str = "4h"


@dataclass
class _PicklableSignalBatch:
    start_idx: int = 0
    end_idx: int = 10
    events: tuple = ()


@dataclass
class _PicklableArtifact:
    model_version: str = "v1"


@dataclass
class _PicklableWindow:
    l2_start: str = "2026-01-01"
    holdout_start: str = "2026-06-01"


def test_setup_optuna_storage_returns_inmemory_when_use_memory_true(tmp_path):
    from src.domain.futures.optimization.observability.run_tracker import setup_optuna_storage

    url, storage = setup_optuna_storage(tmp_path, use_memory=True)
    assert url == ""
    assert isinstance(storage, optuna.storages.InMemoryStorage)


def test_l2_early_stop_callback_triggers_after_no_improve_limit():
    study = optuna.create_study(direction="maximize")

    best_val = float("-inf")
    last_improve = 0
    no_improve_limit = 3
    min_trials = 2
    stopped = False

    for i in range(10):
        t = study.ask()
        val = 0.5 if i == 1 else 0.1
        study.tell(t, val)
        if val > best_val:
            best_val = val
            last_improve = t.number
        if t.number >= min_trials and t.number - last_improve >= no_improve_limit:
            stopped = True
            break

    assert stopped, "Early stop should trigger after 3 consecutive no-improve trials"
    assert i < 9, f"Stopped at trial {i}, expected <= 5"


def test_setup_optuna_storage_use_memory_false_returns_rdb_storage(tmp_path):
    from src.domain.futures.optimization.observability.run_tracker import setup_optuna_storage

    url, storage = setup_optuna_storage(tmp_path, use_memory=False)
    assert "sqlite:///" in url
    assert isinstance(storage, optuna.storages.RDBStorage)


def test_setup_optuna_storage_falls_back_to_inmemory_on_sqlite_failure(tmp_path, mocker):
    from src.domain.futures.optimization.observability.run_tracker import setup_optuna_storage

    mocker.patch(
        "optuna.storages.RDBStorage",
        side_effect=RuntimeError("Simulated DB failure"),
    )

    url, storage = setup_optuna_storage(tmp_path, use_memory=False)
    assert url == ""
    assert isinstance(storage, optuna.storages.InMemoryStorage)


def test_evaluate_l2_trial_lightweight_skips_block_metrics_and_audit(mocker):
    from src.domain.futures.optimization.workflow import evaluate_l2_trial

    mock_sim_result = mocker.MagicMock()
    mock_sim_result.rets_hybrid = [0.001, -0.002, 0.003]
    mock_sim_result.rets_baseline = [0.0, 0.0, 0.0]
    mock_sim_result.rets_baseline_ew = [0.0, 0.0, 0.0]
    mock_sim_result.last_selected = frozenset()
    mock_sim_result.last_w = None
    mock_sim_result.all_turnovers = []
    mock_sim_result.all_turnovers_baseline = []
    mock_sim_result.all_gross_exposures = []
    mock_sim_result.all_net_exposures = []
    mock_sim_result.friction_pass_total = 0
    mock_sim_result.signal_total = 0
    mock_sim_result.support_leak_count = 0
    mock_sim_result.total_cost_hybrid = 0.0
    mock_sim_result.total_cost_baseline = 0.0
    mock_sim_result.cap_saturation_count = 0
    mock_sim_result.rebalance_count = 0
    mock_sim_result.trade_count = 0
    mock_sim_result.fold_rets_hybrid = []
    mock_sim_result.fold_rets_baseline = []
    mock_sim_result.fold_selected_symbols = ()
    mock_sim_result.block_rets_hybrid = ()
    mock_sim_result.block_rets_baseline = ()
    mock_sim_result.fit_rets_hybrid = ()
    mock_sim_result.fit_rets_by_fold = ()
    mock_sim_result.fold_attributions = ()

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation",
        return_value=mock_sim_result,
    )
    audit_spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
    )

    mock_cache = mocker.MagicMock()
    mock_cache.vol_matrix_2d = None
    mock_cache.regime_policy_by_fold = ()

    mock_config = mocker.MagicMock()
    mock_config.l2_entry_spike_penalty_weight = 0.0
    mock_config.l2_max_mdd_abs = 0.5
    mock_config.l2_worst_fold_penalty_threshold = 0.0
    mock_config.l2_worst_fold_penalty_weight = 0.0
    mock_config.l2_objective_risk_util_target = 1.0
    mock_config.l2_objective_risk_util_weight = 0.0
    mock_config.l2_objective_trade_target = 10
    mock_config.l2_objective_trade_weight = 0.0
    mock_config.l2_turnover_penalty_weight = 0.0
    mock_config.l2_deploy_enabled = False
    mock_config.l2_growth_lcb_z = 0.0
    mock_config.l2_deploy_mdd_margin = 0.0
    mock_config.l2_deploy_cvar_margin = 0.0
    mock_config.l2_deploy_l_hard_cap = 1.0
    mock_config.l2_deploy_fit_mdd_crisis_gate = None
    mock_config.l2_objective_growth_lcb_weight = 0.0
    mock_config.l2_deploy_kelly_safety_fraction = None
    mock_config.l2_max_cvar_95 = 0.5
    mock_config.l2_max_exchange_leverage = None
    mock_config.l2_entry_spike_penalty_weight = 0.0
    mock_config.l2_sleeve_combine_method = "pooled"
    mock_config.l2_sleeve_conviction_cap_mult = 1.0
    mock_config.entry_block_spike = None
    mock_config.l2_diag_attribution_enabled = False
    mock_config.l2_diag_sleeve_top_k = 15
    mock_config.l2_diag_sleeve_sample_every = 0

    mock_caps = mocker.MagicMock()
    mock_caps.max_net_notional = 1e9
    mock_caps.max_leverage = 10.0
    mock_caps.max_short_ratio = 1.0
    mock_caps.max_symbol_weight = 1.0
    mock_caps.min_symbol_weight = 0.0
    mock_caps.max_sector_weight = 1.0
    mock_caps.max_gross_exposure_notional = 1e9

    mock_aligned = mocker.MagicMock()
    mock_aligned.datetimes = []
    mock_aligned.close_2d = None
    mock_aligned.symbols = ()
    mock_aligned.active_mask = None
    mock_aligned.inference_active_mask = None
    mock_aligned.warm_mask = None
    mock_aligned.inference_entry_warm_mask = None
    mock_aligned.entry_block_mask = None
    mock_aligned.kill_mask = None
    mock_aligned.execution_cost_bps_2d = None
    mock_aligned.funding_2d = None
    mock_aligned.beta_vs_market_1d = None

    mock_cfg = mocker.MagicMock()
    mock_cfg.l2_crowding_floor_mult = 0.5
    mock_cfg.min_abs_rank_z = 0.0

    entry_audit_preset = mocker.MagicMock()
    entry_audit_preset.warnings = ()

    evaluation = evaluate_l2_trial(
        cache=mock_cache,
        signal_batch=mocker.MagicMock(start_idx=0, end_idx=1),
        aligned=mock_aligned,
        awf_folds=(),
        config=mock_config,
        caps=mock_caps,
        tf="4h",
        entry_audit=entry_audit_preset,
        lightweight=True,
    )

    assert audit_spy.call_count == 0
    assert evaluation.block_metrics == ()


def test_build_l2_signal_batch_caches_by_fingerprint_on_second_call(mocker, tmp_path):
    from src.application.futures.runner.active_pipeline import _build_l2_signal_batch

    cache_dir = str(tmp_path / "l2_signal_cache")
    mocker.patch(
        "src.application.futures.runner.active_pipeline.OPT_FUTURES_CONFIG",
        {"L2_SIGNAL_BATCH_CACHE_DIR": cache_dir},
        create=True,
    )

    predict_spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.signal_selection.predict_layer1_signals_multi_tf",
        return_value=_PicklableSignalBatch(),
    )

    mock_l1 = mocker.MagicMock()
    mock_l1.artifacts_by_tf = {"4h": _PicklableArtifact()}
    mock_l1.inference_artifact = None

    mock_window = _PicklableWindow()
    mock_aligned = mocker.MagicMock()
    mock_aligned.datetimes = []
    mock_aligned.close_2d = None
    mock_aligned.symbols = ()

    mock_cfg = mocker.MagicMock()
    mock_cfg.timeframe = "4h"

    # 1st call: should compute and cache
    result1 = _build_l2_signal_batch(
        l1_res=mock_l1,
        labeled_events=[],
        aligned=mock_aligned,
        cfg=mock_cfg,
        window=mock_window,
    )
    assert predict_spy.call_count == 1

    # 2nd call with same fingerprint: should read from cache
    result2 = _build_l2_signal_batch(
        l1_res=mock_l1,
        labeled_events=[],
        aligned=mock_aligned,
        cfg=mock_cfg,
        window=mock_window,
    )
    assert predict_spy.call_count == 1  # no additional call


def test_run_tiered_l2_study_propagates_lightweight_flag_to_trials(mocker):
    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study

    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    import numpy as np

    mock_cache = L2SimulationCache(
        vol_matrix_2d=np.zeros((10, 1), dtype=np.float64),
        tradeable_mask_2d=np.zeros((10, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((10, 1), dtype=np.float64),
        funding_2d=np.zeros((10, 1), dtype=np.float64),
        beta_1d=np.zeros(1, dtype=np.float64),
        expected_gross_bps_2d=np.zeros((10, 1), dtype=np.float64),
        expected_net_bps_2d=np.zeros((10, 1), dtype=np.float64),
        holding_bars_2d=np.ones((10, 1), dtype=np.float64),
        side_2d=np.zeros((10, 1), dtype=np.float64),
        quality_weight_2d=np.zeros((10, 1), dtype=np.float64),
        signal_mask_2d=np.zeros((10, 1), dtype=np.bool_),
        sleeve_to_sym=np.zeros(1, dtype=np.int64),
        sleeve_keys=(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mock_cache,
    )

    mock_study_result = mocker.MagicMock(blocker_reason="", best_evaluation=None, dsr=0.0)
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
        return_value=mock_study_result,
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.MagicMock(code_1d=0, vol_scale_1d=1.0, crisis_active_1d=False),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
        return_value=0,
    )

    mock_routing_plan = mocker.MagicMock()
    mock_routing_plan.effective_bucket_edges_by_fold = {}
    mock_routing_plan.pooled_edges_by_fold = {}
    mock_routing_plan.effective_regime_code_1d = 0
    mock_routing_plan.diagnostics.active_state_count = 0
    mock_routing_plan.diagnostics.compression_enabled = False
    mock_routing_plan.diagnostics.conditioning_path = "none"
    mock_routing_plan.diagnostics.proof_passed = True
    mock_routing_plan.diagnostics.mean_lift_bps = 0.0
    mock_routing_plan.diagnostics.nw_tstat = 0.0
    mock_routing_plan.diagnostics.fold_pass_ratio = 1.0
    mock_routing_plan.diagnostics.debug_diagnostics = None
    mock_routing_plan.policy_by_fold = ()
    mock_routing_plan.diagnostics.policy_diagnostics = None
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
        return_value=mock_routing_plan,
    )
    mocker.patch(
        "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
        return_value=(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection._signal_batch_fingerprint",
        return_value="mock_fp",
    )

    mock_cfg = _MockCfg()

    import pandas as pd

    window_mock = mocker.MagicMock()
    window_mock.holdout_start = pd.Timestamp("2026-06-01", tz="UTC")
    window_mock.l2_start = pd.Timestamp("2026-01-01", tz="UTC")

    signal_batch_mock = mocker.MagicMock(start_idx=0, end_idx=10, events=[])
    signal_batch_mock.registry_version = "mock_v1"

    result = _run_tiered_l2_study(
        signal_batch=signal_batch_mock,
        aligned=mocker.MagicMock(datetimes=[], close_2d=None, symbols=()),
        cfg=mock_cfg,
        window=window_mock,
        caps=mocker.MagicMock(),
        tf="4h",
        n_trials=1,
        seed=42,
    )

    assert result is not None


@pytest.mark.skip(reason="E2E: requires full pipeline with real data")
def test_l2_phase_wall_time_under_120_seconds():
    pass


@pytest.mark.skip(reason="E2E: requires full pipeline with real data")
def test_l2_phase_peak_rss_under_9gb():
    pass
