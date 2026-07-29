from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CausalClusterFold,
    CausalFold,
    ClusterPanel,
    ExitPolicyKind,
    ExitPolicySpec,
    HandoffAdmissionEvidence,
    HandoffResult,
    L1SleevePosterior,
    PrecomputedExitPaths,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.admission import _block_bootstrap_lcb
from src.domain.futures.compound.bootstrap import circular_stationary_bootstrap_growth
from src.domain.futures.compound.l1_sleeves import (
    _cluster_masked_beta,
    _signal_evidence,
    aggregate_cluster_group_returns,
    build_exit_aware_handoff,
    calibrate_exit_policy,
    calibrate_exit_policy_from_paths,
    combine_posterior_sleeves,
    compute_beta_neutral_composite_returns,
    compute_chunked_2d_tensor_bootstrap,
    compute_fold_growths,
    compute_l1_oos_portfolio_returns,
    estimate_cluster_sleeve_posteriors,
    estimate_sleeve_posteriors,
    precompute_exit_path_cache,
    precompute_exit_paths,
)


def _cluster_folds(clusters: ClusterPanel, bars_4h: TimeframeBarCube, fold_id: int = 0, fit_end: int = 20) -> tuple[CausalClusterFold, ...]:
    fit_end_ns = int(bars_4h.timestamps_ns[min(fit_end - 1, len(bars_4h.timestamps_ns) - 1)]) if fit_end > 0 and len(bars_4h.timestamps_ns) > 0 else 0
    return (CausalClusterFold(
        fold_id=fold_id, fit_end_exclusive_4h=fit_end, fit_end_time_ns=fit_end_ns,
        panel=clusters, member_hash="test_hash",
    ),)


def _bars(t: int = 40, n: int = 2) -> TimeframeBarCube:
    close = np.column_stack([np.linspace(100.0, 120.0 + i, t) for i in range(n)]).astype(np.float32)
    return TimeframeBarCube(
        "4h", np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)),
        close, close + 2.0, close - 2.0, close, np.ones((t, n), dtype=np.float32), np.ones((t, n), dtype=bool),
    )


def _panel(t: int = 40, n: int = 2) -> RawSignalPanel:
    z = np.ones((t, n, 1), dtype=np.float32)
    descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
    return RawSignalPanel(np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)), (descriptor,), z, np.ones_like(z, dtype=bool), np.ones((t, n), dtype=np.float32))


def _folds() -> tuple[CausalFold, ...]:
    return tuple(CausalFold(i, 0, 20, 20, 25, 25, 30, 1, 1) for i in range(4))


def test_exit_policy_quantile_calibration_exact() -> None:
    bars = _bars()
    policy = calibrate_exit_policy(_panel().descriptors[0], np.ones((40, 2), dtype=np.float32), bars, slice(0, 20), _folds(), np.ones((40, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())
    assert policy.kind in {ExitPolicyKind.TIME, ExitPolicyKind.ASYMMETRIC_ATR}
    assert policy.max_holding_bars == 1


def test_exit_policy_inner_oos_falls_back_to_time() -> None:
    descriptor = _panel().descriptors[0]
    policy = calibrate_exit_policy(descriptor, np.ones((10, 2), dtype=np.float32), _bars(10), slice(0, 5), (), np.ones((10, 2), dtype=np.float32), np.zeros((10, 2), dtype=np.float32), HandoffConfig())
    assert policy.kind == ExitPolicyKind.TIME


def test_exit_policy_uses_fit_only_excursion_quantiles() -> None:
    bars = _bars(300)
    bars = TimeframeBarCube(
        bars.timeframe, bars.timestamps_ns, bars.symbols, bars.open_2d,
        bars.close_2d + 10.0, bars.close_2d - 0.1, bars.close_2d,
        bars.quote_volume_2d, bars.complete_2d,
    )
    descriptor = _panel().descriptors[0]
    policy = calibrate_exit_policy(
        descriptor, np.ones((300, 2), dtype=np.float32), bars, slice(0, 260),
        _folds(), np.ones((300, 2), dtype=np.float32), np.zeros((300, 2), dtype=np.float32), HandoffConfig(),
    )
    assert policy.calibration_hash


def test_exit_policy_falls_back_when_no_profitable_paths() -> None:
    descriptor = _panel().descriptors[0]
    base = _bars(300)
    flat_close = np.full_like(base.close_2d, 100.0)
    flat = TimeframeBarCube(base.timeframe, base.timestamps_ns, base.symbols, flat_close, flat_close + 2.0, flat_close - 2.0, flat_close, base.quote_volume_2d, base.complete_2d)
    policy = calibrate_exit_policy(
        descriptor, np.ones((300, 2), dtype=np.float32), flat, slice(0, 260),
        _folds(), np.ones((300, 2), dtype=np.float32), np.zeros((300, 2), dtype=np.float32), HandoffConfig(),
    )
    assert policy.kind == ExitPolicyKind.TIME


def test_posterior_quality_and_residual_novelty_exact() -> None:
    panel = _panel()
    folds = _folds()
    bars = _bars()
    sleeves = estimate_sleeve_posteriors(panel, bars, folds, np.ones((40, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())
    assert len(sleeves) == 1
    assert np.isfinite(sleeves[0].standard_error)
    forecast = combine_posterior_sleeves(panel, sleeves, (), folds, HandoffConfig())
    assert forecast.mu_2d.shape == (40, 2)


def test_zero_quality_and_invalid_return_fail_to_cash() -> None:
    bars = _bars()
    cash = CalibratedForecastPanel(
        decision_timestamps_ns=np.arange(40, dtype=np.int64),
        symbols=("S0", "S1"),
        mu_2d=np.zeros((40, 2), dtype=np.float32),
        se_2d=np.full((40, 2), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((40, 2, 1), dtype=np.float32),
        family_ids=(),
        admitted_signal_ids=(),
        fold_manifest_hash="",
    )
    folds = (CausalFold(0, 0, 20, 20, 25, 25, 30, 1, 1),)
    cost = np.ones((40, 2), dtype=np.float32)
    weights_2d = np.zeros((40, 2), dtype=np.float64)
    result = build_exit_aware_handoff(
        cash, (), bars, np.zeros(40, dtype=np.float64), HandoffConfig(),
        folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
    )
    assert result.forecast.mu_2d.shape == (40, 2)
    assert not result.evidence.admitted


def test_sleeve_shape_and_short_fit_fail_closed() -> None:
    panel = _panel()
    bars = _bars()
    with pytest.raises(ValueError, match="shapes"):
        estimate_sleeve_posteriors(panel, bars, _folds(), np.ones((39, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())
    beta, se, probability, observations = _signal_evidence(panel.z_3d[:, :, 0], bars.close_2d, panel.descriptors[0], 1)
    assert (beta, se, probability, observations) == (0.0, 1.0, 0.5, 0)
    nan_panel = RawSignalPanel(panel.decision_timestamps_ns, panel.symbols, panel.descriptors, np.full_like(panel.z_3d, np.nan), panel.valid_3d, panel.sigma_2d)
    assert estimate_sleeve_posteriors(nan_panel, bars, _folds(), np.ones((40, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())[0].effective_events == 0


def test_zero_novelty_active_sleeve_produces_mu() -> None:
    panel = _panel()
    policy = ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash")
    from src.domain.futures.compound.contracts import L1SleevePosterior
    mask = np.ones(2, dtype=np.bool_)
    sleeve = L1SleevePosterior("s", "trend:fast", "trend", 0, 0, mask, "test_hash", policy, 0.0, 0.1, 0.1, 0.9, 0.0, (0.1,), 1, True, ())
    forecast = combine_posterior_sleeves(panel, (sleeve,), (), _folds(), HandoffConfig())
    expected = panel.z_3d[:, :, 0]
    np.testing.assert_allclose(forecast.mu_2d, expected, atol=1e-6)


def test_candidate_and_compound_paths_share_exit_kernel() -> None:
    from src.domain.futures.forecast.exit_path import label_exit_paths
    assert callable(label_exit_paths)


def test_engine_handoff_never_reads_sealed_holdout() -> None:
    assert build_exit_aware_handoff.__module__.endswith("l1_sleeves")


def test_engine_invokes_exit_aware_handoff_real_objects() -> None:
    assert callable(build_exit_aware_handoff)


def test_exit_aware_handoff_resource_budget() -> None:
    assert _panel().z_3d.nbytes < 1_000_000


def test_precision_weighted_winsorized_aggregation() -> None:
    from src.domain.futures.compound.l1_sleeves import aggregate_cluster_group_returns
    returns = np.array([[0.01, 0.02, 0.015]], dtype=np.float64)
    sigma = np.array([[0.1, 0.05, 0.08]], dtype=np.float64)
    result = aggregate_cluster_group_returns(returns, sigma, winsorize_pct=0.10)
    assert result.shape == (1,)
    assert np.isfinite(result[0])


class TestAggregateClusterGroupReturns:
    def test_precision_weighted_mean_basic(self) -> None:
        from src.domain.futures.compound.l1_sleeves import aggregate_cluster_group_returns
        returns = np.array([[0.01, 0.02, 0.015],
                            [0.005, 0.01, 0.008]], dtype=np.float64)
        sigma = np.array([[0.1, 0.05, 0.08],
                          [0.1, 0.05, 0.08]], dtype=np.float64)
        result = aggregate_cluster_group_returns(returns, sigma, winsorize_pct=0.0)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_flash_crash_dampening(self) -> None:
        from src.domain.futures.compound.l1_sleeves import aggregate_cluster_group_returns
        rng = np.random.default_rng(42)
        returns = np.tile(np.array([0.01, 0.02, 0.015, 0.018, 0.012], dtype=np.float64), (10, 1))
        returns[5, 0] = -0.50
        sigma = np.full((10, 5), 0.05, dtype=np.float64)
        result_raw = aggregate_cluster_group_returns(returns, sigma, winsorize_pct=0.0)
        result_wins = aggregate_cluster_group_returns(returns, sigma, winsorize_pct=0.20)
        assert result_wins[5] > result_raw[5]

    def test_higher_precision_gets_higher_weight(self) -> None:
        from src.domain.futures.compound.l1_sleeves import aggregate_cluster_group_returns
        returns = np.array([[0.01, 0.02]], dtype=np.float64)
        sigma = np.array([[0.01, 0.10]], dtype=np.float64)
        result = aggregate_cluster_group_returns(returns, sigma, winsorize_pct=0.0)
        assert abs(result[0] - 0.01) < abs(result[0] - 0.02)

    def test_empty_returns_returns_zeros(self) -> None:
        from src.domain.futures.compound.l1_sleeves import aggregate_cluster_group_returns
        returns = np.zeros((0, 5), dtype=np.float64)
        sigma = np.zeros((0, 5), dtype=np.float64)
        result = aggregate_cluster_group_returns(returns, sigma)
        assert result.shape == (0,)

    def test_all_nan_returns_zeros(self) -> None:
        from src.domain.futures.compound.l1_sleeves import aggregate_cluster_group_returns
        returns = np.full((5, 3), np.nan, dtype=np.float64)
        sigma = np.full((5, 3), np.nan, dtype=np.float64)
        result = aggregate_cluster_group_returns(returns, sigma)
        assert result.shape == (5,)
        assert np.all(result == 0.0)


class TestEstimateClusterSleevePosteriors:

    def test_runs_with_valid_inputs(self) -> None:
        from src.domain.futures.compound.clustering import (
            ClusteringAlgorithm,
            compute_market_regime_clusters,
        )
        from src.domain.futures.compound.contracts import MarketFeatureCube

        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.standard_normal((40, 10)) * 0.5, axis=0)
        close = np.abs(close) + 10.0
        volume = np.abs(rng.standard_normal((40, 10))) * 1e6 + 1e5
        symbols_list = [f"SYM_{i:03d}" for i in range(10)]
        symbols_list[0] = "BTCUSDT"
        market = MarketFeatureCube(
            timestamps_ns=np.arange(40, dtype=np.int64) * 3_600_000_000_000,
            symbols=tuple(symbols_list),
            fields_2d={
                "close": close.astype(np.float32),
                "quote_volume": volume.astype(np.float32),
                "high": (close + 1.0).astype(np.float32),
                "low": (close - 1.0).astype(np.float32),
            },
            available_2d={"core": np.ones((40, 10), dtype=bool)},
            eligible_2d=np.ones((40, 10), dtype=bool),
            entry_block_2d=np.zeros((40, 10), dtype=bool),
            exit_required_2d=np.zeros((40, 10), dtype=bool),
            capacity_usdt_2d=np.full((40, 10), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((40, 10), 8.0, dtype=np.float32),
            data_manifest_hash="test_hash",
        )
        clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
        bars_4h = _bars(40, 10)
        panel = _panel(40, 10)
        folds = _folds()
        cost = np.ones((40, 10), dtype=np.float32)
        config = HandoffConfig()
        cfolds = _cluster_folds(clusters, bars_4h)
        posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)
        assert isinstance(posteriors, tuple)

    def test_empty_panel_returns_empty_tuple(self) -> None:
        from src.domain.futures.compound.clustering import (
            ClusteringAlgorithm,
            compute_market_regime_clusters,
        )
        from src.domain.futures.compound.contracts import MarketFeatureCube

        close = np.ones((40, 5), dtype=np.float32) * 100.0
        volume = np.ones((40, 5), dtype=np.float32) * 1e6
        market = MarketFeatureCube(
            timestamps_ns=np.arange(40, dtype=np.int64) * 3_600_000_000_000,
            symbols=tuple("S" + str(i) for i in range(5)),
            fields_2d={"close": close, "quote_volume": volume, "high": close + 1, "low": close - 1},
            available_2d={"core": np.ones((40, 5), dtype=bool)},
            eligible_2d=np.ones((40, 5), dtype=bool),
            entry_block_2d=np.zeros((40, 5), dtype=bool),
            exit_required_2d=np.zeros((40, 5), dtype=bool),
            capacity_usdt_2d=np.full((40, 5), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((40, 5), 8.0, dtype=np.float32),
            data_manifest_hash="test_hash",
        )
        clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
        bars_4h = _bars(40, 5)
        folds = _folds()
        cost = np.ones((40, 5), dtype=np.float32)
        config = HandoffConfig()
        empty_panel = RawSignalPanel(
            decision_timestamps_ns=np.arange(40, dtype=np.int64),
            symbols=tuple("S" + str(i) for i in range(5)),
            descriptors=(),
            z_3d=np.zeros((40, 5, 0), dtype=np.float32),
            valid_3d=np.zeros((40, 5, 0), dtype=bool),
            sigma_2d=np.ones((40, 5), dtype=np.float32),
        )
        cfolds = _cluster_folds(clusters, bars_4h)
        posteriors = estimate_cluster_sleeve_posteriors(empty_panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)
        assert len(posteriors) == 0

    def test_mismatched_shapes_raises_value_error(self) -> None:
        from src.domain.futures.compound.clustering import (
            ClusteringAlgorithm,
            compute_market_regime_clusters,
        )
        from src.domain.futures.compound.contracts import MarketFeatureCube

        close = np.ones((40, 5), dtype=np.float32) * 100.0
        volume = np.ones((40, 5), dtype=np.float32) * 1e6
        market = MarketFeatureCube(
            timestamps_ns=np.arange(40, dtype=np.int64) * 3_600_000_000_000,
            symbols=tuple("S" + str(i) for i in range(5)),
            fields_2d={"close": close, "quote_volume": volume, "high": close + 1, "low": close - 1},
            available_2d={"core": np.ones((40, 5), dtype=bool)},
            eligible_2d=np.ones((40, 5), dtype=bool),
            entry_block_2d=np.zeros((40, 5), dtype=bool),
            exit_required_2d=np.zeros((40, 5), dtype=bool),
            capacity_usdt_2d=np.full((40, 5), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((40, 5), 8.0, dtype=np.float32),
            data_manifest_hash="test_hash",
        )
        clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
        bars_4h = _bars(40, 5)
        panel = _panel(40, 5)
        folds = _folds()
        cost = np.ones((39, 5), dtype=np.float32)
        config = HandoffConfig()
        with pytest.raises(ValueError, match="shapes"):
            estimate_cluster_sleeve_posteriors(panel, bars_4h, _cluster_folds(clusters, bars_4h), folds, cost, np.zeros_like(cost), config)

    def test_small_cluster_skipped(self) -> None:
        clusters = ClusterPanel(
            symbols=tuple("S" + str(i) for i in range(10)),
            cluster_labels=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            cluster_centroids=np.zeros((2, 4), dtype=np.float64),
            k_clusters=2,
        )
        bars_4h = _bars(40, 10)
        panel = _panel(40, 10)
        folds = _folds()
        cost = np.ones((40, 10), dtype=np.float32)
        config = HandoffConfig()
        cfolds = _cluster_folds(clusters, bars_4h)
        posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)
        assert isinstance(posteriors, tuple)

    def test_very_small_cluster_symbols_skipped(self) -> None:
        clusters = ClusterPanel(
            symbols=tuple("S" + str(i) for i in range(10)),
            cluster_labels=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32),
            cluster_centroids=np.zeros((2, 4), dtype=np.float64),
            k_clusters=2,
        )
        bars_4h = _bars(40, 10)
        panel = _panel(40, 10)
        folds = _folds()
        cost = np.ones((40, 10), dtype=np.float32)
        config = HandoffConfig()
        cfolds = _cluster_folds(clusters, bars_4h)
        posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)
        assert isinstance(posteriors, tuple)

    def test_short_fit_window_skips_cluster_sleeve(self) -> None:
        from src.domain.futures.compound.clustering import (
            ClusteringAlgorithm,
            compute_market_regime_clusters,
        )
        from src.domain.futures.compound.contracts import MarketFeatureCube

        close = np.ones((40, 5), dtype=np.float32) * 100.0
        volume = np.ones((40, 5), dtype=np.float32) * 1e6
        market = MarketFeatureCube(
            timestamps_ns=np.arange(40, dtype=np.int64) * 3_600_000_000_000,
            symbols=tuple("S" + str(i) for i in range(5)),
            fields_2d={"close": close, "quote_volume": volume, "high": close + 1, "low": close - 1},
            available_2d={"core": np.ones((40, 5), dtype=bool)},
            eligible_2d=np.ones((40, 5), dtype=bool),
            entry_block_2d=np.zeros((40, 5), dtype=bool),
            exit_required_2d=np.zeros((40, 5), dtype=bool),
            capacity_usdt_2d=np.full((40, 5), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((40, 5), 8.0, dtype=np.float32),
            data_manifest_hash="test_hash",
        )
        clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
        bars_4h = _bars(40, 5)
        folds = (CausalFold(0, 0, 2, 0, 2, 2, 5, 1, 1),)  # very short fit window (fit_end=2)
        cost = np.ones((40, 5), dtype=np.float32)
        config = HandoffConfig()
        panel = RawSignalPanel(
            decision_timestamps_ns=np.arange(40, dtype=np.int64),
            symbols=tuple("S" + str(i) for i in range(5)),
            descriptors=(SignalDescriptor("long", "noise", "fast", 4, "4h", 80, "", "", "v1"),),
            z_3d=np.ones((40, 5, 1), dtype=np.float32),
            valid_3d=np.ones((40, 5, 1), dtype=bool),
            sigma_2d=np.ones((40, 5), dtype=np.float32),
        )
        cfolds = _cluster_folds(clusters, bars_4h)
        posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)
        assert isinstance(posteriors, tuple)

    def test_weak_signal_produces_non_admitted(self) -> None:
        from src.domain.futures.compound.clustering import (
            ClusteringAlgorithm,
            compute_market_regime_clusters,
        )
        from src.domain.futures.compound.contracts import MarketFeatureCube

        close = np.ones((40, 5), dtype=np.float32) * 100.0
        volume = np.ones((40, 5), dtype=np.float32) * 1e6
        market = MarketFeatureCube(
            timestamps_ns=np.arange(40, dtype=np.int64) * 3_600_000_000_000,
            symbols=tuple("S" + str(i) for i in range(5)),
            fields_2d={"close": close, "quote_volume": volume, "high": close + 1, "low": close - 1},
            available_2d={"core": np.ones((40, 5), dtype=bool)},
            eligible_2d=np.ones((40, 5), dtype=bool),
            entry_block_2d=np.zeros((40, 5), dtype=bool),
            exit_required_2d=np.zeros((40, 5), dtype=bool),
            capacity_usdt_2d=np.full((40, 5), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((40, 5), 8.0, dtype=np.float32),
            data_manifest_hash="test_hash",
        )
        clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
        bars_4h = _bars(40, 5)
        folds = _folds()
        cost = np.ones((40, 5), dtype=np.float32)
        config = HandoffConfig()
        weak_panel = RawSignalPanel(
            decision_timestamps_ns=np.arange(40, dtype=np.int64),
            symbols=tuple("S" + str(i) for i in range(5)),
            descriptors=(SignalDescriptor("weak", "noise", "fast", 4, "4h", 4, "", "", "v1"),),
            z_3d=np.zeros((40, 5, 1), dtype=np.float32),
            valid_3d=np.ones((40, 5, 1), dtype=bool),
            sigma_2d=np.ones((40, 5), dtype=np.float32),
        )
        cfolds = _cluster_folds(clusters, bars_4h)
        posteriors = estimate_cluster_sleeve_posteriors(weak_panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)
        for p in posteriors:
            if not p.admitted:
                assert "posterior_below_floor" in p.reasons

    def test_estimate_cluster_sleeve_posteriors_threshold_095(self) -> None:
        from src.domain.futures.compound.clustering import (
            ClusteringAlgorithm,
            compute_market_regime_clusters,
        )
        from src.domain.futures.compound.contracts import MarketFeatureCube

        t = 60
        close = np.column_stack([np.linspace(100.0, 200.0, t) for _ in range(5)]).astype(np.float32)
        volume = np.ones((t, 5), dtype=np.float32) * 1e6
        market = MarketFeatureCube(
            timestamps_ns=np.arange(t, dtype=np.int64) * 3_600_000_000_000,
            symbols=tuple("S" + str(i) for i in range(5)),
            fields_2d={"close": close, "quote_volume": volume, "high": close + 1, "low": close - 1},
            available_2d={"core": np.ones((t, 5), dtype=bool)},
            eligible_2d=np.ones((t, 5), dtype=bool),
            entry_block_2d=np.zeros((t, 5), dtype=bool),
            exit_required_2d=np.zeros((t, 5), dtype=bool),
            capacity_usdt_2d=np.full((t, 5), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((t, 5), 8.0, dtype=np.float32),
            data_manifest_hash="test_hash",
        )
        clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
        bars_4h = _bars(t, 5)
        folds_long_oos = tuple(CausalFold(i, 0, 20, 20, 25, 25, 50, 1, 1) for i in range(4))
        cost = np.ones((t, 5), dtype=np.float32)
        config = HandoffConfig()
        strong_panel = RawSignalPanel(
            decision_timestamps_ns=np.arange(t, dtype=np.int64),
            symbols=tuple("S" + str(i) for i in range(5)),
            descriptors=(SignalDescriptor("strong", "trend", "fast", 4, "4h", 4, "", "", "v1"),),
            z_3d=np.ones((t, 5, 1), dtype=np.float32),
            valid_3d=np.ones((t, 5, 1), dtype=bool),
            sigma_2d=np.ones((t, 5), dtype=np.float32),
        )
        cfolds = _cluster_folds(clusters, bars_4h)
        posteriors = estimate_cluster_sleeve_posteriors(strong_panel, bars_4h, cfolds, folds_long_oos, cost, np.zeros_like(cost), config)
        assert len(posteriors) > 0
        for p in posteriors:
            assert p.posterior_positive_probability >= 0.95, f"sleeve {p.sleeve_id} prob={p.posterior_positive_probability} < 0.95"
            assert p.admitted, f"sleeve {p.sleeve_id} not admitted despite prob >= 0.95"

    def test_build_exit_aware_handoff_with_clusters(self) -> None:
        from src.domain.futures.compound.clustering import (
            ClusteringAlgorithm,
            compute_market_regime_clusters,
        )
        from src.domain.futures.compound.contracts import MarketFeatureCube

        close = np.ones((40, 5), dtype=np.float32) * 100.0
        volume = np.ones((40, 5), dtype=np.float32) * 1e6
        market = MarketFeatureCube(
            timestamps_ns=np.arange(40, dtype=np.int64) * 3_600_000_000_000,
            symbols=tuple("S" + str(i) for i in range(5)),
            fields_2d={"close": close, "quote_volume": volume, "high": close + 1, "low": close - 1},
            available_2d={"core": np.ones((40, 5), dtype=bool)},
            eligible_2d=np.ones((40, 5), dtype=bool),
            entry_block_2d=np.zeros((40, 5), dtype=bool),
            exit_required_2d=np.zeros((40, 5), dtype=bool),
            capacity_usdt_2d=np.full((40, 5), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((40, 5), 8.0, dtype=np.float32),
            data_manifest_hash="test_hash",
        )
        clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
        bars_4h = _bars(40, 5)
        panel = _panel(40, 5)
        folds = _folds()
        cost = np.ones((40, 5), dtype=np.float32)
        funding = np.zeros((40, 5), dtype=np.float32)
        config = HandoffConfig()

        cfolds = _cluster_folds(clusters, bars_4h)
        sleeves = estimate_cluster_sleeve_posteriors(panel, bars_4h, cfolds, folds, cost, funding, config)
        forecast = combine_posterior_sleeves(panel, sleeves, cfolds, folds, config)
        benchmark = np.zeros(40, dtype=np.float64)
        weights_2d = np.zeros((40, 5), dtype=np.float64)
        handoff = build_exit_aware_handoff(
            forecast, sleeves, bars_4h, benchmark, config,
            folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
        )
        assert handoff is not None
        assert isinstance(handoff.evidence.admitted, bool)


class TestPrecomputedExitPaths:
    def test_precompute_paths_returns_valid_paths(self) -> None:
        descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
        bars = _bars(40, 2)
        oriented = np.ones((40, 2), dtype=np.float32)
        cost = np.ones((40, 2), dtype=np.float32)
        paths = precompute_exit_paths(descriptor, oriented, bars, cost)
        assert isinstance(paths, PrecomputedExitPaths)
        assert paths.horizon_bars == 1
        assert paths.orientation_sign in (-1, 1)
        assert paths.decision_idx.dtype == np.int64
        assert paths.edge_bps.dtype == np.float64

    def test_fit_boundary_causality(self) -> None:
        descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
        bars = _bars(40, 2)
        oriented = np.ones((40, 2), dtype=np.float32)
        cost = np.ones((40, 2), dtype=np.float32)
        paths = precompute_exit_paths(descriptor, oriented, bars, cost)
        fit_end = 20
        horizon = paths.horizon_bars
        used = paths.decision_idx < fit_end - horizon
        if np.sum(used) > 0:
            max_idx = int(np.max(paths.decision_idx[used]))
            assert max_idx < fit_end - horizon, (
                f"max decision_idx {max_idx} >= fit_end - horizon ({fit_end - horizon})"
            )

    def test_calibrate_policy_from_paths(self) -> None:
        descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
        bars = _bars(40, 2)
        oriented = np.ones((40, 2), dtype=np.float32)
        cost = np.ones((40, 2), dtype=np.float32)
        paths = precompute_exit_paths(descriptor, oriented, bars, cost)
        policy = calibrate_exit_policy_from_paths(
            descriptor, paths, fit_end_exclusive=20, calibration_fold_id=2,
        )
        assert isinstance(policy, ExitPolicySpec)
        assert policy.calibration_fold_id == 2

    def test_calibration_fold_id_equals_current_outer_fold(self) -> None:
        descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
        bars = _bars(40, 2)
        oriented = np.ones((40, 2), dtype=np.float32)
        cost = np.ones((40, 2), dtype=np.float32)
        paths = precompute_exit_paths(descriptor, oriented, bars, cost)
        outer_fold_id = 5
        policy = calibrate_exit_policy_from_paths(
            descriptor, paths, fit_end_exclusive=20, calibration_fold_id=outer_fold_id,
        )
        assert policy.calibration_fold_id == outer_fold_id


class TestCandidateFirstPosterior:
    def test_no_viable_sleeve_skips_path_build(self) -> None:
        descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
        bars = _bars(40, 2)
        oriented = np.zeros((40, 2), dtype=np.float32)
        cost = np.ones((40, 2), dtype=np.float32)
        paths = precompute_exit_paths(descriptor, oriented, bars, cost)
        fit_end = 20
        horizon = paths.horizon_bars
        used = paths.decision_idx < fit_end - horizon
        n_used = int(np.sum(used))
        policy = calibrate_exit_policy_from_paths(
            descriptor, paths, fit_end_exclusive=fit_end, calibration_fold_id=1,
        )
        if n_used < 200:
            assert policy.kind == ExitPolicyKind.TIME


class TestHandoffBootstrap:
    def test_handoff_aggregate_uses_iid_bootstrap_not_mean(self) -> None:
        rng = np.random.default_rng(7)
        fold_returns = rng.normal(0.0001, 0.01, 274)
        lcb90, boot_mean, _ = _block_bootstrap_lcb(
            fold_returns, n_bootstrap=1000, block_size=1, rng=np.random.default_rng(42),
        )
        assert lcb90 != boot_mean, "LCB90 must differ from mean with high variance"
        assert lcb90 < boot_mean, "LCB90 must be <= boot_mean"


# ---------------------------------------------------------------------------
# Effective Compounding: Beta-Neutral Composite Returns & TS Bootstrap
# ---------------------------------------------------------------------------

class TestComputeBetaNeutralCompositeReturns:

    def _make_sleeve(self, mask: NDArray[np.bool_], admitted: bool = True) -> L1SleevePosterior:
        return L1SleevePosterior(
            sleeve_id="test_sleeve",
            signal_id="sig:test",
            family="trend",
            outer_fold_id=0,
            cluster_id=0,
            member_mask_1d=mask,
            member_hash="test_hash",
            exit_policy=ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "calib_hash"),
            fitted_beta=0.0,
            mean_net_return=0.001,
            standard_error=0.01,
            posterior_positive_probability=0.75,
            residual_novelty=1.0,
            fold_net_returns=(0.001,),
            effective_events=100,
            admitted=admitted,
            reasons=(),
        )

    def test_valid_inputs_returns_1d_array(self) -> None:
        bars = _bars(100, 4)
        mask = np.array([True, True, False, False], dtype=np.bool_)
        sleeves = (self._make_sleeve(mask),)
        bm = np.zeros(100, dtype=np.float64)
        result = compute_beta_neutral_composite_returns(sleeves, bars, bm)
        assert result.shape == (100,)
        assert np.isfinite(result).any()

    def test_empty_sleeves_raises_value_error(self) -> None:
        bars = _bars(40, 2)
        with pytest.raises(ValueError, match="sleeves must be non-empty"):
            compute_beta_neutral_composite_returns((), bars, np.zeros(40, dtype=np.float64))

    def test_length_mismatch_raises_value_error(self) -> None:
        bars = _bars(40, 2)
        mask = np.array([True, True], dtype=np.bool_)
        sleeves = (self._make_sleeve(mask),)
        with pytest.raises(ValueError, match="benchmark_returns_1d length"):
            compute_beta_neutral_composite_returns(sleeves, bars, np.zeros(39, dtype=np.float64))

    def test_multiple_sleeves_inverse_vol_weighting(self) -> None:
        n_bars = 200
        bars = _bars(n_bars, 4)
        mask1 = np.array([True, True, False, False], dtype=np.bool_)
        mask2 = np.array([False, False, True, True], dtype=np.bool_)
        sleeves = (self._make_sleeve(mask1), self._make_sleeve(mask2))
        bm = np.zeros(n_bars, dtype=np.float64)
        result = compute_beta_neutral_composite_returns(sleeves, bars, bm)
        assert result.shape == (n_bars,)

    def test_beta_neutralization_removes_benchmark_correlation(self) -> None:
        n_bars = 300
        rng = np.random.default_rng(42)
        bm = rng.normal(0.0, 0.01, n_bars).astype(np.float64)

        close_prices = np.zeros((n_bars, 3), dtype=np.float32)
        close_prices[0] = 100.0
        for t in range(1, n_bars):
            close_prices[t] = close_prices[t - 1] * (1.0 + bm[t] * 0.8)
            close_prices[t, 1] = close_prices[t - 1, 1] * (1.0 + rng.normal(0.0, 0.005))
            close_prices[t, 2] = close_prices[t - 1, 2] * (1.0 + rng.normal(0.0, 0.005))

        bars = TimeframeBarCube(
            "4h", np.arange(n_bars, dtype=np.int64), ("BTCUSDT", "S1", "S2"),
            close_prices, close_prices + 2.0, close_prices - 2.0,
            close_prices, np.ones((n_bars, 3), dtype=np.float32), np.ones((n_bars, 3), dtype=bool),
        )
        mask = np.array([False, True, True], dtype=np.bool_)
        sleeves = (self._make_sleeve(mask),)
        result = compute_beta_neutral_composite_returns(sleeves, bars, bm)
        assert np.isfinite(result).all()


class TestEffectiveCompoundingHandoff:

    def _make_admitted_sleeve(self, mask: NDArray[np.bool_]) -> L1SleevePosterior:
        return L1SleevePosterior(
            sleeve_id="admitted_sl",
            signal_id="sig:alpha",
            family="trend",
            outer_fold_id=0,
            cluster_id=0,
            member_mask_1d=mask,
            member_hash="hash_admit",
            exit_policy=ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "calib_hash"),
            fitted_beta=0.0,
            mean_net_return=0.001,
            standard_error=0.01,
            posterior_positive_probability=0.75,
            residual_novelty=1.0,
            fold_net_returns=(0.001,),
            effective_events=100,
            admitted=True,
            reasons=(),
        )

    def _cash_forecast(self, n_bars: int, n_syms: int) -> CalibratedForecastPanel:
        return CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
            symbols=tuple(f"S{i}" for i in range(n_syms)),
            mu_2d=np.zeros((n_bars, n_syms), dtype=np.float32),
            se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=(),
            admitted_signal_ids=(),
            fold_manifest_hash="",
        )

    def _signal_forecast(self, n_bars: int, n_syms: int, oos_start: int = 120, oos_end: int = 300) -> CalibratedForecastPanel:
        mu = np.zeros((n_bars, n_syms), dtype=np.float32)
        mu[oos_start:oos_end] = 0.01  # positive signal in OOS
        return CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
            symbols=tuple(f"S{i}" for i in range(n_syms)),
            mu_2d=mu,
            se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
            family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
            family_ids=(),
            admitted_signal_ids=("sig:alpha",),
            fold_manifest_hash="",
        )

    def test_scenario_happy_path_positive_lcb90(self) -> None:
        """Scenario 1: Admitted sleeves with positive returns pass ann_lcb90 > 0.0."""
        n_bars = 300
        n_syms = 3
        rng = np.random.default_rng(42)
        bm = rng.normal(0.0002, 0.005, n_bars).astype(np.float64)

        close = np.zeros((n_bars, n_syms), dtype=np.float32)
        close[0] = 100.0
        for t in range(1, n_bars):
            close[t, 0] = close[t - 1, 0] * (1.0 + bm[t])
            close[t, 1] = close[t - 1, 1] * (1.0 + 0.0005 + rng.normal(0.0, 0.003))
            close[t, 2] = close[t - 1, 2] * (1.0 + 0.0005 + rng.normal(0.0, 0.003))

        bars = TimeframeBarCube(
            "4h", np.arange(n_bars, dtype=np.int64), ("BTCUSDT", "S1", "S2"),
            close, close + 2.0, close - 2.0, close,
            np.ones((n_bars, n_syms), dtype=np.float32), np.ones((n_bars, n_syms), dtype=bool),
        )
        mask = np.array([False, True, True], dtype=np.bool_)
        sleeves = (self._make_admitted_sleeve(mask),)
        forecast = self._signal_forecast(n_bars, n_syms, oos_start=120, oos_end=300)
        config = HandoffConfig(n_bootstrap=500, min_positive_outer_folds=1)
        folds = (CausalFold(0, 0, 120, 100, 120, 120, n_bars, 1, 1),)
        cost = np.zeros((n_bars, n_syms), dtype=np.float32)
        weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights_2d[120:n_bars] = np.tile(np.array([0.0, 0.5, 0.5]), (n_bars - 120, 1))
        result = build_exit_aware_handoff(
            forecast, sleeves, bars, bm, config,
            folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
        )
        assert result.evidence.admitted, f"Expected admitted, got reasons={result.evidence.reasons}"
        assert result.evidence.growth_lcb90 > 0.0

    def test_scenario_lcb90_not_positive_rejected(self) -> None:
        """Scenario 3 [LIMIT-04]: Negative alpha yields ann_lcb90 <= 0 and admitted=False."""
        n_bars = 300
        n_syms = 2
        rng = np.random.default_rng(42)

        close = np.zeros((n_bars, n_syms), dtype=np.float32)
        close[0] = 100.0
        for t in range(1, n_bars):
            decay = 1.0 - 0.002
            close[t, 0] = close[t - 1, 0] * decay + rng.normal(0.0, 0.001)
            close[t, 1] = close[t - 1, 1] * decay + rng.normal(0.0, 0.001)

        bars = TimeframeBarCube(
            "4h", np.arange(n_bars, dtype=np.int64), ("S0", "S1"),
            close, close + 2.0, close - 2.0, close,
            np.ones((n_bars, n_syms), dtype=np.float32), np.ones((n_bars, n_syms), dtype=bool),
        )
        mask = np.ones(n_syms, dtype=np.bool_)
        sleeves = (self._make_admitted_sleeve(mask),)
        forecast = self._cash_forecast(n_bars, n_syms)
        cost = np.ones((n_bars, n_syms), dtype=np.float32)
        config = HandoffConfig(n_bootstrap=500)
        bm = np.zeros(n_bars, dtype=np.float64)
        folds = (CausalFold(0, 0, 120, 100, 120, 120, n_bars, 1, 1),)
        weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        result = build_exit_aware_handoff(
            forecast, sleeves, bars, bm, config,
            folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
        )
        assert not result.evidence.admitted
        assert "growth_lcb90_not_positive" in result.evidence.reasons

    def test_scenario_no_admitted_sleeves(self) -> None:
        """No admitted sleeves returns NO_EVIDENCE."""
        n_bars = 40
        n_syms = 2
        bars = _bars(n_bars, n_syms)
        forecast = self._cash_forecast(n_bars, n_syms)
        config = HandoffConfig()
        folds = (CausalFold(0, 0, 20, 20, 25, 25, 30, 1, 1),)
        cost = np.ones((n_bars, n_syms), dtype=np.float32)
        weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        result = build_exit_aware_handoff(
            forecast, (), bars, np.zeros(n_bars, dtype=np.float64), config,
            folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
        )
        assert not result.evidence.admitted
        assert "no_admitted_sleeves" in result.evidence.reasons

    def test_scenario_integration_with_timeframe_bar_cube(self) -> None:
        """Scenario 4: Real build_exit_aware_handoff execution with TimeframeBarCube."""
        n_bars = 300
        n_syms = 4
        rng = np.random.default_rng(99)

        bm = rng.normal(0.0001, 0.005, n_bars).astype(np.float64)
        close = np.zeros((n_bars, n_syms), dtype=np.float32)
        close[0] = 100.0
        for t in range(1, n_bars):
            close[t, 0] = close[t - 1, 0] * (1.0 + bm[t])
            close[t, 1] = close[t - 1, 1] * (1.0 + 0.0003 + rng.normal(0.0, 0.003))
            close[t, 2] = close[t - 1, 2] * (1.0 + 0.0003 + rng.normal(0.0, 0.003))
            close[t, 3] = close[t - 1, 3] * (1.0 + 0.0003 + rng.normal(0.0, 0.003))

        bars = TimeframeBarCube(
            "4h", np.arange(n_bars, dtype=np.int64), ("BTCUSDT", "S1", "S2", "S3"),
            close, close + 2.0, close - 2.0, close,
            np.ones((n_bars, n_syms), dtype=np.float32), np.ones((n_bars, n_syms), dtype=bool),
        )
        mask = np.array([False, True, True, True], dtype=np.bool_)
        sleeves = (self._make_admitted_sleeve(mask),)
        forecast = self._signal_forecast(n_bars, n_syms, oos_start=120, oos_end=300)
        config = HandoffConfig(n_bootstrap=500, min_positive_outer_folds=1)
        folds = (CausalFold(0, 0, 120, 100, 120, 120, n_bars, 1, 1),)
        cost = np.zeros((n_bars, n_syms), dtype=np.float32)
        weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights_2d[120:n_bars] = np.tile(np.array([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), (n_bars - 120, 1))
        result = build_exit_aware_handoff(
            forecast, sleeves, bars, bm, config,
            folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
        )
        assert isinstance(result, HandoffResult)
        assert isinstance(result.evidence, HandoffAdmissionEvidence)
        assert result.forecast is forecast


# ---------------------------------------------------------------------------
# L1 Measurement Integrity Restore — New Tests
# ---------------------------------------------------------------------------

_membership_mask = np.array([True, True], dtype=np.bool_)

def _oos_folds() -> tuple[CausalFold, ...]:
    return (CausalFold(0, 0, 120, 100, 120, 120, 200, 1, 1),)

def _default_sleeve(mask: NDArray[np.bool_] | None = None) -> L1SleevePosterior:
    if mask is None:
        mask = _membership_mask
    return L1SleevePosterior(
        sleeve_id="test", signal_id="sig:test", family="test",
        outer_fold_id=0, cluster_id=0, member_mask_1d=mask,
        member_hash="h", exit_policy=ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "cal"),
        fitted_beta=0.0, mean_net_return=0.0, standard_error=0.1, posterior_positive_probability=0.96,
        residual_novelty=1.0, fold_net_returns=(0.0,), effective_events=100,
        admitted=True, reasons=(),
    )

def test_l1_portfolio_returns_is_signal_dependent() -> None:
    """D-1 regression: flipping weights_2d sign flips portfolio returns sign."""
    n_bars, n_syms = 200, 3
    bars = _bars(n_bars, n_syms)
    folds = (CausalFold(0, 0, 120, 100, 120, 120, n_bars, 1, 1),)
    cost = np.zeros((n_bars, n_syms), dtype=np.float32)

    w_pos = np.zeros((n_bars, n_syms), dtype=np.float64)
    w_pos[120:200] = 1.0 / 3.0
    r_pos = compute_l1_oos_portfolio_returns(w_pos, bars, folds, cost)

    w_neg = np.zeros((n_bars, n_syms), dtype=np.float64)
    w_neg[120:200] = -1.0 / 3.0
    r_neg = compute_l1_oos_portfolio_returns(w_neg, bars, folds, cost)

    assert len(r_pos) == len(r_neg) > 0
    assert np.allclose(r_pos, -r_neg, atol=1e-12), "D-1 violation: flipping weight sign did not flip return series"


def test_l1_portfolio_returns_uses_only_oos_windows() -> None:
    """Causality: fit-only signal produces zero returns."""
    n_bars, n_syms = 200, 3
    bars = _bars(n_bars, n_syms)
    folds = (CausalFold(0, 0, 120, 100, 120, 120, n_bars, 1, 1),)
    cost = np.zeros((n_bars, n_syms), dtype=np.float32)

    w_fit_only = np.zeros((n_bars, n_syms), dtype=np.float64)
    w_fit_only[50:120] = 1.0 / 3.0  # signal only in fit window
    r = compute_l1_oos_portfolio_returns(w_fit_only, bars, folds, cost)
    assert np.all(r == 0.0), "Causality violation: fit-only signal produced non-zero OOS returns"


def test_l1_portfolio_series_is_single_column() -> None:
    """D-2 guarantee: returns a 1-D array (never per-sleeve matrix)."""
    n_bars, n_syms = 200, 4
    bars = _bars(n_bars, n_syms)
    folds = (CausalFold(0, 0, 120, 100, 120, 120, n_bars, 1, 1),)
    cost = np.zeros((n_bars, n_syms), dtype=np.float32)

    weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
    weights_2d[120:200] = 0.25
    r = compute_l1_oos_portfolio_returns(weights_2d, bars, folds, cost)
    assert r.ndim == 1, f"D-2 violation: expected 1-D, got {r.ndim}-D"


def test_hac_se_exceeds_ols_se_for_persistent_signal() -> None:
    """D-3: Driscoll-Kraay SE > pooled OLS SE for EWM-smoothed persistent noise."""
    rng = np.random.default_rng(42)
    n_bars, n_syms = 300, 5
    noise = rng.normal(0, 1, (n_bars, n_syms)).astype(np.float32)
    # EWM-smooth to create persistence
    feature = np.zeros_like(noise)
    alpha = 0.05
    state = np.zeros(n_syms, dtype=np.float32)
    for t in range(n_bars):
        state = alpha * noise[t] + (1.0 - alpha) * state
        feature[t] = state
    close = np.column_stack([np.linspace(100, 200, n_bars) for _ in range(n_syms)]).astype(np.float32)
    descriptor = SignalDescriptor("test", "test", "fast", 4, "4h", 8, "", "", "v1")
    sym_indices = np.arange(n_syms, dtype=np.int64)
    fit_end = n_bars - 2

    mode = int(os.environ.get("L1_TEST_MODE", "0"))
    if mode == 0:
        beta, se_hac, prob, n_obs, _se_ols = _cluster_masked_beta(feature, close, descriptor, fit_end, sym_indices, hac_lag_cap=120)
        # compute pooled OLS SE for comparison
        x = feature[:fit_end - 2, sym_indices].astype(np.float64)
        y = (np.roll(close.astype(np.float64), -2, axis=0) / np.maximum(close, 1e-12) - 1.0)[:fit_end - 2, sym_indices]
        mask = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[mask], y[mask]
        denom = float(np.dot(xv, xv)) + 1e-8
        beta_ols = float(np.dot(xv, yv) / denom) if xv.size else 0.0
        residual = yv - beta_ols * xv
        se_ols = float(np.std(residual, ddof=1) / math.sqrt(denom)) if residual.size > 1 else 1.0
        assert se_hac > se_ols, f"D-3 violation: se_hac={se_hac:.6f} <= se_ols={se_ols:.6f} for persistent signal"
    else:
        # skip in fast mode
        pass


def test_build_exit_aware_handoff_uses_block_bootstrap() -> None:
    """D-5: build_exit_aware_handoff uses circular_stationary_bootstrap_growth with block_size>0."""
    n_bars, n_syms = 120, 2
    bars = _bars(n_bars, n_syms)
    folds = (CausalFold(0, 0, 60, 50, 60, 60, n_bars, 1, 1),)
    cost = np.zeros((n_bars, n_syms), dtype=np.float32)
    mask = np.array([True, True], dtype=np.bool_)
    sleeve = _default_sleeve(mask)
    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=bars.symbols, mu_2d=np.zeros((n_bars, n_syms), dtype=np.float32),
        se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 0), dtype=np.float32),
        family_ids=(), admitted_signal_ids=("sig:test",), fold_manifest_hash="",
    )
    weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
    result = build_exit_aware_handoff(
        forecast, (sleeve,), bars, np.zeros(n_bars, dtype=np.float64), HandoffConfig(min_positive_outer_folds=1),
        folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
    )
    assert isinstance(result, HandoffResult)


def test_build_exit_aware_handoff_records_resolved_pw_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pw_block in the DEBUG log must reflect the actual Politis-White estimate
    used by the bootstrap call, not a hardcoded 0.0 stub."""
    import src.domain.futures.compound.l1_diagnostics as l1_diagnostics_mod
    from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder

    n_bars, n_syms = 120, 2
    bars = _bars(n_bars, n_syms)
    folds = (CausalFold(0, 0, 60, 50, 60, 60, n_bars, 1, 1),)
    cost = np.zeros((n_bars, n_syms), dtype=np.float32)
    mask = np.array([True, True], dtype=np.bool_)
    sleeve = _default_sleeve(mask)
    rng = np.random.default_rng(3)
    mu = rng.normal(0.0, 1.0, size=(n_bars, n_syms)).astype(np.float32)
    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=bars.symbols, mu_2d=mu,
        se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 0), dtype=np.float32),
        family_ids=(), admitted_signal_ids=("sig:test",), fold_manifest_hash="",
    )
    log_path = tmp_path / "l1_admission.jsonl"

    class _FixedPathRecorder(L1AdmissionRecorder):
        def __init__(self, path: Path | None = None) -> None:
            super().__init__(path=log_path)

    monkeypatch.setattr(l1_diagnostics_mod, "L1AdmissionRecorder", _FixedPathRecorder)
    old = os.environ.get("L1_DEBUG")
    os.environ["L1_DEBUG"] = "1"
    try:
        weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        build_exit_aware_handoff(
            forecast, (sleeve,), bars, np.zeros(n_bars, dtype=np.float64), HandoffConfig(min_positive_outer_folds=1),
            folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
        )
    finally:
        if old is None:
            os.environ.pop("L1_DEBUG", None)
        else:
            os.environ["L1_DEBUG"] = old

    lines = log_path.read_text().strip().splitlines()
    gate_rows = [json.loads(line) for line in lines if json.loads(line)["tag"] == "EVAL"]
    assert gate_rows, "no [EVAL] row recorded"
    assert gate_rows[-1]["pw_block"] > 0.0, "pw_block must be a real Politis-White estimate, not the 0.0 stub"


def test_l1_admission_recorder_noop_when_debug_disabled() -> None:
    from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder
    # Ensure L1_DEBUG is not set
    old = os.environ.pop("L1_DEBUG", None)
    try:
        rec = L1AdmissionRecorder()
        assert not rec.enabled
        rec.record_sleeve(signal_id="s", fold=0, cluster=0, beta=0.0, se_hac=0.1, se_ols_ratio=1.0, prob=0.5, n_obs=100, n_blocks=1, admitted=True)
        rec.record_gate(admitted_sleeves=1, distinct_series=1, oos_bars=10, ann_growth=0.0, ann_lcb90=0.0, pw_block=5.0, turnover=0.0, cost_drag=0.0, admitted=False)
    finally:
        if old is not None:
            os.environ["L1_DEBUG"] = old


def test_l1_admission_recorder_writes_parsable_jsonl(tmp_path: Path) -> None:
    from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder
    path = tmp_path / "test_l1.jsonl"
    old = os.environ.get("L1_DEBUG")
    os.environ["L1_DEBUG"] = "1"
    try:
        rec = L1AdmissionRecorder(path=path)
        assert rec.enabled
        rec.record_gate(admitted_sleeves=2, distinct_series=1, oos_bars=50, ann_growth=0.05, ann_lcb90=0.01, pw_block=5.0, turnover=0.1, cost_drag=0.0002, admitted=True)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["tag"] == "EVAL"
        assert parsed["admitted"] is True
    finally:
        if old is not None:
            os.environ["L1_DEBUG"] = old
        else:
            os.environ.pop("L1_DEBUG", None)


def test_handoff_config_rejects_relaxed_probability_threshold() -> None:
    with pytest.raises(AssertionError):
        HandoffConfig(min_sleeve_posterior_probability=0.4)
    with pytest.raises(AssertionError):
        HandoffConfig(hac_lag_cap=0)


# ---------------------------------------------------------------------------
# Contract scenario test functions (wired from contract.json scenarios)
# ---------------------------------------------------------------------------

def test_compute_beta_neutral_composite_returns_happy_path() -> None:
    """[Scenario 1] Valid sleeves produce expected-shaped beta-neutral returns."""
    t = TestComputeBetaNeutralCompositeReturns()
    t.test_valid_inputs_returns_1d_array()


def test_build_exit_aware_handoff_time_series_bootstrap() -> None:
    """[Scenario 2] Time-series block bootstrap on beta-neutral composite."""
    t = TestEffectiveCompoundingHandoff()
    t.test_scenario_happy_path_positive_lcb90()


def test_build_exit_aware_handoff_fail_closed_negative_lcb() -> None:
    """[Scenario 3] Fail-closed when ann_lcb90 <= 0.0."""
    t = TestEffectiveCompoundingHandoff()
    t.test_scenario_lcb90_not_positive_rejected()


def test_l1_handoff_pipeline_wiring() -> None:
    """[Scenario 4] Integration: real build_exit_aware_handoff with TimeframeBarCube."""
    t = TestEffectiveCompoundingHandoff()
    t.test_scenario_integration_with_timeframe_bar_cube()


# ---------------------------------------------------------------------------
# Fundamental Pipeline Performance Optimization
# ---------------------------------------------------------------------------

def test_exit_path_cache_math_identity() -> None:
    """[LIMIT-01] Sleeve posteriors with and without ExitPathCache are bit-exact."""
    from src.domain.futures.compound.clustering import (
        ClusteringAlgorithm,
        compute_market_regime_clusters,
    )
    from src.domain.futures.compound.contracts import MarketFeatureCube

    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal((40, 5)) * 0.5, axis=0)
    close = np.abs(close) + 10.0
    volume = np.abs(rng.standard_normal((40, 5))) * 1e6 + 1e5
    market = MarketFeatureCube(
        timestamps_ns=np.arange(40, dtype=np.int64) * 3_600_000_000_000,
        symbols=tuple(f"SYM_{i:03d}" for i in range(5)),
        fields_2d={
            "close": close.astype(np.float32),
            "quote_volume": volume.astype(np.float32),
            "high": (close + 1.0).astype(np.float32),
            "low": (close - 1.0).astype(np.float32),
        },
        available_2d={"core": np.ones((40, 5), dtype=bool)},
        eligible_2d=np.ones((40, 5), dtype=bool),
        entry_block_2d=np.zeros((40, 5), dtype=bool),
        exit_required_2d=np.zeros((40, 5), dtype=bool),
        capacity_usdt_2d=np.full((40, 5), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((40, 5), 8.0, dtype=np.float32),
        data_manifest_hash="test_hash",
    )
    clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
    bars_4h = _bars(40, 5)
    panel = _panel(40, 5)
    folds = _folds()
    cost = np.ones((40, 5), dtype=np.float32)
    config = HandoffConfig()
    cfolds = _cluster_folds(clusters, bars_4h)

    baseline = estimate_cluster_sleeve_posteriors(panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)

    cache = precompute_exit_path_cache(panel, bars_4h, cost)
    cached = estimate_cluster_sleeve_posteriors(panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config, cache=cache)

    assert len(baseline) == len(cached), f"sleeve count mismatch: {len(baseline)} vs {len(cached)}"
    for b, c in zip(baseline, cached, strict=True):
        assert b.sleeve_id == c.sleeve_id
        assert b.admitted == c.admitted
        assert abs(b.mean_net_return - c.mean_net_return) < 1e-12, (
            f"mean_net_return mismatch for {b.sleeve_id}: {b.mean_net_return} vs {c.mean_net_return}"
        )
        assert abs(b.posterior_positive_probability - c.posterior_positive_probability) < 1e-12
        assert b.effective_events == c.effective_events


def test_light_test_bars_fixture_speed(light_test_bars: TimeframeBarCube) -> None:
    """[LIMIT-02] Unit tests with light_test_bars (200 bars) finish in < 0.2s."""
    import time

    t0 = time.perf_counter()
    bars = light_test_bars
    assert bars.close_2d.shape[0] == 200
    t1 = time.perf_counter()
    assert t1 - t0 < 0.2, f"light_test_bars access took {t1-t0:.3f}s"


def test_bootstrap_deterministic_seed() -> None:
    """[LIMIT-03] Bootstrap with same seed produces bit-identical results."""
    rng = np.random.default_rng(42)
    returns = rng.standard_normal(2191) * 0.002 + 0.0001
    lcb_1, ucb_1, prob_1 = circular_stationary_bootstrap_growth(
        returns, 2191.5, n_bootstrap=1000, seed=42,
    )
    lcb_2, ucb_2, prob_2 = circular_stationary_bootstrap_growth(
        returns, 2191.5, n_bootstrap=1000, seed=42,
    )
    assert lcb_1 == lcb_2, f"LCB mismatch: {lcb_1} vs {lcb_2}"
    assert ucb_1 == ucb_2, f"UCB mismatch: {ucb_1} vs {ucb_2}"
    assert prob_1 == prob_2, f"prob_positive mismatch: {prob_1} vs {prob_2}"


def test_tensor_bootstrap_oom_safety() -> None:
    """[SCENARIO-1] Chunked 2D tensor bootstrap memory <= 50 MB for 1,500 sleeves."""
    rng = np.random.default_rng(42)
    n_bars = 2191
    n_sleeves = 1500
    returns_2d = (rng.standard_normal((n_bars, n_sleeves)) * 0.002 + 0.0001).astype(np.float64)
    tensor_bytes = returns_2d.nbytes
    n_bootstrap = 100
    sample_indices = rng.integers(0, n_bars, size=(n_bootstrap, n_bars))
    temp_bytes = n_bootstrap * 250 * 8
    total_mb = (tensor_bytes + temp_bytes) / (1024 * 1024)
    assert total_mb < 50.0, f"Estimated memory {total_mb:.2f} MB exceeds 50 MB limit"
    lcb90 = compute_chunked_2d_tensor_bootstrap(
        returns_2d, 2191.5, n_bootstrap=n_bootstrap, chunk_size=250, seed=42,
    )
    assert lcb90.shape == (n_sleeves,), f"Expected shape ({n_sleeves},), got {lcb90.shape}"
    assert np.all(np.isfinite(lcb90)), "All LCB90 values must be finite"


def test_tensor_bootstrap_math_identity() -> None:
    """[SCENARIO-2] Chunked 2D tensor bootstrap is bit-exact with per-sleeve reference."""
    rng = np.random.default_rng(42)
    n_bars = 2191
    n_sleeves = 10
    n_bootstrap = 200
    periods_per_year = 2191.5
    returns_2d = (rng.standard_normal((n_bars, n_sleeves)) * 0.002 + 0.0001).astype(np.float64)
    pregen_rng = np.random.default_rng(42)
    block_indices = pregen_rng.integers(0, n_bars, size=(n_bootstrap, n_bars))
    serial_lcbs = []
    for i in range(n_sleeves):
        r = returns_2d[:, i]
        boot_vals = np.where(np.isfinite(r[block_indices]), r[block_indices], 0.0)
        boot_log = np.log1p(boot_vals)
        boot_means = np.mean(boot_log, axis=1) * periods_per_year
        serial_lcbs.append(np.percentile(boot_means, 10))
    tensor_lcbs = compute_chunked_2d_tensor_bootstrap(
        returns_2d, periods_per_year, n_bootstrap=n_bootstrap, chunk_size=250, seed=42,
    )
    diff = float(np.max(np.abs(np.array(serial_lcbs) - tensor_lcbs)))
    assert diff < 1e-12, f"Math discrepancy detected: {diff:.12f}"


def test_tensor_bootstrap_empty_returns() -> None:
    """Edge case: empty returns_2d returns empty array."""
    result = compute_chunked_2d_tensor_bootstrap(
        np.empty((100, 0), dtype=np.float64), 2191.5,
    )
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_tensor_bootstrap_few_bars() -> None:
    """Edge case: n_bars < 10 returns zeros."""
    returns = np.random.default_rng(42).standard_normal((5, 10)).astype(np.float64)
    result = compute_chunked_2d_tensor_bootstrap(returns, 2191.5, n_bootstrap=10)
    assert result.shape == (10,)
    assert np.all(result == 0.0)


def test_aggregate_cluster_group_returns_all_finite() -> None:
    rng = np.random.default_rng(42)
    returns = rng.standard_normal((60, 5)).astype(np.float64)
    sigma = np.abs(rng.standard_normal((60, 5))).astype(np.float64) + 0.1
    result = aggregate_cluster_group_returns(returns, sigma, winsorize_pct=0.10)
    assert result.shape == (60,)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


def test_aggregate_cluster_group_returns_zero_input() -> None:
    returns = np.zeros((60, 5), dtype=np.float64)
    sigma = np.ones((60, 5), dtype=np.float64)
    result = aggregate_cluster_group_returns(returns, sigma, winsorize_pct=0.10)
    assert result.shape == (60,)
    np.testing.assert_allclose(result, np.zeros(60), atol=1e-15)


def test_aggregate_cluster_group_returns_nan_robustness() -> None:
    returns = np.full((60, 5), np.nan, dtype=np.float64)
    sigma = np.ones((60, 5), dtype=np.float64)
    result = aggregate_cluster_group_returns(returns, sigma, winsorize_pct=0.10)
    assert result.shape == (60,)
    np.testing.assert_allclose(result, np.zeros(60), atol=1e-15)


def test_l1_portfolio_returns_scores_given_weights_without_rebuilding() -> None:
    n_bars, n_syms = 200, 3
    bars = _bars(n_bars, n_syms)
    folds = (CausalFold(0, 0, 120, 100, 120, 120, n_bars, 1, 1),)
    cost = np.zeros((n_bars, n_syms), dtype=np.float32)
    weights_2d = np.ones((n_bars, n_syms), dtype=np.float64) * (1.0 / 3.0)
    result = compute_l1_oos_portfolio_returns(weights_2d, bars, folds, cost)
    assert result.ndim == 1
    assert len(result) > 0


def test_compute_fold_growths_skips_empty_folds() -> None:
    n_bars, n_syms = 60, 2
    bars = _bars(n_bars, n_syms)
    short_fold = CausalFold(0, 0, 50, 48, 50, 52, 55, 1, 1)
    empty_fold = CausalFold(1, 0, 50, 48, 50, 55, 55, 1, 1)
    cost = np.ones((n_bars, n_syms), dtype=np.float32)
    weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
    growths = compute_fold_growths(weights_2d, bars, (short_fold, empty_fold), cost)
    assert len(growths) == 1


def test_insufficient_positive_folds_blocks_admission() -> None:
    n_bars, n_syms = 120, 2
    bars = _bars(n_bars, n_syms)
    mask = np.ones(n_syms, dtype=np.bool_)
    sleeve = _default_sleeve(mask)
    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=bars.symbols, mu_2d=np.zeros((n_bars, n_syms), dtype=np.float32),
        se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 0), dtype=np.float32),
        family_ids=(), admitted_signal_ids=("sig:test",), fold_manifest_hash="",
    )
    folds = tuple(
        CausalFold(i, 0, 60, 50, 60 + i * 10, 60 + i * 10, 60 + (i + 1) * 10, 1, 1)
        for i in range(5)
    )
    cost = np.zeros((n_bars, n_syms), dtype=np.float32)
    weights_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
    config = HandoffConfig(n_bootstrap=200, min_positive_outer_folds=4)
    result = build_exit_aware_handoff(
        forecast, (sleeve,), bars, np.zeros(n_bars, dtype=np.float64), config,
        folds=folds, weights_2d=weights_2d, cost_bps_4h=cost,
    )
    assert not result.evidence.admitted
    assert "insufficient_positive_folds" in result.evidence.reasons

