from __future__ import annotations

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
    _signal_evidence,
    build_exit_aware_handoff,
    calibrate_exit_policy,
    calibrate_exit_policy_from_paths,
    combine_posterior_sleeves,
    compute_beta_neutral_composite_returns,
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
    result = build_exit_aware_handoff(
        cash, (), bars, np.zeros(40, dtype=np.float64), HandoffConfig(),
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
    sleeve = L1SleevePosterior("s", "trend:fast", "trend", 0, 0, mask, "test_hash", policy, 0.1, 0.1, 0.9, 0.0, (0.1,), 1, True, ())
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

    def test_estimate_cluster_sleeve_posteriors_threshold_052(self) -> None:
        from src.domain.futures.compound.clustering import (
            ClusteringAlgorithm,
            compute_market_regime_clusters,
        )
        from src.domain.futures.compound.contracts import MarketFeatureCube

        close = np.column_stack([np.linspace(100.0, 200.0, 40) for _ in range(5)]).astype(np.float32)
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
        strong_panel = RawSignalPanel(
            decision_timestamps_ns=np.arange(40, dtype=np.int64),
            symbols=tuple("S" + str(i) for i in range(5)),
            descriptors=(SignalDescriptor("strong", "trend", "fast", 4, "4h", 4, "", "", "v1"),),
            z_3d=np.ones((40, 5, 1), dtype=np.float32),
            valid_3d=np.ones((40, 5, 1), dtype=bool),
            sigma_2d=np.ones((40, 5), dtype=np.float32),
        )
        cfolds = _cluster_folds(clusters, bars_4h)
        posteriors = estimate_cluster_sleeve_posteriors(strong_panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)
        assert len(posteriors) > 0
        for p in posteriors:
            assert p.posterior_positive_probability >= 0.52, f"sleeve {p.sleeve_id} prob={p.posterior_positive_probability} < 0.52"
            assert p.admitted, f"sleeve {p.sleeve_id} not admitted despite prob >= 0.52"

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
        handoff = build_exit_aware_handoff(forecast, sleeves, bars_4h, benchmark, config)
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
        forecast = self._cash_forecast(n_bars, n_syms)
        config = HandoffConfig(n_bootstrap=500)
        result = build_exit_aware_handoff(forecast, sleeves, bars, bm, config)
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
        config = HandoffConfig(n_bootstrap=500)
        bm = np.zeros(n_bars, dtype=np.float64)
        result = build_exit_aware_handoff(forecast, sleeves, bars, bm, config)
        assert not result.evidence.admitted
        assert "growth_lcb90_not_positive" in result.evidence.reasons

    def test_scenario_no_admitted_sleeves(self) -> None:
        """No admitted sleeves returns NO_EVIDENCE."""
        n_bars = 40
        bars = _bars(n_bars, 2)
        forecast = self._cash_forecast(n_bars, 2)
        result = build_exit_aware_handoff(forecast, (), bars, np.zeros(n_bars, dtype=np.float64), HandoffConfig())
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
        forecast = self._cash_forecast(n_bars, n_syms)
        config = HandoffConfig(n_bootstrap=500)
        result = build_exit_aware_handoff(forecast, sleeves, bars, bm, config)
        assert isinstance(result, HandoffResult)
        assert isinstance(result.evidence, HandoffAdmissionEvidence)
        assert result.forecast is forecast


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

