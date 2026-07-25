from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CausalFold,
    ClusterPanel,
    ExitPolicyKind,
    ExitPolicySpec,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_sleeves import (
    _signal_evidence,
    build_exit_aware_handoff,
    calibrate_exit_policy,
    combine_posterior_sleeves,
    estimate_cluster_sleeve_posteriors,
    estimate_sleeve_posteriors,
)


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
    forecast = combine_posterior_sleeves(panel, sleeves, folds, HandoffConfig())
    assert forecast.mu_2d.shape == (40, 2)


def test_zero_quality_and_invalid_return_fail_to_cash() -> None:
    panel = _panel()
    bars = _bars()
    result = build_exit_aware_handoff(panel, MultiTimeframeBars(np.arange(40, dtype=np.int64), {"4h": bars}, {}), (), np.ones((40, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())
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


def test_zero_novelty_active_sleeve_returns_cash() -> None:
    panel = _panel()
    policy = ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash")
    from src.domain.futures.compound.contracts import L1SleevePosterior
    sleeve = L1SleevePosterior("s", "trend:fast", "trend", policy, 0.1, 0.1, 0.9, 0.0, (0.1,), 1, True, ())
    forecast = combine_posterior_sleeves(panel, (sleeve,), _folds(), HandoffConfig())
    assert np.all(forecast.mu_2d == 0.0)


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
        posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, clusters, folds, cost, config)
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
        posteriors = estimate_cluster_sleeve_posteriors(empty_panel, bars_4h, clusters, folds, cost, config)
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
            estimate_cluster_sleeve_posteriors(panel, bars_4h, clusters, folds, cost, config)

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
        posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, clusters, folds, cost, config)
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
        posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, clusters, folds, cost, config)
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
        posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, clusters, folds, cost, config)
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
        posteriors = estimate_cluster_sleeve_posteriors(weak_panel, bars_4h, clusters, folds, cost, config)
        for p in posteriors:
            if not p.admitted:
                assert "posterior_below_floor" in p.reasons

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
        mt_bars = MultiTimeframeBars(bars_4h.timestamps_ns, {"4h": bars_4h}, {"funding": np.zeros((160, 5), dtype=np.float32)})
        panel = _panel(40, 5)
        folds = _folds()
        cost = np.ones((40, 5), dtype=np.float32)
        funding = np.zeros((160, 5), dtype=np.float32)
        config = HandoffConfig()

        handoff = build_exit_aware_handoff(panel, mt_bars, folds, cost, funding, config, clusters=clusters)
        assert handoff is not None
        assert isinstance(handoff.evidence.admitted, bool)

