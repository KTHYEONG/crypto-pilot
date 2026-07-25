from __future__ import annotations

import numpy as np

from src.domain.futures.compound.clustering import (
    ClusteringAlgorithm,
    ClusterPanel,
    _extract_clustering_features,
    _winsorize,
    compute_market_regime_clusters,
)
from src.domain.futures.compound.contracts import MarketFeatureCube


def _dummy_market_cube(n_syms: int = 120, n_bars: int = 200) -> MarketFeatureCube:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal((n_bars, n_syms)) * 0.5, axis=0)
    close = np.abs(close) + 10.0
    volume = np.abs(rng.standard_normal((n_bars, n_syms))) * 1e6 + 1e5
    symbols_list = [f"SYM_{i:03d}" for i in range(n_syms)]
    symbols_list[0] = "BTCUSDT"
    symbols = tuple(symbols_list)
    return MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=symbols,
        fields_2d={
            "close": close.astype(np.float32),
            "quote_volume": volume.astype(np.float32),
            "high": (close + 1.0).astype(np.float32),
            "low": (close - 1.0).astype(np.float32),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=bool)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=bool),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=bool),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=bool),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 8.0, dtype=np.float32),
        data_manifest_hash="test_hash",
    )


def test_robust_kmeans_and_hierarchical_clustering() -> None:
    market = _dummy_market_cube(120)
    kmeans_panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=4)
    ward_panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.HIERARCHICAL_WARD, k_clusters=4)
    assert isinstance(kmeans_panel, ClusterPanel)
    assert isinstance(ward_panel, ClusterPanel)
    assert kmeans_panel.k_clusters == 4
    assert ward_panel.k_clusters == 4
    assert kmeans_panel.cluster_labels.shape == (120,)
    assert ward_panel.cluster_labels.shape == (120,)


class TestClustering:

    def test_robust_kmeans_produces_valid_cluster_panel(self) -> None:
        market = _dummy_market_cube(120)
        panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=4)
        assert isinstance(panel, ClusterPanel)
        assert panel.k_clusters == 4
        assert panel.cluster_labels.shape == (120,)
        assert panel.cluster_centroids.shape == (4, 4)
        assert len(panel.symbols) == 120
        assert len(np.unique(panel.cluster_labels)) == 4

    def test_hierarchical_ward_produces_valid_cluster_panel(self) -> None:
        market = _dummy_market_cube(120)
        panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.HIERARCHICAL_WARD, k_clusters=4)
        assert isinstance(panel, ClusterPanel)
        assert panel.k_clusters == 4
        assert panel.cluster_labels.shape == (120,)
        assert panel.cluster_centroids.shape == (4, 4)

    def test_both_algorithms_produce_same_number_of_clusters(self) -> None:
        market = _dummy_market_cube(120)
        kmeans_panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=4)
        ward_panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.HIERARCHICAL_WARD, k_clusters=4)
        assert kmeans_panel.k_clusters == ward_panel.k_clusters

    def test_all_symbols_get_a_label(self) -> None:
        market = _dummy_market_cube(120)
        panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=4)
        assert np.all(panel.cluster_labels >= 0)
        assert panel.cluster_labels.dtype == np.int32

    def test_min_cluster_size_enforced(self) -> None:
        market = _dummy_market_cube(12)
        panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.HIERARCHICAL_WARD, k_clusters=4, min_cluster_size=5)
        unique, counts = np.unique(panel.cluster_labels, return_counts=True)
        assert all(c >= 5 for c in counts), f"cluster sizes {dict(zip(unique, counts, strict=False))} include < 5"
        assert len(unique) <= 4

    def test_winsorization_clips_extremes(self) -> None:
        features = np.array([[1.0, 1000.0, 3.0, 0.5],
                             [1.1, 2.0, 3.1, 0.6],
                             [0.9, 1.8, 2.9, 0.4],
                             [1.2, 2.2, 3.2, 0.55]], dtype=np.float64)
        winsorized = _winsorize(features, pct=0.25)
        assert winsorized[0, 1] < 1000.0

    def test_extract_features_returns_correct_shape(self) -> None:
        market = _dummy_market_cube(120, 200)
        features = _extract_clustering_features(market)
        assert features.shape == (120, 4)
        assert np.all(np.isfinite(features))

    def test_dbscan_labels_noise_to_minus_one(self) -> None:
        market = _dummy_market_cube(20)
        panel = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.DBSCAN, k_clusters=4)
        assert isinstance(panel, ClusterPanel)
        assert panel.cluster_labels.shape == (20,)
