from __future__ import annotations

import numpy as np

from src.domain.futures.compound.clustering import (
    ClusteringAlgorithm,
    compute_market_regime_clusters,
)
from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CausalClusterFold,
    CausalFold,
    ClusterPanel,
    MarketFeatureCube,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_sleeves import estimate_cluster_sleeve_posteriors


def _dummy_market_cube(n_syms: int = 10, n_bars: int = 40) -> MarketFeatureCube:
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


def _dummy_bars_4h(t: int = 40, n: int = 10) -> TimeframeBarCube:
    close = np.column_stack([np.linspace(100.0, 120.0 + i, t) for i in range(n)]).astype(np.float32)
    return TimeframeBarCube(
        "4h", np.arange(t, dtype=np.int64), tuple(f"SYM_{i:03d}" for i in range(n)),
        close, close + 2.0, close - 2.0, close,
        np.ones((t, n), dtype=np.float32), np.ones((t, n), dtype=bool),
    )


def _dummy_panel(t: int = 40, n: int = 10) -> RawSignalPanel:
    z = np.ones((t, n, 1), dtype=np.float32)
    descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
    return RawSignalPanel(
        np.arange(t, dtype=np.int64), tuple(f"SYM_{i:03d}" for i in range(n)),
        (descriptor,), z, np.ones_like(z, dtype=bool), np.ones((t, n), dtype=np.float32),
    )


def _dummy_folds() -> tuple[CausalFold, ...]:
    return tuple(CausalFold(i, 0, 20, 20, 25, 25, 30, 1, 1) for i in range(4))


def _cluster_folds(clusters: ClusterPanel, bars_4h: TimeframeBarCube) -> tuple[CausalClusterFold, ...]:
    from src.domain.futures.compound.contracts import ClusterPanel
    return (CausalClusterFold(
        fold_id=0, fit_end_exclusive_4h=20,
        fit_end_time_ns=int(bars_4h.timestamps_ns[19]),
        panel=clusters, member_hash="test_hash",
    ),)


def test_cluster_aware_engine_pipeline() -> None:
    from src.domain.futures.compound.contracts import ClusterPanel
    market = _dummy_market_cube(10, 40)
    clusters = compute_market_regime_clusters(market, algorithm=ClusteringAlgorithm.ROBUST_KMEANS, k_clusters=2)
    assert clusters.k_clusters >= 1
    bars_4h = _dummy_bars_4h(40, 10)
    panel = _dummy_panel(40, 10)
    folds = _dummy_folds()
    cost = np.ones((40, 10), dtype=np.float32)
    config = HandoffConfig()
    cfolds = _cluster_folds(clusters, bars_4h)
    posteriors = estimate_cluster_sleeve_posteriors(panel, bars_4h, cfolds, folds, cost, np.zeros_like(cost), config)
    assert isinstance(posteriors, tuple)
