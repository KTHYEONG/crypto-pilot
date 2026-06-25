from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    compute_bucket_realized_edges,
    filter_sleeves_by_bucket,
)


def _make_cache(n_bars: int = 10, n_sleeve: int = 2) -> MagicMock:
    cache = MagicMock()
    cache.signal_mask_2d = np.zeros((n_bars, n_sleeve), dtype=bool)
    cache.side_2d = np.ones((n_bars, n_sleeve), dtype=np.float64)
    cache.holding_bars_2d = np.ones((n_bars, n_sleeve), dtype=np.float64)
    cache.sleeve_to_sym = np.array([0, 0], dtype=np.int64)
    cache.sleeve_ids = (("BTC", "donchian_72_4h"), ("BTC", "trend_pullback_4h"))
    cache.sleeve_to_tf = ("4h", "4h")
    return cache


def _make_aligned(n_bars: int = 10, n_sym: int = 1) -> MagicMock:
    aligned = MagicMock()
    aligned.close_2d = np.ones((n_bars, n_sym), dtype=np.float64)
    return aligned


class TestComputeBucketRealizedEdges:
    """S1-S4: compute_bucket_realized_edges scenarios."""

    def test_positive_when_price_rises(self) -> None:
        n_bars, n_sleeve = 6, 1
        cache = _make_cache(n_bars, n_sleeve)
        cache.signal_mask_2d[1:5, 0] = True
        cache.side_2d[:] = 1.0
        aligned = _make_aligned(n_bars, 1)
        aligned.close_2d = np.array([[1.0], [1.5], [2.0], [2.0], [2.0], [2.0]])
        regime = np.array([1] * n_bars, dtype=np.int8)

        result = compute_bucket_realized_edges(cache, aligned, 0, n_bars, regime, cost_bps=1.0)

        assert len(result) > 0
        key = next(iter(result))
        assert result[key] > 0.0

    def test_negative_when_price_falls(self) -> None:
        n_bars, n_sleeve = 6, 1
        cache = _make_cache(n_bars, n_sleeve)
        cache.signal_mask_2d[1:5, 0] = True
        cache.side_2d[:] = 1.0
        aligned = _make_aligned(n_bars, 1)
        aligned.close_2d = np.array([[2.0], [1.5], [1.0], [1.0], [1.0], [1.0]])
        regime = np.array([1] * n_bars, dtype=np.int8)

        result = compute_bucket_realized_edges(cache, aligned, 0, n_bars, regime, cost_bps=1.0, min_n=1)

        assert len(result) > 0
        key = next(iter(result))
        assert result[key] < 0.0

    def test_shrinkage_toward_family_prior(self) -> None:
        n_bars, n_sleeve = 6, 1
        cache = _make_cache(n_bars, n_sleeve)
        cache.signal_mask_2d[1:4, 0] = True
        cache.side_2d[:] = 1.0
        aligned = _make_aligned(n_bars, 1)
        aligned.close_2d = np.array([[1.0], [2.0], [3.0], [4.0], [4.0], [4.0]])
        regime = np.array([0] * n_bars, dtype=np.int8)

        raw_result = compute_bucket_realized_edges(
            cache, aligned, 0, n_bars, regime, cost_bps=1.0, min_n=1, shrinkage=0.0,
        )
        shrunk_result = compute_bucket_realized_edges(
            cache, aligned, 0, n_bars, regime, cost_bps=1.0, min_n=10, shrinkage=0.3,
        )

        key = next(iter(raw_result))
        assert key in shrunk_result
        assert shrunk_result[key] > 0.0

    def test_empty_when_no_active_sleeves(self) -> None:
        n_bars, n_sleeve = 6, 1
        cache = _make_cache(n_bars, n_sleeve)
        aligned = _make_aligned(n_bars, 1)
        regime = np.array([0] * n_bars, dtype=np.int8)

        result = compute_bucket_realized_edges(cache, aligned, 0, n_bars, regime)
        assert result == {}


class TestFilterSleevesByBucket:
    """S5-S7: filter_sleeves_by_bucket scenarios."""

    def test_passes_above_floor(self) -> None:
        bucket_edges = {
            (1, "donchian_72", "4h"): 500.0,
            (1, "trend_pullback", "4h"): 50.0,
        }
        sig_a = MagicMock()
        sig_b = MagicMock()
        sleeve_sigs = {
            ("BTC", "donchian_72_4h"): sig_a,
            ("BTC", "trend_pullback_4h"): sig_b,
        }

        result = filter_sleeves_by_bucket(
            sleeve_sigs, bucket_edges, regime_now=1, edge_floor_bps=100.0,
        )

        assert ("BTC", "donchian_72_4h") in result
        assert ("BTC", "trend_pullback_4h") not in result

    def test_unknown_bucket_treated_as_zero(self) -> None:
        bucket_edges: dict[tuple[int, str, str], float] = {}
        sleeve_sigs = {("BTC", "donchian_72_4h"): MagicMock()}

        result = filter_sleeves_by_bucket(
            sleeve_sigs, bucket_edges, regime_now=1, edge_floor_bps=100.0,
        )

        assert len(result) == 0

    def test_empty_input(self) -> None:
        result = filter_sleeves_by_bucket(
            {}, {(1, "x", "4h"): 500.0}, regime_now=1, edge_floor_bps=100.0,
        )
        assert result == {}
