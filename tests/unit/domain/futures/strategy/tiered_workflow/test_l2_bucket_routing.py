from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow import l2_meta as l2_meta_module
from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    compute_bucket_realized_edges,
    filter_sleeves_by_bucket,
)
from src.domain.futures.strategy.walk_forward import WFFold


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


def _make_folds() -> tuple[WFFold, ...]:
    return (
        WFFold(fit_start=0, fit_end=3, cal_start=3, cal_end=4, oos_start=4, oos_end=8),
        WFFold(fit_start=0, fit_end=7, cal_start=7, cal_end=8, oos_start=8, oos_end=12),
    )


def _make_routing_sleeve_sigs() -> dict[tuple[str, str], SymbolSignal]:
    sig = SymbolSignal(
        raw_mu=1.0,
        volatility=0.2,
        n_obs=4,
        t_stat=2.0,
        valid=True,
        beta_btc=None,
        quality_weight=1.0,
    )
    return {
        ("BTC", "donchian_72_4h"): sig,
        ("BTC", "trend_pullback_4h"): sig,
    }


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
            cache,
            aligned,
            0,
            n_bars,
            regime,
            cost_bps=1.0,
            min_n=1,
            shrinkage=0.0,
        )
        shrunk_result = compute_bucket_realized_edges(
            cache,
            aligned,
            0,
            n_bars,
            regime,
            cost_bps=1.0,
            min_n=10,
            shrinkage=0.3,
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
            sleeve_sigs,
            bucket_edges,
            regime_now=1,
            edge_floor_bps=100.0,
        )

        assert ("BTC", "donchian_72_4h") in result
        assert ("BTC", "trend_pullback_4h") not in result

    def test_unknown_bucket_treated_as_zero(self) -> None:
        bucket_edges: dict[tuple[int, str, str], float] = {}
        sleeve_sigs = {("BTC", "donchian_72_4h"): MagicMock()}

        result = filter_sleeves_by_bucket(
            sleeve_sigs,
            bucket_edges,
            regime_now=1,
            edge_floor_bps=100.0,
        )

        assert len(result) == 0

    def test_empty_input(self) -> None:
        result = filter_sleeves_by_bucket(
            {},
            {(1, "x", "4h"): 500.0},
            regime_now=1,
            edge_floor_bps=100.0,
        )
        assert result == {}

    def test_filter_sleeves_by_bucket_legacy_mode_keeps_existing_contract(self) -> None:
        bucket_edges = {
            (1, "donchian_72", "4h"): 100.0,
        }
        sleeve_sigs = {
            ("BTC", "donchian_72_4h"): MagicMock(),
            ("BTC", "trend_pullback_4h"): MagicMock(),
        }

        result = filter_sleeves_by_bucket(
            sleeve_sigs,
            bucket_edges,
            regime_now=1,
            edge_floor_bps=100.0,
        )

        assert ("BTC", "donchian_72_4h") not in result
        assert ("BTC", "trend_pullback_4h") not in result


class TestBuildRegimeRoutingPlan:
    """Spec scenarios 1-3 for the new L2 regime routing contract."""

    def test_build_regime_routing_plan_when_compression_enabled_uses_three_states(self) -> None:
        assert hasattr(
            l2_meta_module,
            "build_regime_routing_plan",
        ), "build_regime_routing_plan contract missing from l2_meta"

        cache = _make_cache(n_bars=12, n_sleeve=2)
        cache.signal_mask_2d[:, :] = True
        aligned = _make_aligned(n_bars=12, n_sym=1)
        aligned.close_2d = np.array(
            [
                [100.0],
                [101.0],
                [102.0],
                [103.0],
                [104.0],
                [105.0],
                [106.0],
                [107.0],
                [108.0],
                [109.0],
                [110.0],
                [111.0],
            ],
            dtype=np.float64,
        )
        raw_regime = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.int8)

        plan = l2_meta_module.build_regime_routing_plan(
            cache=cache,
            aligned=aligned,
            awf_folds=_make_folds(),
            raw_regime_code_1d=raw_regime,
            compression_enabled=True,
            proof_enabled=False,
        )

        assert int(plan.effective_regime_code_1d.max()) <= 2
        assert plan.diagnostics.active_state_count == 3
        assert plan.diagnostics.active_state_names == ("bull", "bear", "crisis")

    def test_build_regime_routing_plan_when_proof_fails_replicates_pooled_edges(self) -> None:
        assert hasattr(
            l2_meta_module,
            "build_regime_routing_plan",
        ), "build_regime_routing_plan contract missing from l2_meta"

        cache = _make_cache(n_bars=12, n_sleeve=2)
        cache.signal_mask_2d[:, :] = True
        cache.side_2d[:, :] = 1.0
        aligned = _make_aligned(n_bars=12, n_sym=1)
        aligned.close_2d = np.array(
            [
                [100.0],
                [100.3],
                [100.2],
                [100.4],
                [100.5],
                [100.4],
                [100.6],
                [100.7],
                [100.6],
                [100.8],
                [100.7],
                [100.9],
            ],
            dtype=np.float64,
        )
        raw_regime = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.int8)

        plan = l2_meta_module.build_regime_routing_plan(
            cache=cache,
            aligned=aligned,
            awf_folds=_make_folds(),
            raw_regime_code_1d=raw_regime,
            compression_enabled=True,
            proof_enabled=True,
            fallback_mode="pooled",
            cost_bps=0.0,
            min_n=1,
        )

        assert plan.diagnostics.proof_passed is False
        assert plan.diagnostics.conditioning_path == "pooled_fallback"

        for fold_idx, pooled_edges in enumerate(plan.pooled_edges_by_fold):
            effective_edges = plan.effective_bucket_edges_by_fold[fold_idx]
            for state in range(plan.diagnostics.active_state_count):
                for pooled_key, pooled_edge in pooled_edges.items():
                    fam, tf = pooled_key
                    assert effective_edges[(state, fam, tf)] == pytest.approx(pooled_edge)

        filtered = filter_sleeves_by_bucket(
            _make_routing_sleeve_sigs(),
            plan.effective_bucket_edges_by_fold[0],
            regime_now=0,
            edge_floor_bps=0.0,
        )
        assert filtered, "positive pooled edge should survive pooled fallback"

    def test_build_regime_routing_plan_when_lift_is_consistent_uses_conditioned_edges(self) -> None:
        assert hasattr(
            l2_meta_module,
            "build_regime_routing_plan",
        ), "build_regime_routing_plan contract missing from l2_meta"

        n_bars = 40
        cache = _make_cache(n_bars=n_bars, n_sleeve=1)
        cache.signal_mask_2d[:, :] = True
        cache.side_2d[:, :] = 1.0
        aligned = _make_aligned(n_bars=n_bars, n_sym=1)

        close_values: list[list[float]] = []
        raw_regime_list: list[int] = []
        price = 100.0
        for bar_idx in range(n_bars):
            close_values.append([price])
            if bar_idx < n_bars - 1:
                if bar_idx % 2 == 0:
                    price *= 1.04
                    raw_regime_list.append(0)
                else:
                    price /= 1.04
                    raw_regime_list.append(2)
        raw_regime_list.append(0)
        aligned.close_2d = np.array(close_values, dtype=np.float64)
        raw_regime = np.array(raw_regime_list, dtype=np.int8)
        folds = (
            WFFold(fit_start=0, fit_end=9, cal_start=9, cal_end=10, oos_start=10, oos_end=20),
            WFFold(fit_start=0, fit_end=19, cal_start=19, cal_end=20, oos_start=20, oos_end=30),
            WFFold(fit_start=0, fit_end=29, cal_start=29, cal_end=30, oos_start=30, oos_end=40),
        )

        plan = l2_meta_module.build_regime_routing_plan(
            cache=cache,
            aligned=aligned,
            awf_folds=folds,
            raw_regime_code_1d=raw_regime,
            compression_enabled=True,
            proof_enabled=True,
            cost_bps=0.0,
            min_n=1,
            proof_nw_tstat_threshold=1.0,
            proof_fold_pass_ratio_threshold=0.50,
        )

        assert plan.diagnostics.proof_passed is True
        assert plan.diagnostics.conditioning_path == "regime_conditioned"
        assert plan.diagnostics.mean_lift_bps > 0.0
        assert max(state for fold_map in plan.effective_bucket_edges_by_fold for state, _, _ in fold_map) <= 2
        assert max(state for fold_map in plan.raw_bucket_edges_by_fold for state, _, _ in fold_map) >= 2
