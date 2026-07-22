from __future__ import annotations

import numpy as np

from src.domain.futures.compound.alpha_catalog import (
    build_canonical_alpha_catalog,
    compute_raw_alpha_tape,
)
from src.domain.futures.compound.contracts import MarketFeatureCube


class TestBuildCanonicalAlphaCatalog:
    def test_returns_six_families_three_horizons(self) -> None:
        catalog = build_canonical_alpha_catalog()
        assert len(catalog) == 18
        families = {a.family for a in catalog}
        assert families == {
            "time_series_trend", "breakout", "cross_sectional_momentum",
            "short_term_reversal", "carry_basis", "flow_positioning",
        }
        horizons = {a.horizon_bars for a in catalog}
        assert horizons == {4, 12, 24}

    def test_all_have_causal_lag_one(self) -> None:
        catalog = build_canonical_alpha_catalog()
        assert all(a.causal_lag_bars == 1 for a in catalog)

    def test_flow_recipe_is_conditional_tier(self) -> None:
        catalog = build_canonical_alpha_catalog()
        for a in catalog:
            if a.family == "flow_positioning":
                assert a.data_tier == "conditional"
            else:
                assert a.data_tier == "core"


class TestComputeRawAlphaTape:
    def test_returns_correct_shape(self) -> None:
        n_bars, n_syms = 200, 3
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B", "C"),
            fields_2d={
                "open": np.ones((n_bars, n_syms), dtype=np.float64) * 100,
                "high": np.ones((n_bars, n_syms), dtype=np.float64) * 101,
                "low": np.ones((n_bars, n_syms), dtype=np.float64) * 99,
                "close": np.linspace(100, 110, n_bars * n_syms).reshape(n_bars, n_syms).astype(np.float64),
                "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 1_000_000,
                "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
                "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
                "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 500_000,
            },
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        catalog = build_canonical_alpha_catalog()
        raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
        assert raw.scores_3d.shape == (n_bars, n_syms, 18)
        assert raw.valid_3d.shape == (n_bars, n_syms, 18)
        assert len(raw.recipe_ids) == 18
        assert len(raw.horizon_bars_1d) == 18

    def test_when_metrics_missing_disables_only_flow_recipe(self) -> None:
        n_bars, n_syms = 200, 2
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B"),
            fields_2d={
                "open": np.ones((n_bars, n_syms), dtype=np.float64) * 100,
                "high": np.ones((n_bars, n_syms), dtype=np.float64) * 101,
                "low": np.ones((n_bars, n_syms), dtype=np.float64) * 99,
                "close": np.linspace(100, 110, n_bars * n_syms).reshape(n_bars, n_syms).astype(np.float64),
                "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 1_000_000,
                "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
                "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            },
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        catalog = build_canonical_alpha_catalog()
        raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
        flow_recipe_ids = [i for i, rid in enumerate(raw.recipe_ids) if rid.startswith("flow_")]
        non_flow_recipe_ids = [i for i, rid in enumerate(raw.recipe_ids) if not rid.startswith("flow_")]
        for k in flow_recipe_ids:
            assert not raw.valid_3d[:, :, k].any(), f"flow recipe {raw.recipe_ids[k]} should be all invalid"
        for k in non_flow_recipe_ids:
            assert raw.valid_3d[:, :, k].any(), f"non-flow recipe {raw.recipe_ids[k]} should have valid entries"


def test_alpha_tape_uses_memmap_blocks_not_dense_ram_copy() -> None:
    n_bars, n_syms = 256, 3
    cube = MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("A", "B", "C"),
        fields_2d={
            "open": np.ones((n_bars, n_syms), dtype=np.float64) * 100,
            "high": np.ones((n_bars, n_syms), dtype=np.float64) * 101,
            "low": np.ones((n_bars, n_syms), dtype=np.float64) * 99,
            "close": np.linspace(100, 110, n_bars * n_syms).reshape(n_bars, n_syms).astype(np.float64),
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 1_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_), "conditional": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    catalog = build_canonical_alpha_catalog()
    raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
    assert raw.scores_3d.ndim == 3
    assert raw.scores_3d.dtype == np.float32


def test_build_canonical_alpha_catalog_returns_six_families_three_horizons() -> None:
    catalog = build_canonical_alpha_catalog()
    assert len(catalog) == 18
    families = {a.family for a in catalog}
    assert families == {
        "time_series_trend", "breakout", "cross_sectional_momentum",
        "short_term_reversal", "carry_basis", "flow_positioning",
    }
    horizons = {a.horizon_bars for a in catalog}
    assert horizons == {4, 12, 24}


def test_compute_raw_alpha_tape_when_metrics_missing_disables_only_flow_recipe() -> None:
    n_bars, n_syms = 200, 2
    cube = MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("A", "B"),
        fields_2d={
            "open": np.ones((n_bars, n_syms), dtype=np.float64) * 100,
            "high": np.ones((n_bars, n_syms), dtype=np.float64) * 101,
            "low": np.ones((n_bars, n_syms), dtype=np.float64) * 99,
            "close": np.linspace(100, 110, n_bars * n_syms).reshape(n_bars, n_syms).astype(np.float64),
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 1_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    catalog = build_canonical_alpha_catalog()
    raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
    flow_recipe_ids = [i for i, rid in enumerate(raw.recipe_ids) if rid.startswith("flow_")]
    non_flow_recipe_ids = [i for i, rid in enumerate(raw.recipe_ids) if not rid.startswith("flow_")]
    for k in flow_recipe_ids:
        assert not raw.valid_3d[:, :, k].any(), f"flow recipe {raw.recipe_ids[k]} should be all invalid"
    for k in non_flow_recipe_ids:
        assert raw.valid_3d[:, :, k].any(), f"non-flow recipe {raw.recipe_ids[k]} should have valid entries"
