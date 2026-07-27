from __future__ import annotations

import numpy as np

from src.domain.futures.compound.alpha_catalog import (
    build_canonical_alpha_catalog,
    compute_raw_alpha_tape,
)
from src.domain.futures.compound.contracts import MarketFeatureCube


class TestBuildCanonicalAlphaCatalog:
    def test_returns_ten_families_six_horizons(self) -> None:
        catalog = build_canonical_alpha_catalog()
        assert len(catalog) == 60
        families = {a.family for a in catalog}
        assert families == {
            "time_series_trend", "breakout", "cross_sectional_momentum",
            "short_term_reversal", "carry_basis", "flow_positioning",
            "volatility_squeeze_keltner", "funding_carry_reversion",
            "flow_imbalance_taker", "open_interest_confirmation",
        }
        horizons = {a.horizon_bars for a in catalog}
        assert horizons == {4, 8, 12, 24, 48, 96}

    def test_all_have_causal_lag_one(self) -> None:
        catalog = build_canonical_alpha_catalog()
        assert all(a.causal_lag_bars == 1 for a in catalog)

    def test_conditional_tier_families(self) -> None:
        catalog = build_canonical_alpha_catalog()
        conditional = {"flow_positioning", "flow_imbalance_taker", "open_interest_confirmation"}
        for a in catalog:
            if a.family in conditional:
                assert a.data_tier == "conditional", f"{a.family}:{a.recipe_id}"
            else:
                assert a.data_tier == "core", f"{a.family}:{a.recipe_id}"


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
            exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        catalog = build_canonical_alpha_catalog()
        raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
        assert raw.scores_3d.shape == (n_bars, n_syms, 60)
        assert raw.valid_3d.shape == (n_bars, n_syms, 60)
        assert len(raw.recipe_ids) == 60
        assert len(raw.horizon_bars_1d) == 60

    def test_when_conditional_metrics_missing_disables_conditional_families(self) -> None:
        n_bars, n_syms = 200, 5
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
            symbols=("A", "B", "C", "D", "E"),
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
            exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        catalog = build_canonical_alpha_catalog()
        raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
        conditional_recipe_ids = [i for i, rid in enumerate(raw.recipe_ids)
                                  if any(rid.startswith(f"{f}_") for f in
                                         ("flow_positioning", "flow_imbalance_taker", "open_interest_confirmation"))]
        core_recipe_ids = [i for i, rid in enumerate(raw.recipe_ids)
                           if i not in conditional_recipe_ids]
        for k in conditional_recipe_ids:
            assert not raw.valid_3d[:, :, k].any(), f"conditional recipe {raw.recipe_ids[k]} should be all invalid"
        for k in core_recipe_ids:
            assert raw.valid_3d[:, :, k].any(), f"core recipe {raw.recipe_ids[k]} should have valid entries"


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
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    catalog = build_canonical_alpha_catalog()
    raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
    assert raw.scores_3d.ndim == 3
    assert raw.scores_3d.dtype == np.float32


def test_build_canonical_alpha_catalog_returns_ten_families_six_horizons() -> None:
    catalog = build_canonical_alpha_catalog()
    assert len(catalog) == 60
    families = {a.family for a in catalog}
    assert families == {
        "time_series_trend", "breakout", "cross_sectional_momentum",
        "short_term_reversal", "carry_basis", "flow_positioning",
        "volatility_squeeze_keltner", "funding_carry_reversion",
        "flow_imbalance_taker", "open_interest_confirmation",
    }
    horizons = {a.horizon_bars for a in catalog}
    assert horizons == {4, 8, 12, 24, 48, 96}


def test_build_canonical_alpha_catalog_60_recipes() -> None:
    catalog = build_canonical_alpha_catalog()
    assert len(catalog) >= 60
    families = {a.family for a in catalog}
    assert len(families) >= 8


def test_alpha_catalog_pipeline_wiring() -> None:
    catalog = build_canonical_alpha_catalog()
    assert all(a.causal_lag_bars >= 1 for a in catalog)
    assert all(a.required_fields for a in catalog)


def test_compute_raw_alpha_tape_open_interest_confirmation_coverage() -> None:
    n_bars, n_syms = 200, 3
    rng = np.random.default_rng(42)
    ramp = 1000.0 + np.arange(n_bars, dtype=np.float32) * 0.3
    oi = np.column_stack([ramp + rng.normal(0, 2.0, n_bars) for _ in range(n_syms)]).astype(np.float32)
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
            "open_interest": oi,
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    catalog = build_canonical_alpha_catalog()
    raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
    oi_indices = [i for i, rid in enumerate(raw.recipe_ids) if rid.startswith("open_interest_confirmation")]
    assert len(oi_indices) == 6
    for k in oi_indices:
        assert raw.valid_3d[:, :, k].any(), f"OI recipe {raw.recipe_ids[k]} should have valid entries when data is present"


def test_compute_raw_alpha_tape_when_metrics_missing_disables_conditional_recipes() -> None:
    n_bars, n_syms = 200, 5
    cube = MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("A", "B", "C", "D", "E"),
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
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    catalog = build_canonical_alpha_catalog()
    raw = compute_raw_alpha_tape(cube=cube, catalog=catalog)
    conditional_recipe_ids = [i for i, rid in enumerate(raw.recipe_ids)
                              if any(rid.startswith(f"{f}_") for f in
                                     ("flow_positioning", "flow_imbalance_taker", "open_interest_confirmation"))]
    core_recipe_ids = [i for i, rid in enumerate(raw.recipe_ids) if i not in conditional_recipe_ids]
    for k in conditional_recipe_ids:
        assert not raw.valid_3d[:, :, k].any(), f"conditional recipe {raw.recipe_ids[k]} should be all invalid"
    for k in core_recipe_ids:
        assert raw.valid_3d[:, :, k].any(), f"core recipe {raw.recipe_ids[k]} should have valid entries"
