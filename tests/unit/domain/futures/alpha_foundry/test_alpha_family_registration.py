from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_foundry.recipes import (
    FAMILY_ARCHETYPE,
    FAMILY_EXIT_POLICY,
    FAMILY_MAX_TURNOVER,
    FAMILY_SIDE_RULE,
    RECIPE_DEFINITIONS,
    build_alpha_recipe_catalog,
    map_signal_archetype_to_alpha_archetype,
)
from src.domain.futures.signals.rules import ALL_SIGNAL_FAMILIES as RULES_ALL_SIGNAL_FAMILIES
from src.domain.futures.signals.rules import build_rule_signal_panels as signals_build_panels
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_signals import ALL_SIGNAL_FAMILIES as STRATEGY_ALL_SIGNAL_FAMILIES
from src.domain.futures.strategy.rule_signals import build_rule_signal_panels as strategy_build_panels

NEW_ALPHA_FAMILIES: tuple[str, ...] = (
    "sparse_breakout_retest_v2",
    "trend_pullback_quality_v2",
    "residual_momentum_xs",
    "funding_contra_carry_sparse",
    "oi_price_divergence_unwind",
    "taker_flow_exhaustion",
    "liquidity_vacuum_breakout",
    "volatility_contraction_expansion",
    "btc_regime_relative_strength",
    "mean_reversion_after_liquidation_proxy",
)


class TestNewFamilyRegistration:
    def test_all_10_families_registered_in_recipe_definitions(self) -> None:
        for fam in NEW_ALPHA_FAMILIES:
            assert fam in RECIPE_DEFINITIONS, f"{fam} missing from RECIPE_DEFINITIONS"

    def test_all_10_families_registered_in_archetype_map(self) -> None:
        for fam in NEW_ALPHA_FAMILIES:
            assert fam in FAMILY_ARCHETYPE, f"{fam} missing from FAMILY_ARCHETYPE"

    def test_all_10_families_registered_in_side_rule_map(self) -> None:
        for fam in NEW_ALPHA_FAMILIES:
            assert fam in FAMILY_SIDE_RULE, f"{fam} missing from FAMILY_SIDE_RULE"

    def test_all_10_families_registered_in_exit_policy_map(self) -> None:
        for fam in NEW_ALPHA_FAMILIES:
            assert fam in FAMILY_EXIT_POLICY, f"{fam} missing from FAMILY_EXIT_POLICY"

    def test_all_10_families_registered_in_max_turnover_map(self) -> None:
        for fam in NEW_ALPHA_FAMILIES:
            assert fam in FAMILY_MAX_TURNOVER, f"{fam} missing from FAMILY_MAX_TURNOVER"

    def test_all_10_families_in_rules_all_signal_families(self) -> None:
        for fam in NEW_ALPHA_FAMILIES:
            assert fam in RULES_ALL_SIGNAL_FAMILIES, f"{fam} missing from signals/rules ALL_SIGNAL_FAMILIES"

    def test_all_10_families_in_strategy_all_signal_families(self) -> None:
        for fam in NEW_ALPHA_FAMILIES:
            assert fam in STRATEGY_ALL_SIGNAL_FAMILIES, f"{fam} missing from strategy/rule_signals ALL_SIGNAL_FAMILIES"

    def test_build_recipe_catalog_includes_new_families(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        recipe_families = {r.family for r in recipes}
        for fam in NEW_ALPHA_FAMILIES:
            assert fam in recipe_families, f"{fam} not in built recipe catalog"

    def test_residual_momentum_xs_has_cross_sectional_archetype(self) -> None:
        assert FAMILY_ARCHETYPE["residual_momentum_xs"] == "cross_sectional"

    def test_recipe_definitions_for_first_3_families_have_variants(self) -> None:
        for fam in ("sparse_breakout_retest_v2", "trend_pullback_quality_v2", "residual_momentum_xs"):
            variants = RECIPE_DEFINITIONS[fam]
            assert len(variants) >= 1, f"{fam} has no recipe definitions"


class TestRecipeCatalogEdgeCases:
    def test_include_families_filters_correctly(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h",
            include_families=("sparse_breakout_retest_v2",),
        )
        families = {r.family for r in recipes}
        assert "sparse_breakout_retest_v2" in families
        assert len(families) == 1

    def test_exclude_families_filters_correctly(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h",
            include_families=(
                "sparse_breakout_retest_v2",
                "trend_pullback_quality_v2",
                "residual_momentum_xs",
            ),
            exclude_families=("sparse_breakout_retest_v2",),
        )
        families = {r.family for r in recipes}
        assert "sparse_breakout_retest_v2" not in families
        assert "trend_pullback_quality_v2" in families

    def test_max_recipes_per_family_limit(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h",
            include_families=("sparse_breakout_retest_v2", "trend_pullback_quality_v2"),
            max_recipes_per_family=1,
        )
        from collections import Counter

        counts = Counter(r.family for r in recipes)
        for count in counts.values():
            assert count == 1

    def test_map_signal_archetype_to_alpha(self) -> None:
        assert map_signal_archetype_to_alpha_archetype("trend") == "trend"
        assert map_signal_archetype_to_alpha_archetype("unknown") == "trend"
        assert map_signal_archetype_to_alpha_archetype("xs_alpha") == "cross_sectional"


def _make_signal_aligned(t: int = 400, n: int = 8) -> AlignedMarketData:
    rng = np.random.default_rng(42)
    drift = 0.001 * np.arange(t, dtype=np.float64)
    close = np.column_stack([100.0 * (1.0 + drift + 0.02 * np.sin(0.1 * np.arange(t))) for _ in range(n)])
    close += rng.normal(0, 0.3, (t, n)).astype(np.float64)
    close = np.maximum(close, 1.0)
    datetimes = np.datetime64("2026-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    syms = (*tuple(f"SYM{i}USDT" for i in range(n - 1)), "BTCUSDT")
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=syms,
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((t, n), 1_000.0),
        funding_2d=np.full((t, n), 0.0001),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(mask),
        kill_mask=np.zeros_like(mask),
        oi_2d=np.full((t, n), 10_000.0),
        lsr_2d=np.full((t, n), 1.2),
        taker_buy_2d=np.full((t, n), 500.0),
        trades_2d=np.full((t, n), 100.0),
        execution_cost_bps_2d=np.full((t, n), 2.5),
    )


class TestNewFamilySignalPanelsRun:
    def test_all_new_families_run_without_error(self) -> None:
        aligned = _make_signal_aligned()
        cfg = CandidateStrategyConfig()
        panels = signals_build_panels(
            aligned=aligned,
            cfg=cfg,
            family_filter=("sparse_breakout_retest_v2", "trend_pullback_quality_v2", "residual_momentum_xs"),
        )
        assert isinstance(panels, tuple)
        panel_families = {p.family for p in panels}
        assert "sparse_breakout_retest_v2" in panel_families, f"missing sparse_breakout_retest_v2, got {panel_families}"
        assert "trend_pullback_quality_v2" in panel_families
        assert "residual_momentum_xs" in panel_families

    def test_strategy_all_new_families_run_without_error(self) -> None:
        aligned = _make_signal_aligned()
        cfg = CandidateStrategyConfig()
        panels = strategy_build_panels(
            aligned=aligned,
            cfg=cfg,
            family_filter=("sparse_breakout_retest_v2", "trend_pullback_quality_v2", "residual_momentum_xs"),
        )
        assert isinstance(panels, tuple)
        panel_families = {p.family for p in panels}
        assert "sparse_breakout_retest_v2" in panel_families
        assert "trend_pullback_quality_v2" in panel_families
        assert "residual_momentum_xs" in panel_families

    def test_residual_momentum_xs_cross_sectional_metadata(self) -> None:
        aligned = _make_signal_aligned()
        cfg = CandidateStrategyConfig()
        panels = signals_build_panels(
            aligned=aligned,
            cfg=cfg,
            family_filter=("residual_momentum_xs",),
        )
        for p in panels:
            if p.family == "residual_momentum_xs":
                assert p.metadata.get("archetype") == "xs_alpha"
                assert p.metadata.get("max_abs_btc_beta") == 0.80
