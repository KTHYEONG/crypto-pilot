from __future__ import annotations

import pytest

from src.domain.futures.alpha_foundry.contracts import AlphaRecipe
from src.domain.futures.alpha_foundry.recipes import build_alpha_recipe_catalog

ALL_FAMILIES = (
    "ema_trend",
    "hma_trend",
    "kama_trend",
    "macd_trend",
    "rsi_mean_reversion",
    "stoch_rsi_mean_reversion",
    "bollinger_mean_reversion",
    "keltner_mean_reversion",
    "ichimoku_trend",
    "funding_slope_carry",
    "oi_buildup_flow",
    "lsr_skew_flow",
    "taker_flow_imbalance",
    "xs_momentum",
    "sparse_breakout_retest_liquidity",
    "funding_flow_exhaustion_sparse",
    "oi_lsr_unwind",
    "vol_contraction_breakout",
    "xs_residual_rebalance",
    "carry_net_of_funding",
    "funding_session_orb_flow",
    "liquidity_sweep_reclaim",
    "cvd_vwap_absorption",
    "funding_basis_dislocation",
    "oi_flow_squeeze",
    "xs_residual_flow_rotation",
    "volume_participation_breakout",
    "liquidity_participation_breakout",
    "btc_neutral_residual_reversal",
)

FAMILY_ARCHETYPE_MAP: dict[str, str] = {
    "ema_trend": "trend",
    "hma_trend": "trend",
    "kama_trend": "trend",
    "macd_trend": "trend",
    "rsi_mean_reversion": "mean_reversion",
    "stoch_rsi_mean_reversion": "mean_reversion",
    "bollinger_mean_reversion": "mean_reversion",
    "keltner_mean_reversion": "mean_reversion",
    "ichimoku_trend": "trend",
    "funding_slope_carry": "carry",
    "oi_buildup_flow": "flow",
    "lsr_skew_flow": "flow",
    "taker_flow_imbalance": "flow",
    "xs_momentum": "cross_sectional",
}


class TestBuildAlphaRecipeCatalog:
    def test_contains_diverse_crypto_families(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        families = {r.family for r in recipes}
        archetypes = {r.archetype for r in recipes}

        assert "funding_carry" in families or "funding_slope_carry" in families
        assert "trend" in archetypes
        assert "mean_reversion" in archetypes
        assert "carry" in archetypes
        assert "flow" in archetypes
        assert "cross_sectional" in archetypes
        assert len(recipes) > 0

    def test_no_duplicate_recipe_ids(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        ids = [r.recipe_id for r in recipes]
        assert len(ids) == len(set(ids)), "duplicate recipe_id found"

    def test_all_recipes_have_causal_lag_at_least_one(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        for r in recipes:
            assert r.causal_lag_bars >= 1, f"{r.recipe_id} has causal_lag_bars={r.causal_lag_bars}"

    def test_carry_recipes_require_funding_field(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        for r in recipes:
            if r.archetype == "carry":
                assert "funding" in r.required_fields, f"{r.recipe_id} missing 'funding' in required_fields"

    def test_flow_recipes_require_oi_or_lsr(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        for r in recipes:
            if r.archetype == "flow":
                has_oi = "oi" in r.required_fields
                has_lsr = "lsr" in r.required_fields
                if r.family in ("oi_buildup_flow", "lsr_skew_flow"):
                    assert has_oi or has_lsr, f"{r.recipe_id} missing 'oi' or 'lsr' in required_fields"

    @pytest.mark.parametrize("family", ALL_FAMILIES)
    def test_include_families_filters_correctly(self, family: str) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h", include_families=(family,))
        assert all(r.family == family for r in recipes), f"include_families=({family},) failed"

    @pytest.mark.parametrize("family", ALL_FAMILIES)
    def test_exclude_families_excludes_correctly(self, family: str) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h", exclude_families=(family,))
        assert all(r.family != family for r in recipes), f"exclude_families=({family},) failed"

    def test_max_recipes_per_family_respected(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h", max_recipes_per_family=2)
        from collections import Counter

        family_counts = Counter(r.family for r in recipes)
        assert all(c <= 2 for c in family_counts.values()), "max_recipes_per_family=2 violated"

    def test_empty_result_when_all_families_excluded(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h", include_families=("nonexistent_family",))
        assert len(recipes) == 0

    def test_sparse_liquidity_recipes_in_catalog(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        families = {r.family for r in recipes}
        assert "sparse_breakout_retest_liquidity" in families
        assert "funding_flow_exhaustion_sparse" in families
        assert "oi_lsr_unwind" in families
        assert "vol_contraction_breakout" in families
        # xs_residual_rebalance retired at 4h via FAMILY_TF_RETIREMENT
        assert "carry_net_of_funding" in families

    def test_ltf_native_families_in_catalog(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        families = {r.family for r in recipes}
        assert "funding_session_orb_flow" in families
        assert "liquidity_sweep_reclaim" in families
        assert "cvd_vwap_absorption" in families
        assert "funding_basis_dislocation" in families
        assert "oi_flow_squeeze" in families
        assert "xs_residual_flow_rotation" in families
        assert "volume_participation_breakout" in families

    def test_liquidity_vacuum_breakout_retired_from_catalog(self) -> None:
        """S2-09: After retirement, no liquidity_vacuum_breakout recipe in catalog."""
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        families = {r.family for r in recipes}
        assert "liquidity_vacuum_breakout" not in families
        assert "vol_contraction_breakout" in families  # unaffected

    def test_sparse_recipe_turnover_more_conservative_than_continuous(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        sparse_families = {
            "sparse_breakout_retest_liquidity",
            "funding_flow_exhaustion_sparse",
            "oi_lsr_unwind",
            "vol_contraction_breakout",
        }
        trend_turnovers = [r.max_turnover_per_year for r in recipes if r.family == "trend_ma"]
        sparse_turnovers = [r.max_turnover_per_year for r in recipes if r.family in sparse_families]
        if trend_turnovers and sparse_turnovers:
            assert max(sparse_turnovers) <= max(trend_turnovers), "sparse turnover should be <= continuous trend"


class TestBuildAlphaRecipeCatalogErrors:
    def test_rejects_negative_causal_lag(self) -> None:
        with pytest.raises(ValueError, match="causal_lag_bars must be >= 1"):
            AlphaRecipe(
                recipe_id="bad",
                family="f",
                variant="v",
                timeframe="4h",
                archetype="trend",
                indicator_params={},
                side_rule_id="s",
                exit_policy_id="e",
                required_fields=("close",),
                causal_lag_bars=0,
                max_turnover_per_year=100.0,
            )

    def test_duplicate_recipe_id_raises_error(self) -> None:
        recipes = build_alpha_recipe_catalog(timeframe="4h")
        ids = [r.recipe_id for r in recipes]
        assert len(ids) == len(set(ids))


# ─── S1-04: New panel recipes bound to catalog ──────────────────────────


class TestNewRecipeRegistrations:
    def test_lpb_recipes_have_causal_lag_one(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h", include_families=("liquidity_participation_breakout",)
        )
        assert len(recipes) == 2
        for r in recipes:
            assert r.causal_lag_bars == 1

    def test_bnrr_recipes_have_causal_lag_one(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h", include_families=("btc_neutral_residual_reversal",)
        )
        assert len(recipes) == 2
        for r in recipes:
            assert r.causal_lag_bars == 1

    def test_lpb_recipes_use_catalog_exact_ids(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h", include_families=("liquidity_participation_breakout",)
        )
        variants = {r.variant for r in recipes}
        assert "lpb_40" in variants
        assert "lpb_60" in variants

    def test_bnrr_recipes_use_catalog_exact_ids(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h", include_families=("btc_neutral_residual_reversal",)
        )
        variants = {r.variant for r in recipes}
        assert "bnrr_24" in variants
        assert "bnrr_48" in variants

    def test_lpb_archetype_is_trend(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h", include_families=("liquidity_participation_breakout",)
        )
        for r in recipes:
            assert r.archetype == "trend"

    def test_bnrr_archetype_is_cross_sectional(self) -> None:
        recipes = build_alpha_recipe_catalog(
            timeframe="4h", include_families=("btc_neutral_residual_reversal",)
        )
        for r in recipes:
            assert r.archetype == "cross_sectional"
