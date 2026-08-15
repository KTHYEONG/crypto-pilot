from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_foundry.bridge_helpers import bind_panels_to_alpha_recipes
from src.domain.futures.alpha_foundry.recipes import map_signal_archetype_to_alpha_archetype
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData


def _aligned(t: int = 12, n: int = 2) -> AlignedMarketData:
    dt = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    close = np.linspace(100, 112, t).reshape(-1, 1) * np.ones((1, n))
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full_like(close, 1000.0),
        funding_2d=np.full_like(close, 0.0001),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(close, dtype=np.bool_),
        kill_mask=np.zeros_like(close, dtype=np.bool_),
    )


def _panel(*, family: str, variant: str, archetype: str = "ts_mom") -> CandidateSignalPanel:
    aligned = _aligned()
    t, n = aligned.close_2d.shape
    return CandidateSignalPanel(
        family=family,
        variant=variant,
        params={"window": 24},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=np.ones((t, n), dtype=np.float64),
        side_hint_2d=np.ones((t, n), dtype=np.int8),
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
        metadata={},
        archetype=archetype,
    )


class TestSyntheticRecipeFallback:
    # Scenario 1.7: 카탈로그 미매칭 family, enable_synthetic_recipes=True
    def test_unmatched_family_gets_synthetic_binding(self) -> None:
        panel = _panel(family="dual_momentum", variant="dm_24")
        recipes: dict[str, object] = {}

        bindings = bind_panels_to_alpha_recipes(
            panels=[panel],
            recipes=recipes,
            timeframe="4h",
            max_recipes_per_family=64,
            include_families=(),
            exclude_families=(),
            enable_synthetic_recipes=True,
        )

        assert len(bindings) == 1
        assert bindings[0].source == "synthetic_recipe"
        assert bindings[0].family == "dual_momentum"
        # 합성된 recipe가 실제로 recipes dict에 등록됨(다운스트림 lookup 가능해야 함)
        assert bindings[0].recipe_id in recipes
        synth_recipe = recipes[bindings[0].recipe_id]
        assert synth_recipe.causal_lag_bars == 1
        assert synth_recipe.archetype == "trend"  # ts_mom -> trend 매핑

    # Scenario 2.9: enable_synthetic_recipes=False -> 기존 동작(폐기) 회귀
    def test_disabled_synthetic_recipes_drops_unmatched_panel(self) -> None:
        panel = _panel(family="dual_momentum", variant="dm_24")
        recipes: dict[str, object] = {}

        bindings = bind_panels_to_alpha_recipes(
            panels=[panel],
            recipes=recipes,
            timeframe="4h",
            max_recipes_per_family=64,
            include_families=(),
            exclude_families=(),
            enable_synthetic_recipes=False,
        )

        assert bindings == ()
        assert recipes == {}

    def test_synthetic_recipe_id_deterministic_across_calls(self) -> None:
        panel = _panel(family="vol_breakout", variant="vb_48")
        recipes_a: dict[str, object] = {}
        recipes_b: dict[str, object] = {}

        bindings_a = bind_panels_to_alpha_recipes(
            panels=[panel],
            recipes=recipes_a,
            timeframe="4h",
            max_recipes_per_family=64,
            include_families=(),
            exclude_families=(),
        )
        bindings_b = bind_panels_to_alpha_recipes(
            panels=[panel],
            recipes=recipes_b,
            timeframe="4h",
            max_recipes_per_family=64,
            include_families=(),
            exclude_families=(),
        )
        assert bindings_a[0].recipe_id == bindings_b[0].recipe_id


class TestMapSignalArchetypeToAlphaArchetype:
    # Scenario 2.10: 미인식 archetype -> 안전 폴백
    def test_unrecognized_archetype_falls_back_to_trend(self) -> None:
        assert map_signal_archetype_to_alpha_archetype("unknown_new_archetype") == "trend"

    def test_known_mappings(self) -> None:
        assert map_signal_archetype_to_alpha_archetype("mean_rev") == "mean_reversion"
        assert map_signal_archetype_to_alpha_archetype("flow_rev") == "flow"
        assert map_signal_archetype_to_alpha_archetype("carry_rev") == "carry"
        assert map_signal_archetype_to_alpha_archetype("beta_neut") == "hedge"
        assert map_signal_archetype_to_alpha_archetype("xs_alpha") == "cross_sectional"
