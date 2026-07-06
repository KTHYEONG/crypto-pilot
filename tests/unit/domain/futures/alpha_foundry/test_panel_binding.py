from __future__ import annotations

from src.domain.futures.alpha_foundry.bridge_helpers import bind_panels_to_alpha_recipes
from src.domain.futures.alpha_foundry.contracts import AlphaRecipe
from src.domain.futures.signals.contracts import CandidateSignalPanel


def _make_panel(variant: str = "ema_12_72_4h") -> CandidateSignalPanel:
    import numpy as np

    return CandidateSignalPanel(
        family="trend_ma",
        variant=variant,
        params={"fast": 12, "slow": 72},
        datetimes=np.array(["2026-01-01"], dtype="datetime64[ns]"),
        symbols=("SYM0USDT",),
        signed_score_2d=np.ones((1, 1), dtype=np.float64),
        side_hint_2d=np.ones((1, 1), dtype=np.int8),
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
        valid_mask_2d=np.ones((1, 1), dtype=np.bool_),
        metadata={"recipe_id": "trend_ma__ema_12_72__4h"},
    )


def _make_recipe() -> AlphaRecipe:
    return AlphaRecipe(
        recipe_id="trend_ma__ema_12_72__4h",
        family="trend_ma",
        variant="ema_12_72",
        timeframe="4h",
        archetype="trend",
        indicator_params={"fast": 12, "slow": 72},
        side_rule_id="trend_follow",
        exit_policy_id="atr_trail_2",
        required_fields=("close",),
        causal_lag_bars=1,
        max_turnover_per_year=365.0,
    )


class TestPanelBinding:
    def test_bind_panels_to_recipes_returns_bindings(self) -> None:
        panel = _make_panel()
        recipe = _make_recipe()
        recipes = {recipe.recipe_id: recipe}
        bindings = bind_panels_to_alpha_recipes(
            panels=[panel],
            recipes=recipes,
            timeframe="4h",
            max_recipes_per_family=64,
            include_families=(),
            exclude_families=(),
        )
        assert len(bindings) == 1
