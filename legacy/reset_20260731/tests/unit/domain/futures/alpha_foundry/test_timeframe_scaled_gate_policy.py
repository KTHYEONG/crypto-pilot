from typing import Any

import pytest

from src.domain.futures.alpha_foundry.cheap_gate import resolve_family_timeframe_gate_policy
from src.domain.futures.alpha_foundry.contracts import AlphaRecipe, CheapGateConfig


def _make_recipe(**overrides: Any) -> AlphaRecipe:
    kwargs: dict[str, Any] = {
        "recipe_id": "r1",
        "family": "trend_ma",
        "variant": "ema_12_72",
        "timeframe": "1d",
        "archetype": "trend",
        "indicator_params": {},
        "side_rule_id": "s",
        "exit_policy_id": "e",
        "required_fields": (),
        "causal_lag_bars": 1,
        "max_turnover_per_year": 365.0,
    }
    kwargs.update(overrides)
    return AlphaRecipe(**kwargs)


class TestDailyDensityScaling:
    def test_90_days_window(self) -> None:
        recipe = _make_recipe()
        config = CheapGateConfig(
            daily_event_density=0.30,
            daily_effective_n_density=0.15,
            min_events_floor=10,
            min_effective_n_floor=5.0,
            archetype_event_floors={},
            family_event_floors={},
        )
        policy = resolve_family_timeframe_gate_policy(recipe=recipe, config=config, oos_window_days=90.0)
        assert policy.min_events == 27
        assert policy.min_effective_n == 13.5

    def test_floor_applied_for_small_window(self) -> None:
        recipe = _make_recipe()
        config = CheapGateConfig(
            daily_event_density=0.30,
            daily_effective_n_density=0.15,
            min_events_floor=10,
            min_effective_n_floor=5.0,
            archetype_event_floors={},
            family_event_floors={},
        )
        policy = resolve_family_timeframe_gate_policy(recipe=recipe, config=config, oos_window_days=15.0)
        assert policy.min_events == 10
        assert policy.min_effective_n == 5.0

    def test_invalid_non_positive_window(self) -> None:
        recipe = _make_recipe()
        config = CheapGateConfig()
        with pytest.raises(ValueError, match="oos_window_days must be positive"):
            resolve_family_timeframe_gate_policy(recipe=recipe, config=config, oos_window_days=-5.0)

    def test_zero_window_raises(self) -> None:
        recipe = _make_recipe()
        config = CheapGateConfig()
        with pytest.raises(ValueError, match="oos_window_days must be positive"):
            resolve_family_timeframe_gate_policy(recipe=recipe, config=config, oos_window_days=0.0)
