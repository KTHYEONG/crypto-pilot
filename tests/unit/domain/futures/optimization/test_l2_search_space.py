"""Verify L2_SEARCH_SPACE dict structure."""
from __future__ import annotations

from src.domain.futures.optimization.l2_search_space import L2_SEARCH_SPACE


class TestL2SearchSpace:
    def test_all_entries_have_required_keys(self) -> None:
        for key, spec in L2_SEARCH_SPACE.items():
            assert "type" in spec, f"{key} missing 'type'"
            t = spec["type"]
            if t == "categorical":
                assert "choices" in spec, f"{key} missing 'choices'"
            else:
                assert "low" in spec, f"{key} missing 'low'"
                assert "high" in spec, f"{key} missing 'high'"

    def test_regime_defense_params_present(self) -> None:
        assert "l2_regime_long_short_asymmetry_enabled" in L2_SEARCH_SPACE
        assert "l2_regime_crisis_long_extra_mult" in L2_SEARCH_SPACE
        assert "l2_regime_cap_release_cooldown_bars" in L2_SEARCH_SPACE
        assert "l2_regime_crisis_gross_cap" in L2_SEARCH_SPACE
