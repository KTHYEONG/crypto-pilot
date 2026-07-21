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

    def test_deployed_scale_growth_and_severity_gating_params_present(self) -> None:
        assert "l2_objective_growth_lcb_weight" not in L2_SEARCH_SPACE  # [LIMIT-04] removed
        assert "l2_regime_severity_gating_enabled" in L2_SEARCH_SPACE
        assert L2_SEARCH_SPACE["l2_regime_severity_gating_enabled"]["choices"] == (False, True)

    def test_l2_search_space_excludes_crisis_and_oos_fields(self) -> None:
        excluded = {"l2_deploy_crisis_mdd_margin", "l2_deploy_oos_budget_blend", "l2_deploy_oos_floor_cap"}
        for key in excluded:
            assert key not in L2_SEARCH_SPACE, f"{key} should not be in L2_SEARCH_SPACE"

    def test_l2_deploy_mdd_margin_param_in_search_space(self) -> None:
        assert "l2_deploy_mdd_margin" in L2_SEARCH_SPACE
        spec = L2_SEARCH_SPACE["l2_deploy_mdd_margin"]
        assert spec["type"] == "float"
        assert spec["low"] == 0.05
        assert spec["high"] == 0.30
        assert spec["step"] == 0.05

    def test_l2_search_space_includes_regime_cell_admission_keys(self) -> None:
        assert L2_SEARCH_SPACE["l2_regime_policy_mode"] == {
            "type": "categorical",
            "choices": ("soft", "hybrid"),
        }
        assert L2_SEARCH_SPACE["l2_regime_hard_block_enabled"] == {
            "type": "categorical",
            "choices": (False, True),
        }
        assert L2_SEARCH_SPACE["l2_regime_pooled_is_passthrough"] == {
            "type": "categorical",
            "choices": (False, True),
        }

    def test_l2_search_space_includes_bear_and_crisis_gross_cap(self) -> None:
        assert "l2_regime_crisis_gross_cap" in L2_SEARCH_SPACE
        crisis = L2_SEARCH_SPACE["l2_regime_crisis_gross_cap"]
        assert crisis["type"] == "float"
        assert crisis["low"] == 0.25
        assert crisis["high"] == 0.85

        assert "l2_regime_bear_gross_cap" in L2_SEARCH_SPACE
        bear = L2_SEARCH_SPACE["l2_regime_bear_gross_cap"]
        assert bear["type"] == "float"
        assert bear["low"] == 0.35
        assert bear["high"] == 0.85
