"""Verify the deliberately small causal L2 economic search space."""
from __future__ import annotations

from src.domain.futures.optimization.l2_search_space import L2_SEARCH_SPACE


EXPECTED_KEYS = {
    "K_RANK",
    "REBALANCE_BARS",
    "CS_Z_SCORE_THRESHOLD",
    "deploy_cost_safety_mult",
    "edge_ref_bps",
    "edge_throttle_gamma",
    "risk_budget_floor_ratio",
    "risk_budget_max_scale",
}


class TestL2SearchSpace:
    def test_l2_search_space_contains_only_eight_economic_dimensions(self) -> None:
        assert set(L2_SEARCH_SPACE) == EXPECTED_KEYS

    def test_all_entries_have_required_keys(self) -> None:
        for key, spec in L2_SEARCH_SPACE.items():
            assert "type" in spec, f"{key} missing 'type'"
            if spec["type"] == "categorical":
                assert "choices" in spec, f"{key} missing 'choices'"
            else:
                assert "low" in spec, f"{key} missing 'low'"
                assert "high" in spec, f"{key} missing 'high'"

    def test_l2_search_space_excludes_structural_and_oos_fields(self) -> None:
        forbidden_fragments = ("l2_regime_", "l2_deploy_oos", "l2_deploy_mdd_margin")
        assert all(
            not key.startswith(forbidden_fragments)
            for key in L2_SEARCH_SPACE
        )
