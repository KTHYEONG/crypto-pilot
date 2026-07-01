from __future__ import annotations

import re

import src.domain.futures.allocation.search_space as search_space_module
from src.domain.futures.allocation.search_space import L2_SEARCH_SPACE


class TestSearchSpaceVersioning:
    """Scenario 7: L2 search space replaces versioned aliases."""

    def test_l2_search_space_replaces_versioned_aliases(self) -> None:
        keys = set(L2_SEARCH_SPACE)
        assert "K_RANK" in keys
        assert "REBALANCE_BARS" in keys
        assert "CS_Z_SCORE_THRESHOLD" in keys
        assert "deploy_cost_safety_mult" in keys
        assert "risk_budget_floor_ratio" in keys

    def test_k_rank_spec(self) -> None:
        assert L2_SEARCH_SPACE["K_RANK"] == {"type": "int", "low": 1, "high": 8, "step": 1}

    def test_search_space_has_no_versioned_aliases(self) -> None:
        for name in dir(search_space_module):
            assert re.search(r"L2_ALLOC_SPACE_V\d+", name) is None, f"Found versioned alias: {name}"

    def test_search_space_key_count(self) -> None:
        assert len(L2_SEARCH_SPACE) == 9



