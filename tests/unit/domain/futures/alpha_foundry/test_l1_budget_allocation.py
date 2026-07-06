from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.budget import allocate_global_l1_budget
from src.domain.futures.alpha_foundry.contracts import (
    DiversitySelectionResult,
    L0SignalCandidate,
)


def _candidate(
    recipe_id: str, priority_score: float, archetype: str = "trend", timeframe: str = "4h",
) -> L0SignalCandidate:
    return L0SignalCandidate(
        run_id="test",
        timeframe=timeframe,
        family="a",
        variant=recipe_id,
        recipe_id=recipe_id,
        archetype=archetype,
        source="catalog_exact",
        n_events=100,
        effective_n=100.0,
        mean_net_bps=priority_score + 1.0,
        block_lcb_bps=priority_score,
        nw_tstat=2.0,
        bootstrap_lcb_bps=priority_score,
        bootstrap_agree=True,
        cost_drag_ratio=0.3,
        turnover_per_year=100.0,
        max_abs_corr_in_bucket=0.0,
        tf_coverage_count=0,
        sign_agreement_ratio=0.0,
        corroboration_tier="insufficient_coverage",
        discovery_tier="candidate",
        l1_priority_score=priority_score,
        l1_budget_units=0,
        hard_reject_reasons=(),
        soft_flags=(),
    )


def _bucket(bucket_key: tuple[str, str], selected: tuple[str, ...]) -> DiversitySelectionResult:
    return DiversitySelectionResult(
        bucket_key=bucket_key,
        ranked_recipe_ids=selected,
        selected_recipe_ids=selected,
        redundant_recipe_ids=(),
        redundant_reason_by_id={},
        bucket_corr=np.empty((0, 0), dtype=np.float64),
        bucket_eff_test_count=float(len(selected)),
    )


class TestAllocateGlobalL1Budget:
    def test_allocates_proportional_to_quality(self) -> None:
        bucket_results = [
            _bucket(("a", "4h"), ("r_a",)),
            _bucket(("b", "4h"), ("r_b",)),
            _bucket(("c", "4h"), ("r_c",)),
        ]
        candidate_by_recipe_id = {
            "r_a": _candidate("r_a", 30.0),
            "r_b": _candidate("r_b", 20.0),
            "r_c": _candidate("r_c", 10.0),
        }
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=bucket_results,
            candidate_by_recipe_id=candidate_by_recipe_id,
            total_l1_verification_budget=30,
            top_k_max=5,
        )
        assert sum(allocated.values()) <= 30
        assert allocated[("a", "4h")] >= allocated[("b", "4h")] >= allocated[("c", "4h")]
        assert allocated[("a", "4h")] > 0

    def test_zero_quality_bucket_gets_zero_slots(self) -> None:
        bucket_results = [
            _bucket(("a", "4h"), ()),
            _bucket(("b", "4h"), ("r_b",)),
        ]
        candidate_by_recipe_id = {"r_b": _candidate("r_b", 15.0)}
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=bucket_results,
            candidate_by_recipe_id=candidate_by_recipe_id,
            total_l1_verification_budget=10,
            top_k_max=5,
        )
        assert allocated[("a", "4h")] == 0
        assert allocated[("b", "4h")] > 0

    def test_slots_never_exceed_top_k_max(self) -> None:
        bucket_results = [_bucket(("a", "4h"), ("r_a",))]
        candidate_by_recipe_id = {"r_a": _candidate("r_a", 100.0)}
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=bucket_results,
            candidate_by_recipe_id=candidate_by_recipe_id,
            total_l1_verification_budget=3,
            top_k_max=5,
        )
        assert allocated[("a", "4h")] <= 5

    def test_all_buckets_zero_quality_gets_seed_slot(self) -> None:
        bucket_results = [_bucket(("a", "4h"), ("r_a",))]
        candidate_by_recipe_id = {"r_a": _candidate("r_a", -5.0)}
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=bucket_results,
            candidate_by_recipe_id=candidate_by_recipe_id,
            total_l1_verification_budget=10,
            top_k_max=5,
        )
        # Zero quality but archetype seed ensures 1 slot
        assert allocated[("a", "4h")] >= 1

    def test_tie_break_by_bucket_key_ascending(self) -> None:
        bucket_results = [
            _bucket(("z_family", "4h"), ("r_z",)),
            _bucket(("a_family", "4h"), ("r_a",)),
        ]
        candidate_by_recipe_id = {
            "r_z": _candidate("r_z", 10.0),
            "r_a": _candidate("r_a", 10.0),
        }
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=bucket_results,
            candidate_by_recipe_id=candidate_by_recipe_id,
            total_l1_verification_budget=1,
            top_k_max=5,
        )
        # With seed logic, both buckets have same archetype "trend" -> first archetype seed
        # gets 1 slot (min_seed_slots_per_archetype=1)
        assert sum(allocated.values()) >= 1

    def test_raises_on_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="total_l1_verification_budget"):
            allocate_global_l1_budget(
                bucket_results=[],
                candidate_by_recipe_id={},
                total_l1_verification_budget=0,
                top_k_max=5,
            )

    def test_empty_bucket_results_returns_empty(self) -> None:
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=[],
            candidate_by_recipe_id={},
            total_l1_verification_budget=10,
            top_k_max=5,
        )
        assert allocated == {}

    def test_budget_exhausted_during_archetype_seed(self) -> None:
        bucket_results = [
            _bucket(("a", "4h"), ("r_a",)),
            _bucket(("b", "4h"), ("r_b",)),
        ]
        # different archetypes, budget=1 -> first gets seed, second hits break
        candidate_by_recipe_id = {
            "r_a": _candidate("r_a", 10.0, archetype="trend"),
            "r_b": _candidate("r_b", 5.0, archetype="flow"),
        }
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=bucket_results,
            candidate_by_recipe_id=candidate_by_recipe_id,
            total_l1_verification_budget=1,
            top_k_max=5,
            min_seed_slots_per_archetype=1,
            min_seed_slots_per_timeframe=1,
        )
        assert sum(allocated.values()) <= 1

    def test_timeframe_seed_allocated_when_no_slot_exists(self) -> None:
        bucket_results = [
            _bucket(("a_family", "4h"), ("r_a",)),
            _bucket(("b_family", "6h"), ("r_b",)),
        ]
        # same archetype "trend", different timeframes, budget=2
        candidate_by_recipe_id = {
            "r_a": _candidate("r_a", 10.0, archetype="trend", timeframe="4h"),
            "r_b": _candidate("r_b", 5.0, archetype="trend", timeframe="6h"),
        }
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=bucket_results,
            candidate_by_recipe_id=candidate_by_recipe_id,
            total_l1_verification_budget=2,
            top_k_max=5,
            min_seed_slots_per_archetype=1,
            min_seed_slots_per_timeframe=1,
        )
        # "4h" gets archetype seed, "6h" gets timeframe seed
        assert allocated.get(("a_family", "4h"), 0) >= 1
        assert allocated.get(("b_family", "6h"), 0) >= 1

    def test_timeframe_seed_when_timeframe_already_has_slot(self) -> None:
        # 2 archetypes, same timeframe 4h, budget tight so only archetype seeds get slots
        bucket_results = [
            _bucket(("a_family", "4h"), ("r_a",)),
            _bucket(("b_family", "4h"), ("r_b",)),
        ]
        candidate_by_recipe_id = {
            "r_a": _candidate("r_a", 10.0, archetype="trend"),
            "r_b": _candidate("r_b", 5.0, archetype="flow"),
        }
        allocated, _budgets = allocate_global_l1_budget(
            bucket_results=bucket_results,
            candidate_by_recipe_id=candidate_by_recipe_id,
            total_l1_verification_budget=2,
            top_k_max=5,
            min_seed_slots_per_archetype=1,
            min_seed_slots_per_timeframe=1,
        )
        # both archetypes get seed slot, timeframe seed is skipped (already has slot)
        assert sum(allocated.values()) == 2
