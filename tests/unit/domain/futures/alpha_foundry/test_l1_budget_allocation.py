from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.budget import allocate_global_l1_budget
from src.domain.futures.alpha_foundry.contracts import (
    CheapGateEvidence,
    DiversitySelectionResult,
)


def _evidence(recipe_id: str, block_lcb_bps: float) -> CheapGateEvidence:
    return CheapGateEvidence(
        recipe_id=recipe_id,
        timeframe="4h",
        symbol_scope="symbol",
        n_events=100,
        effective_n=100.0,
        mean_net_bps=block_lcb_bps + 1.0,
        nw_tstat=2.0,
        block_lcb_bps=block_lcb_bps,
        rank_ic=0.05,
        cost_drag_ratio=0.3,
        turnover_per_year=100.0,
        novelty_corr_max=0.0,
        incremental_rank_ic=0.0,
        compute_cost_score=0.0,
        gate_passed=True,
        reject_reasons=(),
        bootstrap_lcb_bps=block_lcb_bps,
        bootstrap_agree=True,
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
    # Scenario 1.6: 품질 비율 3:2:1
    def test_allocates_proportional_to_quality(self) -> None:
        bucket_results = [
            _bucket(("a", "4h"), ("r_a",)),
            _bucket(("b", "4h"), ("r_b",)),
            _bucket(("c", "4h"), ("r_c",)),
        ]
        evidence_by_recipe_id = {
            "r_a": _evidence("r_a", 30.0),
            "r_b": _evidence("r_b", 20.0),
            "r_c": _evidence("r_c", 10.0),
        }
        allocated = allocate_global_l1_budget(
            bucket_results=bucket_results,
            evidence_by_recipe_id=evidence_by_recipe_id,
            total_l1_verification_budget=30,
            top_k_max=5,
        )
        assert sum(allocated.values()) <= 30
        assert allocated[("a", "4h")] >= allocated[("b", "4h")] >= allocated[("c", "4h")]
        assert allocated[("a", "4h")] > 0

    # Scenario 2.6: 품질 0인 버킷은 슬롯 0
    def test_zero_quality_bucket_gets_zero_slots(self) -> None:
        bucket_results = [
            _bucket(("a", "4h"), ()),  # no selected candidates -> quality 0.0
            _bucket(("b", "4h"), ("r_b",)),
        ]
        evidence_by_recipe_id = {"r_b": _evidence("r_b", 15.0)}
        allocated = allocate_global_l1_budget(
            bucket_results=bucket_results,
            evidence_by_recipe_id=evidence_by_recipe_id,
            total_l1_verification_budget=10,
            top_k_max=5,
        )
        assert allocated[("a", "4h")] == 0
        assert allocated[("b", "4h")] > 0

    # Scenario 2.11: total_budget < top_k_max*n_buckets -> 클램프
    def test_slots_never_exceed_top_k_max(self) -> None:
        bucket_results = [_bucket(("a", "4h"), ("r_a",))]
        evidence_by_recipe_id = {"r_a": _evidence("r_a", 100.0)}
        allocated = allocate_global_l1_budget(
            bucket_results=bucket_results,
            evidence_by_recipe_id=evidence_by_recipe_id,
            total_l1_verification_budget=3,
            top_k_max=5,
        )
        assert allocated[("a", "4h")] <= min(5, 3)

    def test_all_buckets_zero_quality_returns_all_zero(self) -> None:
        bucket_results = [_bucket(("a", "4h"), ("r_a",))]
        evidence_by_recipe_id = {"r_a": _evidence("r_a", -5.0)}
        allocated = allocate_global_l1_budget(
            bucket_results=bucket_results,
            evidence_by_recipe_id=evidence_by_recipe_id,
            total_l1_verification_budget=10,
            top_k_max=5,
        )
        assert allocated[("a", "4h")] == 0

    # Scenario 2.12: 동률 처리(bucket_key asc)
    def test_tie_break_by_bucket_key_ascending(self) -> None:
        bucket_results = [
            _bucket(("z_family", "4h"), ("r_z",)),
            _bucket(("a_family", "4h"), ("r_a",)),
        ]
        evidence_by_recipe_id = {
            "r_z": _evidence("r_z", 10.0),
            "r_a": _evidence("r_a", 10.0),
        }
        allocated = allocate_global_l1_budget(
            bucket_results=bucket_results,
            evidence_by_recipe_id=evidence_by_recipe_id,
            total_l1_verification_budget=1,
            top_k_max=5,
        )
        # 동률 소수부 -> bucket_key asc인 a_family가 나머지 1슬롯을 받음
        assert allocated[("a_family", "4h")] == 1
        assert allocated[("z_family", "4h")] == 0

    # Scenario 3.3: total_l1_verification_budget<=0
    def test_raises_on_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="total_l1_verification_budget"):
            allocate_global_l1_budget(
                bucket_results=[],
                evidence_by_recipe_id={},
                total_l1_verification_budget=0,
                top_k_max=5,
            )

    def test_empty_bucket_results_returns_empty(self) -> None:
        allocated = allocate_global_l1_budget(
            bucket_results=[],
            evidence_by_recipe_id={},
            total_l1_verification_budget=10,
            top_k_max=5,
        )
        assert allocated == {}
