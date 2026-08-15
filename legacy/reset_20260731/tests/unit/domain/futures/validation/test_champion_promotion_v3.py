"""Tests for Sequential Champion Promotion and Guard v3.0."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.futures.validation.champion_registry import (
    ChampionMetrics,
    evaluate_sequential_promotion_gate,
    should_promote_candidate,
)


@dataclass
class MockWFResult:
    passed: bool


@dataclass
class MockAtomicResult:
    pass_ratio: float


@dataclass
class MockDualDecayResult:
    passed: bool


def test_sequential_promotion_gate_awf_failure() -> None:
    """Test gate fails when Inner AWF fails."""
    candidate = ChampionMetrics(
        atomic_oos_pass_ratio=0.80,
        capacity_ceiling_usdt=300000.0,
        median_log_growth=0.04,
        worst_block_mdd=0.08,
        absolute_decay_bps_yr=-100.0,
        dsr=0.75,
    )
    wf_result = MockWFResult(passed=False)
    atomic_result = MockAtomicResult(pass_ratio=0.85)
    dual_decay = MockDualDecayResult(passed=True)
    capacity_results = {50000: True, 100000: True, 250000: True}

    res = evaluate_sequential_promotion_gate(
        candidate=candidate,
        champion=None,
        wf_result=wf_result,
        dual_decay=dual_decay,
        atomic_result=atomic_result,
        capacity_results=capacity_results,
        intrabar_tw=1.05,
        intrabar_mdd=0.15,
    )

    assert res.passed is False
    assert "AWF_HARD_GATES" in res.gate_failures
    assert res.promoted_to_champion is False


def test_sequential_promotion_gate_atomic_failure() -> None:
    """Test gate fails when atomic block pass ratio < 0.70."""
    candidate = ChampionMetrics(
        atomic_oos_pass_ratio=0.60,
        capacity_ceiling_usdt=300000.0,
        median_log_growth=0.04,
        worst_block_mdd=0.08,
        absolute_decay_bps_yr=-100.0,
        dsr=0.75,
    )
    wf_result = MockWFResult(passed=True)
    atomic_result = MockAtomicResult(pass_ratio=0.65)  # < 0.70
    dual_decay = MockDualDecayResult(passed=True)
    capacity_results = {50000: True, 100000: True, 250000: True}

    res = evaluate_sequential_promotion_gate(
        candidate=candidate,
        champion=None,
        wf_result=wf_result,
        dual_decay=dual_decay,
        atomic_result=atomic_result,
        capacity_results=capacity_results,
        intrabar_tw=1.05,
        intrabar_mdd=0.15,
    )

    assert res.passed is False
    assert "ATOMIC_PASS_RATIO" in res.gate_failures


def test_sequential_promotion_gate_intrabar_mdd_failure() -> None:
    """Test gate fails when intrabar MDD >= limit."""
    candidate = ChampionMetrics(
        atomic_oos_pass_ratio=0.80,
        capacity_ceiling_usdt=300000.0,
        median_log_growth=0.04,
        worst_block_mdd=0.08,
        absolute_decay_bps_yr=-100.0,
        dsr=0.75,
    )
    wf_result = MockWFResult(passed=True)
    atomic_result = MockAtomicResult(pass_ratio=0.85)
    dual_decay = MockDualDecayResult(passed=True)
    capacity_results = {50000: True, 100000: True, 250000: True}

    res = evaluate_sequential_promotion_gate(
        candidate=candidate,
        champion=None,
        wf_result=wf_result,
        dual_decay=dual_decay,
        atomic_result=atomic_result,
        capacity_results=capacity_results,
        intrabar_tw=1.05,
        intrabar_mdd=0.55,  # >= 0.50
    )

    assert res.passed is False
    assert "INTRABAR_MDD" in res.gate_failures


def test_sequential_promotion_gate_dual_decay_failure() -> None:
    """Test gate fails when dual decay fails."""
    candidate = ChampionMetrics(
        atomic_oos_pass_ratio=0.80,
        capacity_ceiling_usdt=300000.0,
        median_log_growth=0.04,
        worst_block_mdd=0.08,
        absolute_decay_bps_yr=-100.0,
        dsr=0.75,
    )
    wf_result = MockWFResult(passed=True)
    atomic_result = MockAtomicResult(pass_ratio=0.85)
    dual_decay = MockDualDecayResult(passed=False)
    capacity_results = {50000: True, 100000: True, 250000: True}

    res = evaluate_sequential_promotion_gate(
        candidate=candidate,
        champion=None,
        wf_result=wf_result,
        dual_decay=dual_decay,
        atomic_result=atomic_result,
        capacity_results=capacity_results,
        intrabar_tw=1.05,
        intrabar_mdd=0.15,
    )

    assert res.passed is False
    assert "DUAL_DECAY" in res.gate_failures


def test_sequential_promotion_gate_capacity_ladder_failure() -> None:
    """Test gate fails when AUM capacity ladder fails."""
    candidate = ChampionMetrics(
        atomic_oos_pass_ratio=0.80,
        capacity_ceiling_usdt=300000.0,
        median_log_growth=0.04,
        worst_block_mdd=0.08,
        absolute_decay_bps_yr=-100.0,
        dsr=0.75,
    )
    wf_result = MockWFResult(passed=True)
    atomic_result = MockAtomicResult(pass_ratio=0.85)
    dual_decay = MockDualDecayResult(passed=True)
    # 100k tier fails
    capacity_results = {50000: True, 100000: False, 250000: True}

    res = evaluate_sequential_promotion_gate(
        candidate=candidate,
        champion=None,
        wf_result=wf_result,
        dual_decay=dual_decay,
        atomic_result=atomic_result,
        capacity_results=capacity_results,
        intrabar_tw=1.05,
        intrabar_mdd=0.15,
    )

    assert res.passed is False
    assert "CAPACITY_LADDER" in res.gate_failures


def test_sequential_promotion_gate_success_no_champion() -> None:
    """Test automatic promotion if all gates pass and there is no champion."""
    candidate = ChampionMetrics(
        atomic_oos_pass_ratio=0.80,
        capacity_ceiling_usdt=300000.0,
        median_log_growth=0.04,
        worst_block_mdd=0.08,
        absolute_decay_bps_yr=-100.0,
        dsr=0.75,
    )
    wf_result = MockWFResult(passed=True)
    atomic_result = MockAtomicResult(pass_ratio=0.85)
    dual_decay = MockDualDecayResult(passed=True)
    capacity_results = {50000: True, 100000: True, 250000: True}

    res = evaluate_sequential_promotion_gate(
        candidate=candidate,
        champion=None,
        wf_result=wf_result,
        dual_decay=dual_decay,
        atomic_result=atomic_result,
        capacity_results=capacity_results,
        intrabar_tw=1.05,
        intrabar_mdd=0.15,
    )

    assert res.passed is True
    assert len(res.gate_failures) == 0
    assert res.promoted_to_champion is True


def test_should_promote_candidate_priority() -> None:
    """Test hierarchical v3 decision priority logic."""
    champion = ChampionMetrics(
        atomic_oos_pass_ratio=0.70,
        capacity_ceiling_usdt=100000.0,
        median_log_growth=0.02,
        worst_block_mdd=0.15,
        absolute_decay_bps_yr=-200.0,
        dsr=0.65,
    )

    # 1. Candidate superior in atomic_oos_pass_ratio (>= 5%p higher)
    cand_1 = ChampionMetrics(
        atomic_oos_pass_ratio=0.76,
        capacity_ceiling_usdt=100000.0,
        median_log_growth=0.02,
        worst_block_mdd=0.15,
        absolute_decay_bps_yr=-200.0,
        dsr=0.65,
    )
    assert should_promote_candidate(cand_1, champion) is True

    # 2. Candidate has lower atomic ratio but superior capacity ceiling (>= 10% higher)
    cand_2 = ChampionMetrics(
        atomic_oos_pass_ratio=0.70,
        capacity_ceiling_usdt=120000.0,
        median_log_growth=0.02,
        worst_block_mdd=0.15,
        absolute_decay_bps_yr=-200.0,
        dsr=0.65,
    )
    assert should_promote_candidate(cand_2, champion) is True

    # 3. Equal except Candidate has superior DSR (tie-break)
    cand_3 = ChampionMetrics(
        atomic_oos_pass_ratio=0.70,
        capacity_ceiling_usdt=100000.0,
        median_log_growth=0.02,
        worst_block_mdd=0.15,
        absolute_decay_bps_yr=-200.0,
        dsr=0.75,
    )
    assert should_promote_candidate(cand_3, champion) is True
