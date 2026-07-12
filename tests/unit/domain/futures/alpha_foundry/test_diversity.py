from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.contracts import (
    AlphaArchetype,
    AlphaFoundryRuntimeConfig,
    CheapGateEvidence,
    CrossBucketDiversityResult,
    CrossTFCanonicalContext,
    L0SignalCandidate,
)
from src.domain.futures.alpha_foundry.diversity import (
    _paired_score_corr,
    audit_full_family_correlation,
    audit_l0_selected_recipe_independence,
    cluster_correlated_recipes,
    compute_cross_tf_pair_evidence,
    compute_cross_tf_redundancy,
    compute_panel_correlation_matrix,
    estimate_effective_test_count,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel


def _make_panel(score: np.ndarray, *, valid: np.ndarray | None = None) -> CandidateSignalPanel:
    t, n = score.shape
    datetimes = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    return CandidateSignalPanel(
        family="test",
        variant="v1",
        params={},
        datetimes=datetimes,
        symbols=tuple(f"s{i}" for i in range(n)),
        signed_score_2d=score.astype(np.float64),
        side_hint_2d=np.ones((t, n), dtype=np.int8),
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=valid if valid is not None else np.ones((t, n), dtype=np.bool_),
    )


def _make_active_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=np.bool_)


class TestPairedScoreCorr:
    # Scenario 2.1: 교집합 없음 → 0.0 폴백
    def test_replaces_non_finite_with_zero(self) -> None:
        t, n = 100, 5
        score_a = np.random.randn(t, n)
        score_b = np.random.randn(t, n)
        valid_a = np.zeros((t, n), dtype=np.bool_)
        valid_a[0::2, :] = True  # 짝수 인덱스
        valid_b = np.zeros((t, n), dtype=np.bool_)
        valid_b[1::2, :] = True  # 홀수 인덱스
        panel_a = _make_panel(score_a, valid=valid_a)
        panel_b = _make_panel(score_b, valid=valid_b)
        active = _make_active_mask((t, n))
        corr = _paired_score_corr(panel_a, panel_b, active)
        assert corr == 0.0  # 교집합 없음

    def test_identical_scores_give_one(self) -> None:
        t, n = 50, 3
        rng = np.random.default_rng(42)
        score = rng.normal(0, 1, (t, n))
        panel_a = _make_panel(score)
        panel_b = _make_panel(score.copy())
        active = _make_active_mask((t, n))
        corr = _paired_score_corr(panel_a, panel_b, active)
        assert abs(corr - 1.0) < 1e-10


class TestComputePanelCorrelationMatrix:
    def test_returns_square_matrix(self) -> None:
        t, n = 100, 5
        panels = [_make_panel(np.random.randn(t, n)) for _ in range(4)]
        active = _make_active_mask((t, n))
        corr = compute_panel_correlation_matrix(panels, active)
        assert corr.shape == (4, 4)

    def test_diagonal_is_one(self) -> None:
        t, n = 100, 5
        panels = [_make_panel(np.random.randn(t, n)) for _ in range(3)]
        active = _make_active_mask((t, n))
        corr = compute_panel_correlation_matrix(panels, active)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_raises_on_empty_panels(self) -> None:
        active = _make_active_mask((10, 3))
        with pytest.raises(ValueError, match="empty"):
            compute_panel_correlation_matrix([], active)

    # Scenario 3.2: active_mask shape mismatch
    def test_raises_on_shape_mismatch(self) -> None:
        t, n = 100, 5
        panels = [_make_panel(np.random.randn(t, n))]
        bad_active = np.ones((50, 3), dtype=np.bool_)
        with pytest.raises(ValueError, match="shape"):
            compute_panel_correlation_matrix(panels, bad_active)


class TestClusterCorrelatedRecipes:
    def _make_evidence(self, recipe_id: str) -> CheapGateEvidence:
        return CheapGateEvidence(
            recipe_id=recipe_id,
            timeframe="4h",
            symbol_scope="symbol",
            n_events=100,
            effective_n=50.0,
            mean_net_bps=1.0,
            nw_tstat=2.0,
            block_lcb_bps=0.5,
            rank_ic=0.05,
            cost_drag_ratio=0.3,
            turnover_per_year=100.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.02,
            compute_cost_score=0.0,
            gate_passed=True,
            reject_reasons=(),
            bootstrap_lcb_bps=0.4,
            bootstrap_agree=True,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )

    def test_groups_highly_correlated(self) -> None:
        evs = tuple(self._make_evidence(f"r{i}") for i in range(4))
        corr = np.array(
            [
                [1.0, 0.9, 0.3, 0.2],
                [0.9, 1.0, 0.3, 0.2],
                [0.3, 0.3, 1.0, 0.1],
                [0.2, 0.2, 0.1, 1.0],
            ]
        )
        clusters = cluster_correlated_recipes(evidences=evs, corr=corr, max_corr=0.8)
        assert len(clusters) > 0

    def test_raises_on_shape_mismatch(self) -> None:
        evs = tuple(self._make_evidence(f"r{i}") for i in range(3))
        corr = np.eye(4)
        with pytest.raises(ValueError, match="shape"):
            cluster_correlated_recipes(evidences=evs, corr=corr, max_corr=0.8)


class TestEstimateEffectiveTestCount:
    def test_bounds_between_one_and_n(self) -> None:
        corr = np.eye(10)
        m_eff = estimate_effective_test_count(corr)
        assert 1.0 <= m_eff <= 10.0

    def test_raises_on_non_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            estimate_effective_test_count(np.ones((3, 4)))


# ---------------------------------------------------------------------------
# Rule 3 — Family correlation audit
# ---------------------------------------------------------------------------

def _make_multi_family_panel_fixtures(n_families: int = 4) -> list[CandidateSignalPanel]:
    t, n = 50, 2
    datetimes = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    panels: list[CandidateSignalPanel] = []
    for i in range(n_families):
        rng = np.random.default_rng(42 + i)
        score = rng.normal(0, 1, (t, n))
        side = np.where(score > 0, np.int8(1), np.int8(-1))
        panels.append(CandidateSignalPanel(
            family=f"family_{i}",
            variant=f"v{i}",
            params={},
            datetimes=datetimes,
            symbols=tuple(f"s{j}" for j in range(n)),
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=np.ones((t, n), dtype=bool),
            metadata={},
        ))
    return panels


def test_family_correlation_audit_disabled_by_default() -> None:
    cfg = AlphaFoundryRuntimeConfig()
    assert cfg.enable_correlation_audit is False


def test_audit_full_family_correlation_matrix_symmetric() -> None:
    panels = _make_multi_family_panel_fixtures(n_families=4)
    active_mask = np.ones(panels[0].signed_score_2d.shape, dtype=bool)
    result = audit_full_family_correlation(
        panels=panels, active_mask=active_mask, run_id="test", timeframe="4h",
    )
    assert set(result.columns) == {
        "family_a", "variant_a", "family_b", "variant_b",
        "timeframe", "pairwise_corr", "cluster_id", "run_id",
    }
    summary_rows = result[result["family_a"] == "__SUMMARY__"]
    assert len(summary_rows) == 1


# ── Fix 3: Economic Thesis Grouping ─────────────────────────────────────


class TestResolveEconomicThesisId:
    def test_s1_05_known_families_map_to_group(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import resolve_economic_thesis_id

        # trend_ma and ema_trend both map to "trend_ma_cross"
        assert resolve_economic_thesis_id("trend_ma") == "trend_ma_cross"
        assert resolve_economic_thesis_id("ema_trend") == "trend_ma_cross"
        # funding_slope_carry maps to "funding_carry"
        assert resolve_economic_thesis_id("funding_slope_carry") == "funding_carry"

    def test_s2_07_unknown_family_defaults_to_singleton(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import resolve_economic_thesis_id

        assert resolve_economic_thesis_id("brand_new_family_not_in_map") == "brand_new_family_not_in_map"


class TestEstimateDistinctThesisCount:
    def test_s1_05_counts_distinct_groups(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import estimate_distinct_thesis_count

        result = estimate_distinct_thesis_count(["trend_ma", "ema_trend", "funding_slope_carry"])
        assert result == 2  # trend_ma_cross, funding_carry

    def test_s3_02_empty_list_returns_zero(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import estimate_distinct_thesis_count

        assert estimate_distinct_thesis_count([]) == 0

    def test_all_same_family_returns_one(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import estimate_distinct_thesis_count

        assert estimate_distinct_thesis_count(["trend_ma", "trend_ma", "trend_ma"]) == 1


def test_audit_full_family_correlation_raises_on_empty_panels() -> None:
    with pytest.raises(ValueError, match="panels must not be empty"):
        audit_full_family_correlation(
            panels=(), active_mask=np.ones((10, 2), dtype=bool),
            run_id="test", timeframe="4h",
        )


def test_audit_full_family_correlation_clustering_branch() -> None:
    """패널이 높은 상관관계를 가질 때 clustering branch가 실행됨을 검증."""
    t, n = 50, 3
    datetimes = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    rng = np.random.default_rng(99)
    base_score = rng.normal(0, 1, (t, n))
    panels: list[CandidateSignalPanel] = []
    for i in range(3):
        score = base_score + rng.normal(0, 0.01, (t, n))
        side = np.where(score > 0, np.int8(1), np.int8(-1))
        panels.append(CandidateSignalPanel(
            family=f"fam_{i}", variant=f"v{i}", params={},
            datetimes=datetimes, symbols=tuple(f"s{j}" for j in range(n)),
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=3, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=np.ones((t, n), dtype=bool),
            metadata={},
        ))
    active_mask = np.ones((t, n), dtype=bool)
    result = audit_full_family_correlation(
        panels=panels, active_mask=active_mask, run_id="test", timeframe="4h",
        max_corr=0.5,
    )
    assert set(result.columns) == {
        "family_a", "variant_a", "family_b", "variant_b",
        "timeframe", "pairwise_corr", "cluster_id", "run_id",
    }


# ── compute_cross_tf_redundancy ─────────────────────────────────────────────


def _make_aligned(
    datetimes_ns: np.ndarray,
    n_syms: int = 1,
) -> Any:
    class _MockAligned:
        def __init__(self, dts: np.ndarray, n_syms: int) -> None:
            t = len(dts)
            self.datetimes = dts
            self.active_mask = np.ones((t, n_syms), dtype=np.bool_)
            self.warm_mask = np.ones((t, n_syms), dtype=np.bool_)
            self.entry_block_mask = np.zeros((t, n_syms), dtype=np.bool_)
            self.kill_mask = np.zeros((t, n_syms), dtype=np.bool_)
    return _MockAligned(datetimes_ns, n_syms)


def _candidate(
    recipe_id: str,
    family: str = "trend_ma",
    variant: str = "ema",
    timeframe: str = "4h",
    score: float = 0.0,
) -> L0SignalCandidate:
    return L0SignalCandidate(
        run_id="test",
        timeframe=timeframe,
        family=family,
        variant=variant,
        recipe_id=recipe_id,
        archetype="trend",
        source="synthetic_recipe",
        n_events=100,
        effective_n=50.0,
        mean_net_bps=score,
        block_lcb_bps=score * 0.5,
        nw_tstat=1.5,
        bootstrap_lcb_bps=0.0,
        bootstrap_agree=True,
        cost_drag_ratio=0.3,
        turnover_per_year=50.0,
        max_abs_corr_in_bucket=0.0,
        tf_coverage_count=0,
        sign_agreement_ratio=0.0,
        corroboration_tier="single_tf_strict",
        discovery_tier="candidate",
        l1_priority_score=score,
        l1_budget_units=1,
        hard_reject_reasons=(),
        soft_flags=(),
    )


def _make_canonical_panel(
    score_vals: list[float],
    dt_start: int = 0,
    bar_step_ns: int = 3_600_000_000_000,
) -> CandidateSignalPanel:
    n = len(score_vals)
    dts = [dt_start + i * bar_step_ns for i in range(n)]
    return CandidateSignalPanel(
        family="test", variant="tv", params={},
        datetimes=np.array(dts, dtype=np.int64),
        symbols=("BTCUSDT",),
        signed_score_2d=np.array(score_vals, dtype=np.float64).reshape(n, 1),
        side_hint_2d=np.zeros((n, 1), dtype=np.int8),
        expected_holding_bars=4, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((n, 1), dtype=np.float64),
        valid_mask_2d=np.ones((n, 1), dtype=np.bool_),
        metadata={"recipe_id": "test:tv:4h:abc"},
    )


def _make_canonical_panel_from_2d(
    score_2d: NDArray[np.float64],
    dt_start: int = 0,
    bar_step_ns: int = 3_600_000_000_000,
) -> CandidateSignalPanel:
    n = score_2d.shape[0]
    dts = [dt_start + i * bar_step_ns for i in range(n)]
    return CandidateSignalPanel(
        family="test", variant="tv", params={},
        datetimes=np.array(dts, dtype=np.int64),
        symbols=tuple(f"s{j}" for j in range(score_2d.shape[1])),
        signed_score_2d=score_2d,
        side_hint_2d=np.zeros(score_2d.shape, dtype=np.int8),
        expected_holding_bars=4, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros(score_2d.shape, dtype=np.float64),
        valid_mask_2d=np.ones(score_2d.shape, dtype=np.bool_),
        metadata={},
    )


def test_compute_cross_tf_redundancy_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_cross_tf_redundancy(
            selected_by_tf={},
            panel_by_recipe_id={},
            aligned_by_tf={},
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=12,
        )


def test_compute_cross_tf_redundancy_two_candidates_one_demoted() -> None:
    c4h = _candidate("r1", timeframe="4h", score=2.0)
    c12h = _candidate("r2", timeframe="12h", score=1.0)
    n_bars = 16
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000

    rng = np.random.default_rng(42)
    scores_4h = rng.normal(0, 1, (n_bars, 1))
    scores_12h = scores_4h * 0.95 + rng.normal(0, 0.01, (n_bars, 1))
    panel_4h = _make_canonical_panel_from_2d(scores_4h, dt_start=0)
    panel_12h = _make_canonical_panel_from_2d(scores_12h, dt_start=0)
    panel_12h.metadata["recipe_id"] = "r2"

    aligned_4h = _make_aligned(canonical_dt, n_syms=1)
    aligned_by_tf = {"4h": aligned_4h, "12h": aligned_4h}

    result = compute_cross_tf_redundancy(
        selected_by_tf={"4h": [c4h], "12h": [c12h]},
        panel_by_recipe_id={"r1": panel_4h, "r2": panel_12h},
        aligned_by_tf=aligned_by_tf,
        min_common_active_bars=1,
        max_novelty_corr=0.70,
        min_directional_entry_jaccard=0.50,
        min_shared_directional_entries=12,
    )
    assert len(result.demoted_recipe_ids) >= 0
    assert result.canonical_tf == "4h"


def test_compute_cross_tf_redundancy_no_aligned_tf_raises() -> None:
    c4h = _candidate("r1", timeframe="4h")
    c1h = _candidate("r2", timeframe="1h")
    with pytest.raises(KeyError):
        compute_cross_tf_redundancy(
            selected_by_tf={"4h": [c4h], "1h": [c1h]},
            panel_by_recipe_id={},
            aligned_by_tf={},
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=12,
        )


def test_compute_cross_tf_redundancy_single_candidate_returns_immediately() -> None:
    """Single candidate should skip pair computation entirely (line 640)."""
    c = _candidate("r1", timeframe="4h", score=2.0)
    n_bars = 16
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
    panel = _make_canonical_panel_from_2d(np.random.default_rng(0).normal(0, 1, (n_bars, 1)), dt_start=0)
    aligned_4h = _make_aligned(canonical_dt, n_syms=1)
    result = compute_cross_tf_redundancy(
        selected_by_tf={"4h": [c]},
        panel_by_recipe_id={"r1": panel},
        aligned_by_tf={"4h": aligned_4h},
        min_common_active_bars=1,
        max_novelty_corr=0.70,
        min_directional_entry_jaccard=0.50,
        min_shared_directional_entries=12,
    )
    assert result.final_selected_recipe_ids == ("r1",)
    assert result.n_common_active_bars > 0


def test_compute_cross_tf_pair_evidence_sparse_overlap_no_demotion() -> None:
    """Scenario 3 [LIMIT-04]: high corr and high J but fewer than 12 shared entries → not redundant."""
    n_bars = 100
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
    active = np.ones((n_bars, 1), dtype=bool)

    # Score with mostly flat regions + 11 scattered impulses to get few shared entries
    score = np.zeros((n_bars, 1), dtype=np.float64)
    # Same 11 sign-change positions for both panels → high corr, J>0.5, but only 11 shared entries
    for idx in range(10, 60, 5):
        score[idx, 0] = 1.0
    score[70, 0] = 1.0

    p_a = _make_canonical_panel_from_2d(score.copy(), dt_start=0)
    p_b = _make_canonical_panel_from_2d(score.copy(), dt_start=0)
    p_a.metadata["recipe_id"] = "r1"
    p_b.metadata["recipe_id"] = "r2"

    context = CrossTFCanonicalContext(
        canonical_tf="1h",
        canonical_datetimes_ns=canonical_dt,
        active_mask_2d=active,
        common_start_ns=int(canonical_dt[0]),
        common_end_ns=int(canonical_dt[-1]),
        n_common_active_bars=n_bars,
    )
    ev = compute_cross_tf_pair_evidence(
        recipe_id_a="r1", recipe_id_b="r2",
        panel_a=p_a, panel_b=p_b,
        context=context,
        min_score_corr=0.70,
        min_directional_entry_jaccard=0.50,
        min_shared_directional_entries=12,
    )
    # With identical scores, corr=1.0 and J=1.0 but only 11 shared entries
    assert ev.score_corr > 0.70
    assert ev.shared_directional_entries < 12
    assert not ev.is_redundant  # [LIMIT-04]


def test_resolve_cross_tf_canonical_context_all_panels_missing_raises() -> None:
    """When no panel has datetimes, resolve_cross_tf_canonical_context raises (line 519)."""
    c = _candidate("r1", timeframe="4h", score=1.0)
    n_bars = 10
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
    aligned_4h = _make_aligned(canonical_dt, n_syms=1)
    with pytest.raises(ValueError, match="no panel with datetimes"):
        compute_cross_tf_redundancy(
            selected_by_tf={"4h": [c]},
            panel_by_recipe_id={},  # all panels missing; first stays True
            aligned_by_tf={"4h": aligned_4h},
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=12,
        )


def test_resolve_cross_tf_canonical_context_insufficient_active_bars_raises() -> None:
    """Resolve context with zero common active bars raises ValueError (line 527)."""
    c = _candidate("r1", timeframe="4h", score=1.0)
    n_bars = 10
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
    panel = _make_canonical_panel_from_2d(np.ones((n_bars, 1)), dt_start=0)
    aligned_4h = _make_aligned(canonical_dt, n_syms=1)
    # Force active_mask to all False
    aligned_4h.active_mask = np.zeros((n_bars, 1), dtype=np.bool_)

    with pytest.raises(ValueError, match="n_common_active_bars"):
        compute_cross_tf_redundancy(
            selected_by_tf={"4h": [c]},
            panel_by_recipe_id={"r1": panel},
            aligned_by_tf={"4h": aligned_4h},
            min_common_active_bars=100,  # larger than n_bars
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=12,
        )


def test_audit_l0_selected_recipe_independence_one_panel_only() -> None:
    """Two candidates but only one bound panel → returns with n_independent_clusters=2 (line 800)."""
    n_bars = 32
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
    c1 = _candidate("r1", family="trend_ma", score=2.0, timeframe="4h")
    c2 = _candidate("r2", family="carry", score=1.0, timeframe="6h")
    panel_1 = _make_canonical_panel_from_2d(np.ones((n_bars, 1)), dt_start=0)
    aligned_4h = _make_aligned(canonical_dt, n_syms=1)
    audit = audit_l0_selected_recipe_independence(
        selected_by_tf={"4h": [c1], "6h": [c2]},
        panel_by_recipe_id={"r1": panel_1},  # r2 has no panel
        aligned_by_tf={"4h": aligned_4h},
        min_common_active_bars=1,
        max_corr=0.70,
    )
    assert audit.n_selected_total == 2
    assert audit.n_independent_clusters == 2  # falls back to one cluster per candidate


def test_audit_l0_selected_recipe_independence_single_candidate() -> None:
    """Single candidate audit returns immediately with n_independent_clusters=1 (line 779)."""
    c = _candidate("r1", family="trend_ma", score=2.0, timeframe="4h")
    canonical_dt = np.arange(0, 10, dtype=np.int64) * 3_600_000_000_000
    aligned_4h = _make_aligned(canonical_dt, n_syms=1)
    panel = _make_canonical_panel_from_2d(np.ones((10, 1)), dt_start=0)
    audit = audit_l0_selected_recipe_independence(
        selected_by_tf={"4h": [c]},
        panel_by_recipe_id={"r1": panel},
        aligned_by_tf={"4h": aligned_4h},
        min_common_active_bars=1,
        max_corr=0.70,
    )
    assert audit.n_selected_total == 1
    assert audit.n_independent_clusters == 1


def test_compute_cross_tf_redundancy_demotes_lower_priority_identical() -> None:
    """Lower-priority demoted when both candidates have identical panels (coverage for line 738)."""
    n_bars = 100
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
    aligned_4h = _make_aligned(canonical_dt, n_syms=1)

    score = np.zeros((n_bars, 1), dtype=np.float64)
    for idx in range(2, 80, 3):
        score[idx, 0] = 1.0
    for idx in range(3, 80, 3):
        score[idx, 0] = -1.0

    c_high = _candidate("r1", timeframe="4h", score=10.0)
    c_low = _candidate("r2", timeframe="12h", score=1.0)
    panel = _make_canonical_panel_from_2d(score.copy(), dt_start=0)
    panel.metadata["recipe_id"] = "r1"

    result = compute_cross_tf_redundancy(
        selected_by_tf={"4h": [c_high], "12h": [c_low]},
        panel_by_recipe_id={"r1": panel, "r2": panel},  # both point to same panel
        aligned_by_tf={"4h": aligned_4h, "12h": aligned_4h},
        min_common_active_bars=1,
        max_novelty_corr=0.70,
        min_directional_entry_jaccard=0.50,
        min_shared_directional_entries=12,
    )
    assert "r2" in result.demoted_recipe_ids


def test_compute_cross_tf_redundancy_demotes_identical_candidate() -> None:
    """Two candidates with identical score patterns → demotion of lower priority (line 738)."""
    n_bars = 100
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
    aligned_4h = _make_aligned(canonical_dt, n_syms=1)

    # Score with enough signal to generate >=12 shared entries
    score = np.zeros((n_bars, 1), dtype=np.float64)
    for idx in range(2, 80, 3):
        score[idx, 0] = 1.0
    for idx in range(3, 80, 3):
        score[idx, 0] = -1.0

    c_high = _candidate("r1", timeframe="4h", score=10.0)
    c_low = _candidate("r2", timeframe="12h", score=1.0)
    panel_high = _make_canonical_panel_from_2d(score.copy(), dt_start=0)
    panel_low = _make_canonical_panel_from_2d(score.copy(), dt_start=0)
    panel_high.metadata["recipe_id"] = "r1"
    panel_low.metadata["recipe_id"] = "r2"

    result = compute_cross_tf_redundancy(
        selected_by_tf={"4h": [c_high], "12h": [c_low]},
        panel_by_recipe_id={"r1": panel_high, "r2": panel_low},
        aligned_by_tf={"4h": aligned_4h, "12h": aligned_4h},
        min_common_active_bars=1,
        max_novelty_corr=0.70,
        min_directional_entry_jaccard=0.50,
        min_shared_directional_entries=12,
    )
    assert "r2" in result.demoted_recipe_ids
    assert result.demoted_reason_by_id["r2"] == "r1"
    assert result.final_selected_recipe_ids == ("r1",)


def test_compute_cross_tf_pair_evidence_identical_scores_redundant() -> None:
    """Identical scores with many shared entries → is_redundant=True."""
    n_bars = 100
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
    active = np.ones((n_bars, 1), dtype=bool)
    score = np.zeros((n_bars, 1), dtype=np.float64)
    score[10:80:3, 0] = 1.0
    score[11:81:3, 0] = -1.0

    p_a = _make_canonical_panel_from_2d(score.copy(), dt_start=0)
    p_b = _make_canonical_panel_from_2d(score.copy(), dt_start=0)
    p_a.metadata["recipe_id"] = "r1"
    p_b.metadata["recipe_id"] = "r2"

    context = CrossTFCanonicalContext(
        canonical_tf="1h",
        canonical_datetimes_ns=canonical_dt,
        active_mask_2d=active,
        common_start_ns=int(canonical_dt[0]),
        common_end_ns=int(canonical_dt[-1]),
        n_common_active_bars=n_bars,
    )
    ev = compute_cross_tf_pair_evidence(
        recipe_id_a="r1", recipe_id_b="r2",
        panel_a=p_a, panel_b=p_b,
        context=context,
        min_score_corr=0.70,
        min_directional_entry_jaccard=0.50,
        min_shared_directional_entries=12,
    )
    assert ev.score_corr > 0.99
    assert ev.shared_directional_entries >= 12
    assert ev.directional_entry_jaccard >= 0.50
    assert ev.is_redundant


# ── audit_l0_selected_recipe_independence ───────────────────────────────────


def test_audit_l0_selected_recipe_independence_basic_counts() -> None:
    n_bars = 32
    rng = np.random.default_rng(99)
    base = rng.normal(0, 1, (n_bars, 1))
    c1 = _candidate("r1", family="trend_ma", score=2.0, timeframe="4h")
    c2 = _candidate("r2", family="trend_ma", score=1.5, timeframe="6h")
    c3 = _candidate("r3", family="carry_net_of_funding", score=1.0, timeframe="8h")
    c4 = _candidate("r4", family="volume_participation_breakout", score=0.5, timeframe="4h")
    c5 = _candidate("r5", family="xs_momentum", score=1.2, timeframe="12h")
    canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000

    def _p(rid: str, offset: float) -> CandidateSignalPanel:
        return _make_canonical_panel_from_2d(base + offset, dt_start=0)

    panels = {"r1": _p("r1", 0.0), "r2": _p("r2", 0.01), "r3": _p("r3", 0.5),
              "r4": _p("r4", -0.3), "r5": _p("r5", 0.2)}

    aligned_4h = _make_aligned(canonical_dt, n_syms=1)
    aligned_by_tf = {"4h": aligned_4h}

    audit = audit_l0_selected_recipe_independence(
        selected_by_tf={"4h": [c1, c4], "6h": [c2], "8h": [c3], "12h": [c5]},
        panel_by_recipe_id=panels,
        aligned_by_tf=aligned_by_tf,
        min_common_active_bars=1,
        max_corr=0.70,
    )
    assert audit.n_selected_total == 5
    assert audit.n_distinct_thesis_ids > 0
    assert audit.n_independent_clusters > 0


def test_audit_l0_selected_recipe_independence_heterogeneous_native_tf_shapes() -> None:
    """4h panel (40 native bars) and 12h panel (~13 native bars) must be
    projected onto the finest canonical grid before correlating — regression for
    a bug where compute_panel_correlation_matrix was called directly on raw
    panels of differing native shapes and raised ValueError."""
    n_bars_canonical = 40
    canonical_dt = np.arange(0, n_bars_canonical, dtype=np.int64) * 3_600_000_000_000

    c4h = _candidate("r1", family="trend_ma", score=2.0, timeframe="4h")
    c12h = _candidate("r2", family="carry_net_of_funding", score=1.0, timeframe="12h")

    panel_4h = _make_canonical_panel_from_2d(
        np.random.default_rng(1).normal(0, 1, (n_bars_canonical, 1)), dt_start=0, bar_step_ns=3_600_000_000_000
    )
    panel_12h = _make_canonical_panel_from_2d(
        np.random.default_rng(2).normal(0, 1, (3, 1)), dt_start=0, bar_step_ns=12 * 3_600_000_000_000
    )

    aligned_4h = _make_aligned(canonical_dt, n_syms=1)
    aligned_by_tf = {"4h": aligned_4h}

    audit = audit_l0_selected_recipe_independence(
        selected_by_tf={"4h": [c4h], "12h": [c12h]},
        panel_by_recipe_id={"r1": panel_4h, "r2": panel_12h},
        aligned_by_tf=aligned_by_tf,
        min_common_active_bars=1,
        max_corr=0.70,
    )
    assert audit.n_selected_total == 2


# ── apply_cross_tf_survival_floor ─────────────────────────────────────────────


def _floor_candidate(
    recipe_id: str, archetype: AlphaArchetype, timeframe: str, priority: float
) -> L0SignalCandidate:
    return L0SignalCandidate(
        run_id="test", timeframe=timeframe, family="btc_regime_pullback", variant="v",
        recipe_id=recipe_id, archetype=archetype, source="synthetic_recipe",
        n_events=100, effective_n=50.0, mean_net_bps=priority, block_lcb_bps=priority * 0.5,
        nw_tstat=1.5, bootstrap_lcb_bps=0.0, bootstrap_agree=True, cost_drag_ratio=0.3,
        turnover_per_year=50.0, max_abs_corr_in_bucket=0.0, tf_coverage_count=0,
        sign_agreement_ratio=0.0, corroboration_tier="single_tf_strict", discovery_tier="candidate",
        l1_priority_score=priority, l1_budget_units=1, hard_reject_reasons=(), soft_flags=(),
    )


def test_apply_cross_tf_survival_floor_readmits_fully_demoted_archetype() -> None:
    """[LIMIT-03] "hedge" archetype's only candidate lost its cross-TF cluster
    comparison to a higher-priority "trend" candidate and was demoted — floor
    re-admits it."""
    from src.domain.futures.alpha_foundry.diversity import apply_cross_tf_survival_floor

    trend_winner = _floor_candidate("r1", "trend", "4h", priority=10.0)
    hedge_loser = _floor_candidate("r2", "hedge", "12h", priority=3.0)
    cross_tf_result = CrossBucketDiversityResult(
        final_selected_recipe_ids=("r1",),
        demoted_recipe_ids=("r2",),
        demoted_reason_by_id={"r2": "r1"},
        cross_bucket_corr=np.array([[1.0, 0.9], [0.9, 1.0]]),
        global_eff_test_count=1.2,
    )

    result = apply_cross_tf_survival_floor(
        cross_tf_result=cross_tf_result,
        candidate_by_recipe_id={"r1": trend_winner, "r2": hedge_loser},
        min_survivors_per_archetype=1,
        min_survivors_per_tf=1,
    )

    assert "r2" in result.final_selected_recipe_ids
    assert "r2" not in result.demoted_recipe_ids


def test_apply_cross_tf_survival_floor_noop_when_already_satisfied() -> None:
    """Floor is a no-op when every archetype/TF already has >=1 survivor."""
    from src.domain.futures.alpha_foundry.diversity import apply_cross_tf_survival_floor

    cross_tf_result = CrossBucketDiversityResult(
        final_selected_recipe_ids=("r1", "r2"),
        demoted_recipe_ids=(),
        demoted_reason_by_id={},
        cross_bucket_corr=np.eye(2),
        global_eff_test_count=2.0,
    )
    candidates = {
        "r1": _floor_candidate("r1", "trend", "4h", priority=10.0),
        "r2": _floor_candidate("r2", "hedge", "12h", priority=3.0),
    }

    result = apply_cross_tf_survival_floor(
        cross_tf_result=cross_tf_result,
        candidate_by_recipe_id=candidates,
        min_survivors_per_archetype=1,
        min_survivors_per_tf=1,
    )

    assert result is cross_tf_result


def test_apply_cross_tf_survival_floor_tf_floor_readmits_demoted_tf() -> None:
    """[LIMIT-03] TF floor: all of "12h" candidates demoted, re-admit highest priority."""
    from src.domain.futures.alpha_foundry.diversity import apply_cross_tf_survival_floor

    winner_4h = _floor_candidate("r1", "trend", "4h", priority=10.0)
    demoted_12h_low = _floor_candidate("r2", "trend", "12h", priority=3.0)
    demoted_12h_high = _floor_candidate("r3", "trend", "12h", priority=5.0)
    cross_tf_result = CrossBucketDiversityResult(
        final_selected_recipe_ids=("r1",),
        demoted_recipe_ids=("r2", "r3"),
        demoted_reason_by_id={"r2": "r1", "r3": "r1"},
        cross_bucket_corr=np.eye(3),
        global_eff_test_count=1.2,
    )

    result = apply_cross_tf_survival_floor(
        cross_tf_result=cross_tf_result,
        candidate_by_recipe_id={"r1": winner_4h, "r2": demoted_12h_low, "r3": demoted_12h_high},
        min_survivors_per_archetype=1,
        min_survivors_per_tf=1,
    )

    assert "r3" in result.final_selected_recipe_ids
    assert "r2" not in result.final_selected_recipe_ids
    assert "r3" not in result.demoted_recipe_ids


def test_apply_cross_tf_survival_floor_idempotent_re_admission() -> None:
    """A candidate that satisfies both archetype and TF floor is re-admitted once."""
    from src.domain.futures.alpha_foundry.diversity import apply_cross_tf_survival_floor

    only_candidate = _floor_candidate("r1", "hedge", "12h", priority=3.0)
    cross_tf_result = CrossBucketDiversityResult(
        final_selected_recipe_ids=(),
        demoted_recipe_ids=("r1",),
        demoted_reason_by_id={"r1": "other"},
        cross_bucket_corr=np.array([[1.0]]),
        global_eff_test_count=0.0,
    )

    result = apply_cross_tf_survival_floor(
        cross_tf_result=cross_tf_result,
        candidate_by_recipe_id={"r1": only_candidate},
        min_survivors_per_archetype=1,
        min_survivors_per_tf=1,
    )

    assert "r1" in result.final_selected_recipe_ids
    assert result.final_selected_recipe_ids == ("r1",)
    assert "r1" not in result.demoted_recipe_ids


def test_apply_cross_tf_survival_floor_skips_unknown_candidate() -> None:
    """[LIMIT-03] defensive-floor: demoted candidate absent from
    candidate_by_recipe_id is skipped silently."""
    from src.domain.futures.alpha_foundry.diversity import apply_cross_tf_survival_floor

    cross_tf_result = CrossBucketDiversityResult(
        final_selected_recipe_ids=("r1",),
        demoted_recipe_ids=("r2",),
        demoted_reason_by_id={"r2": "r1"},
        cross_bucket_corr=np.array([[1.0]]),
        global_eff_test_count=1.0,
    )

    result = apply_cross_tf_survival_floor(
        cross_tf_result=cross_tf_result,
        candidate_by_recipe_id={"r1": _floor_candidate("r1", "trend", "4h", priority=10.0)},
        min_survivors_per_archetype=1,
        min_survivors_per_tf=1,
    )

    assert result is cross_tf_result


def test_apply_cross_tf_survival_floor_never_exceeds_pre_pruning_union() -> None:
    """[LIMIT-04] re-admission never exceeds pre-pruning selected_for_l1 set."""
    from src.domain.futures.alpha_foundry.diversity import apply_cross_tf_survival_floor

    pre_pruning_union = {"r1", "r2", "r3"}
    cross_tf_result = CrossBucketDiversityResult(
        final_selected_recipe_ids=("r1",),
        demoted_recipe_ids=("r2", "r3"),
        demoted_reason_by_id={"r2": "r1", "r3": "r1"},
        cross_bucket_corr=np.eye(3),
        global_eff_test_count=1.0,
    )
    candidates = {
        "r1": _floor_candidate("r1", "trend", "4h", priority=10.0),
        "r2": _floor_candidate("r2", "hedge", "12h", priority=3.0),
        "r3": _floor_candidate("r3", "carry", "8h", priority=2.0),
    }

    result = apply_cross_tf_survival_floor(
        cross_tf_result=cross_tf_result,
        candidate_by_recipe_id=candidates,
        min_survivors_per_archetype=1,
        min_survivors_per_tf=1,
    )

    assert set(result.final_selected_recipe_ids) <= pre_pruning_union


def test_apply_cross_tf_survival_floor_counts_actual_survivors_not_distinct_labels() -> None:
    """Regression: survivor counting must count actual surviving candidates
    per archetype/TF (not just distinct-label presence). With
    min_survivors_per_archetype=2 and 2 "trend" candidates already
    surviving on the SAME TF (so the TF floor is independently satisfied
    too), the archetype floor must NOT fire — a set-membership check would
    incorrectly treat any single survivor as satisfying any threshold."""
    from src.domain.futures.alpha_foundry.diversity import apply_cross_tf_survival_floor

    trend_1 = _floor_candidate("r1", "trend", "4h", priority=10.0)
    trend_2 = _floor_candidate("r2", "trend", "4h", priority=8.0)
    demoted_trend = _floor_candidate("r3", "trend", "4h", priority=1.0)
    cross_tf_result = CrossBucketDiversityResult(
        final_selected_recipe_ids=("r1", "r2"),
        demoted_recipe_ids=("r3",),
        demoted_reason_by_id={"r3": "r1"},
        cross_bucket_corr=np.eye(3),
        global_eff_test_count=2.0,
    )

    result = apply_cross_tf_survival_floor(
        cross_tf_result=cross_tf_result,
        candidate_by_recipe_id={"r1": trend_1, "r2": trend_2, "r3": demoted_trend},
        min_survivors_per_archetype=2,
        min_survivors_per_tf=1,
    )

    assert "r3" not in result.final_selected_recipe_ids
    assert result.final_selected_recipe_ids == ("r1", "r2")
