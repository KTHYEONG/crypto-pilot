from __future__ import annotations

import logging
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
    CrossTFSharedContext,
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
    resolve_cross_tf_shared_context,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel


@pytest.fixture(autouse=True)
def _restore_logger_classes() -> None:
    """Prevent CategorizedLogger mutation from leaking to downstream tests."""
    from src.core.utils.utils import CategorizedLogger

    for name in ("opt_main_futures", "src.domain.futures.strategy_runtime.bridge"):
        lg = logging.getLogger(name)
        if isinstance(lg, CategorizedLogger):
            lg.__class__ = logging.Logger
    yield
    for name in ("opt_main_futures", "src.domain.futures.strategy_runtime.bridge"):
        lg = logging.getLogger(name)
        if isinstance(lg, CategorizedLogger):
            lg.__class__ = logging.Logger


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
        panels.append(
            CandidateSignalPanel(
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
            )
        )
    return panels


def test_family_correlation_audit_disabled_by_default() -> None:
    cfg = AlphaFoundryRuntimeConfig()
    assert cfg.enable_correlation_audit is False


def test_audit_full_family_correlation_matrix_symmetric() -> None:
    panels = _make_multi_family_panel_fixtures(n_families=4)
    active_mask = np.ones(panels[0].signed_score_2d.shape, dtype=bool)
    result = audit_full_family_correlation(
        panels=panels,
        active_mask=active_mask,
        run_id="test",
        timeframe="4h",
    )
    assert set(result.columns) == {
        "family_a",
        "variant_a",
        "family_b",
        "variant_b",
        "timeframe",
        "pairwise_corr",
        "cluster_id",
        "run_id",
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
            panels=(),
            active_mask=np.ones((10, 2), dtype=bool),
            run_id="test",
            timeframe="4h",
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
        panels.append(
            CandidateSignalPanel(
                family=f"fam_{i}",
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
            )
        )
    active_mask = np.ones((t, n), dtype=bool)
    result = audit_full_family_correlation(
        panels=panels,
        active_mask=active_mask,
        run_id="test",
        timeframe="4h",
        max_corr=0.5,
    )
    assert set(result.columns) == {
        "family_a",
        "variant_a",
        "family_b",
        "variant_b",
        "timeframe",
        "pairwise_corr",
        "cluster_id",
        "run_id",
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
        family="test",
        variant="tv",
        params={},
        datetimes=np.array(dts, dtype=np.int64),
        symbols=("BTCUSDT",),
        signed_score_2d=np.array(score_vals, dtype=np.float64).reshape(n, 1),
        side_hint_2d=np.zeros((n, 1), dtype=np.int8),
        expected_holding_bars=4,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
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
        family="test",
        variant="tv",
        params={},
        datetimes=np.array(dts, dtype=np.int64),
        symbols=tuple(f"s{j}" for j in range(score_2d.shape[1])),
        signed_score_2d=score_2d,
        side_hint_2d=np.zeros(score_2d.shape, dtype=np.int8),
        expected_holding_bars=4,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
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
        recipe_id_a="r1",
        recipe_id_b="r2",
        panel_a=p_a,
        panel_b=p_b,
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
        recipe_id_a="r1",
        recipe_id_b="r2",
        panel_a=p_a,
        panel_b=p_b,
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

    panels = {"r1": _p("r1", 0.0), "r2": _p("r2", 0.01), "r3": _p("r3", 0.5), "r4": _p("r4", -0.3), "r5": _p("r5", 0.2)}

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


def _floor_candidate(recipe_id: str, archetype: AlphaArchetype, timeframe: str, priority: float) -> L0SignalCandidate:
    return L0SignalCandidate(
        run_id="test",
        timeframe=timeframe,
        family="btc_regime_pullback",
        variant="v",
        recipe_id=recipe_id,
        archetype=archetype,
        source="synthetic_recipe",
        n_events=100,
        effective_n=50.0,
        mean_net_bps=priority,
        block_lcb_bps=priority * 0.5,
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
        l1_priority_score=priority,
        l1_budget_units=1,
        hard_reject_reasons=(),
        soft_flags=(),
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


def test_compute_cross_tf_redundancy_handles_non_nested_tf_calendar_ranges() -> None:
    """[ADR_20260712_L0_CROSS_TF_CANONICAL_CALENDAR_CONTAINMENT_FIX] Regression:
    previously raised when a non-canonical TF's panel calendar range extended
    beyond the (dynamically finest-selected) canonical TF's own range."""
    bar_4h_ns = 4 * 3_600_000_000_000
    bar_1h_ns = 3_600_000_000_000
    dt_a = np.arange(0, 100, dtype=np.int64) * bar_4h_ns  # hours [0, 396] step 4, 100 bars
    dt_b = np.arange(30, 70, dtype=np.int64) * bar_1h_ns  # hours [30, 69] step 1, 40 bars

    aligned_by_tf = {
        "4h": _make_aligned(dt_a, n_syms=1),
        "1h": _make_aligned(dt_b, n_syms=1),
    }
    candidate_a = _candidate("a", timeframe="4h", score=1.0)
    candidate_b = _candidate("b", timeframe="1h", score=1.0)

    panel_a = _make_canonical_panel_from_2d(
        np.ones((len(dt_a), 1), dtype=np.float64),
        dt_start=0,
        bar_step_ns=bar_4h_ns,
    )
    panel_b = _make_canonical_panel_from_2d(
        np.ones((len(dt_b), 1), dtype=np.float64),
        dt_start=int(dt_b[0]),
        bar_step_ns=bar_1h_ns,
    )

    result = compute_cross_tf_redundancy(
        selected_by_tf={"4h": (candidate_a,), "1h": (candidate_b,)},
        panel_by_recipe_id={"a": panel_a, "b": panel_b},
        aligned_by_tf=aligned_by_tf,
        min_common_active_bars=10,
        max_novelty_corr=0.70,
        min_directional_entry_jaccard=0.50,
        min_shared_directional_entries=1,
    )

    assert set(result.final_selected_recipe_ids) | set(result.demoted_recipe_ids) == {"a", "b"}


# ── CrossTFSharedContext / resolve_cross_tf_shared_context ─────────────────


def _build_four_recipe_fixture() -> tuple[
    dict[str, list[L0SignalCandidate]],
    dict[str, CandidateSignalPanel],
    dict[str, Any],
]:
    """4 recipes across 2 TFs (4h, 12h), 2 each, with valid panels."""
    bar_4h_ns = 4 * 3_600_000_000_000
    dt_4h = np.arange(0, 40, dtype=np.int64) * bar_4h_ns
    dt_12h = np.arange(0, 13, dtype=np.int64) * (12 * 3_600_000_000_000)

    rng = np.random.default_rng(42)
    p1 = _make_canonical_panel_from_2d(rng.normal(0, 1, (40, 1)), dt_start=0)
    p2 = _make_canonical_panel_from_2d(rng.normal(0, 1, (40, 1)), dt_start=0)
    p3 = _make_canonical_panel_from_2d(rng.normal(0, 1, (13, 1)), dt_start=0)
    p4 = _make_canonical_panel_from_2d(rng.normal(0, 1, (13, 1)), dt_start=0)

    c1 = _candidate("r1", timeframe="4h", score=2.0)
    c2 = _candidate("r2", timeframe="4h", score=1.5)
    c3 = _candidate("r3", timeframe="12h", score=1.0)
    c4 = _candidate("r4", timeframe="12h", score=0.5)

    aligned_4h = _make_aligned(dt_4h, n_syms=1)
    aligned_12h = _make_aligned(dt_12h, n_syms=1)

    return (
        {"4h": [c1, c2], "12h": [c3, c4]},
        {"r1": p1, "r2": p2, "r3": p3, "r4": p4},
        {"4h": aligned_4h, "12h": aligned_12h},
    )


class TestCrossTFSharedContext:
    def test_resolve_cross_tf_shared_context_matches_inline_computation(self) -> None:
        """Scenario 1: shared_context.corr matches inline corr from compute_cross_tf_redundancy."""
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()

        shared = resolve_cross_tf_shared_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
        )
        result = compute_cross_tf_redundancy(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
        )

        assert shared.corr.shape == result.cross_bucket_corr.shape
        np.testing.assert_allclose(shared.corr, result.cross_bucket_corr, atol=1e-10)

    def test_resolve_cross_tf_shared_context_raises_when_memory_budget_exceeded(
        self,
        mocker: Any,
    ) -> None:
        """Scenario 2 [LIMIT-05]: memory budget exceeded raises ValueError."""
        mocker.patch(
            "src.domain.futures.alpha_foundry.diversity.admit_memory_stage",
            return_value=False,
        )
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()

        with pytest.raises(ValueError, match="memory_budget"):
            resolve_cross_tf_shared_context(
                selected_by_tf=selected_by_tf,
                panel_by_recipe_id=panel_by_recipe_id,
                aligned_by_tf=aligned_by_tf,
                min_common_active_bars=1,
            )

    def test_resolve_cross_tf_shared_context_propagates_min_common_active_bars_valueerror(
        self,
    ) -> None:
        """Scenario 3: ValueError propagation from resolve_cross_tf_canonical_context."""
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()

        with pytest.raises(ValueError, match="n_common_active_bars"):
            resolve_cross_tf_shared_context(
                selected_by_tf=selected_by_tf,
                panel_by_recipe_id=panel_by_recipe_id,
                aligned_by_tf=aligned_by_tf,
                min_common_active_bars=10_000,  # impossible
            )


class TestCrossTFPairEvidencePrecomputed:
    def test_compute_cross_tf_pair_evidence_with_precomputed_inputs_matches_self_computed(
        self,
    ) -> None:
        """Scenario 1: precomputed path yields byte-identical results to self-computed."""
        n_bars = 100
        canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
        active = np.ones((n_bars, 1), dtype=bool)
        rng = np.random.default_rng(42)
        score_a = rng.normal(0, 1, (n_bars, 1))
        score_b = score_a * 0.9 + rng.normal(0, 0.1, (n_bars, 1))

        p_a = _make_canonical_panel_from_2d(score_a, dt_start=0)
        p_b = _make_canonical_panel_from_2d(score_b, dt_start=0)

        context = CrossTFCanonicalContext(
            canonical_tf="1h",
            canonical_datetimes_ns=canonical_dt,
            active_mask_2d=active,
            common_start_ns=int(canonical_dt[0]),
            common_end_ns=int(canonical_dt[-1]),
            n_common_active_bars=n_bars,
        )

        # Self-computed
        ev_self = compute_cross_tf_pair_evidence(
            recipe_id_a="a",
            recipe_id_b="b",
            panel_a=p_a,
            panel_b=p_b,
            context=context,
            min_score_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
        )

        # Precomputed: build projections and side/entry manually
        from src.domain.futures.alpha_foundry.multi_tf_fusion import project_signal_to_canonical_grid

        proj_a = project_signal_to_canonical_grid(panel=p_a, canonical_datetimes=canonical_dt, causal_lag_bars=1)
        proj_b = project_signal_to_canonical_grid(panel=p_b, canonical_datetimes=canonical_dt, causal_lag_bars=1)
        from src.domain.futures.alpha_foundry.diversity import _causal_projected_side_and_entry

        se_a = _causal_projected_side_and_entry(proj_a[0], proj_a[1], active)
        se_b = _causal_projected_side_and_entry(proj_b[0], proj_b[1], active)

        valid_ab = proj_a[1] & proj_b[1] & active
        flat_a = proj_a[0][valid_ab]
        flat_b = proj_b[0][valid_ab]
        c = float(np.corrcoef(flat_a, flat_b)[0, 1])
        precomputed_corr = c if np.isfinite(c) else 0.0

        ev_pre = compute_cross_tf_pair_evidence(
            recipe_id_a="a",
            recipe_id_b="b",
            panel_a=p_a,
            panel_b=p_b,
            context=context,
            min_score_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
            precomputed_proj_a=proj_a,
            precomputed_proj_b=proj_b,
            precomputed_side_entry_a=se_a,
            precomputed_side_entry_b=se_b,
            precomputed_score_corr=precomputed_corr,
        )

        assert ev_self.score_corr == pytest.approx(ev_pre.score_corr, rel=1e-5)
        assert ev_self.shared_directional_entries == ev_pre.shared_directional_entries
        assert ev_self.directional_entry_jaccard == pytest.approx(ev_pre.directional_entry_jaccard, rel=1e-5)
        assert ev_self.is_redundant == ev_pre.is_redundant


class TestCrossTFRedundancyDedup:
    def test_compute_cross_tf_redundancy_corr_matrix_is_symmetric_computed_once(
        self,
        mocker: Any,
    ) -> None:
        """Scenario 2 [LIMIT-03]: corr matrix uses i<j only, mirror to j,i.
        np.corrcoef calls should be N*(N-1)/2 = 6 for 4 recipes."""

        spy = mocker.spy(np, "corrcoef")
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()

        compute_cross_tf_redundancy(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
        )

        # 4*3/2 = 6 unique pairs + 4*3/2 = 6 from pair_evidence (jaccard uses
        # precomputed_score_corr, so no extra corrcoef calls) = 6 corrcoef calls total
        n_pairs = 4 * 3 // 2
        assert spy.call_count == n_pairs

    def test_compute_cross_tf_redundancy_projects_each_recipe_exactly_once(
        self,
        mocker: Any,
    ) -> None:
        """Scenario 2 [LIMIT-01][LIMIT-02]: proj + side/entry each called N times, not N + 2*C(N,2)."""
        from src.domain.futures.alpha_foundry import multi_tf_fusion as _fusion_mod
        from src.domain.futures.alpha_foundry.diversity import (
            _causal_projected_side_and_entry as _real_side_entry,
        )

        call_count = {"project": 0, "side_entry": 0}

        def _counting_project(
            *,
            panel: Any,
            canonical_datetimes: Any,
            causal_lag_bars: int,
        ) -> Any:
            call_count["project"] += 1
            return _fusion_mod.project_signal_to_canonical_grid(
                panel=panel,
                canonical_datetimes=canonical_datetimes,
                causal_lag_bars=causal_lag_bars,
            )

        def _counting_side_entry(
            score: Any,
            valid: Any,
            active: Any,
        ) -> Any:
            call_count["side_entry"] += 1
            return _real_side_entry(score, valid, active)

        mocker.patch(
            "src.domain.futures.alpha_foundry.diversity.project_signal_to_canonical_grid",
            side_effect=_counting_project,
        )
        mocker.patch(
            "src.domain.futures.alpha_foundry.diversity._causal_projected_side_and_entry",
            side_effect=_counting_side_entry,
        )

        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()

        compute_cross_tf_redundancy(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
        )

        assert call_count["project"] == 4
        assert call_count["side_entry"] == 4

    def test_compute_cross_tf_redundancy_uses_precomputed_shared_context(
        self,
    ) -> None:
        """Providing precomputed_shared_context should produce identical results."""
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()

        shared = resolve_cross_tf_shared_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
        )

        result_self = compute_cross_tf_redundancy(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
        )

        result_shared = compute_cross_tf_redundancy(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
            precomputed_shared_context=shared,
        )

        assert result_self.final_selected_recipe_ids == result_shared.final_selected_recipe_ids
        assert result_self.demoted_recipe_ids == result_shared.demoted_recipe_ids
        assert result_self.demoted_reason_by_id == result_shared.demoted_reason_by_id
        np.testing.assert_allclose(result_self.cross_bucket_corr, result_shared.cross_bucket_corr, atol=1e-10)

    def test_compute_cross_tf_redundancy_audit_uses_precomputed_shared_context(
        self,
    ) -> None:
        """audit_l0_selected_recipe_independence with precomputed_shared_context produces identical results."""
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()

        shared = resolve_cross_tf_shared_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
        )

        audit_self = audit_l0_selected_recipe_independence(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_corr=0.70,
        )

        audit_shared = audit_l0_selected_recipe_independence(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_corr=0.70,
            precomputed_shared_context=shared,
        )

        assert audit_self.n_selected_total == audit_shared.n_selected_total
        assert audit_self.n_independent_clusters == audit_shared.n_independent_clusters
        assert len(audit_self.cluster_members) == len(audit_shared.cluster_members)


class TestCrossTFBatchAcceleration:
    def test_batch_jaccard_matches_per_pair_on_synthetic(self) -> None:
        """OPT-2 batch jaccard path yields byte-identical results to per-pair fallback path."""
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()

        shared_with_batch = resolve_cross_tf_shared_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
        )
        # shared_with_batch has entry_pos_flat populated → triggers batch path
        result_batch = compute_cross_tf_redundancy(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
            precomputed_shared_context=shared_with_batch,
        )

        # Build a shared context WITHOUT entry_pos_flat → triggers per-pair fallback
        shared_no_batch = CrossTFSharedContext(
            canonical_context=shared_with_batch.canonical_context,
            proj_cache=shared_with_batch.proj_cache,
            side_entry_cache=shared_with_batch.side_entry_cache,
            corr=shared_with_batch.corr,
            recipe_order=shared_with_batch.recipe_order,
        )
        result_per_pair = compute_cross_tf_redundancy(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
            max_novelty_corr=0.70,
            min_directional_entry_jaccard=0.50,
            min_shared_directional_entries=1,
            precomputed_shared_context=shared_no_batch,
        )

        assert result_batch.final_selected_recipe_ids == result_per_pair.final_selected_recipe_ids
        assert result_batch.demoted_recipe_ids == result_per_pair.demoted_recipe_ids
        assert result_batch.global_eff_test_count == result_per_pair.global_eff_test_count
        np.testing.assert_allclose(result_batch.cross_bucket_corr, result_per_pair.cross_bucket_corr, atol=1e-12)

    def test_batch_corr_upper_triangle_symmetric(self) -> None:
        """OPT-1-a: corr matrix built with valid_stack precompute is symmetric."""
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()
        shared = resolve_cross_tf_shared_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
        )
        np.testing.assert_allclose(shared.corr, shared.corr.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(shared.corr), np.ones(shared.corr.shape[0]), atol=1e-12)

    def test_valid_stack_precompute_n1_no_error(self) -> None:
        """N=1 when valid_stack precompute should not raise."""
        rng = np.random.default_rng(42)
        dt = np.arange(0, 40, dtype=np.int64) * (4 * 3_600_000_000_000)
        panel = _make_canonical_panel_from_2d(rng.normal(0, 1, (40, 1)), dt_start=0)
        candidate = _candidate("single", timeframe="4h", score=1.0)
        aligned = _make_aligned(dt, n_syms=1)

        ctx = resolve_cross_tf_shared_context(
            selected_by_tf={"4h": [candidate]},
            panel_by_recipe_id={"single": panel},
            aligned_by_tf={"4h": aligned},
            min_common_active_bars=1,
        )
        assert ctx.corr.shape == (1, 1)
        assert ctx.corr[0, 0] == 1.0
        assert ctx.entry_pos_flat
        assert ctx.n_entries["single"] >= 0

    def test_corr_dtype_is_float64_after_optimization(self) -> None:
        """corr dtype float64 preserved (quant rule §2)."""
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()
        ctx = resolve_cross_tf_shared_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
        )
        assert ctx.corr.dtype == np.float64

    def test_batch_jaccard_self_consistency(self) -> None:
        """dir_shared[i,i] == n_entries[i]."""
        selected_by_tf, panel_by_recipe_id, aligned_by_tf = _build_four_recipe_fixture()
        ctx = resolve_cross_tf_shared_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=1,
        )
        recipe_ids = list(ctx.recipe_order)
        n = len(recipe_ids)
        pos_flat = np.array(
            [ctx.entry_pos_flat.get(rid, np.array([], dtype=np.int8)) for rid in recipe_ids],
            dtype=np.int16,
        )
        neg_flat = np.array(
            [ctx.entry_neg_flat.get(rid, np.array([], dtype=np.int8)) for rid in recipe_ids],
            dtype=np.int16,
        )
        dir_shared = pos_flat @ pos_flat.T + neg_flat @ neg_flat.T
        for i in range(n):
            rid = recipe_ids[i]
            expected = ctx.n_entries[rid]
            msg = f"dir_shared[{i},{i}]={dir_shared[i, i]} != n_entries[{rid}]={expected}"
            assert int(dir_shared[i, i]) == expected, msg


class TestProjectSignalToCanonicalGridFloat32:
    def test_project_signal_to_canonical_grid_returns_float32_dtype(self) -> None:
        """Scenario 2 [LIMIT-05]: projected array dtype is float32."""
        from src.domain.futures.alpha_foundry.multi_tf_fusion import project_signal_to_canonical_grid

        n_bars = 50
        canonical_dt = np.arange(0, n_bars, dtype=np.int64) * 3_600_000_000_000
        panel = _make_canonical_panel_from_2d(np.random.default_rng(0).normal(0, 1, (n_bars, 1)), dt_start=0)

        projected, valid = project_signal_to_canonical_grid(
            panel=panel,
            canonical_datetimes=canonical_dt,
            causal_lag_bars=1,
        )

        assert projected.dtype == np.float32
        assert valid.dtype == np.bool_


class TestBatchPairwiseCorr:
    """OPT-4 [ADR_20260712_L0_CROSS_TF_BATCH_CORRELATION]: batch correlation tests."""

    @staticmethod
    def _make_stack(
        n: int,
        n_bars: int,
        n_syms: int,
        seed: int = 42,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]:
        rng = np.random.default_rng(seed)
        P = rng.normal(0, 1, (n, n_bars, n_syms)).astype(np.float32)
        M = rng.random((n, n_bars, n_syms)) > 0.2
        A = np.ones((n_bars, n_syms), dtype=np.bool_)
        return P, M, A

    @staticmethod
    def _per_pair_corrcoef(
        projected_stack: NDArray[np.float32],
        mask_stack: NDArray[np.bool_],
        active_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        n = projected_stack.shape[0]
        corr = np.full((n, n), np.nan, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                mask = mask_stack[i] & mask_stack[j] & active_mask
                a = projected_stack[i][mask]
                b = projected_stack[j][mask]
                if len(a) < 2:
                    corr[i, j] = 0.0
                    corr[j, i] = 0.0
                else:
                    cv = float(np.corrcoef(a, b)[0, 1])
                    corr[i, j] = cv if np.isfinite(cv) else 0.0
                    corr[j, i] = cv if np.isfinite(cv) else 0.0
            corr[i, i] = 1.0
        return corr

    def test_batch_agrees_with_per_pair_corrcoef(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import _batch_pairwise_corr

        P, M, A = self._make_stack(4, 13, 1, seed=42)
        batch = _batch_pairwise_corr(projected_stack=P, mask_stack=M, active_mask=A)
        per_pair = self._per_pair_corrcoef(P, M, A)
        np.testing.assert_allclose(batch, per_pair, atol=1e-12)
        assert batch.dtype == np.float64
        assert batch.shape == (4, 4)

    def test_n1_returns_identity(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import _batch_pairwise_corr

        P, M, A = self._make_stack(1, 40, 1, seed=42)
        corr = _batch_pairwise_corr(projected_stack=P, mask_stack=M, active_mask=A)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == 1.0

    def test_disjoint_masks_yield_zero_corr(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import _batch_pairwise_corr

        P = np.random.default_rng(1).normal(0, 1, (2, 10, 1)).astype(np.float32)
        M = np.zeros((2, 10, 1), dtype=np.bool_)
        M[0, :5, 0] = True
        M[1, 5:, 0] = True
        A = np.ones((10, 1), dtype=np.bool_)
        corr = _batch_pairwise_corr(projected_stack=P, mask_stack=M, active_mask=A)
        assert corr[0, 1] == 0.0
        assert corr[1, 0] == 0.0

    def test_output_dtype_is_float64(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import _batch_pairwise_corr

        P, M, A = self._make_stack(3, 5, 2)
        corr = _batch_pairwise_corr(projected_stack=P, mask_stack=M, active_mask=A)
        assert corr.dtype == np.float64

    def test_symmetry_and_diagonal(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import _batch_pairwise_corr

        P, M, A = self._make_stack(4, 13, 1, seed=99)
        corr = _batch_pairwise_corr(projected_stack=P, mask_stack=M, active_mask=A)
        np.testing.assert_allclose(corr, corr.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(corr), np.ones(4), atol=1e-12)

    def test_empty_input_returns_empty_matrix(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import _batch_pairwise_corr

        P = np.empty((0, 0), dtype=np.float32)
        M = np.empty((0, 0), dtype=np.bool_)
        A = np.ones((0,), dtype=np.bool_)
        corr = _batch_pairwise_corr(projected_stack=P, mask_stack=M, active_mask=A)
        assert corr.shape == (0, 0)
        assert corr.dtype == np.float64

    def test_raises_on_1d_input(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import _batch_pairwise_corr

        with pytest.raises(ValueError, match="must be at least 2D"):
            _batch_pairwise_corr(
                projected_stack=np.array([1.0, 2.0], dtype=np.float32),
                mask_stack=np.array([True, False]),
                active_mask=np.ones(2, dtype=np.bool_),
            )

    def test_raises_on_shape_mismatch(self) -> None:
        from src.domain.futures.alpha_foundry.diversity import _batch_pairwise_corr

        P = np.zeros((2, 10, 3), dtype=np.float32)
        M = np.zeros((2, 10, 4), dtype=np.bool_)
        A = np.ones((10, 3), dtype=np.bool_)
        with pytest.raises(ValueError, match="shape mismatch"):
            _batch_pairwise_corr(projected_stack=P, mask_stack=M, active_mask=A)
