from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    CheapGateEvidence,
)
from src.domain.futures.alpha_foundry.diversity import (
    _paired_score_corr,
    audit_full_family_correlation,
    cluster_correlated_recipes,
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
    panels: list = []
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
