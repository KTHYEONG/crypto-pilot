from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.contracts import CheapGateEvidence
from src.domain.futures.alpha_foundry.diversity import (
    cluster_correlated_recipes,
    compute_panel_correlation_matrix,
    estimate_effective_test_count,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel


def _make_panel(score: np.ndarray) -> CandidateSignalPanel:
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
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
    )


class TestComputePanelCorrelationMatrix:
    def test_returns_square_matrix(self) -> None:
        panels = [_make_panel(np.random.randn(100, 5)) for _ in range(4)]
        corr = compute_panel_correlation_matrix(panels)
        assert corr.shape == (4, 4)

    def test_diagonal_is_one(self) -> None:
        panels = [_make_panel(np.random.randn(100, 5)) for _ in range(3)]
        corr = compute_panel_correlation_matrix(panels)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_raises_on_empty_panels(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_panel_correlation_matrix([])

    def test_replaces_non_finite_with_zero(self) -> None:
        panels = [_make_panel(np.full((100, 5), np.nan)) for _ in range(2)]
        corr = compute_panel_correlation_matrix(panels)
        assert np.all(np.isfinite(corr))


class TestClusterCorrelatedRecipes:
    def test_groups_highly_correlated(self) -> None:
        evs = tuple(
            CheapGateEvidence(
                recipe_id=f"r{i}",
                timeframe="4h",
                symbol_scope="symbol",
                n_events=100,
                effective_n=50.0,
                mean_net_bps=1.0,
                nw_tstat=2.0,
                block_lcb_bps=0.5,
                rank_ic=0.05,
                monotonic_bucket_score=0.0,
                regime_edges_bps={},
                cost_drag_ratio=0.3,
                turnover_per_year=100.0,
                novelty_corr_max=0.0,
                incremental_rank_ic=0.02,
                compute_cost_score=0.0,
                gate_passed=True,
                reject_reasons=(),
            )
            for i in range(4)
        )
        corr = np.array([
            [1.0, 0.9, 0.3, 0.2],
            [0.9, 1.0, 0.3, 0.2],
            [0.3, 0.3, 1.0, 0.1],
            [0.2, 0.2, 0.1, 1.0],
        ])
        clusters = cluster_correlated_recipes(
            evidences=evs, corr=corr, max_corr=0.8
        )
        assert len(clusters) > 0

    def test_raises_on_shape_mismatch(self) -> None:
        evs = tuple(
            CheapGateEvidence(
                recipe_id=f"r{i}",
                timeframe="4h",
                symbol_scope="symbol",
                n_events=100,
                effective_n=50.0,
                mean_net_bps=1.0,
                nw_tstat=2.0,
                block_lcb_bps=0.5,
                rank_ic=0.05,
                monotonic_bucket_score=0.0,
                regime_edges_bps={},
                cost_drag_ratio=0.3,
                turnover_per_year=100.0,
                novelty_corr_max=0.0,
                incremental_rank_ic=0.02,
                compute_cost_score=0.0,
                gate_passed=True,
                reject_reasons=(),
            )
            for i in range(3)
        )
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
