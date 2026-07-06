"""compute_diversification_ratio TDD tests (Scenarios A1-A4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import (
    compute_diversification_ratio,
)


class TestComputeDiversificationRatio:
    """Choueifaty-Coignard diversification ratio: DR = sum|w_i|*sigma_i / sqrt(w' Sigma w)."""

    def test_fully_correlated_returns_one(self) -> None:
        """A1: 완전상관 2자산 등가중 → DR≈1.0."""
        w = np.array([0.5, 0.5], dtype=np.float64)
        sigma_diag = np.array([0.02, 0.02], dtype=np.float64)
        sigma_mat = np.array([[0.0004, 0.0004], [0.0004, 0.0004]], dtype=np.float64)

        dr = compute_diversification_ratio(w, sigma_diag, sigma_mat)
        assert dr == pytest.approx(1.0, rel=1e-6)

    def test_uncorrelated_equal_weight_returns_sqrt_n(self) -> None:
        """A2: 완전무상관 2자산 등가중 → DR≈sqrt(2)."""
        w = np.array([0.5, 0.5], dtype=np.float64)
        sigma_diag = np.array([0.02, 0.02], dtype=np.float64)
        sigma_mat = np.diag(np.array([0.0004, 0.0004], dtype=np.float64))

        dr = compute_diversification_ratio(w, sigma_diag, sigma_mat)
        assert dr == pytest.approx(np.sqrt(2.0), rel=1e-6)

    def test_zero_weight_returns_one_degenerate(self) -> None:
        """A3: 전량 zero weight → degenerate fallback 1.0."""
        w = np.zeros(3, dtype=np.float64)
        sigma_diag = np.array([0.02, 0.02, 0.02], dtype=np.float64)
        sigma_mat = np.eye(3, dtype=np.float64) * 0.0004

        dr = compute_diversification_ratio(w, sigma_diag, sigma_mat)
        assert dr == pytest.approx(1.0, rel=1e-6)

    def test_hedged_book_exceeds_uncorrelated_baseline(self) -> None:
        """A4: 음의 상관 롱-온리(long-only) → DR > sqrt(N) (헤지 효과)."""
        w = np.array([0.5, 0.5], dtype=np.float64)
        sigma_diag = np.array([0.02, 0.02], dtype=np.float64)
        sigma_mat = np.array([[0.0004, -0.00038], [-0.00038, 0.0004]], dtype=np.float64)

        dr = compute_diversification_ratio(w, sigma_diag, sigma_mat)
        assert dr > np.sqrt(2.0)
