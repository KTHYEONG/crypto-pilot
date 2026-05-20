"""Phase 2: DSR entropy effective rank 테스트.

사양서 §4.4 — weighted_corr 기반 entropy effective rank.
"""

from __future__ import annotations

import numpy as np

from src.domain.futures.optimization.evaluator import calc_n_trials_eff_entropy


class TestDsrEntropy:
    """calc_n_trials_eff_entropy 함수 검증."""

    def test_identical_signatures_n_eff_close_to_one(self) -> None:
        """동일한 signature를 가진 trial 100개 → n_trials_eff ≈ 1.0."""
        identical = np.tile(np.array([0.05] * 11, dtype=np.float64), (100, 1))
        weights = np.ones(100, dtype=np.float64)

        n_eff = calc_n_trials_eff_entropy(identical, weights)

        assert n_eff < 2.0, (
            f"동일한 basin에서 유효 독립 검정은 최소여야 함: n_eff={n_eff}"
        )

    def test_independent_signatures_n_eff_large(self) -> None:
        """완전 독립 signature → n_trials_eff ≈ n_trials."""
        rng = np.random.default_rng(0)
        indep = rng.normal(0, 1, (100, 11))
        weights = np.ones(100, dtype=np.float64)

        n_eff = calc_n_trials_eff_entropy(indep, weights)

        assert n_eff > 5.0, (
            f"독립 signature에서 n_eff가 충분히 커야 함: n_eff={n_eff}"
        )

    def test_pruned_weight_reduces_n_eff(self) -> None:
        """Pruned trial weight < 1 적용 시 n_eff 감소."""
        rng = np.random.default_rng(42)
        sigs = rng.normal(0, 1, (100, 11))

        w_partial = np.array([0.5] * 50 + [1.0] * 50, dtype=np.float64)
        w_full = np.ones(100, dtype=np.float64)

        n_eff_partial = calc_n_trials_eff_entropy(sigs, w_partial)
        n_eff_full = calc_n_trials_eff_entropy(sigs, w_full)

        # partial weight → 기여 감소 → n_eff ≤ full
        assert n_eff_partial <= n_eff_full + 1e-6, (
            f"partial weight n_eff({n_eff_partial:.3f}) > full n_eff({n_eff_full:.3f})"
        )

    def test_single_trial_returns_one(self) -> None:
        """단일 trial → n_eff = 1.0 (엔트로피 최소)."""
        sig = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1]])
        w = np.array([1.0])

        n_eff = calc_n_trials_eff_entropy(sig, w)

        # 단일 trial: p_i = 1.0 → entropy = 0 → n_eff = exp(0) = 1.0
        assert abs(n_eff - 1.0) < 0.1, f"단일 trial n_eff ≈ 1.0이어야 함: {n_eff}"

    def test_output_is_finite_positive(self) -> None:
        """출력값이 항상 유한하고 양수."""
        rng = np.random.default_rng(7)
        sigs = rng.normal(0, 1, (50, 11))
        w = rng.uniform(0.1, 1.0, 50)

        n_eff = calc_n_trials_eff_entropy(sigs, w)

        assert np.isfinite(n_eff), f"n_eff가 유한해야 함: {n_eff}"
        assert n_eff > 0.0, f"n_eff가 양수여야 함: {n_eff}"
