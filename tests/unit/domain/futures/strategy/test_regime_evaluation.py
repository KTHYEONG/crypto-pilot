"""Unit tests for C2-C5 regime evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.regime_evaluation import (
    RegimeLiftProofResult,
    RegimeScoreCard,
    _compute_c2,
    _compute_c3,
    _compute_c4,
    _compute_c5,
    deflated_sharpe_ratio,
    evaluate_regime_classifier,
    evaluate_regime_lift_proof,
    newey_west_tstat,
)

_N_REGIMES = 6  # _REGIME_NAMES length


# ---------------------------------------------------------------------------
# C2 — Persistence
# ---------------------------------------------------------------------------


def test_c2_stable_regime_has_high_dwell() -> None:
    """100 bars of code 0 then 100 of code 1 → dwell ≥ 6, transition_rate ≈ 0.

    Both micro codes 0/1 map to macro class 0 (bull), so macro_dwell = 200.
    Score is computed from macro_dwell ≥ 8 → dwell component = 6.
    """
    code = np.array([0] * 100 + [1] * 100, dtype=np.int8)
    dwell, tr, _, score, macro_dwell, _macro_tr = _compute_c2(code)
    assert dwell >= 6.0
    assert tr <= 0.01
    # macro_dwell: 200 bars of macro class 0 → single run of 200
    assert macro_dwell >= 100.0
    assert score >= 7.0


def test_c2_whipsaw_has_low_score() -> None:
    """Alternating micro codes 0/1 stay in macro class 0 (bull) → macro_dwell high, score from tr."""
    code = np.array([0, 1] * 50, dtype=np.int8)
    dwell, tr, _, score, macro_dwell, _macro_tr = _compute_c2(code)
    assert dwell == pytest.approx(1.0)
    assert tr == pytest.approx(1.0, rel=0.01)
    # macro class stays 0 (bull) the whole time → macro_dwell = 100
    assert macro_dwell >= 50.0
    # score from transition_rate (micro tr=1.0 → 0 points) + macro_dwell (≥8 → 6 points) = 6.0
    assert score >= 5.0


def test_c2_empty_returns_zero_dwell() -> None:
    dwell, tr, _ent, score, macro_dwell, macro_tr = _compute_c2(np.array([], dtype=np.int8))
    assert dwell == 0.0
    assert tr == 1.0
    assert score == 0.0
    assert macro_dwell == 0.0
    assert macro_tr == 1.0


def test_c2_single_bar_returns_zero() -> None:
    dwell, _tr, _ent, _score, macro_dwell, _macro_tr = _compute_c2(np.array([2], dtype=np.int8))
    assert dwell == 0.0
    assert macro_dwell == 0.0


def test_c2_medium_dwell_partial_score() -> None:
    """Micro blocks of 4 bars alternating 0/1 → same macro class (bull) → macro_dwell=160."""
    code = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 20).astype(np.int8)
    dwell, _, _, score, macro_dwell, _macro_tr = _compute_c2(code)
    # micro dwell = 4 (blocks of 4)
    assert 3.0 <= dwell <= 5.0
    # macro: all codes 0/1 → macro class 0 → single run of 160 → score ≥ 6.0
    assert macro_dwell >= 100.0
    assert score >= 5.0


def test_c2_macro_cross_direction_whipsaw() -> None:
    """Alternating bull(0)/bear(2) blocks of 3 → macro transitions are real direction flips."""
    code = np.tile([0, 0, 0, 2, 2, 2], 20).astype(np.int8)
    dwell, _tr, _, score, macro_dwell, _macro_tr = _compute_c2(code)
    # micro dwell = 3 → partial
    assert dwell == pytest.approx(3.0)
    # macro dwell = 3 (each block of 3 bull then 3 bear)
    assert macro_dwell == pytest.approx(3.0)
    # score: macro_dwell=3 (≥2 →1.5) + tr≈0.33 (>0.25 →0) = 1.5
    assert score == pytest.approx(1.5)


def test_c2_macro_transition_state_excluded_from_direction() -> None:
    """Transition state (code 4) maps to macro class 2, not bull/bear — no direction flip."""
    # 50 bars bull(0), 50 bars transition(4), 50 bars bull(0)
    code = np.array([0] * 50 + [4] * 50 + [0] * 50, dtype=np.int8)
    _dwell, _tr, _ent, score, macro_dwell, macro_tr = _compute_c2(code)
    # macro: bull(0)→transition(2)→bull(0): 2 macro transitions out of 149 steps
    assert macro_tr == pytest.approx(2 / 149, rel=0.01)
    # macro dwell: runs of 50, 50, 50 → median = 50
    assert macro_dwell == pytest.approx(50.0)
    assert score >= 5.0


# ---------------------------------------------------------------------------
# C3 — Distinctness
# ---------------------------------------------------------------------------


def test_c3_distinct_regimes_pass_kw_and_sign_flip() -> None:
    """Code 0 → large positive edge, code 1 → large negative edge: clear separation."""
    rng = np.random.default_rng(42)
    n = 300
    code = np.array([0] * n + [1] * n, dtype=np.int8)
    fwd = np.concatenate([rng.normal(20.0, 3.0, n), rng.normal(-20.0, 3.0, n)])
    pval, flip, _mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert pval < 0.001
    assert flip is True
    assert score >= 7.0
    assert mag_sep > 0.0


def test_c3_indistinct_regimes_high_pvalue() -> None:
    """Same distribution across codes → high p-value, low score."""
    rng = np.random.default_rng(0)
    n = 100
    code = np.array([0] * n + [1] * n + [2] * n, dtype=np.int8)
    fwd = rng.normal(0.0, 5.0, 3 * n)
    pval, _flip, _mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert pval > 0.05 or score <= 5.0
    assert np.isfinite(mag_sep)


def test_c3_nan_values_filtered_safely() -> None:
    """NaN entries in fwd do not crash computation."""
    rng = np.random.default_rng(1)
    n = 150
    code = np.array([0] * n + [1] * n, dtype=np.int8)
    fwd = np.concatenate([rng.normal(10.0, 2.0, n), rng.normal(-10.0, 2.0, n)])
    fwd[::3] = np.nan
    pval, _flip, mi, _score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert np.isfinite(pval)
    assert np.isfinite(mi)
    assert np.isfinite(mag_sep)


def test_c3_insufficient_groups_returns_zeros() -> None:
    """Only one regime with enough obs → two-group test impossible."""
    code = np.zeros(50, dtype=np.int8)  # all code 0
    fwd = np.random.default_rng(5).normal(5.0, 2.0, 50)
    pval, _flip, _mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert pval == pytest.approx(1.0)
    assert score == 0.0
    assert mag_sep == pytest.approx(0.0)


def test_c3_magnitude_separation_without_sign_flip_achieves_directional() -> None:
    """Magnitude separation >= 1.5 without sign_flip still qualifies as directional."""
    rng = np.random.default_rng(7)
    n = 300
    # Both groups positive but with large mean separation relative to pooled std
    code = np.array([0] * n + [1] * n, dtype=np.int8)
    fwd = np.concatenate([rng.normal(30.0, 2.0, n), rng.normal(2.0, 2.0, n)])
    pval, flip, _mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert flip is False  # both means positive → no sign flip
    assert mag_sep >= 1.5  # large magnitude relative to pooled std
    assert pval < 0.01
    assert score >= 7.0  # directional via magnitude_sep → high score


# ---------------------------------------------------------------------------
# C4 — OOS Stability
# ---------------------------------------------------------------------------


def test_c4_stable_sharpe_rank_high_rho() -> None:
    """IS and OOS return same regime ranking -> rho >= 0.5, score >= 7."""
    rng = np.random.default_rng(7)
    T = 600
    # 4 regimes cycling, IS = first half, OOS = second half
    raw_code = np.tile([0, 1, 2, 3], T // 4).astype(np.int8)
    means: dict[int, float] = {0: 8.0, 1: 4.0, 2: -2.0, 3: -6.0}
    fwd = np.array([rng.normal(means[int(c)], 1.5) for c in raw_code])
    is_m = np.array([True] * (T // 2) + [False] * (T // 2))
    oos_m = ~is_m
    rho, n, score = _compute_c4(raw_code, fwd, is_m, oos_m, min_n_per_regime=10)
    assert rho >= 0.5
    assert score >= 7.0
    assert n >= 2


def test_c4_reversed_rank_negative_rho() -> None:
    """OOS ranking reverses IS -> rho < 0, score <= 1."""
    rng = np.random.default_rng(11)
    T = 400
    code = np.tile([0, 1, 2, 3], T // 4).astype(np.int8)
    is_m = np.array([True] * (T // 2) + [False] * (T // 2))
    oos_m = ~is_m
    # IS: 0 > 1 > 2 > 3  /  OOS: opposite
    is_means: dict[int, float] = {0: 8.0, 1: 4.0, 2: -2.0, 3: -6.0}
    oos_means: dict[int, float] = {0: -6.0, 1: -2.0, 2: 4.0, 3: 8.0}
    fwd = np.array([rng.normal(is_means[int(c)] if is_m[i] else oos_means[int(c)], 1.0) for i, c in enumerate(code)])
    rho, _n, score = _compute_c4(code, fwd, is_m, oos_m, min_n_per_regime=10)
    assert rho < 0
    assert score <= 1.0


def test_c4_single_common_regime_insufficient() -> None:
    """Only 1 common regime → n_common < 2, score ≤ 2."""
    T = 100
    code = np.zeros(T, dtype=np.int8)
    fwd = np.random.default_rng(3).normal(2.0, 1.0, T)
    is_m = np.array([True] * 50 + [False] * 50)
    _rho, n, score = _compute_c4(code, fwd, is_m, ~is_m, min_n_per_regime=5)
    assert n < 2
    assert score <= 2.0


# ---------------------------------------------------------------------------
# C5 — Coverage
# ---------------------------------------------------------------------------


def test_c5_balanced_four_regimes() -> None:
    """Equal split across 4 codes → occupancy=0.25, n_eff ≈ 4, score ≥ 8."""
    code = np.tile([0, 1, 2, 3], 50).astype(np.int8)
    min_occ, max_occ, n_eff, score = _compute_c5(code)
    assert min_occ == pytest.approx(0.25, rel=0.01)
    assert max_occ == pytest.approx(0.25, rel=0.01)
    assert n_eff >= 3.5
    assert score >= 8.0


def test_c5_dominant_regime_fails_coverage() -> None:
    """One regime 95% of bars → max_occ > 0.60 → score < 8."""
    code = np.array([0] * 95 + [1] * 5, dtype=np.int8)
    _min_occ, max_occ, _n_eff, score = _compute_c5(code)
    assert max_occ == pytest.approx(0.95, rel=0.01)
    assert score < 8.0


def test_c5_empty_code_returns_zeros() -> None:
    _min_occ, _max_occ, n_eff, score = _compute_c5(np.array([], dtype=np.int8))
    assert score == 0.0
    assert n_eff == 0.0


def test_c5_out_of_range_codes_ignored() -> None:
    """Invalid code values (>= _N_REGIMES) must not crash."""
    code = np.array([0, 1, 10, -1, 2], dtype=np.int8)
    _min_occ, _max_occ, _n_eff, score = _compute_c5(code)
    assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Integration — evaluate_regime_classifier
# ---------------------------------------------------------------------------


def test_evaluate_regime_classifier_returns_scorecard() -> None:
    """Full pipeline on random data returns a valid RegimeScoreCard."""
    rng = np.random.default_rng(42)
    T = 600
    all_codes = rng.integers(0, 4, T).astype(np.int8)
    N = 200
    ev_codes = rng.integers(0, 4, N).astype(np.int8)
    ev_edges = rng.normal(0.0, 5.0, N)
    is_m = np.array([True] * (N // 2) + [False] * (N // 2))
    oos_m = ~is_m

    sc = evaluate_regime_classifier(
        all_codes_1d=all_codes,
        event_codes=ev_codes,
        event_edges_bps=ev_edges,
        is_event_mask=is_m,
        oos_event_mask=oos_m,
    )
    assert isinstance(sc, RegimeScoreCard)
    assert np.isfinite(sc.c2_dwell_median)
    assert np.isfinite(sc.c3_kw_pvalue)
    assert np.isfinite(sc.c4_spearman_rho)
    assert np.isfinite(sc.c5_min_occupancy)
    assert 0.0 <= sc.weighted_c2_to_c5 <= 1.0
    assert len(sc.occupancy_by_regime) == _N_REGIMES
    assert len(sc.regime_names) == _N_REGIMES


def test_evaluate_regime_classifier_detects_distinctness() -> None:
    """Clearly distinct regimes receive C3 score ≥ 7 and sign_flip=True."""
    rng = np.random.default_rng(99)
    n = 250
    all_codes = np.array([0] * n + [1] * n, dtype=np.int8)
    ev_codes = all_codes.copy()
    ev_edges = np.concatenate([rng.normal(15.0, 2.0, n), rng.normal(-15.0, 2.0, n)])
    is_m = np.ones(2 * n, dtype=bool)
    oos_m = np.zeros(2 * n, dtype=bool)

    sc = evaluate_regime_classifier(
        all_codes_1d=all_codes,
        event_codes=ev_codes,
        event_edges_bps=ev_edges,
        is_event_mask=is_m,
        oos_event_mask=oos_m,
    )
    assert sc.c3_kw_pvalue < 0.01
    assert sc.c3_has_sign_flip is True
    assert sc.c3_score >= 7.0


def test_evaluate_regime_classifier_empty_events() -> None:
    """Empty event arrays don't raise; returns low C3/C4 scores."""
    all_codes = np.zeros(100, dtype=np.int8)
    ev_codes = np.array([], dtype=np.int8)
    ev_edges = np.array([], dtype=np.float64)
    is_m = np.array([], dtype=bool)
    oos_m = np.array([], dtype=bool)

    sc = evaluate_regime_classifier(
        all_codes_1d=all_codes,
        event_codes=ev_codes,
        event_edges_bps=ev_edges,
        is_event_mask=is_m,
        oos_event_mask=oos_m,
    )
    assert sc.c3_score == 0.0
    assert sc.c4_n_regimes_evaluated == 0


def test_evaluate_regime_classifier_weighted_in_unit_range() -> None:
    """Weighted partial score is always in [0, 1]."""
    rng = np.random.default_rng(77)
    for _ in range(5):
        T = rng.integers(50, 300)
        N = rng.integers(20, 150)
        all_c = rng.integers(0, 6, T).astype(np.int8)
        ev_c = rng.integers(0, 6, N).astype(np.int8)
        ev_e = rng.normal(0.0, 10.0, N)
        split = N // 2
        is_m = np.array([True] * split + [False] * (N - split))
        oos_m = ~is_m
        sc = evaluate_regime_classifier(
            all_codes_1d=all_c,
            event_codes=ev_c,
            event_edges_bps=ev_e,
            is_event_mask=is_m,
            oos_event_mask=oos_m,
        )
        assert 0.0 <= sc.weighted_c2_to_c5 <= 1.0


# ---------------------------------------------------------------------------
# P0 — Regime Lift Proof Gate
# ---------------------------------------------------------------------------


class TestNeweyWestTstat:
    """S3 — NW vs IID 보수성 (양의 serial-corr에서 NW가 더 보수적)."""

    def test_nw_more_conservative_than_iid_with_positive_autocorr(self) -> None:
        """Arrange: rho=0.7 positive serial-corr + positive mean.
        Assert: nw_se > naive_se, nw_tstat <= naive_tstat.
        """
        # Arrange
        rng = np.random.default_rng(42)
        n = 300
        e = rng.standard_normal(n)
        diff = np.zeros(n)
        diff[0] = e[0]
        for i in range(1, n):
            diff[i] = 0.7 * diff[i - 1] + e[i]
        diff += 1.0  # positive mean

        # Act
        nw_t, nw_se, _n_eff = newey_west_tstat(diff, max_lag=6)
        naive_se = float(np.std(diff, ddof=1) / np.sqrt(n))
        naive_t = float(np.mean(diff)) / naive_se

        # Assert
        assert nw_se > naive_se
        assert nw_t <= naive_t

    def test_n_eff_reduced_by_autocorr(self) -> None:
        """S5 — N_eff autocorr 보정: rho_hat~0.8 -> n_eff << N."""
        # Arrange
        rng = np.random.default_rng(7)
        n = 100
        diff_ar = np.zeros(n)
        e = rng.standard_normal(n)
        for i in range(1, n):
            diff_ar[i] = 0.8 * diff_ar[i - 1] + e[i]
        diff_ar += 0.5

        # Act
        _nw_t, _nw_se, n_eff = newey_west_tstat(diff_ar, max_lag=6)

        # Assert
        assert n_eff < n / 3  # theory ~ N * 0.2/1.8 ~ N/9; allow tolerance


class TestDeflatedSharpeRatio:
    """S4 — DSR 다중검정 보정."""

    def test_dsr_negative_with_many_trials_and_modest_sr(self) -> None:
        """n_trials=60, N=200, SR=0.3 → DSR < 0 (다중검정 후 기각)."""
        # Arrange / Act
        dsr = deflated_sharpe_ratio(sr=0.3, n_obs=200, n_trials=60, skewness=0.0, kurtosis=0.0)

        # Assert
        assert dsr < 0

    def test_dsr_positive_with_very_high_sr(self) -> None:
        """SR=5.0, n_trials=2 → DSR > 0 (강한 edge는 통과)."""
        dsr = deflated_sharpe_ratio(sr=5.0, n_obs=500, n_trials=2, skewness=0.0, kurtosis=0.0)
        assert dsr > 0


class TestEvaluateRegimeLiftProof:
    """S1, S2, S6, S7 — evaluate_regime_lift_proof integration tests."""

    def test_pooled_fallback_with_zero_mean_noise(self) -> None:
        """S1 — pooled fallback: AR(1) noise, true mean=0 → proof_passed=False."""
        # Arrange
        rng = np.random.default_rng(42)
        n = 200
        e = rng.standard_normal(n)
        lift = np.zeros(n)
        for i in range(1, n):
            lift[i] = 0.4 * lift[i - 1] + e[i]  # mean=0
        fold_ids = np.repeat(np.arange(4), 50).astype(np.int32)

        # Act
        result = evaluate_regime_lift_proof(
            regime_cond_edges=np.ones(n, dtype=np.float64),
            pooled_edges=np.zeros(n),
            realized_edges=lift,
            fold_ids=fold_ids,
            proof_enabled=True,
            nw_tstat_threshold=1.5,
        )

        # Assert
        assert not result.proof_passed
        assert result.conditioning_path == "pooled_fallback"
        assert result.nw_tstat < result.nw_tstat_threshold

    def test_regime_conditioned_with_strong_consistent_lift(self) -> None:
        """S2 — 진짜 edge: +20bps consistent lift → proof_passed=True."""
        # Arrange
        rng = np.random.default_rng(0)
        n = 400
        lift = np.full(n, 20.0) + rng.standard_normal(n) * 2.0
        fold_ids = np.repeat(np.arange(4), 100).astype(np.int32)

        # Act
        result = evaluate_regime_lift_proof(
            regime_cond_edges=np.ones(n, dtype=np.float64),
            pooled_edges=np.zeros(n),
            realized_edges=lift,
            fold_ids=fold_ids,
        )

        # Assert
        assert result.proof_passed
        assert result.conditioning_path == "regime_conditioned"
        assert result.nw_tstat >= 1.5

    def test_fold_pass_ratio_majority_passes(self) -> None:
        """S6A — 3/4 folds pass (75% >= 60%) -> proof_passed=True.

        Uses n_regime_cells=4 to keep DSR penalty small enough for strong signal.
        """
        # Arrange: 3 signal folds + 1 noise fold
        rng = np.random.default_rng(1)
        n_per_fold = 200
        signal = np.full(n_per_fold, 20.0) + rng.standard_normal(n_per_fold) * 1.5
        noise = rng.standard_normal(n_per_fold)  # mean~0
        lift = np.concatenate([signal, signal, signal, noise])
        fold_ids = np.repeat(np.arange(4), n_per_fold).astype(np.int32)

        # Act
        result = evaluate_regime_lift_proof(
            regime_cond_edges=np.ones(len(lift), dtype=np.float64),
            pooled_edges=np.zeros(len(lift)),
            realized_edges=lift,
            fold_ids=fold_ids,
            nw_tstat_threshold=1.5,
            fold_pass_ratio_threshold=0.60,
            n_regime_cells=4,  # small n_trials keeps DSR penalty manageable
        )

        # Assert
        assert result.proof_passed
        assert result.fold_pass_ratio >= 0.60

    def test_fold_pass_ratio_minority_fails(self) -> None:
        """S6B — 2/4 folds pass (50% < 60%) → proof_passed=False."""
        # Arrange: 2 signal folds + 2 noise folds
        rng = np.random.default_rng(2)
        n_per_fold = 100
        signal = np.full(n_per_fold, 20.0) + rng.standard_normal(n_per_fold) * 1.5
        noise_neg = np.full(n_per_fold, -5.0) + rng.standard_normal(n_per_fold) * 1.5
        lift = np.concatenate([signal, signal, noise_neg, noise_neg])
        fold_ids = np.repeat(np.arange(4), n_per_fold).astype(np.int32)

        # Act
        result = evaluate_regime_lift_proof(
            regime_cond_edges=np.ones(len(lift), dtype=np.float64),
            pooled_edges=np.zeros(len(lift)),
            realized_edges=lift,
            fold_ids=fold_ids,
            nw_tstat_threshold=1.5,
            fold_pass_ratio_threshold=0.60,
        )

        # Assert
        assert not result.proof_passed
        assert result.fold_pass_ratio < 0.60

    def test_proof_disabled_always_passes(self) -> None:
        """S7 — proof_enabled=False → proof_passed=True 무조건."""
        # Arrange: random noise (would normally fail)
        rng = np.random.default_rng(99)
        n = 100
        lift = rng.standard_normal(n)
        fold_ids = np.repeat(np.arange(4), 25).astype(np.int32)

        # Act
        result = evaluate_regime_lift_proof(
            regime_cond_edges=np.ones(n, dtype=np.float64),
            pooled_edges=np.zeros(n),
            realized_edges=lift,
            fold_ids=fold_ids,
            proof_enabled=False,
        )

        # Assert
        assert result.proof_passed is True
        assert result.conditioning_path == "regime_conditioned"

    def test_result_fields_are_populated(self) -> None:
        """RegimeLiftProofResult fields are all finite and within valid ranges."""
        rng = np.random.default_rng(5)
        n = 120
        lift = np.full(n, 10.0) + rng.standard_normal(n) * 3.0
        fold_ids = np.repeat(np.arange(3), 40).astype(np.int32)

        result = evaluate_regime_lift_proof(
            regime_cond_edges=np.ones(n, dtype=np.float64),
            pooled_edges=np.zeros(n),
            realized_edges=lift,
            fold_ids=fold_ids,
        )

        assert isinstance(result, RegimeLiftProofResult)
        assert np.isfinite(result.mean_lift_bps)
        assert np.isfinite(result.nw_tstat)
        assert np.isfinite(result.deflated_sharpe)
        assert 0.0 <= result.fold_pass_ratio <= 1.0
        assert result.n_eff > 0
        assert result.n_folds_evaluated == 3


# --- Audit regression: DSR SR-variance must use correct Bailey-LdP coefficient ---
def test_deflated_sharpe_ratio_sr_variance_positive_contribution() -> None:
    """SR estimation variance must add positive uncertainty (Bailey-LdP sign).

    Regression: a buggy coefficient (0.5*(k_ex-1)) yields a NEGATIVE SR^2 term
    for normal moments, collapsing sr_std to ~0 and inflating DSR toward the
    naive (sr-e_max)/sqrt(v_max). The corrected (k_ex+2)/4 keeps sr_std > 0,
    so DSR must be strictly below the naive value for a positive numerator.
    """
    import math as _math

    from scipy.stats import norm as _norm

    from src.domain.futures.strategy.regime_evaluation import deflated_sharpe_ratio

    sr, n_obs, n_trials = 3.0, 100, 10
    dsr = deflated_sharpe_ratio(sr=sr, n_obs=n_obs, n_trials=n_trials, skewness=0.0, kurtosis=0.0)

    gamma_e = 0.5772156649
    z1 = float(_norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(_norm.ppf(1.0 - 1.0 / (n_trials * _math.e)))
    e_max = (1.0 - gamma_e) * z1 + gamma_e * z2
    v_max = max((z1**2 - z2**2) * (1.0 - gamma_e) ** 2 / 2.0, 0.0)
    naive = (sr - e_max) / max(_math.sqrt(v_max + 1e-12), 1e-12)

    assert sr - e_max > 0.0  # positive numerator precondition
    assert dsr < naive  # corrected sr_std>0 strictly shrinks DSR vs naive
