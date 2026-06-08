"""Unit tests for C2-C5 regime evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.regime_evaluation import (
    RegimeScoreCard,
    _compute_c2,
    _compute_c3,
    _compute_c4,
    _compute_c5,
    evaluate_regime_classifier,
)

_N_REGIMES = 6  # _REGIME_NAMES length


# ---------------------------------------------------------------------------
# C2 — Persistence
# ---------------------------------------------------------------------------


def test_c2_stable_regime_has_high_dwell():
    """100 bars of code 0 then 100 of code 1 → dwell ≥ 6, transition_rate ≈ 0.

    Both micro codes 0/1 map to macro class 0 (bull), so macro_dwell = 200.
    Score is computed from macro_dwell ≥ 8 → dwell component = 6.
    """
    code = np.array([0] * 100 + [1] * 100, dtype=np.int8)
    dwell, tr, _, score, macro_dwell, macro_tr = _compute_c2(code)
    assert dwell >= 6.0
    assert tr <= 0.01
    # macro_dwell: 200 bars of macro class 0 → single run of 200
    assert macro_dwell >= 100.0
    assert score >= 7.0


def test_c2_whipsaw_has_low_score():
    """Alternating micro codes 0/1 stay in macro class 0 (bull) → macro_dwell high, score from tr."""
    code = np.array([0, 1] * 50, dtype=np.int8)
    dwell, tr, _, score, macro_dwell, macro_tr = _compute_c2(code)
    assert dwell == pytest.approx(1.0)
    assert tr == pytest.approx(1.0, rel=0.01)
    # macro class stays 0 (bull) the whole time → macro_dwell = 100
    assert macro_dwell >= 50.0
    # score from transition_rate (micro tr=1.0 → 0 points) + macro_dwell (≥8 → 6 points) = 6.0
    assert score >= 5.0


def test_c2_empty_returns_zero_dwell():
    dwell, tr, ent, score, macro_dwell, macro_tr = _compute_c2(np.array([], dtype=np.int8))
    assert dwell == 0.0
    assert tr == 1.0
    assert score == 0.0
    assert macro_dwell == 0.0
    assert macro_tr == 1.0


def test_c2_single_bar_returns_zero():
    dwell, tr, ent, score, macro_dwell, macro_tr = _compute_c2(np.array([2], dtype=np.int8))
    assert dwell == 0.0
    assert macro_dwell == 0.0


def test_c2_medium_dwell_partial_score():
    """Micro blocks of 4 bars alternating 0/1 → same macro class (bull) → macro_dwell=160."""
    code = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 20).astype(np.int8)
    dwell, _, _, score, macro_dwell, macro_tr = _compute_c2(code)
    # micro dwell = 4 (blocks of 4)
    assert 3.0 <= dwell <= 5.0
    # macro: all codes 0/1 → macro class 0 → single run of 160 → score ≥ 6.0
    assert macro_dwell >= 100.0
    assert score >= 5.0


def test_c2_macro_cross_direction_whipsaw():
    """Alternating bull(0)/bear(2) blocks of 3 → macro transitions are real direction flips."""
    code = np.tile([0, 0, 0, 2, 2, 2], 20).astype(np.int8)
    dwell, tr, _, score, macro_dwell, macro_tr = _compute_c2(code)
    # micro dwell = 3 → partial
    assert dwell == pytest.approx(3.0)
    # macro dwell = 3 (each block of 3 bull then 3 bear)
    assert macro_dwell == pytest.approx(3.0)
    # score: macro_dwell=3 (≥2 →1.5) + tr≈0.33 (>0.25 →0) = 1.5
    assert score == pytest.approx(1.5)


def test_c2_macro_transition_state_excluded_from_direction():
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


def test_c3_distinct_regimes_pass_kw_and_sign_flip():
    """Code 0 → large positive edge, code 1 → large negative edge: clear separation."""
    rng = np.random.default_rng(42)
    n = 300
    code = np.array([0] * n + [1] * n, dtype=np.int8)
    fwd = np.concatenate([rng.normal(20.0, 3.0, n), rng.normal(-20.0, 3.0, n)])
    pval, flip, mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert pval < 0.001
    assert flip is True
    assert score >= 7.0
    assert mag_sep > 0.0


def test_c3_indistinct_regimes_high_pvalue():
    """Same distribution across codes → high p-value, low score."""
    rng = np.random.default_rng(0)
    n = 100
    code = np.array([0] * n + [1] * n + [2] * n, dtype=np.int8)
    fwd = rng.normal(0.0, 5.0, 3 * n)
    pval, flip, mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert pval > 0.05 or score <= 5.0
    assert np.isfinite(mag_sep)


def test_c3_nan_values_filtered_safely():
    """NaN entries in fwd do not crash computation."""
    rng = np.random.default_rng(1)
    n = 150
    code = np.array([0] * n + [1] * n, dtype=np.int8)
    fwd = np.concatenate([rng.normal(10.0, 2.0, n), rng.normal(-10.0, 2.0, n)])
    fwd[::3] = np.nan
    pval, flip, mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert np.isfinite(pval)
    assert np.isfinite(mi)
    assert np.isfinite(mag_sep)


def test_c3_insufficient_groups_returns_zeros():
    """Only one regime with enough obs → two-group test impossible."""
    code = np.zeros(50, dtype=np.int8)  # all code 0
    fwd = np.random.default_rng(5).normal(5.0, 2.0, 50)
    pval, flip, mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert pval == pytest.approx(1.0)
    assert score == 0.0
    assert mag_sep == pytest.approx(0.0)


def test_c3_magnitude_separation_without_sign_flip_achieves_directional():
    """Magnitude separation >= 1.5 without sign_flip still qualifies as directional."""
    rng = np.random.default_rng(7)
    n = 300
    # Both groups positive but with large mean separation relative to pooled std
    code = np.array([0] * n + [1] * n, dtype=np.int8)
    fwd = np.concatenate([rng.normal(30.0, 2.0, n), rng.normal(2.0, 2.0, n)])
    pval, flip, mi, score, mag_sep = _compute_c3(code, fwd, min_n_per_group=10)
    assert flip is False  # both means positive → no sign flip
    assert mag_sep >= 1.5  # large magnitude relative to pooled std
    assert pval < 0.01
    assert score >= 7.0  # directional via magnitude_sep → high score


# ---------------------------------------------------------------------------
# C4 — OOS Stability
# ---------------------------------------------------------------------------


def test_c4_stable_sharpe_rank_high_rho():
    """IS and OOS return same regime ranking → ρ ≥ 0.5, score ≥ 7."""
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


def test_c4_reversed_rank_negative_rho():
    """OOS ranking reverses IS → ρ < 0, score ≤ 1."""
    rng = np.random.default_rng(11)
    T = 400
    code = np.tile([0, 1, 2, 3], T // 4).astype(np.int8)
    is_m = np.array([True] * (T // 2) + [False] * (T // 2))
    oos_m = ~is_m
    # IS: 0 > 1 > 2 > 3  /  OOS: opposite
    is_means: dict[int, float] = {0: 8.0, 1: 4.0, 2: -2.0, 3: -6.0}
    oos_means: dict[int, float] = {0: -6.0, 1: -2.0, 2: 4.0, 3: 8.0}
    fwd = np.array([
        rng.normal(is_means[int(c)] if is_m[i] else oos_means[int(c)], 1.0)
        for i, c in enumerate(code)
    ])
    rho, n, score = _compute_c4(code, fwd, is_m, oos_m, min_n_per_regime=10)
    assert rho < 0
    assert score <= 1.0


def test_c4_single_common_regime_insufficient():
    """Only 1 common regime → n_common < 2, score ≤ 2."""
    T = 100
    code = np.zeros(T, dtype=np.int8)
    fwd = np.random.default_rng(3).normal(2.0, 1.0, T)
    is_m = np.array([True] * 50 + [False] * 50)
    rho, n, score = _compute_c4(code, fwd, is_m, ~is_m, min_n_per_regime=5)
    assert n < 2
    assert score <= 2.0


# ---------------------------------------------------------------------------
# C5 — Coverage
# ---------------------------------------------------------------------------


def test_c5_balanced_four_regimes():
    """Equal split across 4 codes → occupancy=0.25, n_eff ≈ 4, score ≥ 8."""
    code = np.tile([0, 1, 2, 3], 50).astype(np.int8)
    min_occ, max_occ, n_eff, score = _compute_c5(code)
    assert min_occ == pytest.approx(0.25, rel=0.01)
    assert max_occ == pytest.approx(0.25, rel=0.01)
    assert n_eff >= 3.5
    assert score >= 8.0


def test_c5_dominant_regime_fails_coverage():
    """One regime 95% of bars → max_occ > 0.60 → score < 8."""
    code = np.array([0] * 95 + [1] * 5, dtype=np.int8)
    min_occ, max_occ, n_eff, score = _compute_c5(code)
    assert max_occ == pytest.approx(0.95, rel=0.01)
    assert score < 8.0


def test_c5_empty_code_returns_zeros():
    min_occ, max_occ, n_eff, score = _compute_c5(np.array([], dtype=np.int8))
    assert score == 0.0
    assert n_eff == 0.0


def test_c5_out_of_range_codes_ignored():
    """Invalid code values (>= _N_REGIMES) must not crash."""
    code = np.array([0, 1, 10, -1, 2], dtype=np.int8)
    min_occ, max_occ, n_eff, score = _compute_c5(code)
    assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Integration — evaluate_regime_classifier
# ---------------------------------------------------------------------------


def test_evaluate_regime_classifier_returns_scorecard():
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


def test_evaluate_regime_classifier_detects_distinctness():
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


def test_evaluate_regime_classifier_empty_events():
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


def test_evaluate_regime_classifier_weighted_in_unit_range():
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
