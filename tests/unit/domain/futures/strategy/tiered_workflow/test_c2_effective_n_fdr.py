"""Unit tests for C2 effective-N FDR correction.

Covers:
  - _by_q_values: m_eff parameter reduces BY q-values for correlated hypotheses
  - _compute_probe_m_eff: effective test count from probe TF correlation clusters

Time Complexity: O(n log n) per _by_q_values call (dominated by argsort).
Space Complexity: O(n) per call.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _by_q_values,
    _compute_probe_m_eff,
)

# ---------------------------------------------------------------------------
# S1: Correlated multi-test → q-values decrease (core correctness)
# ---------------------------------------------------------------------------


def test_by_q_values_correlated_tf_reduces_q_values() -> None:
    """S1: High inter-TF correlation (r̄=0.95) → m_eff < m → smaller q-values."""
    # Arrange
    p_values: NDArray[np.float64] = np.array([0.04, 0.045, 0.05], dtype=np.float64)
    # m_eff = 3 / (1 + 2 * 0.95) = 3 / 2.9 ≈ 1.034
    m_eff = 3.0 / (1.0 + 2.0 * 0.95)

    # Act
    q_eff = _by_q_values(p_values, m_eff=m_eff)
    q_naive = _by_q_values(p_values)

    # Assert — effective-N correction must relax (never tighten) every q-value
    assert np.all(q_eff <= q_naive + 1e-9), (
        f"Expected q_eff ≤ q_naive element-wise; got q_eff={q_eff}, q_naive={q_naive}"
    )
    # At least the smallest p must strictly improve
    assert q_eff[0] < q_naive[0], (
        f"Expected strict reduction for smallest p; got q_eff[0]={q_eff[0]}, q_naive[0]={q_naive[0]}"
    )


# ---------------------------------------------------------------------------
# S2: Independent hypotheses (r̄=0) → q-values unchanged
# ---------------------------------------------------------------------------


def test_by_q_values_independent_hypotheses_unchanged() -> None:
    """S2: When diversity_corr is empty → m_eff ≈ m → q-values match naive BY."""
    # Arrange
    groups = [
        ("BTC", "rr:rr_16_4h", "all"),
        ("BTC", "rr:rr_16_6h", "all"),
        ("BTC", "rr:rr_16_1h", "all"),
    ]
    diversity_corr: dict[str, float] = {}
    p_values: NDArray[np.float64] = np.array([0.04, 0.045, 0.05], dtype=np.float64)

    # Act
    m_eff = _compute_probe_m_eff(groups=groups, diversity_corr=diversity_corr)
    q_eff = _by_q_values(p_values, m_eff=m_eff)
    q_naive = _by_q_values(p_values)

    # Assert
    assert m_eff == pytest.approx(3.0, abs=1e-9), (
        f"Expected m_eff≈3.0 for zero correlation; got {m_eff}"
    )
    assert np.allclose(q_eff, q_naive, atol=1e-9), (
        f"Expected q_eff≈q_naive for independent tests; got q_eff={q_eff}, q_naive={q_naive}"
    )


# ---------------------------------------------------------------------------
# S3: Non-probe hypotheses each contribute 1 to m_eff
# ---------------------------------------------------------------------------


def test_compute_probe_m_eff_non_probe_groups_count_as_one_each() -> None:
    """S3: 3 non-probe + 2 correlated probes (r̄=0) → m_eff == 5.0."""
    # Arrange — 2 probes share (BTC, rr) cluster; 3 non-probes have no TF suffix
    groups = [
        ("BTC", "rr:rr_16_4h", "all"),   # probe
        ("BTC", "rr:rr_16_6h", "all"),   # probe (same cluster)
        ("ETH", "momentum:base", "all"),  # non-probe
        ("BTC", "vol:base", "all"),       # non-probe
        ("SOL", "arb:spread", "all"),     # non-probe
    ]
    diversity_corr: dict[str, float] = {}

    # Act
    m_eff = _compute_probe_m_eff(groups=groups, diversity_corr=diversity_corr)

    # Assert — probe cluster r̄=0 → contributes 2.0; non-probes contribute 3.0
    assert m_eff == pytest.approx(5.0, abs=1e-9), (
        f"Expected m_eff=5.0; got {m_eff}"
    )


# ---------------------------------------------------------------------------
# S4: Empty diversity_corr with probe groups → no ZeroDivisionError, m_eff == m
# ---------------------------------------------------------------------------


def test_compute_probe_m_eff_empty_diversity_corr_no_exception() -> None:
    """S4: Probe groups exist but diversity_corr={} → m_eff == len(groups), no error."""
    # Arrange
    groups = [
        ("BTC", "rr:rr_16_4h", "all"),
        ("BTC", "rr:rr_16_6h", "all"),
        ("ETH", "rr:rr_16_1h", "all"),
    ]
    diversity_corr: dict[str, float] = {}

    # Act
    m_eff = _compute_probe_m_eff(groups=groups, diversity_corr=diversity_corr)

    # Assert — no correlation data → all treated as independent
    assert m_eff == pytest.approx(float(len(groups)), abs=1e-9), (
        f"Expected m_eff={float(len(groups))}; got {m_eff}"
    )


# ---------------------------------------------------------------------------
# S5: m_eff=None is backward-compatible with omitting the argument
# ---------------------------------------------------------------------------


def test_by_q_values_m_eff_none_matches_default() -> None:
    """S5: _by_q_values(p, m_eff=None) == _by_q_values(p) (backward compat)."""
    # Arrange
    p_values: NDArray[np.float64] = np.array(
        [0.01, 0.03, 0.07, 0.12, 0.25], dtype=np.float64
    )

    # Act
    q_none = _by_q_values(p_values, m_eff=None)
    q_default = _by_q_values(p_values)

    # Assert
    assert np.allclose(q_none, q_default, atol=1e-12), (
        f"Expected q_none == q_default; diff={q_none - q_default}"
    )


# ---------------------------------------------------------------------------
# S6: Bidirectional key lookup — b~a fallback when a~b absent
# ---------------------------------------------------------------------------


def test_compute_probe_m_eff_reverse_key_lookup() -> None:
    """S6: diversity_corr stores '6h~4h' (reverse order) — r̄ still resolved to 0.9."""
    # Arrange — key stored as reverse order of (4h, 6h)
    diversity_corr: dict[str, float] = {"BTC:rr:6h~4h": 0.9}
    groups = [
        ("BTC", "rr:rr_16_4h", "all"),
        ("BTC", "rr:rr_16_6h", "all"),
    ]

    # Act
    m_eff = _compute_probe_m_eff(groups=groups, diversity_corr=diversity_corr)

    # Assert — m_eff = 2 / (1 + 1 * 0.9) = 2/1.9 ≈ 1.0526 < 2.0
    expected = 2.0 / (1.0 + 1.0 * 0.9)
    assert m_eff == pytest.approx(expected, rel=1e-6), (
        f"Expected m_eff≈{expected} from reverse-key lookup; got {m_eff}"
    )
    assert m_eff < 2.0, "m_eff must be < naive m=2 when correlation is non-zero"
