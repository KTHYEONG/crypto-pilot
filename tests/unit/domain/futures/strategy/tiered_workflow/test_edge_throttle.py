"""Tests for edge-conditional throttle (layer2-edge-conditional-deployment spec).

Scenarios:
  S1 - Full deploy: score >= ref → m == 1.0
  S2 - Partial throttle: floor < score < ref → 0 < m < 1
  S3 - Flat: score <= floor → m == 0.0
  S4 - Distribution reshape: throttled stream Sharpe > unthrottled
  S5 - Scale-invariance guard: global scalar does NOT move DSR
  S6 - Config plumbing: edge_throttle_enabled=False → m≡1
  S7 - Edge: empty book or NaN score → m == 0.0
  S8 - BVA gamma: convex/concave boundary behaviour
"""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _book_edge_score,
    _edge_throttle_multiplier,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

# ---------------------------------------------------------------------------
# _edge_throttle_multiplier
# ---------------------------------------------------------------------------

class TestEdgeThrottleMultiplier:
    def test_s1_full_deploy_at_ref(self) -> None:
        """S1: score == ref → m == 1.0."""
        m = _edge_throttle_multiplier(5.0, floor_bps=0.0, ref_bps=5.0, gamma=1.0)
        assert m == pytest.approx(1.0)

    def test_s1_full_deploy_above_ref(self) -> None:
        """S1: score > ref → m clamped to 1.0."""
        m = _edge_throttle_multiplier(10.0, floor_bps=0.0, ref_bps=5.0, gamma=1.0)
        assert m == pytest.approx(1.0)

    def test_s2_partial_linear(self) -> None:
        """S2: floor=0, ref=5, score=2, gamma=1 → m == 0.4."""
        m = _edge_throttle_multiplier(2.0, floor_bps=0.0, ref_bps=5.0, gamma=1.0)
        assert m == pytest.approx(0.4, rel=1e-6)

    def test_s3_flat_at_floor(self) -> None:
        """S3: score == floor → m == 0.0."""
        m = _edge_throttle_multiplier(0.0, floor_bps=0.0, ref_bps=5.0, gamma=1.0)
        assert m == pytest.approx(0.0)

    def test_s3_flat_below_floor(self) -> None:
        """S3: score < floor → m == 0.0."""
        m = _edge_throttle_multiplier(-1.0, floor_bps=0.0, ref_bps=5.0, gamma=1.0)
        assert m == pytest.approx(0.0)

    def test_s7_nan_score(self) -> None:
        """S7: NaN score → m == 0.0, no exception."""
        m = _edge_throttle_multiplier(float("nan"), floor_bps=0.0, ref_bps=5.0, gamma=1.0)
        assert m == pytest.approx(0.0)

    def test_s8_convex_gamma(self) -> None:
        """S8: gamma=2, x=0.5 → m == 0.25."""
        m = _edge_throttle_multiplier(2.5, floor_bps=0.0, ref_bps=5.0, gamma=2.0)
        assert m == pytest.approx(0.25, rel=1e-6)

    def test_s8_zero_gamma_guard(self) -> None:
        """S8: gamma=0 → guarded to 1e-9; any x>0 → m≈1.0."""
        m = _edge_throttle_multiplier(2.0, floor_bps=0.0, ref_bps=5.0, gamma=0.0)
        assert m == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# _book_edge_score
# ---------------------------------------------------------------------------

class TestBookEdgeScore:
    def test_s7_empty_book(self) -> None:
        """S7: all weights zero → 0.0."""
        w = np.zeros(4, dtype=np.float64)
        mu = np.array([3.0, 5.0, 2.0, 7.0], dtype=np.float64)
        hurdle = np.full(4, 1.0, dtype=np.float64)
        assert _book_edge_score(w, mu, hurdle) == pytest.approx(0.0)

    def test_equal_weight_simple(self) -> None:
        """Gross-weighted avg net-of-cost: |mu| - hurdle, no negatives."""
        w = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        mu = np.array([6.0, 4.0, 10.0], dtype=np.float64)
        hurdle = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        # net_edge: [5.0, 3.0, 0.0]; gross-weighted: (0.5*5 + 0.5*3)/(0.5+0.5) = 4.0
        assert _book_edge_score(w, mu, hurdle) == pytest.approx(4.0, rel=1e-6)

    def test_hurdle_clips_negative_to_zero(self) -> None:
        """mu < hurdle → net_edge=0 for that position."""
        w = np.array([1.0, 1.0], dtype=np.float64)
        mu = np.array([0.5, 8.0], dtype=np.float64)
        hurdle = np.array([2.0, 2.0], dtype=np.float64)
        # net: [0.0, 6.0]; weighted: 3.0
        assert _book_edge_score(w, mu, hurdle) == pytest.approx(3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# S4 - Distribution reshape
# ---------------------------------------------------------------------------

class TestDistributionReshape:
    """S4: throttle reduces bad-fold contribution → Sharpe improves."""

    def _make_stream(self, bad_factor: float) -> list[float]:
        rng = np.random.default_rng(42)
        good: list[float] = [float(x) for x in rng.normal(loc=0.002, scale=0.01, size=200)]
        bad: list[float] = [float(x) * bad_factor for x in rng.normal(loc=-0.005, scale=0.015, size=100)]
        return good + bad

    def _sharpe(self, rets: list[float]) -> float:
        arr = np.asarray(rets, dtype=np.float64)
        return float(float(np.mean(arr)) / (float(np.std(arr, ddof=1)) + 1e-9)) * float(np.sqrt(2190))

    def test_throttle_improves_sharpe(self) -> None:
        """Bad-fold down-weighted stream yields higher Sharpe than full-weight stream."""
        unthrottled = self._make_stream(1.0)
        throttled = self._make_stream(0.2)  # bad fold scaled by 0.2 (low m)
        assert self._sharpe(throttled) > self._sharpe(unthrottled)


# ---------------------------------------------------------------------------
# S5 - Scale-invariance: global scalar ≠ DSR mover
# ---------------------------------------------------------------------------

class TestScaleInvariance:
    """S5: Sharpe is scale-invariant → confirms throttle must reshape distribution."""

    def test_sharpe_scale_invariant(self) -> None:
        rng = np.random.default_rng(7)
        r = rng.normal(loc=0.001, scale=0.01, size=500)
        arr = r.astype(np.float64)

        def sharpe(x: np.ndarray) -> float:
            return float(np.mean(x) / (np.std(x, ddof=1) + 1e-12))

        assert sharpe(arr * 0.1) == pytest.approx(sharpe(arr), rel=1e-6)
        assert sharpe(arr * 3.0) == pytest.approx(sharpe(arr), rel=1e-6)


# ---------------------------------------------------------------------------
# S6 - Config plumbing: disabled → m≡1
# ---------------------------------------------------------------------------

class TestConfigPlumbing:
    def test_s6_disabled_via_from_mapping(self) -> None:
        """S6: edge_throttle_enabled=False parsed correctly."""
        cfg = Layer2AllocationConfig.from_mapping({"edge_throttle_enabled": False})
        assert cfg.edge_throttle_enabled is False

    def test_s6_default_enabled(self) -> None:
        """Default: throttle enabled."""
        cfg = Layer2AllocationConfig.from_mapping({})
        assert cfg.edge_throttle_enabled is True

    def test_s6_throttle_disabled_multiplier_is_one(self) -> None:
        """When disabled, caller uses m=1.0 (contract: no scaling applied)."""
        # Simulate caller logic: if not enabled, skip _edge_throttle_multiplier
        cfg = Layer2AllocationConfig.from_mapping({"edge_throttle_enabled": False})
        # m is 1.0 by convention when disabled
        m = 1.0 if not cfg.edge_throttle_enabled else _edge_throttle_multiplier(
            0.0, floor_bps=cfg.edge_floor_bps, ref_bps=cfg.edge_ref_bps,
            gamma=cfg.edge_throttle_gamma,
        )
        assert m == pytest.approx(1.0)

    def test_throttle_params_from_mapping(self) -> None:
        """All 4 throttle params round-trip through from_mapping."""
        cfg = Layer2AllocationConfig.from_mapping({
            "edge_throttle_enabled": True,
            "edge_floor_bps": 1.5,
            "edge_ref_bps": 8.0,
            "edge_throttle_gamma": 2.0,
        })
        assert cfg.edge_floor_bps == pytest.approx(1.5)
        assert cfg.edge_ref_bps == pytest.approx(8.0)
        assert cfg.edge_throttle_gamma == pytest.approx(2.0)
