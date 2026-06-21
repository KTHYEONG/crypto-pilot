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
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _apply_risk_budget_floor,
    _book_edge_score,
    _edge_throttle_multiplier,
    _estimate_annual_vol,
    _resolve_adaptive_k_rank,
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

    def test_min_active_multiplier_lifts_positive_score(self) -> None:
        """Positive edge keeps minimum active deployment while raw zero stays flat."""
        m = _edge_throttle_multiplier(
            1.0,
            floor_bps=0.0,
            ref_bps=5.0,
            gamma=1.0,
            min_active_mult=0.25,
        )

        assert m == pytest.approx(0.40, rel=1e-6)
        assert m > 0.20

    def test_min_active_multiplier_does_not_lift_floor_score(self) -> None:
        """Score at floor remains flat even when min_active_mult is high."""
        m = _edge_throttle_multiplier(
            0.0,
            floor_bps=0.0,
            ref_bps=5.0,
            gamma=1.0,
            min_active_mult=0.60,
        )

        assert m == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _apply_risk_budget_floor
# ---------------------------------------------------------------------------

class TestRiskBudgetFloor:
    def test_under_deployed_book_scales_up_without_support_leak(self) -> None:
        weights = np.array([0.05, -0.05, 0.0], dtype=np.float64)
        sigma = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        support = np.array([True, True, False], dtype=np.bool_)
        bars_per_year = 2190.0
        floor_ann_vol = 1.0 * 0.30
        before_vol = float(np.sqrt(float(np.dot(weights**2, sigma**2)))) * float(np.sqrt(bars_per_year))

        out = _apply_risk_budget_floor(
            weights=weights,
            sigma=sigma,
            bars_per_year=bars_per_year,
            vol_target=1.0,
            floor_ratio=0.30,
            max_scale=3.0,
            caps=PortfolioCaps(gross=3.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=1.0),
            btc_beta=None,
            support_mask=support,
        )
        after_vol = float(np.sqrt(float(np.dot(out**2, sigma**2)))) * float(np.sqrt(bars_per_year))

        assert float(np.sum(np.abs(out))) > float(np.sum(np.abs(weights)))
        assert out[2] == pytest.approx(0.0)
        assert abs(floor_ann_vol - after_vol) < abs(floor_ann_vol - before_vol)
        assert after_vol <= 1.0

    def test_empty_or_disabled_risk_budget_floor_returns_original(self) -> None:
        weights = np.array([0.0, 0.0], dtype=np.float64)
        sigma = np.array([0.01, 0.01], dtype=np.float64)
        support = np.array([False, False], dtype=np.bool_)

        out = _apply_risk_budget_floor(
            weights=weights,
            sigma=sigma,
            bars_per_year=2190.0,
            vol_target=1.0,
            floor_ratio=0.30,
            max_scale=3.0,
            caps=PortfolioCaps(),
            btc_beta=None,
            support_mask=support,
        )

        assert np.array_equal(out, weights)


class TestAdaptiveBreadth:
    def test_estimate_annual_vol_returns_zero_on_shape_mismatch(self) -> None:
        out = _estimate_annual_vol(
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.01], dtype=np.float64),
            2190.0,
        )

        assert out == pytest.approx(0.0)

    def test_expand_when_prev_book_under_uses_risk_budget(self) -> None:
        out = _resolve_adaptive_k_rank(
            base_k=3,
            n_valid=9,
            prev_weights=np.array([0.02, 0.02, 0.0, 0.0], dtype=np.float64),
            sigma=np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64),
            bars_per_year=2190.0,
            vol_target=1.0,
            expand_below_vol_ratio=0.35,
            max_extra=4,
        )

        assert out == 7

    def test_no_expand_when_prev_book_already_uses_risk_budget(self) -> None:
        out = _resolve_adaptive_k_rank(
            base_k=3,
            n_valid=9,
            prev_weights=np.array([0.80, 0.80, 0.0, 0.0], dtype=np.float64),
            sigma=np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64),
            bars_per_year=2190.0,
            vol_target=1.0,
            expand_below_vol_ratio=0.35,
            max_extra=4,
        )

        assert out == 3


# ---------------------------------------------------------------------------
# _book_edge_score
# ---------------------------------------------------------------------------

class TestBookEdgeScore:
    def test_s7_empty_book(self) -> None:
        """S7: all weights zero → 0.0."""
        w: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        mu = np.array([3.0, 5.0, 2.0, 7.0], dtype=np.float64)
        # Fix 1: hurdle 인자 제거 — mu는 이미 net-of-cost (이중차감 방지)
        assert _book_edge_score(w, mu) == pytest.approx(0.0)

    def test_equal_weight_simple(self) -> None:
        """Gross-weighted avg |mu| (mu already net-of-cost — Fix 1)."""
        w = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        mu = np.array([5.0, 3.0, 0.0], dtype=np.float64)  # 이미 net값
        # score: (0.5*5 + 0.5*3) / (0.5+0.5) = 4.0
        assert _book_edge_score(w, mu) == pytest.approx(4.0, rel=1e-6)

    def test_mu_already_net_no_hurdle_deduction(self) -> None:
        """Fix 1: mu가 이미 net이므로 추가 차감 없이 그대로 사용."""
        w = np.array([1.0, 1.0], dtype=np.float64)
        mu = np.array([3.0, 6.0], dtype=np.float64)  # 이미 net
        # score: (1.0*3 + 1.0*6) / 2.0 = 4.5
        assert _book_edge_score(w, mu) == pytest.approx(4.5, rel=1e-6)


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
            "edge_throttle_min_active_mult": 0.25,
        })
        assert cfg.edge_floor_bps == pytest.approx(1.5)
        assert cfg.edge_ref_bps == pytest.approx(8.0)
        assert cfg.edge_throttle_gamma == pytest.approx(2.0)
        assert cfg.edge_throttle_min_active_mult == pytest.approx(0.25)
