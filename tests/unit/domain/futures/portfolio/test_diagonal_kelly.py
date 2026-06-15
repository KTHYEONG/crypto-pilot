"""diagonal_kelly_weights 단위 테스트.

T3: Friction filter (mu < hurdle → w=0)
T4: Diagonal Kelly sigma-edge (VOL_FLOOR 근처, zero mu)
T5: No-trade band (유지/리밸런스/band=0)
Additional: output shape, gross cap 준수
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    diagonal_kelly_weights,
)
from src.domain.futures.strategy.cs_rank import VOL_FLOOR

# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------

_DEFAULT_CAPS = PortfolioCaps(
    gross=3.0,
    per_symbol=0.10,
    net=0.30,
    beta=0.50,
    target_ann_vol=0.20,
)

_SIGMA_NORMAL = np.array([0.002, 0.002], dtype=np.float64)  # ~0.2% per-bar
_HURDLE = np.array([3.8, 3.8], dtype=np.float64)  # bps
_PREV_ZERO = np.zeros(2, dtype=np.float64)
_BAND = 0.01


# ---------------------------------------------------------------------------
# T3 — Friction Filter
# ---------------------------------------------------------------------------


class TestFrictionFilter:
    """T3: mu < hurdle 심볼은 w=0으로 강제."""

    def test_diagonal_kelly_friction_filter_blocks_below_hurdle(self) -> None:
        # Arrange
        mu_bps = np.array([2.0, 5.0], dtype=np.float64)  # [0] < hurdle, [1] >= hurdle

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
        )

        # Assert
        assert float(w[0]) == pytest.approx(0.0, abs=1e-9), "mu < hurdle → w=0"
        assert float(w[1]) > 0.0, "mu >= hurdle → w > 0"

    def test_diagonal_kelly_friction_filter_blocks_short_below_hurdle(self) -> None:
        # Arrange: 음수 mu (숏 신호) — |mu| < hurdle → w=0
        mu_bps = np.array([-2.0, -5.0], dtype=np.float64)  # [0] |<| hurdle, [1] |>=| hurdle

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
        )

        # Assert: |mu[0]|=2.0 < hurdle=3.8 → w=0; |mu[1]|=5.0 >= hurdle → w < 0 (숏)
        assert float(w[0]) == pytest.approx(0.0, abs=1e-9), "short mu < hurdle → w=0"
        assert float(w[1]) < 0.0, "short mu >= hurdle → w < 0 (short position)"

    def test_diagonal_kelly_friction_all_blocked(self) -> None:
        # Arrange: 모든 mu < hurdle
        mu_bps = np.array([1.0, 2.0], dtype=np.float64)

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
        )

        # Assert: 전부 0 (현금 보유)
        np.testing.assert_allclose(w, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# T4 — Diagonal Kelly sigma-edge
# ---------------------------------------------------------------------------


class TestSigmaEdge:
    """T4: VOL_FLOOR 근처 sigma, zero mu edge case."""

    def test_diagonal_kelly_sigma_near_vol_floor_gives_finite_weight(self) -> None:
        # Arrange: sigma = VOL_FLOOR * 1.01 (floor 바로 위)
        sigma_near_floor = np.full(2, float(VOL_FLOOR) * 1.01, dtype=np.float64)
        mu_bps = np.array([10.0, 10.0], dtype=np.float64)  # > hurdle

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma_near_floor,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
        )

        # Assert: 유한값, caps.per_symbol 이하
        assert np.all(np.isfinite(w)), "VOL_FLOOR 근처 sigma도 유한 weight"
        assert np.all(np.abs(w) <= _DEFAULT_CAPS.per_symbol + 1e-9), "per_symbol cap 준수"

    def test_diagonal_kelly_zero_mu_gives_zero_weight(self) -> None:
        # Arrange: mu=0 (friction mask 통과해도 Kelly=0)
        mu_bps = np.array([0.0, 0.0], dtype=np.float64)
        hurdle_zero = np.array([0.0, 0.0], dtype=np.float64)  # hurdle=0 → mask 통과

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle_zero,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
        )

        # Assert
        np.testing.assert_allclose(w, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# T5 — No-trade band
# ---------------------------------------------------------------------------


class TestNoTradeBand:
    """T5: no_trade_band 로직 검증."""

    def test_diagonal_kelly_no_trade_band_keeps_prev_when_small_delta(self) -> None:
        # Arrange: prev=[0.10, 0.10], 목표가 0.105 수준 → |Δw|=0.005 < band=0.01
        prev_w = np.array([0.10, 0.10], dtype=np.float64)
        # mu가 작아 Kelly raw가 0.105 근방이 되도록 역산:
        # w_raw = 0.25 * mu_ret / sigma^2, sigma=0.002 → var=4e-6
        # 목표 w_raw_capped ≈ 0.105 → mu_ret ≈ 0.105 * 4e-6 / 0.25 = 1.68e-6 → mu_bps ≈ 0.0168
        # 하지만 hurdle=3.8 bps 이상 필요하므로 mu_bps=4.0으로 하고
        # sigma를 크게 잡아 w_raw 자체를 0.105 근방으로 만든다.
        # 0.105 = 0.25 * 4.0e-4 / sigma^2 → sigma^2 = 0.25*4e-4/0.105 ≈ 9.52e-4 → sigma ≈ 0.03086
        sigma_tuned = np.full(2, 0.03086, dtype=np.float64)
        mu_bps = np.array([4.0, 4.0], dtype=np.float64)
        hurdle_low = np.array([3.8, 3.8], dtype=np.float64)

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma_tuned,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle_low,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.01,
        )

        # Assert: |Δw| < 0.01 → prev_w 유지
        np.testing.assert_allclose(w, prev_w, atol=1e-6)

    def test_diagonal_kelly_no_trade_band_rebalances_when_large_delta(self) -> None:
        # Arrange: prev=[0.00, 0.00], mu=10.0 bps (충분히 큼), band=0.01
        mu_bps = np.array([10.0, 10.0], dtype=np.float64)

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.01,
        )

        # Assert: |Δw| >> 0.01 → 리밸런스 발생 (w ≠ prev_w)
        assert not np.allclose(w, _PREV_ZERO, atol=1e-6), "large delta → 리밸런스"

    def test_diagonal_kelly_no_trade_band_zero_allows_all_rebalance(self) -> None:
        # Arrange: band=0.0 → 항상 new weight 적용
        mu_bps = np.array([10.0, 10.0], dtype=np.float64)
        prev_w = np.array([0.05, 0.05], dtype=np.float64)

        # Act
        w_band0 = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )
        w_band_large = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=1.0,  # 매우 큰 band → 항상 prev 유지
        )

        # Assert: band=0이면 new weight, large band이면 prev 유지
        assert not np.allclose(w_band0, prev_w, atol=1e-6), "band=0 → new weight"
        np.testing.assert_allclose(w_band_large, prev_w, atol=1e-6)


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    """출력 shape, gross cap, vol_target, btc_beta."""

    def test_diagonal_kelly_output_shape_matches_input(self) -> None:
        # Arrange: N=5
        n = 5
        mu_bps = np.full(n, 5.0, dtype=np.float64)
        sigma = np.full(n, 0.002, dtype=np.float64)
        hurdle = np.full(n, 3.8, dtype=np.float64)
        prev_w = np.zeros(n, dtype=np.float64)

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        # Assert
        assert w.shape == (n,), f"expected ({n},), got {w.shape}"
        assert w.dtype == np.float64

    def test_diagonal_kelly_caps_respected_gross(self) -> None:
        # Arrange: 극단적 mu → raw weight 합 >> caps.gross
        n = 10
        mu_bps = np.full(n, 1000.0, dtype=np.float64)  # 극단적
        sigma = np.full(n, 0.001, dtype=np.float64)
        hurdle = np.zeros(n, dtype=np.float64)
        prev_w = np.zeros(n, dtype=np.float64)
        caps = PortfolioCaps(gross=3.0, per_symbol=0.10, net=0.30, beta=0.50, target_ann_vol=0.20)

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        # Assert
        gross = float(np.sum(np.abs(w)))
        assert gross <= caps.gross + 1e-6, f"gross={gross:.4f} > caps.gross={caps.gross}"

    def test_diagonal_kelly_vol_target_override_applies(self) -> None:
        # Arrange: vol_target=0.05 (작은 값) → w 크게 축소
        mu_bps = np.array([10.0, 10.0], dtype=np.float64)
        sigma_large = np.array([0.05, 0.05], dtype=np.float64)  # 큰 sigma
        prev_w = _PREV_ZERO

        # Act
        w_tight = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma_large,
            kelly_fraction=0.25,
            vol_target=0.01,  # 매우 작은 vol target
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )
        w_loose = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma_large,
            kelly_fraction=0.25,
            vol_target=0.50,  # 큰 vol target (축소 안 함)
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        # Assert: tight vol_target이 loose보다 weight 절대값이 작거나 같다
        assert float(np.sum(np.abs(w_tight))) <= float(np.sum(np.abs(w_loose))) + 1e-9

    def test_diagonal_kelly_btc_beta_none_defaults_to_zero(self) -> None:
        # Arrange: btc_beta=None vs zeros → 동일 결과
        mu_bps = np.array([5.0, 5.0], dtype=np.float64)

        # Act
        w_none = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
            btc_beta=None,
        )
        w_zero = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=_HURDLE,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
            btc_beta=np.zeros(2, dtype=np.float64),
        )

        # Assert
        np.testing.assert_allclose(w_none, w_zero, atol=1e-12)


# ---------------------------------------------------------------------------
# S1-S4 — Amortized Friction Hurdle (spec: layer2-signal-utilization.md §2.1)
# ---------------------------------------------------------------------------


class TestAmortizedFrictionHurdle:
    """amortized hurdle = round_trip / holding_bars * safety_mult."""

    def test_s1_long_holding_makes_weak_signal_pass(self) -> None:
        # Arrange: mu=2.0bps, hurdle=3.8bps → 기존 로직 FAIL, amortized(6bars) PASS
        mu_bps = np.array([2.0], dtype=np.float64)
        hurdle = np.array([3.8], dtype=np.float64)
        sigma = np.array([0.002], dtype=np.float64)
        prev_w = np.zeros(1, dtype=np.float64)
        caps = PortfolioCaps(gross=3.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=0.5)

        # Act: holding_bars=6 → eff_hurdle = 3.8/6 ≈ 0.63 < 2.0 → PASS
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle,
            holding_bars=6,
            friction_safety_mult=1.0,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        # Assert: w_i > 0 (통과)
        assert w[0] > 0.0

    def test_s2_short_holding_blocks_weak_signal(self) -> None:
        # Arrange: holding_bars=1 → eff_hurdle=3.8 → mu=2.0 FAIL
        mu_bps = np.array([2.0], dtype=np.float64)
        hurdle = np.array([3.8], dtype=np.float64)
        sigma = np.array([0.002], dtype=np.float64)
        prev_w = np.zeros(1, dtype=np.float64)
        caps = PortfolioCaps(gross=3.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=0.5)

        # Act
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle,
            holding_bars=1,
            friction_safety_mult=1.0,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        # Assert: w_i = 0 (차단)
        assert w[0] == pytest.approx(0.0)

    def test_s3_safety_mult_boundary(self) -> None:
        # Arrange: friction_safety_mult=2.0, holding_bars=6, hurdle=3.8
        # eff_hurdle = 3.8 * 2.0 / 6 ≈ 1.267
        # mu=1.0 → FAIL (1.0 < 1.267), mu=1.5 → PASS (1.5 >= 1.267)
        hurdle = np.array([3.8, 3.8], dtype=np.float64)
        sigma = np.array([0.002, 0.002], dtype=np.float64)
        prev_w = np.zeros(2, dtype=np.float64)
        caps = PortfolioCaps(gross=3.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=0.5)

        mu_fail = np.array([1.0, 0.0], dtype=np.float64)
        mu_pass = np.array([0.0, 1.5], dtype=np.float64)

        w_fail = diagonal_kelly_weights(
            mu_bps=mu_fail,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle,
            holding_bars=6,
            friction_safety_mult=2.0,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
        )
        w_pass = diagonal_kelly_weights(
            mu_bps=mu_pass,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle,
            holding_bars=6,
            friction_safety_mult=2.0,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        assert w_fail[0] == pytest.approx(0.0)
        assert w_pass[1] > 0.0

    def test_s4_holding_bars_zero_guard_no_zero_division(self) -> None:
        # Arrange: holding_bars=0 → max(.,1) 가드 → ZeroDivision 없음
        mu_bps = np.array([5.0], dtype=np.float64)
        hurdle = np.array([3.8], dtype=np.float64)
        sigma = np.array([0.002], dtype=np.float64)
        prev_w = np.zeros(1, dtype=np.float64)
        caps = PortfolioCaps(gross=3.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=0.5)

        # Act: should not raise
        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle,
            holding_bars=0,
            friction_safety_mult=1.0,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        # Assert: finite output, no crash
        assert np.all(np.isfinite(w))


class TestS8DefaultBackwardCompatibility:
    """S8: use_portfolio_kelly 제거 후 diagonal 경로 기본 동작 불변 (회귀)."""

    def test_s8_default_params_give_same_result_as_before(self) -> None:
        # Arrange: holding_bars=1(default), friction_safety_mult=1.0(default)
        # 기존 로직: |mu| >= hurdle → holding=1, mult=1 → eff_hurdle=hurdle
        mu_bps = np.array([5.0, 2.0], dtype=np.float64)
        sigma = np.array([0.002, 0.002], dtype=np.float64)
        hurdle = np.array([3.8, 3.8], dtype=np.float64)
        prev_w = np.zeros(2, dtype=np.float64)
        caps = PortfolioCaps(gross=3.0, per_symbol=0.5, net=1.0, beta=2.0, target_ann_vol=0.5)

        # Act: with defaults (holding_bars=1, friction_safety_mult=1.0)
        w_default = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            friction_hurdle_bps=hurdle,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        # Assert: [0] 5.0>=3.8 → PASS (w>0), [1] 2.0<3.8 → FAIL (w=0)
        assert w_default[0] > 0.0
        assert w_default[1] == pytest.approx(0.0)
