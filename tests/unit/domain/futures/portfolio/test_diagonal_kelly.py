"""diagonal_kelly_weights 단위 테스트.

Friction filter (Step 1) was removed: mu_bps is already net-of-cost.
Tests updated accordingly — friction filter tests removed, amortized hurdle tests removed.
"""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    diagonal_kelly_weights,
)
from src.domain.futures.strategy.cs_rank import VOL_FLOOR

_DEFAULT_CAPS = PortfolioCaps(
    gross=3.0,
    per_symbol=0.10,
    net=0.30,
    beta=0.50,
    target_ann_vol=0.20,
)

_SIGMA_NORMAL = np.array([0.002, 0.002], dtype=np.float64)
_PREV_ZERO = np.zeros(2, dtype=np.float64)
_BAND = 0.01


class TestSigmaEdge:
    """Diagonal Kelly sigma-edge (VOL_FLOOR 근처, zero mu)."""

    def test_diagonal_kelly_sigma_near_vol_floor_gives_finite_weight(self) -> None:
        sigma_near_floor = np.full(2, float(VOL_FLOOR) * 1.01, dtype=np.float64)
        mu_bps = np.array([10.0, 10.0], dtype=np.float64)

        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma_near_floor,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
        )

        assert np.all(np.isfinite(w)), "VOL_FLOOR 근처 sigma도 유한 weight"
        assert np.all(np.abs(w) <= _DEFAULT_CAPS.per_symbol + 1e-9), "per_symbol cap 준수"

    def test_diagonal_kelly_zero_mu_gives_zero_weight(self) -> None:
        mu_bps = np.array([0.0, 0.0], dtype=np.float64)

        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
        )

        np.testing.assert_allclose(w, 0.0, atol=1e-12)


class TestNoTradeBand:
    """No-trade band 로직 검증."""

    def test_diagonal_kelly_no_trade_band_keeps_prev_when_small_delta(self) -> None:
        prev_w = np.array([0.10, 0.10], dtype=np.float64)
        sigma_tuned = np.full(2, 0.03086, dtype=np.float64)
        mu_bps = np.array([4.0, 4.0], dtype=np.float64)

        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma_tuned,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.01,
        )

        np.testing.assert_allclose(w, prev_w, atol=1e-6)

    def test_diagonal_kelly_no_trade_band_rebalances_when_large_delta(self) -> None:
        mu_bps = np.array([10.0, 10.0], dtype=np.float64)

        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.01,
        )

        assert not np.allclose(w, _PREV_ZERO, atol=1e-6), "large delta → 리밸런스"

    def test_diagonal_kelly_no_trade_band_zero_allows_all_rebalance(self) -> None:
        mu_bps = np.array([10.0, 10.0], dtype=np.float64)
        prev_w = np.array([0.05, 0.05], dtype=np.float64)

        w_band0 = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )
        w_band_large = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=1.0,
        )

        assert not np.allclose(w_band0, prev_w, atol=1e-6), "band=0 → new weight"
        np.testing.assert_allclose(w_band_large, prev_w, atol=1e-6)


class TestAdditionalEdgeCases:
    """출력 shape, gross cap, vol_target, btc_beta."""

    def test_diagonal_kelly_output_shape_matches_input(self) -> None:
        n = 5
        mu_bps = np.full(n, 5.0, dtype=np.float64)
        sigma = np.full(n, 0.002, dtype=np.float64)
        prev_w = np.zeros(n, dtype=np.float64)

        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        assert w.shape == (n,), f"expected ({n},), got {w.shape}"
        assert w.dtype == np.float64

    def test_diagonal_kelly_caps_respected_gross(self) -> None:
        n = 10
        mu_bps = np.full(n, 1000.0, dtype=np.float64)
        sigma = np.full(n, 0.001, dtype=np.float64)
        prev_w = np.zeros(n, dtype=np.float64)
        caps = PortfolioCaps(gross=3.0, per_symbol=0.10, net=0.30, beta=0.50, target_ann_vol=0.20)

        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        gross = float(np.sum(np.abs(w)))
        assert gross <= caps.gross + 1e-6, f"gross={gross:.4f} > caps.gross={caps.gross}"

    def test_diagonal_kelly_vol_target_override_applies(self) -> None:
        mu_bps = np.array([10.0, 10.0], dtype=np.float64)
        sigma_large = np.array([0.05, 0.05], dtype=np.float64)
        prev_w = _PREV_ZERO

        w_tight = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma_large,
            kelly_fraction=0.25,
            vol_target=0.01,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )
        w_loose = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma_large,
            kelly_fraction=0.25,
            vol_target=0.50,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )

        assert float(np.sum(np.abs(w_tight))) <= float(np.sum(np.abs(w_loose))) + 1e-9

    def test_diagonal_kelly_btc_beta_none_defaults_to_zero(self) -> None:
        mu_bps = np.array([5.0, 5.0], dtype=np.float64)

        w_none = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=_SIGMA_NORMAL,
            kelly_fraction=0.25,
            vol_target=None,
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
            caps=_DEFAULT_CAPS,
            prev_w=_PREV_ZERO,
            no_trade_band=0.0,
            btc_beta=np.zeros(2, dtype=np.float64),
        )

        np.testing.assert_allclose(w_none, w_zero, atol=1e-12)


class TestCSAmplification:
    """CS Z-Score → Mu Amplification (anti-Kelly=EW-convergence)."""

    def test_happy_path_top_z_gets_higher_weight(self) -> None:
        n = 3
        mu_bps = np.full(n, 0.5, dtype=np.float64)
        sigma = np.full(n, 0.005, dtype=np.float64)
        prev_w = np.zeros(n, dtype=np.float64)
        caps = PortfolioCaps(gross=5.0, per_symbol=2.0, net=3.0, beta=2.0, target_ann_vol=0.20)
        z_scores = np.array([2.0, 0.5, 0.0], dtype=np.float64)

        w = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            caps=caps,
            prev_w=prev_w,
            no_trade_band=0.0,
            z_scores=z_scores,
            cs_amp_alpha=2.0,
        )

        assert w[0] > w[1] > 0.0, f"expected w[0] > w[1] > 0, got w={w}"
        np.testing.assert_allclose(w[1], w[2], atol=1e-12, err_msg="z=0.5 and z=0.0 equal weight (both below median)")

    def test_all_negative_z_no_amplification(self) -> None:
        n = 3
        mu_bps = np.full(n, 30.0, dtype=np.float64)
        sigma = np.full(n, 0.002, dtype=np.float64)
        prev_w = np.zeros(n, dtype=np.float64)
        z_scores = np.array([-1.0, -0.5, -0.2], dtype=np.float64)

        w_null = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
        )
        w_amp = diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=None,
            caps=_DEFAULT_CAPS,
            prev_w=prev_w,
            no_trade_band=0.0,
            z_scores=z_scores,
            cs_amp_alpha=2.0,
        )

        np.testing.assert_allclose(w_amp, w_null, atol=1e-12)

    def test_single_symbol_amplification_identity(self) -> None:
        n = 1
        mu_bps = np.full(n, 30.0, dtype=np.float64)
        sigma = np.full(n, 0.002, dtype=np.float64)
        prev_w = np.zeros(n, dtype=np.float64)
        z_scores = np.array([3.0], dtype=np.float64)

        w_null = diagonal_kelly_weights(
            mu_bps=mu_bps, sigma=sigma, kelly_fraction=0.25,
            vol_target=None, caps=_DEFAULT_CAPS, prev_w=prev_w, no_trade_band=0.0,
        )
        w_amp = diagonal_kelly_weights(
            mu_bps=mu_bps, sigma=sigma, kelly_fraction=0.25,
            vol_target=None, caps=_DEFAULT_CAPS, prev_w=prev_w, no_trade_band=0.0,
            z_scores=z_scores, cs_amp_alpha=2.0,
        )

        np.testing.assert_allclose(w_amp, w_null, atol=1e-12)

    def test_z_scores_none_backward_compat(self) -> None:
        n = 4
        mu_bps = np.full(n, 10.0, dtype=np.float64)
        sigma = np.full(n, 0.002, dtype=np.float64)
        prev_w = np.zeros(n, dtype=np.float64)

        w_base = diagonal_kelly_weights(
            mu_bps=mu_bps, sigma=sigma, kelly_fraction=0.25,
            vol_target=None, caps=_DEFAULT_CAPS, prev_w=prev_w, no_trade_band=0.0,
        )
        w_compat = diagonal_kelly_weights(
            mu_bps=mu_bps, sigma=sigma, kelly_fraction=0.25,
            vol_target=None, caps=_DEFAULT_CAPS, prev_w=prev_w, no_trade_band=0.0,
            z_scores=None, cs_amp_alpha=2.0,
        )

        np.testing.assert_allclose(w_compat, w_base, atol=1e-12)
