"""Phase 5: 5-cap 투영 검증.

사양서 §7.5 기준.
"""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    RiskSnapshot,
    precompute_rebalance_weights,
    project_all_caps,
)


class TestCapsProjection:
    """5-cap 투영 함수 검증."""

    def _default_caps(self) -> PortfolioCaps:
        return PortfolioCaps()

    def test_gross_cap_enforced(self) -> None:
        """Gross = 3.6 > 3.0 → 투영 후 gross ≤ 3.0."""
        w = np.array([0.6] * 6, dtype=np.float64)  # gross = 3.6
        btc_beta = np.zeros(6, dtype=np.float64)
        sigma_port = 0.01
        bars_per_year = 2190.0

        w_proj = project_all_caps(
            w,
            btc_beta=btc_beta,
            sigma_port=sigma_port,
            bars_per_year=bars_per_year,
        )
        assert float(np.sum(np.abs(w_proj))) <= 3.0 + 1e-6, (
            f"gross cap 초과: {np.sum(np.abs(w_proj)):.4f}"
        )

    def test_net_cap_enforced(self) -> None:
        """Net = 1.6 >> 0.30 → 투영 후 |net| ≤ 0.30."""
        w = np.array([0.2] * 8, dtype=np.float64)  # net = 1.6
        btc_beta = np.zeros(8, dtype=np.float64)
        sigma_port = 0.01
        bars_per_year = 2190.0

        w_proj = project_all_caps(
            w,
            btc_beta=btc_beta,
            sigma_port=sigma_port,
            bars_per_year=bars_per_year,
        )
        assert abs(float(np.sum(w_proj))) <= 0.30 + 1e-6, f"net cap 초과: {np.sum(w_proj):.4f}"

    def test_beta_cap_enforced(self) -> None:
        """beta_exposure = 1.0 > 0.50 → 투영 후 |beta_exp| ≤ 0.50."""
        n = 10
        w = np.ones(n, dtype=np.float64) * 0.1  # sum = 1.0
        btc_beta = np.ones(n, dtype=np.float64)  # 모든 심볼 beta=1
        sigma_port = 0.01
        bars_per_year = 2190.0

        w_proj = project_all_caps(
            w,
            btc_beta=btc_beta,
            sigma_port=sigma_port,
            bars_per_year=bars_per_year,
        )
        beta_exp = float(np.abs(np.dot(w_proj, btc_beta)))
        assert beta_exp <= 0.50 + 1e-6, f"beta cap 초과: {beta_exp:.4f}"

    def test_per_symbol_cap_enforced(self) -> None:
        """w[0] = 0.4 > 0.10 → 투영 후 max|w_i| ≤ 0.10."""
        w = np.array([0.4, 0.1, 0.1], dtype=np.float64)
        btc_beta = np.zeros(3, dtype=np.float64)
        sigma_port = 0.01
        bars_per_year = 2190.0

        w_proj = project_all_caps(
            w,
            btc_beta=btc_beta,
            sigma_port=sigma_port,
            bars_per_year=bars_per_year,
        )
        assert float(np.max(np.abs(w_proj))) <= 0.10 + 1e-6, (
            f"per_symbol cap 초과: {np.max(np.abs(w_proj)):.4f}"
        )

    def test_zero_weights_remain_zero(self) -> None:
        """모든 weight=0이면 투영 후에도 0."""
        w = np.zeros(5, dtype=np.float64)
        btc_beta = np.ones(5, dtype=np.float64)
        sigma_port = 0.01
        bars_per_year = 2190.0

        w_proj = project_all_caps(
            w,
            btc_beta=btc_beta,
            sigma_port=sigma_port,
            bars_per_year=bars_per_year,
        )
        np.testing.assert_allclose(w_proj, 0.0, atol=1e-12)

    def test_custom_caps_respected(self) -> None:
        """커스텀 PortfolioCaps 적용 시 해당 caps 사용."""
        caps = PortfolioCaps(gross=2.0, per_symbol=0.20, net=0.50, beta=1.0, target_ann_vol=0.30)
        w = np.array([0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float64)  # gross=2.0
        btc_beta = np.zeros(5, dtype=np.float64)
        sigma_port = 0.01
        bars_per_year = 2190.0

        w_proj = project_all_caps(
            w,
            btc_beta=btc_beta,
            sigma_port=sigma_port,
            bars_per_year=bars_per_year,
            caps=caps,
        )
        assert float(np.sum(np.abs(w_proj))) <= caps.gross + 1e-6

    def test_precompute_uses_non_zero_beta_vector_for_projection(self) -> None:
        """Precompute 경로에서 non-zero beta 입력 시 beta cap 투영이 적용된다."""
        n_bars, n_syms = 8, 2
        close_2d = np.full((n_bars, n_syms), 100.0, dtype=np.float64)
        xs_long = np.zeros((n_bars, n_syms), dtype=np.float64)
        xs_short = np.zeros((n_bars, n_syms), dtype=np.float64)
        xs_long[0, :] = 1.0  # rebalance bar(i=1)에서 양(+) 가중치 유도
        sigma_3d = np.tile(np.eye(n_syms, dtype=np.float64)[None, :, :] * 1e-8, (n_bars, 1, 1))
        beta_2d = np.ones((n_bars, n_syms), dtype=np.float64)

        out = precompute_rebalance_weights(
            close_2d,
            xs_long,
            xs_short,
            rebalance_bars=1,
            lookback=5,
            bars_per_year=1.0,
            kappa=0.0,
            f_kelly_max=10.0,
            sigma_target_ann=1.0,
            gross_cap=3.0,
            per_symbol_cap=2.0,
            sigma_3d=sigma_3d,
            risk_snapshot=RiskSnapshot(covariance_3d=sigma_3d, beta_2d=beta_2d),
        )

        beta_exp = float(np.dot(out[1], beta_2d[1]))
        assert abs(beta_exp) <= 0.50 + 1e-6

    def test_precompute_fallback_without_beta_keeps_legacy_behavior(self) -> None:
        """Beta 미입력이면 zero-vector fallback으로 기존 동작을 유지한다."""
        n_bars, n_syms = 8, 2
        close_2d = np.full((n_bars, n_syms), 100.0, dtype=np.float64)
        xs_long = np.zeros((n_bars, n_syms), dtype=np.float64)
        xs_short = np.zeros((n_bars, n_syms), dtype=np.float64)
        xs_long[0, :] = 1.0
        sigma_3d = np.tile(np.eye(n_syms, dtype=np.float64)[None, :, :] * 1e-8, (n_bars, 1, 1))

        out = precompute_rebalance_weights(
            close_2d,
            xs_long,
            xs_short,
            rebalance_bars=1,
            lookback=5,
            bars_per_year=1.0,
            kappa=0.0,
            f_kelly_max=10.0,
            sigma_target_ann=1.0,
            gross_cap=3.0,
            per_symbol_cap=2.0,
            sigma_3d=sigma_3d,
        )
        out_with_zero_beta = precompute_rebalance_weights(
            close_2d,
            xs_long,
            xs_short,
            rebalance_bars=1,
            lookback=5,
            bars_per_year=1.0,
            kappa=0.0,
            f_kelly_max=10.0,
            sigma_target_ann=1.0,
            gross_cap=3.0,
            per_symbol_cap=2.0,
            sigma_3d=sigma_3d,
            risk_snapshot=RiskSnapshot(
                covariance_3d=sigma_3d,
                beta_2d=np.zeros((n_bars, n_syms), dtype=np.float64),
            ),
        )
        np.testing.assert_allclose(out, out_with_zero_beta, atol=1e-12)
