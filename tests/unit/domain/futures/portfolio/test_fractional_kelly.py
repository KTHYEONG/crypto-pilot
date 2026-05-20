"""Phase 5: Fractional Kelly 0.25x 검증.

사양서 §7.4 기준.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import (
    KELLY_FRACTION,
    _kelly_raw,
    _kelly_scaled,
)


class TestFractionalKelly:
    """_kelly_scaled = _kelly_raw × 0.25 검증."""

    def test_scaled_equals_raw_times_fraction(self) -> None:
        """full Kelly 대비 0.25x 검증."""
        mu = np.array([0.001, -0.0005, 0.002, 0.0, -0.001])
        sigma_diag = np.array([0.01, 0.015, 0.008, 0.012, 0.02])

        full = _kelly_raw(mu, sigma_diag, f_kelly_max=10.0)
        scaled = _kelly_scaled(mu, sigma_diag, f_kelly_max=10.0)

        np.testing.assert_allclose(scaled, full * KELLY_FRACTION, rtol=1e-9)

    def test_kelly_fraction_constant_is_025(self) -> None:
        """KELLY_FRACTION 모듈 상수 = 0.25."""
        assert KELLY_FRACTION == 0.25, f"KELLY_FRACTION은 0.25여야 함: {KELLY_FRACTION}"

    def test_fraction_not_injectable(self) -> None:
        """KELLY_FRACTION이 외부 주입 불가 (_kelly_scaled 시그니처에 없어야 함)."""
        sig = inspect.signature(_kelly_scaled)
        forbidden = ["fraction", "kelly_fraction", "f_fraction", "frac"]
        for p in forbidden:
            assert p not in sig.parameters, (
                f"파라미터 '{p}'가 외부 주입 가능해서는 안 됨"
            )

    def test_scaled_output_shape_preserved(self) -> None:
        """출력 shape가 입력 shape와 동일."""
        n = 20
        rng = np.random.default_rng(0)
        mu = rng.normal(0, 0.001, n)
        sigma_diag = rng.uniform(0.005, 0.02, n)

        result = _kelly_scaled(mu, sigma_diag, f_kelly_max=5.0)
        assert result.shape == (n,), f"출력 shape 불일치: {result.shape} vs ({n},)"

    def test_zero_mu_yields_zero_weight(self) -> None:
        """μ=0이면 weight=0."""
        mu = np.zeros(5)
        sigma_diag = np.ones(5) * 0.01

        result = _kelly_scaled(mu, sigma_diag, f_kelly_max=10.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)
