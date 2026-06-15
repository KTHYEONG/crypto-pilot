"""Unit tests for tiered_workflow.metrics internal functions."""
import pytest
import numpy as np

from src.domain.futures.strategy.tiered_workflow.metrics import _cagr, _sharpe, _mdd


class TestCagrCompound:
    """_cagr 복리 계산 검증."""

    def test_cagr_compound_vs_arithmetic(self) -> None:
        # Arrange
        rets = [0.1, -0.05, 0.1]
        bars_per_year = len(rets)  # 1년 = 3 bars -> CAGR^1 = compound - 1

        # Act
        result = _cagr(rets, bars_per_year=float(bars_per_year))

        # Assert: 복리 base = 1.1 * 0.95 * 1.1 = 1.1495, CAGR^1 = 1.1495 - 1 = 0.1495
        expected = 1.1 * 0.95 * 1.1 - 1.0
        assert result == pytest.approx(expected, rel=1e-6)
        # 산술합 기반 (0.15) 과 달라야 함
        assert abs(result - 0.15) > 1e-4

    def test_cagr_empty_returns_zero(self) -> None:
        assert _cagr([]) == 0.0

    def test_cagr_total_loss(self) -> None:
        # prod(1 + [-1.0]) = 0.0 <= 0 → -1.0
        assert _cagr([-1.0]) == -1.0

    def test_cagr_near_total_loss(self) -> None:
        # prod이 0 이하가 되지 않는 경우: 0.4^3 = 0.064 > 0 → 정상
        rets = [-0.6, -0.6, -0.6]
        result = _cagr(rets, bars_per_year=3.0)
        assert result == pytest.approx(0.064 - 1.0, rel=1e-6)

    def test_cagr_positive_sequence(self) -> None:
        # 단조증가: 복리 > 산술
        rets = [0.01] * 100
        bars_per_year = 100.0
        result = _cagr(rets, bars_per_year=bars_per_year)
        # base = 1.01^100, CAGR^1 = 1.01^100 - 1
        expected = (1.01 ** 100) - 1.0
        assert result == pytest.approx(expected, rel=1e-6)
