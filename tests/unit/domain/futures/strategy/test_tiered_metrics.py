"""Unit tests for tiered_workflow.metrics internal functions."""
import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.metrics import (
    _bars_per_year_for_tf,
    _block_log_growth,
    _cagr,
    _cvar_95,
    _effective_sample_size_hac,
    _growth_lcb,
    _hac_sharpe,
    _sharpe,
)


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


def test_hac_sharpe_iid_close_to_naive() -> None:
    rng = np.random.default_rng(7)
    rets = rng.normal(loc=0.001, scale=0.01, size=2000).astype(np.float64)

    naive = _sharpe(rets.tolist(), bars_per_year=2190.0)
    hac = _hac_sharpe(rets, bars_per_year=2190.0)

    assert hac == pytest.approx(naive, rel=0.15)


def test_hac_effective_sample_size_shrinks_under_autocorrelation() -> None:
    rng = np.random.default_rng(11)
    noise = rng.normal(loc=0.0, scale=0.01, size=2000).astype(np.float64)
    rets = np.zeros_like(noise)
    for idx in range(1, noise.size):
        rets[idx] = 0.6 * rets[idx - 1] + noise[idx]

    n_eff = _effective_sample_size_hac(rets)

    assert n_eff < float(rets.size)
    assert n_eff >= 1.0


def test_block_growth_and_lcb_fail_closed_on_total_loss() -> None:
    blocks = _block_log_growth([-1.0, 0.01, 0.02], bars_per_year=365.0, block_size=2)

    assert blocks.size == 0


def test_growth_lcb_and_cvar_are_finite_for_regular_series() -> None:
    rets = np.asarray([0.01, -0.02, 0.005, -0.015, 0.02, -0.03], dtype=np.float64)
    blocks = _block_log_growth(rets, bars_per_year=365.0, block_size=2)

    growth_lcb = _growth_lcb(blocks, z_lcb=1.0)
    cvar_95 = _cvar_95(rets)

    assert np.isfinite(growth_lcb)
    assert cvar_95 > 0.0


def test_bars_per_year_uses_timeframe() -> None:
    assert _bars_per_year_for_tf("1h") == pytest.approx(24.0 * 365.0)
    assert _bars_per_year_for_tf("4h") == pytest.approx((24.0 * 365.0) / 4.0)
