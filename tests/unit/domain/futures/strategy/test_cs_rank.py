"""Unit tests for cs_rank module.

Covers: SymbolSignal, neutralize_cross_section, rank_and_select.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.domain.futures.strategy.cs_rank import (
    VOL_FLOOR,
    SymbolSignal,
    neutralize_cross_section,
    rank_and_select,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    raw_mu: float,
    volatility: float = 0.01,
    n_obs: int = 100,
    t_stat: float = 2.0,
    valid: bool = True,
) -> SymbolSignal:
    return SymbolSignal(
        raw_mu=raw_mu,
        volatility=volatility,
        n_obs=n_obs,
        t_stat=t_stat,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# T1 — CS Neutralization (beta_btc=None, demean)
# ---------------------------------------------------------------------------

def test_neutralize_cross_section_demean_preserves_rank() -> None:
    """CS demean: 결과 평균≈0, 입력 순위 보존."""
    # Arrange
    mu = np.array([2.0, 1.0, 0.0], dtype=np.float64)

    # Act
    result = neutralize_cross_section(mu)

    # Assert
    assert pytest.approx(float(result.mean()), abs=1e-10) == 0.0
    assert result[0] > result[1] > result[2]


def test_neutralize_cross_section_demean_output_shape() -> None:
    """출력 shape이 입력과 동일."""
    # Arrange
    mu = np.array([5.0, 3.0, 1.0, -1.0], dtype=np.float64)

    # Act
    result = neutralize_cross_section(mu)

    # Assert
    assert result.shape == mu.shape


# ---------------------------------------------------------------------------
# T1b — CS Neutralization (beta_btc 사용)
# ---------------------------------------------------------------------------

def test_neutralize_cross_section_beta_neutralize() -> None:
    """beta_btc=[1,1,1]: 결과 평균≈0 (μ_mkt 전량 차감)."""
    # Arrange
    mu = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    beta = np.ones(3, dtype=np.float64)

    # Act
    result = neutralize_cross_section(mu, beta_btc=beta)

    # Assert — β_i*μ_mkt = 1.0*1.0 = 1.0 전부 차감 → mean≈0
    assert pytest.approx(float(result.mean()), abs=1e-10) == 0.0


def test_neutralize_cross_section_beta_neutralize_zero_beta() -> None:
    """beta=0이면 mu 그대로 + demean만 적용."""
    # Arrange
    mu = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    beta = np.zeros(3, dtype=np.float64)

    # Act
    result = neutralize_cross_section(mu, beta_btc=beta)

    # Assert: β=0 → mu_neutral = mu - 0*mu_mkt = mu; 평균≈2.0 (demean 미적용)
    # 실제로 beta_btc=zeros → mu_neutral = mu - 0 = mu → mean=2.0 (not zero)
    assert pytest.approx(float(result.mean()), abs=1e-10) == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# T1c — edge: N<2이면 그대로
# ---------------------------------------------------------------------------

def test_neutralize_cross_section_single_symbol() -> None:
    """N=1이면 입력값 그대로 반환."""
    # Arrange
    mu = np.array([3.0], dtype=np.float64)

    # Act
    result = neutralize_cross_section(mu)

    # Assert
    assert pytest.approx(float(result[0])) == 3.0


def test_neutralize_cross_section_empty_array() -> None:
    """N=0이면 빈 배열 반환."""
    # Arrange
    mu = np.array([], dtype=np.float64)

    # Act
    result = neutralize_cross_section(mu)

    # Assert
    assert result.size == 0


# ---------------------------------------------------------------------------
# T2 — Top-K Hysteresis
# ---------------------------------------------------------------------------

def test_rank_and_select_hysteresis_keeps_within_buffer() -> None:
    """prev에 있고 rank ≤ k_rank+rank_buffer이면 유지."""
    # Arrange: 5개 심볼, A의 z-score가 4위 (k_rank=3, rank_buffer=1 → 버퍼 내)
    # A를 의도적으로 4위에 배치: A=0.5, B=1.0, C=2.0, D=3.0, E=4.0 (E가 1위)
    signals: dict[str, SymbolSignal] = {
        "E": _make_signal(4.0),
        "D": _make_signal(3.0),
        "C": _make_signal(2.0),
        "A": _make_signal(0.5),   # 4위 → rank_buffer=1이면 유지
        "B": _make_signal(-1.0),  # 5위
    }
    prev = frozenset({"A"})

    # Act
    selected, _z = rank_and_select(
        signals,
        k_rank=3,
        sector_cap=10,
        prev_selection=prev,
        rank_buffer=1,
    )

    # Assert: A가 4위(≤ 3+1=4) → 유지
    assert "A" in selected
    assert len(selected) == 3


def test_rank_and_select_hysteresis_evicts_outside_buffer() -> None:
    """rank > k_rank+rank_buffer이면 prev 심볼 교체."""
    # Arrange: A가 5위 (k_rank=3, rank_buffer=1 → 버퍼=4, 5위 초과)
    signals: dict[str, SymbolSignal] = {
        "E": _make_signal(5.0),
        "D": _make_signal(4.0),
        "C": _make_signal(3.0),
        "F": _make_signal(2.0),   # 4위 (버퍼 내지만 prev에 없음)
        "A": _make_signal(0.0),   # 5위 → evict
    }
    prev = frozenset({"A"})

    # Act
    selected, _ = rank_and_select(
        signals,
        k_rank=3,
        sector_cap=10,
        prev_selection=prev,
        rank_buffer=1,
    )

    # Assert: A가 5위(> 3+1=4) → 제거됨
    assert "A" not in selected
    assert len(selected) == 3


def test_rank_and_select_no_hysteresis_when_empty_prev() -> None:
    """prev=frozenset() → 순수 Top-K 선택."""
    # Arrange
    signals: dict[str, SymbolSignal] = {
        "A": _make_signal(3.0),
        "B": _make_signal(2.0),
        "C": _make_signal(1.0),
        "D": _make_signal(0.0),
    }

    # Act
    selected, z_scores = rank_and_select(
        signals,
        k_rank=2,
        sector_cap=10,
        prev_selection=frozenset(),
        rank_buffer=1,
    )

    # Assert: Top-2 = A, B
    assert selected == frozenset({"A", "B"})
    assert len(z_scores) == 4  # 모든 valid 심볼 z_score 반환


# ---------------------------------------------------------------------------
# T2c — valid=False 제외
# ---------------------------------------------------------------------------

def test_rank_and_select_excludes_invalid_signals() -> None:
    """valid=False 심볼은 선택 불가."""
    # Arrange
    signals: dict[str, SymbolSignal] = {
        "A": _make_signal(10.0, valid=False),  # invalid — 절대 선택 불가
        "B": _make_signal(2.0),
        "C": _make_signal(1.0),
    }

    # Act
    selected, z_scores = rank_and_select(
        signals,
        k_rank=2,
        sector_cap=10,
        prev_selection=frozenset(),
        rank_buffer=0,
    )

    # Assert
    assert "A" not in selected
    assert "A" not in z_scores
    assert selected == frozenset({"B", "C"})


def test_rank_and_select_all_invalid_returns_empty() -> None:
    """모든 심볼이 invalid이면 빈 결과 반환."""
    # Arrange
    signals: dict[str, SymbolSignal] = {
        "A": _make_signal(1.0, valid=False),
        "B": _make_signal(2.0, valid=False),
    }

    # Act
    selected, z_scores = rank_and_select(
        signals,
        k_rank=2,
        sector_cap=10,
        prev_selection=frozenset(),
        rank_buffer=0,
    )

    # Assert
    assert selected == frozenset()
    assert z_scores == {}


def test_rank_and_select_empty_signals_returns_empty() -> None:
    """빈 signals 딕셔너리 → 빈 결과 반환."""
    # Act
    selected, z_scores = rank_and_select(
        {},
        k_rank=3,
        sector_cap=10,
        prev_selection=frozenset(),
        rank_buffer=1,
    )

    # Assert
    assert selected == frozenset()
    assert z_scores == {}


# ---------------------------------------------------------------------------
# T3 — SymbolSignal dataclass
# ---------------------------------------------------------------------------

def test_symbol_signal_frozen() -> None:
    """frozen=True: attribute 수정 시 FrozenInstanceError."""
    # Arrange
    sig = SymbolSignal(raw_mu=1.0, volatility=0.01, n_obs=50, t_stat=1.5, valid=True)

    # Act / Assert
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        sig.raw_mu = 2.0  # type: ignore[misc]


def test_symbol_signal_vol_floor_contract() -> None:
    """VOL_FLOOR import 가능, 양수 확인."""
    # Assert
    assert isinstance(VOL_FLOOR, float)
    assert VOL_FLOOR > 0.0


def test_symbol_signal_slots() -> None:
    """slots=True: __dict__ 없음."""
    # Arrange
    sig = SymbolSignal(raw_mu=0.0, volatility=0.01, n_obs=1, t_stat=0.0, valid=True)

    # Assert
    assert not hasattr(sig, "__dict__")


def test_symbol_signal_equality() -> None:
    """동일 값으로 생성된 두 인스턴스는 동등."""
    # Arrange
    sig_a = SymbolSignal(raw_mu=1.0, volatility=0.01, n_obs=100, t_stat=2.0, valid=True)
    sig_b = SymbolSignal(raw_mu=1.0, volatility=0.01, n_obs=100, t_stat=2.0, valid=True)

    # Assert
    assert sig_a == sig_b


# ---------------------------------------------------------------------------
# T4 — BTC-beta neutralize via rank_and_select
# ---------------------------------------------------------------------------

class TestNeutralizeWithBtcBeta:
    """T4: BTC-beta single-factor neutralize rank_and_select 통합 검증."""

    def test_rank_and_select_beta_btc_propagated_to_neutralize(self) -> None:
        """mu=[3,2,1], beta=[1,1,1] → neutral=[1,0,-1] → BTC, ETH 선택."""
        # Arrange: 3종목, beta_btc=1.0 균일 → mu_neutral = mu - mu_mkt
        sigs = {
            "BTC": SymbolSignal(raw_mu=3.0, volatility=0.002, n_obs=100, t_stat=2.5, valid=True, beta_btc=1.0),
            "ETH": SymbolSignal(raw_mu=2.0, volatility=0.002, n_obs=100, t_stat=2.5, valid=True, beta_btc=1.0),
            "SOL": SymbolSignal(raw_mu=1.0, volatility=0.002, n_obs=100, t_stat=2.5, valid=True, beta_btc=1.0),
        }

        # Act
        selected, z_scores = rank_and_select(
            sigs, k_rank=2, sector_cap=10,
            prev_selection=frozenset(), rank_buffer=0,
        )

        # Assert: BTC, ETH 선택 (neutral 기준 상위 2개)
        assert "BTC" in selected
        assert "ETH" in selected
        assert "SOL" not in selected
        # z_score 순서: BTC > ETH > SOL
        assert z_scores["BTC"] > z_scores["ETH"] > z_scores["SOL"]

    def test_rank_and_select_all_beta_none_falls_back_to_demean(self) -> None:
        """beta_btc=None → demean fallback, mu 높은 A 선택."""
        # Arrange
        sigs = {
            "A": SymbolSignal(raw_mu=5.0, volatility=0.001, n_obs=50, t_stat=2.0, valid=True),
            "B": SymbolSignal(raw_mu=1.0, volatility=0.001, n_obs=50, t_stat=2.0, valid=True),
        }

        # Act
        selected, z_scores = rank_and_select(
            sigs, k_rank=1, sector_cap=10,
            prev_selection=frozenset(), rank_buffer=0,
        )

        # Assert: A 선택 (mu 높음)
        assert "A" in selected
        assert z_scores["A"] > z_scores["B"]

    def test_symbol_signal_beta_btc_default_is_none(self) -> None:
        """beta_btc 미지정 시 기본값 None — backward compat 보장."""
        # Arrange
        sig = SymbolSignal(raw_mu=1.0, volatility=0.001, n_obs=50, t_stat=2.0, valid=True)

        # Assert
        assert sig.beta_btc is None
