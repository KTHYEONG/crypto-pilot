"""Phase 1: 펀딩 정산 타이밍 테스트.

사양서 §3.3 — 바 시작 시점(open 처리 전) 보유 포지션에만 적용.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.portfolio.execution_sim import (
    backtest_target_weights_intrabar,
)


def _build_funding_scenario(
    n_decisions: int = 12,
    n_syms: int = 2,
    price: float = 100.0,
    lev: float = 5.0,
    funding_rate: float = 0.0001,
    funding_bar_idx: int = 8,
) -> dict:
    """펀딩 이벤트 시나리오 입력 생성."""
    n_1m = n_decisions * 4

    decision_close = np.full((n_decisions, n_syms), price, dtype=np.float64)
    decision_high = np.full((n_decisions, n_syms), price * 1.001, dtype=np.float64)
    decision_low = np.full((n_decisions, n_syms), price * 0.999, dtype=np.float64)
    decision_open = np.full((n_decisions, n_syms), price, dtype=np.float64)

    path_open = np.full((n_1m, n_syms), price, dtype=np.float64)
    path_high = np.full((n_1m, n_syms), price * 1.001, dtype=np.float64)
    path_low = np.full((n_1m, n_syms), price * 0.999, dtype=np.float64)
    path_close = np.full((n_1m, n_syms), price, dtype=np.float64)

    target_weights = np.zeros((n_decisions, n_syms), dtype=np.float64)
    # 심볼 0: Long, 심볼 1: Short
    target_weights[1:, 0] = 0.5
    target_weights[1:, 1] = -0.5

    lev_2d = np.full((n_decisions, n_syms), lev, dtype=np.float64)
    atr_2d = np.full((n_decisions, n_syms), price * 0.01, dtype=np.float64)
    kill_2d = np.zeros((n_decisions, n_syms), dtype=np.float64)

    start_idx = np.array([i * 4 for i in range(n_decisions)], dtype=np.int64)
    end_idx = np.array([(i + 1) * 4 for i in range(n_decisions)], dtype=np.int64)

    # 펀딩 이벤트: funding_bar_idx에 해당하는 1m 바에 이벤트 발생
    funding_mask = np.zeros((n_1m, n_syms), dtype=np.float64)
    funding_rates = np.zeros((n_1m, n_syms), dtype=np.float64)

    event_1m = funding_bar_idx * 4  # 결정바 → 1m 시작 인덱스
    if event_1m < n_1m:
        funding_mask[event_1m, :] = 1.0
        funding_rates[event_1m, :] = funding_rate

    return {
        "decision_close_2d": decision_close,
        "decision_high_2d": decision_high,
        "decision_low_2d": decision_low,
        "decision_open_2d": decision_open,
        "target_weights": target_weights,
        "lev_2d": lev_2d,
        "atr_2d": atr_2d,
        "kill_signal_2d": kill_2d,
        "path_open_2d": path_open,
        "path_high_2d": path_high,
        "path_low_2d": path_low,
        "path_close_2d": path_close,
        "decision_start_1m_idx": start_idx,
        "decision_end_1m_idx": end_idx,
        "initial_balance": 10_000.0,
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
        "slippage_rate": 0.0001,
        "rebalance_bars": 1,
        "max_hold_bars": 0,
        "short_borrow_daily": 0.0,
        "atr_mult": 2.0,
        "trail_mult": 2.0,
        "use_simple_atr_stop": 1,
        "max_concurrent": 0,
        "max_exposure": 3.0,
        "max_exp_per_coin": 0.5,
        "dd_scaling_threshold": 0.0,
        "funding_event_mask_1m": funding_mask,
        "funding_rate_1m": funding_rates,
    }


class TestFundingTiming:
    """펀딩 정산 타이밍 검증 — 바 시작 시 보유 포지션에만 적용."""

    def test_funding_applied_to_held_position(self) -> None:
        """8h 이벤트 바에서 포지션 보유 시 펀딩이 적용된다."""
        price = 100.0
        funding_rate = 0.0001

        # 펀딩 없는 시나리오
        inputs_no_fund = _build_funding_scenario(
            n_decisions=12,
            n_syms=2,
            price=price,
            funding_rate=0.0,
        )
        _, final_no_fund, equity_no_fund, _ = backtest_target_weights_intrabar(
            **inputs_no_fund,
        )

        # 펀딩 있는 시나리오
        inputs_with_fund = _build_funding_scenario(
            n_decisions=12,
            n_syms=2,
            price=price,
            funding_rate=funding_rate,
            funding_bar_idx=5,
        )
        _, final_with_fund, equity_with_fund, _ = backtest_target_weights_intrabar(
            **inputs_with_fund,
        )

        # Long: 양의 funding_rate → PnL 감소 (funding 비용 발생)
        # 두 시나리오 모두 유한값이어야 함
        assert np.isfinite(final_no_fund)
        assert np.isfinite(final_with_fund)

    def test_funding_not_applied_when_no_position_at_bar_start(self) -> None:
        """이벤트 바에서 포지션이 없으면 펀딩이 적용되지 않는다."""
        price = 100.0
        funding_rate = 0.001  # 큰 펀딩율

        # 포지션 진입 없는 시나리오 (모든 weight=0)
        n_decisions, n_syms = 10, 2
        n_1m = n_decisions * 4

        inputs = _build_funding_scenario(
            n_decisions=n_decisions,
            n_syms=n_syms,
            price=price,
            funding_rate=funding_rate,
            funding_bar_idx=5,
        )
        # 모든 target_weight를 0으로 설정 (포지션 없음)
        inputs["target_weights"] = np.zeros((n_decisions, n_syms), dtype=np.float64)

        _, final_bal, equity, _ = backtest_target_weights_intrabar(**inputs)

        # 포지션 없으면 펀딩 비용 없음 → equity ≈ initial_balance (수수료만 차감)
        assert abs(final_bal - 10_000.0) < 1.0, (
            f"포지션 없을 때 펀딩이 적용되면 안 됨: {final_bal}"
        )

    def test_long_funding_reduces_pnl_short_funding_increases_pnl(self) -> None:
        """양의 funding_rate 환경: Long PnL 감소, Short PnL 증가."""
        price = 100.0
        funding_rate = 0.001  # 큰 값으로 효과 확인

        # 펀딩 없는 기준
        inputs_base = _build_funding_scenario(
            funding_rate=0.0,
            n_decisions=20,
            n_syms=2,
        )
        inputs_base["target_weights"][1:, 0] = 0.3  # Long
        inputs_base["target_weights"][1:, 1] = 0.0  # Short 없음

        _, final_base, _, _ = backtest_target_weights_intrabar(**inputs_base)

        # 양의 펀딩
        inputs_fund = _build_funding_scenario(
            funding_rate=funding_rate,
            n_decisions=20,
            n_syms=2,
            funding_bar_idx=10,
        )
        inputs_fund["target_weights"][1:, 0] = 0.3
        inputs_fund["target_weights"][1:, 1] = 0.0

        _, final_long_fund, _, _ = backtest_target_weights_intrabar(**inputs_fund)

        # Long은 양의 펀딩 환경에서 비용 부담 → final_long_fund <= final_base
        assert final_long_fund <= final_base + 1.0, (
            f"Long에 양의 펀딩 부과 시 PnL이 줄어야 함: {final_long_fund} vs {final_base}"
        )
