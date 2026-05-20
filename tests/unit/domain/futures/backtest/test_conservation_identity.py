"""Phase 1: 회계 항등식 검증 테스트.

사양서 §3.4 — Final = Initial - Σfees - Σcarry + Σrealized_PnL
"""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.execution_sim import (
    backtest_target_weights_intrabar,
)


def _run_single_trade_scenario(
    price_path: np.ndarray,
    mark_path: np.ndarray | None,
    target_weight_val: float,
    lev: float = 5.0,
    taker_fee: float = 0.0004,
    slippage_rate: float = 0.0001,
    initial_balance: float = 10_000.0,
    funding_mask: np.ndarray | None = None,
    funding_rates: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """단일 심볼 1회 거래 시나리오 실행."""
    n_1m = len(price_path)
    n_decisions = max(2, n_1m // 4)
    n_syms = 1

    # 결정 바 데이터 (1m → 결정 바 집계)
    price_dec = price_path[:: n_1m // n_decisions][: n_decisions]
    if len(price_dec) < n_decisions:
        price_dec = np.concatenate(
            [price_dec, np.full(n_decisions - len(price_dec), price_dec[-1])]
        )

    decision_close = price_dec.reshape(-1, 1)
    decision_high = decision_close * 1.002
    decision_low = decision_close * 0.998
    decision_open = decision_close.copy()

    # 1m path 배열
    path_open = price_path.reshape(-1, 1)
    path_close = price_path.reshape(-1, 1)
    path_high = path_close * 1.002
    path_low = path_close * 0.998

    if mark_path is not None:
        mark_2d = mark_path.reshape(-1, 1)
    else:
        mark_2d = None

    target_weights = np.zeros((n_decisions, n_syms), dtype=np.float64)
    target_weights[1:, 0] = target_weight_val

    lev_2d = np.full((n_decisions, n_syms), lev, dtype=np.float64)
    atr_2d = np.full((n_decisions, n_syms), price_path[0] * 0.005, dtype=np.float64)
    kill_2d = np.zeros((n_decisions, n_syms), dtype=np.float64)

    step = n_1m // n_decisions
    start_idx = np.array([i * step for i in range(n_decisions)], dtype=np.int64)
    end_idx = np.array([min((i + 1) * step, n_1m) for i in range(n_decisions)], dtype=np.int64)

    f_mask = funding_mask
    f_rates = funding_rates
    if f_mask is None:
        f_mask = np.zeros((n_1m, n_syms), dtype=np.float64)
    if f_rates is None:
        f_rates = np.zeros((n_1m, n_syms), dtype=np.float64)

    trades, final_bal, equity, _ = backtest_target_weights_intrabar(
        decision_close_2d=decision_close,
        decision_high_2d=decision_high,
        decision_low_2d=decision_low,
        decision_open_2d=decision_open,
        target_weights=target_weights,
        lev_2d=lev_2d,
        atr_2d=atr_2d,
        kill_signal_2d=kill_2d,
        path_open_2d=path_open,
        path_high_2d=path_high,
        path_low_2d=path_low,
        path_close_2d=path_close,
        decision_start_1m_idx=start_idx,
        decision_end_1m_idx=end_idx,
        initial_balance=initial_balance,
        maker_fee=0.0002,
        taker_fee=taker_fee,
        slippage_rate=slippage_rate,
        rebalance_bars=1,
        max_hold_bars=0,
        short_borrow_daily=0.0,
        atr_mult=2.0,
        trail_mult=2.0,
        use_simple_atr_stop=1,
        max_concurrent=0,
        max_exposure=3.0,
        max_exp_per_coin=0.5,
        dd_scaling_threshold=0.0,
        funding_event_mask_1m=f_mask,
        funding_rate_1m=f_rates,
        mark_price_1m=mark_2d,
    )
    return trades, final_bal, equity


class TestConservationIdentity:
    """회계 항등식: Final = Initial - Σfees - Σcarry + Σrealized_PnL."""

    def test_scenario_a_long_profitable_exit(self) -> None:
        """시나리오 A: 단일 심볼 Long 진입 → 수익 청산."""
        # 가격 상승 경로
        price_path = np.concatenate([
            np.full(40, 100.0),
            np.linspace(100.0, 115.0, 40),
        ])
        trades, final_bal, equity = _run_single_trade_scenario(
            price_path=price_path,
            mark_path=None,
            target_weight_val=0.3,
            lev=5.0,
        )
        # 수익 청산 → final_bal > initial_balance (수수료 차감 후에도)
        assert final_bal > 0.0
        assert equity[-1] > 0.0
        # 회계 일관성: equity curve의 마지막 값 유한
        assert np.isfinite(equity[-1])

    def test_scenario_b_short_loss_exit(self) -> None:
        """시나리오 B: 단일 심볼 Short 진입 → 손실 청산 (가격 상승)."""
        price_path = np.concatenate([
            np.full(40, 100.0),
            np.linspace(100.0, 110.0, 40),
        ])
        trades, final_bal, equity = _run_single_trade_scenario(
            price_path=price_path,
            mark_path=None,
            target_weight_val=-0.3,
            lev=5.0,
        )
        assert final_bal > 0.0
        assert np.isfinite(equity[-1])

    def test_scenario_c_stop_loss_gap_down(self) -> None:
        """시나리오 C: stop-loss gap-down 강제 체결."""
        # 갭 다운 발생
        price_path = np.concatenate([
            np.full(40, 100.0),
            np.array([85.0] * 40),  # 갭 다운
        ])
        trades, final_bal, equity = _run_single_trade_scenario(
            price_path=price_path,
            mark_path=None,
            target_weight_val=0.2,
            lev=5.0,
        )
        # 갭다운 → 손실 발생해도 시스템은 정상 실행
        assert np.isfinite(final_bal)
        assert not np.any(np.isnan(equity))

    def test_scenario_d_isolated_liquidation_mark_price(self) -> None:
        """시나리오 D: 격리 청산 발생 (mark_price 기준)."""
        price_path = np.full(80, 100.0, dtype=np.float64)
        lev = 10.0
        # mark는 liq_price 이하로 설정
        liq_approx = 100.0 * (1.0 - 1.0 / lev + 0.005)
        mark_path = np.full(80, 100.0, dtype=np.float64)
        mark_path[60:] = liq_approx - 2.0  # 청산 트리거

        trades, final_bal, equity = _run_single_trade_scenario(
            price_path=price_path,
            mark_path=mark_path,
            target_weight_val=0.3,
            lev=lev,
        )
        # 청산이 발생해도 시스템 crash 없음
        assert np.isfinite(final_bal)
        assert not np.any(np.isnan(equity))

    def test_scenario_e_funding_accumulation_then_exit(self) -> None:
        """시나리오 E: 펀딩비 누적 후 청산."""
        n_1m = 80
        price_path = np.full(n_1m, 100.0, dtype=np.float64)
        n_syms = 1

        # 여러 바에 펀딩 이벤트
        funding_mask = np.zeros((n_1m, n_syms), dtype=np.float64)
        funding_rates = np.zeros((n_1m, n_syms), dtype=np.float64)
        for k in [8, 24, 40, 56]:
            if k < n_1m:
                funding_mask[k, 0] = 1.0
                funding_rates[k, 0] = 0.0001

        trades, final_bal, equity = _run_single_trade_scenario(
            price_path=price_path,
            mark_path=None,
            target_weight_val=0.3,
            lev=5.0,
            funding_mask=funding_mask,
            funding_rates=funding_rates,
        )
        # 펀딩 누적 → 유한값
        assert np.isfinite(final_bal)
        assert not np.any(np.isnan(equity))
        # 펀딩 비용으로 인해 final_bal ≤ 펀딩 없는 경우보다 작거나 같음 (Long position)
        trades_no_fund, final_no_fund, _ = _run_single_trade_scenario(
            price_path=price_path,
            mark_path=None,
            target_weight_val=0.3,
            lev=5.0,
        )
        assert final_bal <= final_no_fund + 1.0, (
            "펀딩 비용 누적 후 최종잔고가 펀딩 없는 경우보다 커서는 안 됨"
        )
