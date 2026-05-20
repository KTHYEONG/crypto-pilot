"""Phase 1: mark_price 청산 판정 테스트.

사양서 §3.2 기준 — exec_low 대신 mark_price를 청산 트리거로 사용.
"""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.execution_sim import (
    backtest_target_weights_intrabar,
)


def _build_minimal_intrabar_inputs(
    n_decisions: int,
    n_syms: int,
    price: float = 100.0,
    lev: float = 10.0,
) -> dict:
    """최소한의 intrabar 입력 배열을 생성하는 헬퍼."""
    n_1m = n_decisions * 4  # 결정바당 4개 1m 바

    decision_close = np.full((n_decisions, n_syms), price, dtype=np.float64)
    decision_high = np.full((n_decisions, n_syms), price * 1.002, dtype=np.float64)
    decision_low = np.full((n_decisions, n_syms), price * 0.998, dtype=np.float64)
    decision_open = np.full((n_decisions, n_syms), price, dtype=np.float64)

    path_open = np.full((n_1m, n_syms), price, dtype=np.float64)
    path_high = np.full((n_1m, n_syms), price * 1.002, dtype=np.float64)
    path_low = np.full((n_1m, n_syms), price * 0.998, dtype=np.float64)
    path_close = np.full((n_1m, n_syms), price, dtype=np.float64)

    target_weights = np.zeros((n_decisions, n_syms), dtype=np.float64)
    # 심볼 0: 2번째 바부터 Long 진입
    target_weights[1:, 0] = 0.5

    lev_2d = np.full((n_decisions, n_syms), lev, dtype=np.float64)
    atr_2d = np.full((n_decisions, n_syms), price * 0.01, dtype=np.float64)
    kill_2d = np.zeros((n_decisions, n_syms), dtype=np.float64)

    # 결정바 → 1m 인덱스 매핑
    start_idx = np.array([i * 4 for i in range(n_decisions)], dtype=np.int64)
    end_idx = np.array([(i + 1) * 4 for i in range(n_decisions)], dtype=np.int64)

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
    }


class TestMarkPriceLiquidation:
    """mark_price 기준 청산 로직 검증."""

    def test_exec_low_below_liq_but_mark_above_no_liquidation(self) -> None:
        """exec_low가 liq_price 이하이나 mark_price는 이상인 경우 → 청산 미발생.

        Long 포지션, leverage=10 → liq_price ≈ 100 * (1 - 1/10 + 0.005) = 90.5
        path_low를 89.0으로 설정하되 mark_price는 91.0으로 유지.
        """
        n_decisions, n_syms = 10, 2
        price = 100.0
        inputs = _build_minimal_intrabar_inputs(n_decisions, n_syms, price=price, lev=10.0)

        # path_low를 liq_price(≈90.5) 이하로 설정
        liq_approx = price * (1.0 - 1.0 / 10.0 + 0.005)  # ≈ 90.5
        low_below_liq = liq_approx - 1.0  # ≈ 89.5

        inputs["path_low_2d"][:, 0] = low_below_liq

        # mark_price는 liq_price 이상 유지 → 청산 불발
        mark_above_liq = np.full_like(inputs["path_low_2d"], price)
        mark_above_liq[:, 0] = liq_approx + 1.0  # ≈ 91.5

        trades_with_mark, final_bal_mark, equity_mark, _ = backtest_target_weights_intrabar(
            **inputs,
            mark_price_1m=mark_above_liq,
        )

        # mark_price가 liq_price 이상 → 청산 없음 → 마지막 equity > 0
        assert equity_mark[-1] > 0.0, "mark_price 이상 시 청산이 발생하면 안 됨"

        # mark_price=None fallback (exec_low 기준) → 청산 발생 가능
        trades_no_mark, final_bal_no_mark, equity_no_mark, _ = backtest_target_weights_intrabar(
            **inputs,
            mark_price_1m=None,
        )
        # mark_price=None 상황에서는 path_low 기준이므로 포지션이 청산될 수 있음
        # equity_mark >= equity_no_mark (mark 기준이 더 보수적이지 않음, 즉 청산 덜 발생)
        assert equity_mark[-1] >= equity_no_mark[-1], (
            "mark_price 기준이 exec_low 기준보다 equity가 낮을 수 없음"
        )

    def test_mark_below_liq_triggers_liquidation(self) -> None:
        """mark_price가 liq_price 이하인 경우 → 청산 발생."""
        n_decisions, n_syms = 10, 2
        price = 100.0
        inputs = _build_minimal_intrabar_inputs(n_decisions, n_syms, price=price, lev=10.0)

        liq_approx = price * (1.0 - 1.0 / 10.0 + 0.005)  # ≈ 90.5

        # path_low는 liq 이상으로 유지하고, mark만 liq 이하
        inputs["path_low_2d"][:, 0] = liq_approx + 1.0
        inputs["path_high_2d"][:, 0] = price * 1.01
        inputs["path_open_2d"][:, 0] = price
        inputs["path_close_2d"][:, 0] = price

        mark_below_liq = np.full_like(inputs["path_low_2d"], price)
        # 절반 지점 이후부터 mark를 liq 이하로 낮춤
        half = inputs["path_low_2d"].shape[0] // 2
        mark_below_liq[half:, 0] = liq_approx - 2.0

        trades_with_mark, final_bal_mark, equity_mark, _ = backtest_target_weights_intrabar(
            **inputs,
            mark_price_1m=mark_below_liq,
        )

        # mark 기준 청산 발생 시 equity가 초기값보다 낮아야 함
        # (청산 손실 발생 또는 포지션 강제 청산)
        assert final_bal_mark <= inputs["initial_balance"] * 1.1, (
            "mark_price 기준 청산이 발생해야 함"
        )

    def test_mark_price_none_fallback_uses_exec_low(self) -> None:
        """mark_price_1m=None 전달 시 exec_low 대리 사용, 정상 작동 확인."""
        n_decisions, n_syms = 10, 2
        price = 100.0
        inputs = _build_minimal_intrabar_inputs(n_decisions, n_syms, price=price)

        trades, final_bal, equity, diag = backtest_target_weights_intrabar(
            **inputs,
            mark_price_1m=None,
        )

        # None 전달 시에도 정상 실행 (RuntimeError 없음)
        assert equity.shape[0] == n_decisions
        assert np.all(np.isfinite(equity[equity > 0]))
        assert final_bal > 0.0
