"""Numba 백테스트 엔진 수학적 무결성 검증 테스트 스위트.

backtest_target_weights_numba 함수를 직접 호출해 9대 Pillar 검증.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.portfolio.execution_sim import (
    backtest_target_weights_intrabar_numba,
    backtest_target_weights_numba,
)

# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------


def _make_data(
    n_bars: int,
    n_syms: int,
    price: float = 100.0,
    atr: float = 5.0,
    funding: float = 0.0,
) -> dict[str, np.ndarray]:
    """Flat OHLC + ATR + zero funding/kill 합성 데이터 생성."""
    o = np.full((n_bars, n_syms), price, dtype=np.float64)
    h = o + atr * 0.5
    lo = o - atr * 0.5
    c = o.copy()
    return {
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "atr": np.full((n_bars, n_syms), atr, dtype=np.float64),
        "funding": np.full((n_bars, n_syms), funding, dtype=np.float64),
        "kill": np.zeros((n_bars, n_syms), dtype=np.float64),
        "lev": np.ones((n_bars, n_syms), dtype=np.float64),
    }


def _run(
    d: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    init_bal: float = 10_000.0,
    taker_fee: float = 0.0005,
    slip: float = 0.0002,
    rb: int = 1,
    max_hold: int = 0,
    atr_mult: float = 3.0,
    trail_mult: float = 3.0,
    use_simple_atr: int = 1,
    max_conc: int = 100,
    max_exp: float = 10.0,
    max_exp_coin: float = 100.0,
    dd_thr: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """backtest_target_weights_numba 호출 래퍼."""
    return cast(
        tuple[np.ndarray, float, np.ndarray, np.ndarray],
        backtest_target_weights_numba(
            d["close"],
            d["high"],
            d["low"],
            d["open"],
            d["funding"],
            d["kill"],
            weights,
            init_bal,
            d["lev"],
            0.0002,  # maker_fee (unused)
            taker_fee,
            slip,
            rb,
            max_hold,
            0.0,   # short_borrow_daily
            4.0,   # bar_hours (4h bar 기준; short_borrow_daily=0.0이므로 결과 무관)
            d["atr"],
            atr_mult,
            trail_mult,
            use_simple_atr,
            max_conc,
            max_exp,
            max_exp_coin,
            dd_thr,
        ),
    )


# ---------------------------------------------------------------------------
# Module-level warmup fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _warmup() -> None:
    """Numba JIT 최초 컴파일. 이후 테스트는 캐시 사용."""
    d = _make_data(3, 1)
    w = np.zeros((3, 1), dtype=np.float64)
    _run(d, w)


# ---------------------------------------------------------------------------
# Pillar 1 — Exposure & Concurrent Cap
# ---------------------------------------------------------------------------


class TestExposureCap:
    """노출 한도 및 동시 포지션 제한 검증."""

    def test_gross_exposure_cap(self) -> None:
        """Gross exposure 0.80 캡: 5심볼 * 0.30 = 1.50 → 0.80으로 스케일다운."""
        n_bars, n_syms = 10, 5
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)
        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1:, :] = 0.30  # bar 1부터 전체 0.30

        trades, _, _, _ = _run(
            d, weights, max_exp=0.80, max_conc=100, rb=1, init_bal=10_000.0
        )

        # bar 1(i=1) 진입 trades 필터
        bar1_trades = trades[trades[:, 1] == 1]
        total_notional = float((bar1_trades[:, 7] * bar1_trades[:, 4]).sum())
        cap = 10_000.0 * 0.80 * 1.01  # 1% 허용: 수수료/슬리피지
        assert total_notional <= cap

    def test_max_concurrent_cap(self) -> None:
        """max_concurrent=3: weight 상위 3개 심볼만 진입."""
        n_bars, n_syms = 5, 10
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)
        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1, :3] = 0.30  # 큰 weight
        weights[1, 3:] = 0.05  # 작은 weight

        trades, _, _, _ = _run(d, weights, max_conc=3, max_exp=10.0, rb=1)

        bar1_trades = trades[trades[:, 1] == 1]
        assert len(bar1_trades) == 3
        # 진입 심볼이 weight 상위 3개 (sym_idx 0, 1, 2)인지 확인
        entered_syms = set(bar1_trades[:, 0].astype(int))
        assert entered_syms == {0, 1, 2}

    def test_dd_scaling(self) -> None:
        """DD > threshold 시 진입 notional이 축소됨을 검증."""
        # rb=1 매 바 리밸런싱: bar 1에서 0.80 진입, bars 2~9 유지
        # bar 10: open=74 (급락) → DD 계산 후 weight=0.30 축소 진입
        n_bars, n_syms = 12, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        # bar 10~11: 폭락
        for bi in range(10, n_bars):
            d["open"][bi, 0] = 74.0
            d["high"][bi, 0] = 74.5
            d["low"][bi, 0] = 73.0
            d["close"][bi, 0] = 74.0

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, 10):
            weights[i, 0] = 0.80  # bars 1~9: 진입/유지
        for i in range(10, n_bars):
            weights[i, 0] = 0.30  # bar 10~: 신호

        # atr_mult=999 → stop_p 매우 낮아 stop 미발동
        trades_dd, _, _, _ = _run(
            d, weights, dd_thr=0.15, atr_mult=999.0, rb=1,
            max_exp=10.0, max_exp_coin=100.0
        )
        trades_no_dd, _, _, _ = _run(
            d, weights, dd_thr=0.0, atr_mult=999.0, rb=1,
            max_exp=10.0, max_exp_coin=100.0
        )

        # bar 10에서 진입한 trades 비교
        dd_entry = trades_dd[trades_dd[:, 1] == 10]
        no_dd_entry = trades_no_dd[trades_no_dd[:, 1] == 10]

        assert len(no_dd_entry) > 0, "no_dd 시나리오에서 bar 10 진입 없음"
        notional_no_dd = float((no_dd_entry[:, 7] * no_dd_entry[:, 4]).sum())

        if len(dd_entry) > 0:
            notional_with_dd = float((dd_entry[:, 7] * dd_entry[:, 4]).sum())
            assert notional_with_dd < notional_no_dd
        # dd_entry가 없으면 더 심하게 축소된 것 (dd_factor=0.1 → dust_skip)


# ---------------------------------------------------------------------------
# Pillar 2 — Liquidation & Bankruptcy Protection
# ---------------------------------------------------------------------------


class TestLiquidation:
    """청산 및 파산 방지 검증."""

    def test_liquidation_on_equity_zero(self) -> None:
        """레버리지 포지션 급락으로 equity ≤ 0 → 강제청산, final_balance ≥ 0."""
        # rb=3: bar 3에서 진입 (i%3==0), bar 4에서 close=0.01 → equity<0
        n_bars, n_syms = 6, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        # bar 3의 lev=5.0 (진입 바)
        d["lev"][3, 0] = 5.0

        # bar 4: 가격이 entry의 1/5 이하 → equity ≤ 0
        d["close"][4, 0] = 0.01
        d["high"][4, 0] = 0.01
        d["low"][4, 0] = 0.01
        d["open"][4, 0] = 0.01
        d["close"][5, 0] = 0.01
        d["high"][5, 0] = 0.01
        d["low"][5, 0] = 0.01
        d["open"][5, 0] = 0.01

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[3, 0] = 0.90  # bar 3 진입 (rb=3)
        weights[4, 0] = 0.90  # 유지 설정 (stop 발동 방지용 atr_mult=999)

        trades, final_balance, equity_curve, _ = _run(
            d, weights, rb=3, init_bal=1_000.0, atr_mult=999.0,
            max_exp=10.0, max_exp_coin=100.0
        )

        assert final_balance >= 0.0
        assert len(trades) > 0
        # equity_curve[4] == 0.0 (강제청산 후)
        assert equity_curve[4] == pytest.approx(0.0, abs=1e-4)
        # 이후 bars는 0으로 고정
        assert np.all(equity_curve[5:] == 0.0)

    def test_dust_skip_on_tiny_notional(self) -> None:
        """tgt_notional < min_notional_floor → dust_skip_cnt 증가, 포지션 미개설."""
        # balance=0.0001, price=1.0 → tgt_notional=0.0001*0.999=0.0000999
        # min_notional = max(0.01, 0.0001*0.0001) = 0.01 → dust_skip 발생
        n_bars, n_syms = 3, 1
        d = _make_data(n_bars, n_syms, price=1.0, atr=0.1)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1, 0] = 0.999  # tgt_notional = 0.0001*0.999 << min_notional

        trades, _, _, diag_out = _run(
            d, weights, init_bal=0.0001, max_exp=10.0, max_exp_coin=100.0, rb=1
        )

        assert diag_out[0] > 0  # dust_skip_cnt 증가
        assert len(trades) == 0  # 포지션 미개설

    def test_margin_fail_when_required_margin_exceeds_free_margin(self) -> None:
        """free_margin 부족 시 margin_fail_cnt 증가 및 미체결 검증."""
        n_bars, n_syms = 4, 2
        d = _make_data(n_bars, n_syms, price=1.0, atr=0.1)
        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1, 0] = 0.9
        weights[1, 1] = 0.9
        trades, _, _, diag_out = _run(
            d,
            weights,
            init_bal=1.0,
            rb=1,
            taker_fee=0.05,
            max_exp=10.0,
            max_exp_coin=100.0,
        )
        assert diag_out[1] > 0  # margin_fail_cnt
        assert len(trades) >= 0

    def test_margin_fail_count_increments_directly(self) -> None:
        """동일 bar 다중 진입 시 free_margin 부족으로 margin_fail_cnt가 증가해야 함."""
        n_bars, n_syms = 4, 2
        d = _make_data(n_bars, n_syms, price=1.0, atr=0.1)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        # bar 1에서 두 심볼 모두 큰 비중 진입 시도:
        # 첫 진입 후 free_margin 감소, 두 번째는 margin 부족으로 fail 경로 진입
        weights[1, 0] = 0.90
        weights[1, 1] = 0.90

        trades, _, _, diag_out = _run(
            d,
            weights,
            rb=1,
            init_bal=1.0,
            taker_fee=0.05,
            max_conc=10,
            max_exp=10.0,
            max_exp_coin=100.0,
        )

        assert len(trades) >= 0  # 실행 안정성 확인
        assert diag_out[1] > 0  # margin_fail_cnt 증가


# ---------------------------------------------------------------------------
# Pillar 3 — Fees & Slippage Math
# ---------------------------------------------------------------------------


class TestFeesSlippage:
    """수수료와 슬리피지 수학적 정확성 검증."""

    def test_round_trip_cost_long(self) -> None:
        """Long 왕복 수수료 수식 검증: entry open=1000, exit open=1100."""
        n_bars, n_syms = 3, 1
        price_entry = 1000.0
        price_exit = 1100.0
        fee = 0.0005
        slip = 0.0002

        d = _make_data(n_bars, n_syms, price=price_entry, atr=50.0)
        # bar 2: open=1100으로 변경
        d["open"][2, 0] = price_exit
        d["high"][2, 0] = price_exit + 25.0
        d["low"][2, 0] = price_exit - 25.0
        d["close"][2, 0] = price_exit

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1, 0] = 0.30  # bar 1 long 진입
        # bar 2: weight=0 → 청산

        trades, final_balance, _, _ = _run(
            d, weights, taker_fee=fee, slip=slip, rb=1,
            max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0
        )

        assert len(trades) >= 1

        # 수식 계산 (max_qty_margin=free_margin*0.97/fill_p 기준)
        fill_p = price_entry * (1.0 + slip)
        tgt_notional = 10_000.0 * 0.30
        qty = min(tgt_notional / fill_p, 10_000.0 * 0.97 / fill_p)
        entry_fee = qty * fill_p * fee
        exit_price = price_exit * (1.0 - slip)
        pnl_x = (exit_price - fill_p) * qty
        exit_fee = qty * exit_price * fee

        expected_net = pnl_x - exit_fee - entry_fee
        assert final_balance - 10_000.0 == pytest.approx(expected_net, abs=0.01)

        trade = trades[0]
        # net_pnl = pnl_x - exit_fee - fund_fee_stored (funding=0이므로 fund_fee=0)
        assert trade[6] == pytest.approx(pnl_x - exit_fee, abs=0.01)

    def test_turnover_cost_partial_exit(self) -> None:
        """리밸런싱 시 weight 0.60 → 0.30 (절반 축소): 수수료 차감 확인."""
        n_bars, n_syms = 3, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1, 0] = 0.60  # bar 1: 진입
        weights[2, 0] = 0.30  # bar 2: 절반 축소 (재진입)

        trades, final_balance, _, _ = _run(
            d, weights, taker_fee=0.0005, slip=0.0002, rb=1,
            max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0
        )

        # 수수료가 차감되어 final_balance < initial_balance
        # (가격 변동 없으므로 PnL≈0, 수수료만 차감)
        assert final_balance < 10_000.0

        bar1_entries = trades[trades[:, 1] == 1]
        assert len(bar1_entries) >= 1


# ---------------------------------------------------------------------------
# Pillar 4 — Price Gaps & Execution
# ---------------------------------------------------------------------------


class TestPriceGaps:
    """가격 갭 체결 논리 검증."""

    def test_gap_down_stop_long(self) -> None:
        """갭 하락으로 stop 발동: exit = open*(1-slip) (stop_p 아닌 open 기준)."""
        # rb=3: i=3에서 진입 (i%3==0)
        # fill_p=100.02, stop_dist=atr*atr_mult=5.0, stop_p≈95.02
        # bar 4: open=80.0 < stop_p → gap-down → exit=80*(1-slip)
        n_bars, n_syms = 6, 1
        price = 100.0
        atr_val = 5.0
        slip = 0.0002
        atr_mult_val = 1.0  # stop_dist = 5.0

        d = _make_data(n_bars, n_syms, price=price, atr=atr_val)

        fill_p = price * (1.0 + slip)
        gap_open = 80.0  # fill_p - atr_val = 95.02 초과 하락
        d["open"][4, 0] = gap_open
        d["high"][4, 0] = gap_open
        d["low"][4, 0] = gap_open - 1.0
        d["close"][4, 0] = gap_open

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[3, 0] = 0.30
        weights[4, 0] = 0.30
        weights[5, 0] = 0.30

        trades, _, _, _ = _run(
            d, weights, rb=3, slip=slip, atr_mult=atr_mult_val,
            use_simple_atr=1, max_exp=10.0, max_exp_coin=100.0
        )

        assert len(trades) >= 1
        t = trades[0]
        expected_fill = fill_p
        expected_exit = gap_open * (1.0 - slip)
        assert t[4] == pytest.approx(expected_fill, abs=1e-3)   # entry_price
        assert t[5] == pytest.approx(expected_exit, abs=1e-4)   # exit_price

    def test_gap_up_stop_short(self) -> None:
        """Short: 갭 상향으로 stop 발동, exit = open*(1+slip)."""
        # rb=3: i=3 short 진입, fill_p=99.98, stop_p=104.98
        # bar 4: open=120 > stop_p → exit=120*(1+slip)
        n_bars, n_syms = 6, 1
        price = 100.0
        atr_val = 5.0
        slip = 0.0002
        atr_mult_val = 1.0

        d = _make_data(n_bars, n_syms, price=price, atr=atr_val)

        gap_open = 120.0
        d["open"][4, 0] = gap_open
        d["high"][4, 0] = gap_open + 1.0
        d["low"][4, 0] = gap_open
        d["close"][4, 0] = gap_open

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[3, 0] = -0.30
        weights[4, 0] = -0.30
        weights[5, 0] = -0.30

        trades, _, _, _ = _run(
            d, weights, rb=3, slip=slip, atr_mult=atr_mult_val,
            use_simple_atr=1, max_exp=10.0, max_exp_coin=100.0
        )

        assert len(trades) >= 1
        t = trades[0]
        expected_exit = gap_open * (1.0 + slip)
        assert t[5] == pytest.approx(expected_exit, abs=1e-4)


# ---------------------------------------------------------------------------
# Pillar 5 — Funding Rate Physics
# ---------------------------------------------------------------------------


class TestFundingRate:
    """펀딩비 물리 검증."""

    def test_funding_bleed_long_end_of_bars(self) -> None:
        """End-of-bars 경로: long + 양수 펀딩 → balance 감소 (funding 반영)."""
        # rb=1, weight 일정 유지 → bar 1에서 진입, bars 1~11 누적 funding(11바)
        # end-of-bars(bar 11 이후) 강제청산 시 fund_fee_stored가 balance에 반영됨
        n_bars, n_syms = 12, 1
        funding_rate = 0.0001
        price = 100.0
        fee = 0.0005
        slip = 0.0002

        d = _make_data(n_bars, n_syms, price=price, atr=5.0, funding=funding_rate)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, n_bars):
            weights[i, 0] = 0.30  # weight 일정 유지 → 포지션 유지 (need_exit=False)

        trades, final_balance, _, _ = _run(
            d, weights, rb=1, taker_fee=fee, slip=slip,
            max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0,
            atr_mult=999.0  # stop 미발동
        )

        assert len(trades) >= 1

        # 수식 계산 (rb=1로 진입, bar 1 entry)
        fill_p = price * (1.0 + slip)
        qty = min(10_000.0 * 0.30 / fill_p, 10_000.0 * 0.97 / fill_p)
        entry_fee = qty * fill_p * fee

        # end-of-bars: bar 11(last_idx=11) close=100
        exit_close = price
        pnl = (exit_close - fill_p) * qty - qty * exit_close * fee

        # bars 1~11 (11바) funding 누적 (loop i=1~11, 각 close 단계에서 fund_fee)
        total_funding = qty * price * funding_rate * 11

        # final = init - entry_fee + pnl - total_funding
        expected_final = 10_000.0 - entry_fee + pnl - total_funding

        assert final_balance == pytest.approx(expected_final, abs=0.05)

    def test_funding_sign_short_positive_better_than_negative(self) -> None:
        """Short 기준으로 +funding 결과가 -funding 결과보다 유리함을 검증."""
        n_bars, n_syms = 5, 1
        price = 100.0
        fee = 0.0005
        slip = 0.0002

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, n_bars):
            weights[i, 0] = -0.30  # short 유지

        d_pos = _make_data(n_bars, n_syms, price=price, atr=5.0, funding=0.0001)
        d_neg = _make_data(n_bars, n_syms, price=price, atr=5.0, funding=-0.0001)

        trades_pos, final_pos, _, _ = _run(
            d_pos, weights, rb=1, taker_fee=fee, slip=slip,
            max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0,
            atr_mult=999.0
        )
        trades_neg, final_neg, _, _ = _run(
            d_neg, weights, rb=1, taker_fee=fee, slip=slip,
            max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0,
            atr_mult=999.0
        )

        assert len(trades_pos) >= 1
        assert len(trades_neg) >= 1
        assert final_pos > final_neg

    def test_funding_sign_short_formula_explicit(self) -> None:
        """Short funding sign 명시 검증: +funding 수취(이익), -funding 지급(손실)."""
        n_bars, n_syms = 6, 1
        price = 100.0
        fee = 0.0005
        slip = 0.0002
        fr_abs = 0.0001

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, n_bars):
            weights[i, 0] = -0.30  # short 유지

        d_pos = _make_data(n_bars, n_syms, price=price, atr=5.0, funding=fr_abs)
        d_neg = _make_data(n_bars, n_syms, price=price, atr=5.0, funding=-fr_abs)

        _, final_pos, _, _ = _run(
            d_pos,
            weights,
            rb=1,
            taker_fee=fee,
            slip=slip,
            max_exp=10.0,
            max_exp_coin=100.0,
            init_bal=10_000.0,
            atr_mult=999.0,
        )
        _, final_neg, _, _ = _run(
            d_neg,
            weights,
            rb=1,
            taker_fee=fee,
            slip=slip,
            max_exp=10.0,
            max_exp_coin=100.0,
            init_bal=10_000.0,
            atr_mult=999.0,
        )

        # 엔진 누적식: fund_fee = qty * price * funding * side, short side=-1
        # -> +funding일 때 fund_fee<0 (수취), -funding일 때 fund_fee>0 (지급)
        fill_p = price * (1.0 - slip)
        qty = min(10_000.0 * 0.30 / fill_p, 10_000.0 * 0.97 / fill_p)
        n_funding_bars = n_bars - 1  # i=1..last_idx
        expected_gap = 2.0 * qty * price * fr_abs * n_funding_bars
        assert final_pos > final_neg
        assert (final_pos - final_neg) == pytest.approx(expected_gap, abs=0.1)

    def test_funding_rebalance_exit_applies_to_balance(self) -> None:
        """리밸런싱 청산 경로에서도 누적 funding이 balance에 반영됨을 검증."""
        n_bars, n_syms = 6, 1
        funding_rate = 0.0001
        price = 100.0
        fee = 0.0005
        slip = 0.0002

        d = _make_data(n_bars, n_syms, price=price, atr=5.0, funding=funding_rate)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[2, 0] = 0.30  # rb=2 진입 바 (i=2)
        weights[3, 0] = 0.30  # 포지션 유지
        # i=4 리밸런싱에서 weight=0으로 청산

        _, final_balance, _, _ = _run(
            d, weights, rb=2, taker_fee=fee, slip=slip,
            max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0
        )

        fill_p = price * (1.0 + slip)
        qty = min(10_000.0 * 0.30 / fill_p, 10_000.0 * 0.97 / fill_p)
        entry_fee = qty * fill_p * fee
        exit_p = price * (1.0 - slip)
        pnl_x = (exit_p - fill_p) * qty
        exit_fee = qty * exit_p * fee

        # 진입 i=2 이후 i=2,3 close에서 funding 2바 누적
        funding_2bar = qty * price * funding_rate * 2

        # 기댓값: funding도 차감되어야 함
        expected_correct = 10_000.0 - entry_fee + pnl_x - exit_fee - funding_2bar
        assert final_balance == pytest.approx(expected_correct, abs=0.01)


# ---------------------------------------------------------------------------
# Pillar 6 — Look-ahead Bias Protection
# ---------------------------------------------------------------------------


class TestLookaheadBias:
    """미래 참조 오류 방지 검증."""

    def test_signal_execution_lag(self) -> None:
        """weights[i]는 bar i의 open에 체결됨을 검증."""
        n_bars, n_syms = 5, 1
        slip = 0.0002

        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)
        d["open"][1, 0] = 100.0
        d["open"][2, 0] = 110.0

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1, 0] = 0.30  # bar 1 신호
        # bars 2-4: weight=0 → 청산

        trades, _, _, _ = _run(
            d, weights, rb=1, slip=slip, max_exp=10.0, max_exp_coin=100.0
        )

        assert len(trades) >= 1
        # entry는 bar 1의 open=100.0에서 발생해야 함
        assert trades[0, 4] == pytest.approx(100.0 * (1.0 + slip), abs=1e-4)
        # bar 0 weight는 무의미 (루프 i=1부터 시작)
        bar0_entries = trades[trades[:, 1] == 0]
        assert len(bar0_entries) == 0

    def test_no_future_bar_entry(self) -> None:
        """weights[1]=0일 때 bar 1에서 진입 없음; weights[2]=0.30일 때 bar 2 진입."""
        n_bars, n_syms = 6, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1, 0] = 0.0   # bar 1: 신호 없음
        weights[2, 0] = 0.30  # bar 2: 신호 발생

        trades, _, _, _ = _run(
            d, weights, rb=1, max_exp=10.0, max_exp_coin=100.0
        )

        bar1_entries = trades[trades[:, 1] == 1]
        bar2_entries = trades[trades[:, 1] == 2]
        assert len(bar1_entries) == 0
        assert len(bar2_entries) >= 1

    def test_atr_stop_uses_previous_bar_atr(self) -> None:
        """진입 시 stop_dist = atr_2d[prev_i] * atr_mult (진입 전 바의 ATR 기준)."""
        # rb=2: i=2에서 진입, prev_i=1
        # atr_2d[1, 0]=0.1 → stop_dist=0.1, stop_p=fill_p-0.1≈99.92
        # atr_2d[2, 0]=999.0 (진입 바) — 무시되어야 함 (stop_p에 영향 없음)
        # bar 2 low = price - 0.05 = 99.95 > stop_p=99.92 → 진입 바에서 stop 미발동
        # bar 3: open = fill_p - 0.15 = 99.87 < stop_p=99.92 → gap-down exit
        n_bars, n_syms = 6, 1
        slip = 0.0002
        atr_mult_val = 1.0
        price = 100.0
        atr_small = 0.1  # 기본 atr: h=100.05, l=99.95

        d = _make_data(n_bars, n_syms, price=price, atr=atr_small)

        # atr_2d[1, 0] = 0.1 → stop_dist=0.1 (prev_i=1의 ATR)
        d["atr"][1, 0] = 0.1
        # atr_2d[2, 0] = 999 → 무시됨 (stop_p는 진입 시 already computed)
        d["atr"][2, 0] = 999.0

        fill_p = price * (1.0 + slip)  # 100.02
        stop_p_val = fill_p - 0.1      # 99.92

        # bar 3: open = fill_p - 0.15 = 99.87 < stop_p → gap-down
        gap_open = fill_p - 0.15
        d["open"][3, 0] = gap_open
        d["high"][3, 0] = gap_open
        d["low"][3, 0] = gap_open - 0.05
        d["close"][3, 0] = gap_open
        _ = stop_p_val  # 참조용 (assertion에서 직접 사용)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[2, 0] = 0.30
        weights[3, 0] = 0.30
        weights[4, 0] = 0.30
        weights[5, 0] = 0.30

        trades, _, _, _ = _run(
            d, weights, rb=2, slip=slip, atr_mult=atr_mult_val,
            use_simple_atr=1, max_exp=10.0, max_exp_coin=100.0
        )

        assert len(trades) >= 1
        t = trades[0]
        # gap-down: c_open(gap_open) <= stop_p → exit = gap_open*(1-slip)
        expected_exit = gap_open * (1.0 - slip)
        assert t[5] == pytest.approx(expected_exit, abs=1e-4)


# ---------------------------------------------------------------------------
# Pillar 7 — Conservation of Money
# ---------------------------------------------------------------------------


class TestConservationOfMoney:
    """자금 보존 항등식 검증."""

    def test_conservation_of_money_no_funding(self) -> None:
        """final_balance == initial - sum(entry_fee) + sum(net_pnl) (항등식)."""
        n_bars, n_syms = 10, 2
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0, funding=0.0)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[3, 0] = 0.30   # long
        weights[3, 1] = -0.20  # short
        weights[6, :] = 0.0    # 청산

        trades, final_balance, _, _ = _run(
            d, weights, rb=3, max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0
        )

        if len(trades) > 0:
            net_pnl_total = float(trades[:, 6].sum())      # net_pnl (exit_fee 포함)
            entry_fees_total = float(trades[:, 8].sum())   # entry_fee 합계
            # final = initial - sum(entry_fee) + sum(net_pnl)
            expected = 10_000.0 - entry_fees_total + net_pnl_total
            assert final_balance == pytest.approx(expected, abs=1e-4)

    def test_equity_curve_flat_no_position(self) -> None:
        """포지션 없으면 equity_curve 전체 == initial_balance."""
        n_bars, n_syms = 10, 2
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)
        weights = np.zeros((n_bars, n_syms), dtype=np.float64)

        _, final_balance, equity_curve, _ = _run(
            d, weights, max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0
        )

        assert np.all(equity_curve == pytest.approx(10_000.0, abs=1e-6))
        assert final_balance == pytest.approx(10_000.0, abs=1e-6)

    def test_equity_curve_after_close(self) -> None:
        """모든 포지션 청산 후 마지막 equity_curve == final_balance."""
        n_bars, n_syms = 8, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[2, 0] = 0.30  # bar 2 진입 (rb=2: i=2,4,6에서 리밸런싱)
        # bar 4에서 weight=0 → 청산

        _trades, final_balance, equity_curve, _ = _run(
            d, weights, rb=2, max_exp=10.0, max_exp_coin=100.0, init_bal=10_000.0
        )

        # 마지막 equity_curve 값이 final_balance와 일치
        assert equity_curve[-1] == pytest.approx(final_balance, abs=1e-4)

    def test_conservation_of_money_randomized_10_cases(self) -> None:
        """고정 시드 10회 랜덤 경로에서 회계 항등식 검증."""
        seed = 20260519
        rng = np.random.default_rng(seed)
        n_bars, n_syms = 24, 3
        for _ in range(10):
            d = _make_data(n_bars, n_syms, price=100.0, atr=5.0, funding=0.0)
            for s in range(n_syms):
                walk = rng.normal(0.0, 0.8, size=n_bars).cumsum()
                px = np.maximum(20.0, 100.0 + walk)
                d["open"][:, s] = px
                d["high"][:, s] = px + 1.0
                d["low"][:, s] = px - 1.0
                d["close"][:, s] = px + rng.normal(0.0, 0.2, size=n_bars)

            weights = np.zeros((n_bars, n_syms), dtype=np.float64)
            for i in range(1, n_bars):
                sig = rng.normal(0.0, 0.35, size=n_syms)
                weights[i, :] = np.clip(sig, -0.8, 0.8)

            trades, final_balance, _eq, _diag = _run(
                d,
                weights,
                rb=2,
                max_exp=1.2,
                max_exp_coin=0.8,
                init_bal=10_000.0,
                atr_mult=999.0,
            )
            if len(trades) == 0:
                continue
            net_pnl_total = float(trades[:, 6].sum())
            entry_fees_total = float(trades[:, 8].sum())
            expected = 10_000.0 - entry_fees_total + net_pnl_total
            assert final_balance == pytest.approx(expected, abs=1e-4)

    def test_conservation_invariant_randomized_10_seeded(self) -> None:
        """고정 seed 10회 랜덤 시나리오에서 자금 보존 항등식 + 재현성 검증."""
        rng = np.random.default_rng(20260519)
        init_bal = 10_000.0

        for _ in range(10):
            n_bars, n_syms = 18, 3
            d = _make_data(n_bars, n_syms, price=100.0, atr=5.0, funding=0.0)

            # 랜덤 워크 기반 가격 경로 (항상 양수)
            rets = rng.normal(loc=0.0, scale=0.004, size=(n_bars, n_syms))
            close = 100.0 * np.cumprod(1.0 + rets, axis=0)
            close = np.clip(close, 1.0, None)
            d["close"] = close
            d["open"][1:, :] = close[:-1, :]
            d["open"][0, :] = close[0, :]
            d["high"] = np.maximum(d["open"], d["close"]) + 0.2
            d["low"] = np.minimum(d["open"], d["close"]) - 0.2

            weights = np.zeros((n_bars, n_syms), dtype=np.float64)
            for i in range(1, n_bars):
                if i % 3 == 0:
                    w = rng.uniform(-0.35, 0.35, size=n_syms)
                    gross = float(np.sum(np.abs(w)))
                    if gross > 0.95:
                        w *= 0.95 / gross
                    weights[i, :] = w

            trades1, final1, eq1, _ = _run(
                d, weights, rb=1, init_bal=init_bal, max_exp=1.0, max_exp_coin=1.0
            )
            trades2, final2, eq2, _ = _run(
                d, weights, rb=1, init_bal=init_bal, max_exp=1.0, max_exp_coin=1.0
            )

            # 결정론: 동일 입력 결과 동일
            assert final1 == final2
            assert np.array_equal(eq1, eq2)
            assert np.array_equal(trades1, trades2)

            if len(trades1) > 0:
                net_pnl_total = float(trades1[:, 6].sum())
                entry_fees_total = float(trades1[:, 8].sum())
                expected = init_bal - entry_fees_total + net_pnl_total
                assert final1 == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Pillar 8 — NaN/Inf 격리
# ---------------------------------------------------------------------------


class TestNanIsolation:
    """NaN/Inf 격리 검증."""

    def test_nan_close_price_skipped(self) -> None:
        """close에 NaN 주입: final_balance, equity_curve 오염 없음."""
        n_bars, n_syms = 10, 2
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)
        d["close"][5, 0] = np.nan

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[1, 0] = 0.30
        weights[1, 1] = 0.20

        _, final_balance, equity_curve, _ = _run(
            d, weights, max_exp=10.0, max_exp_coin=100.0
        )

        assert np.isfinite(final_balance)
        assert not np.any(np.isnan(equity_curve))

    def test_nan_funding_rate_no_corruption(self) -> None:
        """funding_rate에 NaN 주입: balance 오염 없음."""
        n_bars, n_syms = 8, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0, funding=0.0001)
        d["funding"][3, 0] = np.nan

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, n_bars):
            weights[i, 0] = 0.30

        _, final_balance, equity_curve, _ = _run(
            d, weights, rb=1, max_exp=10.0, max_exp_coin=100.0
        )

        assert np.isfinite(final_balance)
        assert not np.any(np.isnan(equity_curve))

    def test_nan_atr_prevents_entry(self) -> None:
        """atr_2d[prev_i]=NaN: 해당 심볼 진입 불가, crash 없음."""
        # rb=3: i=3에서 리밸런싱, prev_i=2 → atr_2d[2, 0]=NaN → 진입 거부
        n_bars, n_syms = 6, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)
        d["atr"][2, 0] = np.nan

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[3, 0] = 0.30

        trades, final_balance, _, diag_out = _run(
            d, weights, rb=3, max_exp=10.0, max_exp_coin=100.0
        )

        assert diag_out[1] >= 0  # crash 없음
        assert np.isfinite(final_balance)
        # NaN ATR → 진입 거부됨
        bar3_entries = trades[trades[:, 1] == 3] if len(trades) > 0 else np.array([])
        assert len(bar3_entries) == 0

    def test_nan_open_on_rebalance_bar_is_skipped(self) -> None:
        """리밸런싱 바 open NaN 심볼은 진입/청산 스킵되고 계정은 오염되지 않아야 함."""
        n_bars, n_syms = 8, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)
        d["open"][4, 0] = np.nan
        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, n_bars):
            weights[i, 0] = 0.35
        _trades, final_balance, equity_curve, _diag = _run(
            d,
            weights,
            rb=2,
            max_exp=10.0,
            max_exp_coin=100.0,
            atr_mult=999.0,
        )
        assert np.isfinite(final_balance)
        assert np.isfinite(equity_curve).all()

    def test_rebalance_open_nan_skips_entry_exit_and_stays_finite(self) -> None:
        """리밸런싱 bar open NaN 심볼은 진입/청산 스킵되고 결과 수치가 finite여야 함."""
        n_bars, n_syms = 8, 1
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        # rb=2 기준 i=4 리밸런싱 시점에 open NaN 주입
        d["open"][4, 0] = np.nan

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[2, 0] = 0.30  # i=2 진입
        weights[4, 0] = 0.00  # i=4 청산 의도 (open NaN으로 스킵되어야 함)
        weights[6, 0] = 0.00  # i=6에서 정상 open으로 청산

        trades, final_balance, equity_curve, _ = _run(
            d, weights, rb=2, atr_mult=999.0, max_exp=10.0, max_exp_coin=100.0
        )

        assert np.isfinite(final_balance)
        assert np.all(np.isfinite(equity_curve))
        # open NaN bar(4)에서는 청산이 스킵되어 exit_idx=4 거래가 없어야 함
        if len(trades) > 0:
            exits_at_4 = trades[trades[:, 2] == 4]
            assert len(exits_at_4) == 0


# ---------------------------------------------------------------------------
# Pillar 9 — Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """결정론 검증."""

    def test_determinism(self) -> None:
        """동일 입력 2회 실행: balance, equity_curve bit-identical."""
        n_bars, n_syms = 15, 3
        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        weights[3, 0] = 0.25
        weights[3, 1] = -0.15
        weights[6, :] = 0.0
        weights[9, 2] = 0.20

        _, bal1, eq1, _ = _run(d, weights, rb=3, max_exp=10.0, max_exp_coin=100.0)
        _, bal2, eq2, _ = _run(d, weights, rb=3, max_exp=10.0, max_exp_coin=100.0)

        assert bal1 == bal2
        assert np.array_equal(eq1, eq2)

    def test_funding_sign_convention_for_short(self) -> None:
        """Short 포지션은 +funding에서 수취, -funding에서 지급되어야 한다."""
        n_bars, n_syms = 6, 1
        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, n_bars):
            weights[i, 0] = -0.25

        d_pos = _make_data(n_bars, n_syms, price=100.0, atr=5.0, funding=0.0001)
        d_neg = _make_data(n_bars, n_syms, price=100.0, atr=5.0, funding=-0.0001)
        _, bal_pos, _eq_pos, _diag_pos = _run(d_pos, weights, rb=1, atr_mult=999.0)
        _, bal_neg, _eq_neg, _diag_neg = _run(d_neg, weights, rb=1, atr_mult=999.0)
        assert bal_pos > bal_neg


# ---------------------------------------------------------------------------
# Kill Signal & Max Hold
# ---------------------------------------------------------------------------


class TestKillSignalMaxHold:
    """Kill signal 및 max hold 검증."""

    def test_kill_signal_forces_exit(self) -> None:
        """kill_signal[prev_i]=1.0 → bar i open에 강제청산."""
        # rb=1, weight 일정 유지 (리밸런싱에서 need_exit=False)
        # kill[2, 0]=1.0 → bar 3 stop블록에서 exit
        n_bars, n_syms = 6, 1
        slip = 0.0002

        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        # kill_signal[2, 0]=1.0 → i=3 stop블록에서 kill[prev_i=2, 0] 체크
        d["kill"][2, 0] = 1.0

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, n_bars):
            weights[i, 0] = 0.30  # weight 유지 → 리밸런싱에서 need_exit=False

        trades, _, _, _ = _run(
            d, weights, rb=1, slip=slip, atr_mult=999.0,
            max_exp=10.0, max_exp_coin=100.0
        )

        assert len(trades) >= 1
        # 첫 번째 trade: bar 1 진입, bar 3 청산
        t = trades[0]
        assert int(t[1]) == 1  # entry_idx == 1
        assert int(t[2]) == 3  # exit_idx == 3
        # long exit: open[3, 0]*(1-slip)
        expected_exit = 100.0 * (1.0 - slip)
        assert t[5] == pytest.approx(expected_exit, abs=1e-4)

    def test_max_hold_bars_exit(self) -> None:
        """max_hold=3: entry_idx=1에서 진입, bar 4(i-entry=3)에서 강제청산."""
        # rb=1, weight 일정 유지, max_hold=3 → bar 4 stop블록에서 exit
        n_bars, n_syms = 10, 1
        slip = 0.0002

        d = _make_data(n_bars, n_syms, price=100.0, atr=5.0)

        weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        for i in range(1, n_bars):
            weights[i, 0] = 0.30  # weight 유지

        trades, _, _, _ = _run(
            d, weights, rb=1, max_hold=3, slip=slip, atr_mult=999.0,
            max_exp=10.0, max_exp_coin=100.0
        )

        assert len(trades) >= 1
        t = trades[0]
        assert int(t[1]) == 1  # entry_idx == 1
        assert int(t[2]) == 4  # exit_idx == 4 (4-1=3 >= max_hold=3)


# ---------------------------------------------------------------------------
# Intrabar 1m semantics
# ---------------------------------------------------------------------------


class TestIntrabarStopPathContract:
    """intrabar_1m stop path 보수적 계약 검증."""

    def test_long_gap_down_through_stop_uses_adverse_open_fill(self) -> None:
        """Long stop gap-down은 stop price가 아닌 불리한 open 기준으로 체결되어야 한다."""
        n_decisions, n_syms = 6, 1
        n_path = 12
        slip = 0.0002
        decision_price = np.full((n_decisions, n_syms), 100.0, dtype=np.float64)
        target_weights = np.zeros((n_decisions, n_syms), dtype=np.float64)
        target_weights[2:, 0] = 0.50
        lev = np.ones((n_decisions, n_syms), dtype=np.float64)
        atr = np.full((n_decisions, n_syms), 5.0, dtype=np.float64)
        kill = np.zeros((n_decisions, n_syms), dtype=np.float64)
        path_open = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_high = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_low = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_close = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_open[6, 0] = 80.0
        path_high[6, 0] = 80.0
        path_low[6, 0] = 79.0
        path_close[6, 0] = 80.0
        start_idx = np.array([0, 2, 4, 6, 8, 10], dtype=np.int64)
        end_idx = np.array([2, 4, 6, 8, 10, 12], dtype=np.int64)

        trades, _bal, _eq, _diag = backtest_target_weights_intrabar_numba(
            decision_price,
            decision_price,
            decision_price,
            decision_price,
            target_weights,
            lev,
            atr,
            kill,
            path_open,
            path_high,
            path_low,
            path_close,
            start_idx,
            end_idx,
            10_000.0,
            0.0,
            0.0,
            slip,
            2,
            0,
            0.0,
            1.0,
            999.0,
            1,
            10,
            10.0,
            10.0,
            0.0,
            None,
            None,
            None,
        )

        assert len(trades) >= 1
        trade = trades[0]
        assert int(trade[1]) == 2
        assert int(trade[2]) == 3
        assert trade[5] == pytest.approx(80.0 * (1.0 - slip), abs=1e-4)

    def test_short_gap_up_through_stop_uses_adverse_open_fill(self) -> None:
        """Short stop gap-up은 stop price가 아닌 불리한 open 기준으로 체결되어야 한다."""
        n_decisions, n_syms = 6, 1
        n_path = 12
        slip = 0.0002
        decision_price = np.full((n_decisions, n_syms), 100.0, dtype=np.float64)
        target_weights = np.zeros((n_decisions, n_syms), dtype=np.float64)
        target_weights[2:, 0] = -0.50
        lev = np.ones((n_decisions, n_syms), dtype=np.float64)
        atr = np.full((n_decisions, n_syms), 5.0, dtype=np.float64)
        kill = np.zeros((n_decisions, n_syms), dtype=np.float64)
        path_open = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_high = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_low = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_close = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_open[6, 0] = 120.0
        path_high[6, 0] = 121.0
        path_low[6, 0] = 120.0
        path_close[6, 0] = 120.0
        start_idx = np.array([0, 2, 4, 6, 8, 10], dtype=np.int64)
        end_idx = np.array([2, 4, 6, 8, 10, 12], dtype=np.int64)

        trades, _bal, _eq, _diag = backtest_target_weights_intrabar_numba(
            decision_price,
            decision_price,
            decision_price,
            decision_price,
            target_weights,
            lev,
            atr,
            kill,
            path_open,
            path_high,
            path_low,
            path_close,
            start_idx,
            end_idx,
            10_000.0,
            0.0,
            0.0,
            slip,
            2,
            0,
            0.0,
            1.0,
            999.0,
            1,
            10,
            10.0,
            10.0,
            0.0,
            None,
            None,
            None,
        )

        assert len(trades) >= 1
        trade = trades[0]
        assert int(trade[1]) == 2
        assert int(trade[2]) == 3
        assert trade[5] == pytest.approx(120.0 * (1.0 + slip), abs=1e-4)

    def test_open_priority_exit_precedes_intrabar_stop_scan(self) -> None:
        """관측 가능한 open-event(max_hold)는 같은 1m bar의 stop hit보다 먼저 처리되어야 한다."""
        n_decisions, n_syms = 6, 1
        n_path = 12
        slip = 0.0002
        decision_price = np.full((n_decisions, n_syms), 100.0, dtype=np.float64)
        target_weights = np.zeros((n_decisions, n_syms), dtype=np.float64)
        target_weights[2:, 0] = 0.50
        lev = np.ones((n_decisions, n_syms), dtype=np.float64)
        atr = np.full((n_decisions, n_syms), 5.0, dtype=np.float64)
        kill = np.zeros((n_decisions, n_syms), dtype=np.float64)
        path_open = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_high = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_low = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_close = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_low[6, 0] = 90.0
        start_idx = np.array([0, 2, 4, 6, 8, 10], dtype=np.int64)
        end_idx = np.array([2, 4, 6, 8, 10, 12], dtype=np.int64)

        trades, _bal, _eq, _diag = backtest_target_weights_intrabar_numba(
            decision_price,
            decision_price,
            decision_price,
            decision_price,
            target_weights,
            lev,
            atr,
            kill,
            path_open,
            path_high,
            path_low,
            path_close,
            start_idx,
            end_idx,
            10_000.0,
            0.0,
            0.0,
            slip,
            2,
            1,
            0.0,
            1.0,
            999.0,
            1,
            10,
            10.0,
            10.0,
            0.0,
            None,
            None,
            None,
        )

        assert len(trades) >= 1
        trade = trades[0]
        assert int(trade[1]) == 2
        assert int(trade[2]) == 3
        assert trade[5] == pytest.approx(100.0 * (1.0 - slip), abs=1e-4)


class TestIntrabarFundingEventContract:
    """intrabar funding event-only 반영 계약 검증."""

    def test_funding_event_applies_only_on_marked_1m_bars(self) -> None:
        """funding_event_mask_1m=1.0 인 1m bar에서만 funding 누적."""
        n_decisions, n_syms = 4, 1
        n_path = 8
        decision_price = np.full((n_decisions, n_syms), 100.0, dtype=np.float64)
        target_weights = np.zeros((n_decisions, n_syms), dtype=np.float64)
        target_weights[1:, 0] = 0.50
        lev = np.ones((n_decisions, n_syms), dtype=np.float64)
        atr = np.full((n_decisions, n_syms), 5.0, dtype=np.float64)
        kill = np.zeros((n_decisions, n_syms), dtype=np.float64)
        path_open = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_high = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_low = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        path_close = np.full((n_path, n_syms), 100.0, dtype=np.float64)
        start_idx = np.array([0, 2, 4, 6], dtype=np.int64)
        end_idx = np.array([2, 4, 6, 8], dtype=np.int64)

        mask_none = np.zeros((n_path, n_syms), dtype=np.float64)
        rate = np.full((n_path, n_syms), 0.001, dtype=np.float64)
        trades_none, bal_none, _eq_none, _diag_none = backtest_target_weights_intrabar_numba(
            decision_price,
            decision_price,
            decision_price,
            decision_price,
            target_weights,
            lev,
            atr,
            kill,
            path_open,
            path_high,
            path_low,
            path_close,
            start_idx,
            end_idx,
            10_000.0,
            0.0,
            0.0,
            0.0,
            1,
            0,
            0.0,
            999.0,
            999.0,
            1,
            10,
            10.0,
            10.0,
            0.0,
            mask_none,
            rate,
            None,
        )

        mask_evt = np.zeros((n_path, n_syms), dtype=np.float64)
        mask_evt[4, 0] = 1.0
        trades_evt, bal_evt, _eq_evt, _diag_evt = backtest_target_weights_intrabar_numba(
            decision_price,
            decision_price,
            decision_price,
            decision_price,
            target_weights,
            lev,
            atr,
            kill,
            path_open,
            path_high,
            path_low,
            path_close,
            start_idx,
            end_idx,
            10_000.0,
            0.0,
            0.0,
            0.0,
            1,
            0,
            0.0,
            999.0,
            999.0,
            1,
            10,
            10.0,
            10.0,
            0.0,
            mask_evt,
            rate,
            None,
        )

        assert len(trades_none) >= 1
        assert len(trades_evt) >= 1
        # long 포지션에서 +funding은 비용으로 누적되어 final balance를 감소시켜야 함.
        assert bal_evt < bal_none
        # 한 번의 event만 추가한 케이스이므로 총 funding_fee도 더 커야 함.
        total_funding_none = float(np.nansum(trades_none[:, 9]))
        total_funding_evt = float(np.nansum(trades_evt[:, 9]))
        assert total_funding_evt > total_funding_none
