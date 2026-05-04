import numpy as np
from numba import njit


@njit(inline="always")  # type: ignore[untyped-decorator]
def process_long_scale_out(
    c_open: float,
    c_high: float,
    entry_price: float,
    pos_atr: float,
    l_scale_atr: float,
    amount: float,
    fee_rate: float,
) -> tuple[bool, float, float, float, float]:
    scale_target = entry_price + (pos_atr * l_scale_atr)
    if c_high >= scale_target:
        sc_price = c_open if c_open >= scale_target else scale_target
        sc_amount = amount / 2.0
        pnl = (sc_price - entry_price) * sc_amount
        fee = sc_amount * sc_price * fee_rate
        return True, sc_price, sc_amount, pnl, fee
    return False, 0.0, 0.0, 0.0, 0.0


@njit(inline="always")  # type: ignore[untyped-decorator]
def process_short_scale_out(
    c_open: float,
    c_low: float,
    entry_price: float,
    pos_atr: float,
    s_tp_mult: float,
    amount: float,
    fee_rate: float,
) -> tuple[bool, float, float, float, float]:
    tp_price = entry_price - (pos_atr * s_tp_mult)
    if c_open <= tp_price or c_low <= tp_price:
        sc_price = c_open if c_open <= tp_price else tp_price
        sc_amount = amount / 2.0
        pnl = (entry_price - sc_price) * sc_amount
        fee = sc_amount * sc_price * fee_rate
        return True, sc_price, sc_amount, pnl, fee
    return False, 0.0, 0.0, 0.0, 0.0


@njit(inline="always")  # type: ignore[untyped-decorator]
def check_long_exit(
    c_open: float,
    c_low: float,
    highest: float,
    pos_atr: float,
    stop_price: float,
    l_trail_mult: float,
    slippage_rate: float,
) -> tuple[bool, float, float]:
    if c_open <= stop_price:
        return True, c_open * (1.0 - slippage_rate), stop_price
    elif c_low <= stop_price:
        return True, stop_price * (1.0 - slippage_rate), stop_price

    new_stop = highest - (pos_atr * l_trail_mult)
    if new_stop > stop_price:
        stop_price = new_stop
    return False, 0.0, stop_price


@njit(inline="always")  # type: ignore[untyped-decorator]
def check_short_exit(
    c_open: float,
    c_high: float,
    lowest: float,
    pos_atr: float,
    stop_price: float,
    s_trail_mult: float,
    slippage_rate: float,
) -> tuple[bool, float, float]:
    if c_open >= stop_price:
        return True, c_open * (1.0 + slippage_rate), stop_price
    elif c_high >= stop_price:
        return True, stop_price * (1.0 + slippage_rate), stop_price

    new_stop = lowest + (pos_atr * s_trail_mult)
    if new_stop < stop_price:
        stop_price = new_stop

    return False, 0.0, stop_price


@njit(inline="always")  # type: ignore[untyped-decorator]
def calculate_position_size(
    fill_price: float,
    asset_atr_pct: float,        # 코인의 내재 변동성 (ATR / Price)
    current_equity_for_risk: float,
    available_margin: float,
    risk_per_trade: float,       # 포트폴리오 타겟 변동성 (기존 risk_per_trade 활용)
    leverage: float,
    sf: float,                   # Confidence multiplier (Z-Score driven Alpha multiplier)
    gk: float,
    max_exposure_per_coin: float = 1.5,
) -> float:
    """[RE-ENGINEERED] Alpha-Driven Kelly & Dynamic Portfolio Scaling.

    Fuses ML Z-Score Alpha (sf) with Target Volatility Sizing.
    """
    # 0. NaN Protection
    if np.isnan(asset_atr_pct) or np.isnan(current_equity_for_risk) or np.isnan(fill_price):
        return 0.0
    
    # 1. Target Volatility 기반 명목 자본 할당 (Notional Allocation)
    # asset_atr_pct가 높을수록 할당 금액이 감소 (단일 패널티)
    vol_scalar = risk_per_trade / max(asset_atr_pct, 0.001)
    
    # [NEW] Alpha-Driven Confidence mapping
    # sf는 이미 engine_multi에서 (Z - Thr)/(3 - Thr)로 계산되어 옴.
    conf_mult = max(min(sf, 1.0), 0.0)
    
    # Garch-Kelly inhibitor (optional but kept for robustness)
    gk_use = max(min(gk, 1.0), 0.0)
    
    target_notional = current_equity_for_risk * vol_scalar * conf_mult * gk_use
    
    # 2. [ROBUST MARGIN PROTECTION]
    # 전체 Equity 대비 70% 캡 + 가용 증거금 대비 80% 캡 중 보수적인 값 선택
    # 이는 급격한 변동성 상황에서 마진콜을 방지하기 위함
    max_safe_by_equity = max(current_equity_for_risk, 0.0) * leverage * 0.70
    max_safe_by_margin = max(available_margin, 0.0) * leverage * 0.80
    
    target_notional = min(target_notional, min(max_safe_by_equity, max_safe_by_margin))

    # 3. 명목 한도 캡 (Max Exposure / Anti-Gap Protection)
    max_qty_by_exposure = (current_equity_for_risk * max_exposure_per_coin) / fill_price
    target_qty = min(target_notional / fill_price, max_qty_by_exposure)

    # 4. 가용 증거금 실질 한도 캡 (Margin Constraint) - 수수료 예비분 3% 제외 (더 보수적으로 변경)
    max_qty_by_margin = (available_margin * 0.97 * leverage) / fill_price
    if max_qty_by_margin < 0:
        max_qty_by_margin = 0.0
    target_qty = min(target_qty, max_qty_by_margin)

    # 5. 소액 계좌(예: $1000)를 위한 최소 먼지(Dust) 한도 보정 ($6.0 보장)
    if target_qty > 0.0 and (target_qty * fill_price) < 6.0:
        min_qty = 6.0 / fill_price
        if min_qty <= max_qty_by_margin:
            target_qty = min_qty
        else:
            target_qty = 0.0
    
    return target_qty


@njit(inline="always")  # type: ignore[untyped-decorator]
def check_intra_bar_stop(
    pos_side: int,
    c_high: float,
    c_low: float,
    stop_price: float,
    entry_price: float,
    amount: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[bool, float, float, float]:
    if pos_side == 1 and c_low <= stop_price:
        intra_exit_price = stop_price * (1.0 - slippage_rate)
        pnl = (intra_exit_price - entry_price) * amount
        exit_fee = amount * intra_exit_price * fee_rate
        return True, intra_exit_price, pnl, exit_fee
    elif pos_side == -1 and c_high >= stop_price:
        intra_exit_price = stop_price * (1.0 + slippage_rate)
        pnl = (entry_price - intra_exit_price) * amount
        exit_fee = amount * intra_exit_price * fee_rate
        return True, intra_exit_price, pnl, exit_fee
    return False, 0.0, 0.0, 0.0
