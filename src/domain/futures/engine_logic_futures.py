import numpy as np
from numba import njit


@njit(inline="always")
def process_long_scale_out(
    c_open: float,
    c_high: float,
    entry_price: float,
    pos_atr: float,
    l_scale_atr: float,
    amount: float,
    fee_rate: float,
):
    scale_target = entry_price + (pos_atr * l_scale_atr)
    if c_high >= scale_target:
        sc_price = c_open if c_open >= scale_target else scale_target
        sc_amount = amount / 2.0
        pnl = (sc_price - entry_price) * sc_amount
        fee = sc_amount * sc_price * fee_rate
        return True, sc_price, sc_amount, pnl, fee
    return False, 0.0, 0.0, 0.0, 0.0


@njit(inline="always")
def process_short_scale_out(
    c_open: float,
    c_low: float,
    entry_price: float,
    pos_atr: float,
    s_tp_mult: float,
    amount: float,
    fee_rate: float,
):
    tp_price = entry_price - (pos_atr * s_tp_mult)
    if c_open <= tp_price or c_low <= tp_price:
        sc_price = c_open if c_open <= tp_price else tp_price
        sc_amount = amount / 2.0
        pnl = (entry_price - sc_price) * sc_amount
        fee = sc_amount * sc_price * fee_rate
        return True, sc_price, sc_amount, pnl, fee
    return False, 0.0, 0.0, 0.0, 0.0


@njit(inline="always")
def check_long_exit(
    c_open: float,
    c_low: float,
    highest: float,
    pos_atr: float,
    stop_price: float,
    l_trail_mult: float,
    slippage_rate: float,
):
    if c_open <= stop_price:
        return True, c_open * (1.0 - slippage_rate), stop_price
    elif c_low <= stop_price:
        return True, stop_price * (1.0 - slippage_rate), stop_price

    new_stop = highest - (pos_atr * l_trail_mult)
    if new_stop > stop_price:
        stop_price = new_stop
    return False, 0.0, stop_price


@njit(inline="always")
def check_short_exit(
    c_open: float,
    c_high: float,
    lowest: float,
    pos_atr: float,
    stop_price: float,
    s_trail_mult: float,
    slippage_rate: float,
):
    if c_open >= stop_price:
        return True, c_open * (1.0 + slippage_rate), stop_price
    elif c_high >= stop_price:
        return True, stop_price * (1.0 + slippage_rate), stop_price

    new_stop = lowest + (pos_atr * s_trail_mult)
    if new_stop < stop_price:
        stop_price = new_stop

    return False, 0.0, stop_price


@njit(inline="always")
def calculate_position_size(
    fill_price: float,
    asset_atr_pct: float,        # 코인의 내재 변동성 (ATR / Price)
    current_equity_for_risk: float,
    available_margin: float,
    risk_per_trade: float,       # 포트폴리오 타겟 변동성 (기존 risk_per_trade 활용)
    leverage: float,
    sf: float,
    gk: float,
    max_exposure_per_coin: float = 1.5,
) -> float:
    """
    [RE-ENGINEERED] Target Volatility Sizing (Alternative 1)
    Decouples Sizing from Stop-Loss distance to prevent 1/ATR^2 penalty.
    """
    # 0. NaN Protection
    if np.isnan(asset_atr_pct) or np.isnan(current_equity_for_risk) or np.isnan(fill_price):
        return 0.0
    if np.isnan(gk):
        gk = 0.0
    if gk < 0.0:
        gk = 0.0
    
    # 1. Target Volatility 기반 명목 자본 할당 (Notional Allocation)
    # asset_atr_pct가 높을수록(변동성이 클수록) 할당 금액이 비례하여 감소 (단일 패널티)
    vol_scalar = risk_per_trade / max(asset_atr_pct, 0.001)
    target_notional = current_equity_for_risk * vol_scalar * gk
    
    # [SAFETY] 전체 가용 레버리지 용량의 70%를 초과하는 명목 가치 설정 금지 (Margin Fail 방지 핵심)
    max_safe_notional = max(current_equity_for_risk, 0.0) * leverage * 0.70
    target_notional = min(target_notional, max_safe_notional)

    # 2. 명목 한도 캡 (Max Exposure / Anti-Gap Protection)
    max_qty_by_exposure = (current_equity_for_risk * max_exposure_per_coin) / fill_price
    target_qty = min(target_notional / fill_price, max_qty_by_exposure)

    # 3. 가용 증거금 한도 캡 (Margin Constraint) - 수수료 예비분 2% 제외
    max_qty_by_margin = (available_margin * 0.98 * leverage) / fill_price
    if max_qty_by_margin < 0:
        max_qty_by_margin = 0.0
    target_qty = min(target_qty, max_qty_by_margin)

    # 4. Sizing module's confidence multiplier (sf)
    sf_c = sf if sf <= 1.0 else 1.0
    if sf_c < 0.0:
        sf_c = 0.0
    target_qty *= sf_c
    
    return target_qty


@njit(inline="always")
def check_intra_bar_stop(
    pos_side: int,
    c_high: float,
    c_low: float,
    stop_price: float,
    entry_price: float,
    amount: float,
    fee_rate: float,
    slippage_rate: float,
):
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
