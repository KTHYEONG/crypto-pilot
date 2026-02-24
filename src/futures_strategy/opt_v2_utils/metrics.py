"""
백테스트 결과인 손익 데이터를 바탕으로 수익률, MDD, RoMaD 등 정량적 성과 지표를 계산함.
전략의 우수성을 판단하기 위해 단순 수익률뿐만 아니라 리스크 대비 효율성을 점수화하는 역할을 수행함.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from config.settings import FUTURES_INITIAL_BALANCE

def calc_romad(pnl_series: pd.Series, n_trades: int, tf: str) -> Tuple[float, float, float]:
    """
    Calculate Return on Max Drawdown (RoMaD) as the primary risk-adjusted metric.
    Returns: (romad_score, return_pct, mdd_pct)
    """
    if pnl_series.empty or n_trades == 0:
        return -20.0, 0.0, 0.0

    equity: pd.Series = FUTURES_INITIAL_BALANCE + pnl_series.cumsum()
    end_equity: float = float(equity.iloc[-1])
    ret_pct: float = ((end_equity / FUTURES_INITIAL_BALANCE) - 1.0) * 100.0

    running_max: np.ndarray = np.maximum.accumulate(equity.values)
    running_max[running_max == 0] = 1e-9
    drawdown: np.ndarray = (equity.values - running_max) / running_max * 100.0
    mdd_pct: float = abs(float(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    span_days: float = float((pnl_series.index[-1] - pnl_series.index[0]).total_seconds() / 86400.0)
    if span_days < 1.0:
        span_days = 1.0
    annual_return: float = ret_pct * (365.0 / span_days)

    romad: float = annual_return / max(mdd_pct, 5.0)

    # Statistical Significance Parameters
    min_trades_target: int = 120 if tf == "4h" else 40
    trade_ratio: float = float(n_trades) / float(min_trades_target)
    
    # Financial Engineering Penalty: Deflated Sharpe/RoMaD Logic
    penalty_multiplier: float
    if trade_ratio < 1.0:
        penalty_multiplier = max(trade_ratio ** 2.0, 0.05)
    else:
        penalty_multiplier = min(trade_ratio ** 0.5, 1.2)

    # Adjust base score protecting against negative bias
    adjusted_romad: float = (romad * penalty_multiplier) if romad > 0.0 else (romad / penalty_multiplier)
    final_score: float = adjusted_romad - max(0.0, mdd_pct - 40.0) * 0.3
    
    return final_score, ret_pct, mdd_pct

def calc_romad_from_metrics(
    ret_pct: float,
    mdd_pct: float,
    n_trades: int,
    tf: str,
    span_days: float,
) -> Tuple[float, float, float]:
    """
    RoMaD score from precomputed return and MDD.
    Returns: (romad_score, return_pct, mdd_pct).
    """
    if n_trades == 0:
        return -20.0, ret_pct, mdd_pct
        
    mdd_abs: float = abs(mdd_pct)
    days: float = max(float(span_days), 1.0)
    annual_return: float = ret_pct * (365.0 / days)
    romad: float = annual_return / max(mdd_abs, 5.0)
    
    # Statistical Significance Parameters
    min_trades_target: int = 120 if tf == "4h" else 40
    trade_ratio: float = float(n_trades) / float(min_trades_target)
    
    # Financial Engineering Penalty: Deflated Sharpe/RoMaD Logic
    penalty_multiplier: float
    if trade_ratio < 1.0:
        penalty_multiplier = max(trade_ratio ** 2.0, 0.05)
    else:
        penalty_multiplier = min(trade_ratio ** 0.5, 1.2)
    
    # Adjust base score protecting against negative bias
    adjusted_romad: float = (romad * penalty_multiplier) if romad > 0.0 else (romad / penalty_multiplier)
    final_score: float = adjusted_romad - max(0.0, mdd_abs - 40.0) * 0.3
    
    return final_score, ret_pct, mdd_abs
