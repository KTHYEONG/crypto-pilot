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
        
    total_ret_ratio: float = 1.0 + (ret_pct / 100.0)
    if total_ret_ratio <= 0:
        annual_return: float = -100.0
    else:
        annual_return = (pow(total_ret_ratio, 365.0 / span_days) - 1.0) * 100.0

    mdd_abs: float = max(abs(mdd_pct), 1.0)
    romad: float = annual_return / mdd_abs

    # Dynamic Statistical Significance Parameters
    trades_per_year_floor: float = 120.0 if tf == "4h" else 60.0
    dynamic_min_trades: float = max(trades_per_year_floor * (span_days / 365.0), 20.0)
    trade_ratio: float = float(n_trades) / dynamic_min_trades
    
    # Sigmoid Penalty Function
    k: float = 6.0
    penalty_multiplier: float = 1.0 / (1.0 + float(np.exp(-k * (trade_ratio - 1.0))))
    penalty_multiplier = min(penalty_multiplier, 1.0)
    
    # Adjust base score protecting against negative bias
    adjusted_romad: float = (romad * penalty_multiplier) if romad > 0.0 else (romad / penalty_multiplier)
    
    # Deflated RoMaD Logic
    deflation_factor: float = 1.0 - (1.0 / float(np.sqrt(max(float(n_trades), 1.0))))
    adjusted_romad *= deflation_factor

    final_score: float = adjusted_romad - max(0.0, mdd_abs - 40.0) * 0.3
    
    return final_score, ret_pct, mdd_abs

def calc_profit_factor(trades_df: pd.DataFrame) -> float:
    """Calculate Profit Factor (Gross Profit / Gross Loss)."""
    if trades_df.empty:
        return 1.0
        
    gains: pd.Series = trades_df["pnl"][trades_df["pnl"] > 0]
    losses: pd.Series = trades_df["pnl"][trades_df["pnl"] < 0]
    
    gross_profit: float = float(gains.sum()) if not gains.empty else 0.0
    gross_loss: float = abs(float(losses.sum())) if not losses.empty else 0.0
    
    if gross_loss == 0.0:
        return 999.0 if gross_profit > 0 else 1.0
        
    return gross_profit / gross_loss

def calc_romad_from_metrics(
    ret_pct: float,
    mdd_pct: float,
    n_trades: int,
    tf: str,
    span_days: float,
    n_trials: int = 1,
) -> Tuple[float, float, float]:
    """
    RoMaD score from precomputed return and MDD.
    Returns: (romad_score, return_pct, mdd_pct).
    
    [Theoretical Score Grades (n_trials=1750 기준)]
    - S (Elite)    : 8.0+ (금융공학적 상위 0.1%, 압도적 강건성 및 수익성)
    - A (Excellent): 4.0 ~ 8.0 (실전 투입 적극 권장, 높은 리스크 대비 수익)
    - B (Good)     : 1.5 ~ 4.0 (실전 투입 가능, 안정적 운용 기대)
    - C (Poor)     : 0.5 ~ 1.5 (낮은 통계적 유의성, 과적합 경계 필요)
    - D (Failing)  : < 0.5 (실전 투입 불가, 전략 재수립 필요)
    
    ※ Multiple Testing Correction(MTC) 및 Profit Factor Gate 페널티가 반영된 수치임.
    """
    if n_trades == 0:
        return -20.0, ret_pct, mdd_pct
        
    mdd_abs: float = max(abs(mdd_pct), 1.0)
    days: float = max(float(span_days), 1.0)

    total_ret_ratio: float = 1.0 + (ret_pct / 100.0)
    if total_ret_ratio <= 0:
        annual_return: float = -100.0
    else:
        annual_return = (pow(total_ret_ratio, 365.0 / days) - 1.0) * 100.0

    romad: float = annual_return / mdd_abs
    
    # Dynamic Statistical Significance Parameters
    trades_per_year_floor: float = 120.0 if tf == "4h" else 60.0
    dynamic_min_trades: float = max(trades_per_year_floor * (days / 365.0), 20.0)
    trade_ratio: float = float(n_trades) / dynamic_min_trades
    
    # Sigmoid Penalty Function
    k: float = 6.0
    penalty_multiplier: float = 1.0 / (1.0 + float(np.exp(-k * (trade_ratio - 1.0))))
    penalty_multiplier = min(penalty_multiplier, 1.0)
    
    # Adjust base score protecting against negative bias
    adjusted_romad: float = (romad * penalty_multiplier) if romad > 0.0 else (romad / penalty_multiplier)
    
    # Deflated RoMaD Logic
    deflation_factor: float = 1.0 - (1.0 / float(np.sqrt(max(float(n_trades), 1.0))))
    adjusted_romad *= deflation_factor
    
    # Multiple Testing Correction (Bailey & López de Prado)
    # n_trials가 늘어나면 스코어 패널티를 줌 (우연적 과적합 방지)
    mt_correction: float = 1.0 - (float(np.log(n_trials)) / (2.0 * max(float(n_trades), 1.0)))
    mt_correction = max(mt_correction, 0.5)
    adjusted_romad *= mt_correction

    final_score: float = adjusted_romad - max(0.0, mdd_abs - 40.0) * 0.3
    
    return final_score, ret_pct, mdd_abs
