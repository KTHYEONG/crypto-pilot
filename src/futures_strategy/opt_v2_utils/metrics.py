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
    Standardized to mirror the enhanced logic in calc_romad_from_metrics.
    """
    if pnl_series.empty or n_trades == 0:
        return -10.0, 0.0, 0.0

    equity: pd.Series = FUTURES_INITIAL_BALANCE + pnl_series.cumsum()
    end_equity: float = float(equity.iloc[-1])
    ret_pct: float = ((end_equity / FUTURES_INITIAL_BALANCE) - 1.0) * 100.0

    running_max: np.ndarray = np.maximum.accumulate(equity.values)
    running_max[running_max == 0] = 1e-9
    drawdown: np.ndarray = (equity.values - running_max) / running_max * 100.0
    mdd_pct: float = abs(float(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    span_days: float = float((pnl_series.index[-1] - pnl_series.index[0]).total_seconds() / 86400.0)
    win_rate: float = float((pnl_series > 0).mean() * 100.0) if not pnl_series.empty else 0.0

    return calc_romad_from_metrics(
        ret_pct=ret_pct,
        mdd_pct=mdd_pct,
        n_trades=n_trades,
        tf=tf,
        span_days=span_days,
        n_trials=1,
        win_rate=win_rate
    )

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
    win_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """
    RoMaD score from precomputed return and MDD with strict financial engineering constraints.
    Returns: (score, return_pct, mdd_pct).
    
    ※ Multiple Testing Correction(MTC), Annualization Cap, Trade Gate, and MDD Penalty included.
    """
    days: float = max(float(span_days), 1.0)
    trades_per_year_floor: float = 40.0 if tf == "4h" else 20.0
    dynamic_min_trades: float = max(trades_per_year_floor * (days / 365.0), 15.0)

    # 1. Hard Gate: Scale absolute 20 trades limit to the actual span length (548 days = 18m)
    # This prevents the gate from rejecting valid CV-folds (6 months) or OOS folds (3 months).
    scaled_min_trades: float = max(5.0, 20.0 * (days / 548.0))
    if n_trades < scaled_min_trades:
        return -10.0, ret_pct, mdd_pct
        
    mdd_abs: float = max(abs(mdd_pct), 1.0)
    total_ret_ratio: float = 1.0 + (ret_pct / 100.0)
    
    if total_ret_ratio <= 0:
        annual_return: float = -100.0
    else:
        # 2. Authentic Annualization: Reward compound growth, cap small-duration luck natively
        raw_annual: float = (pow(total_ret_ratio, 365.0 / days) - 1.0) * 100.0
        annual_return = raw_annual if days > 90 else (ret_pct * (365.0/days))

    # 3. Smoother MTC & Credibility (Bailey et al.)
    trade_ratio: float = float(n_trades) / dynamic_min_trades
    k: float = 3.0 # Flatter sigmoid to permit a wider viable search space
    significance: float = 1.0 / (1.0 + float(np.exp(-k * (trade_ratio - 1.0))))
    
    # MTC moderated to not destroy the intrinsic score of long-tail systems
    mt_correction: float = 1.0 - 0.5 * (np.sqrt(np.log(max(float(n_trials), 2.0))) / np.sqrt(max(float(n_trades), 1.0)))
    mt_correction = max(mt_correction, 0.5)
    
    credibility: float = max(significance * mt_correction, 0.2)
    
    # 4. Win Rate Penalty Gate: as per image (win_rate < 25.0)
    if win_rate > 0 and win_rate < 25.0:
        credibility *= 0.5
    
    # 5. Trade Frequency Bonus: Precisely trade_ratio > 1.5
    frequency_bonus: float = 0.3 if trade_ratio > 1.5 else 0.0
    
    base_romad: float = annual_return / mdd_abs
    
    # Mathematical Fix: Shrink positive returns by credibility, but EXPAND negative returns
    if base_romad > 0:
        adjusted_romad: float = base_romad * credibility + frequency_bonus
    else:
        adjusted_romad: float = base_romad * (1.0 / max(credibility, 0.1))

    # 6. Strict MDD Discipline: 20% Threshold, 1.0x weight
    mdd_penalty: float = max(0.0, mdd_abs - 20.0) * 1.0
    final_score: float = adjusted_romad - mdd_penalty
    
    return float(final_score), ret_pct, mdd_abs
