"""
백테스트 결과인 손익 데이터를 바탕으로 수익률, MDD, RoMaD 등 정량적 성과 지표를 계산함.
전략의 우수성을 판단하기 위해 단순 수익률뿐만 아니라 리스크 대비 효율성을 점수화하는 역할을 수행함.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from config.settings import FUTURES_INITIAL_BALANCE

# Maximum annualization factor is no longer used for capping in V2.7 Time-Normalized logic.
_MAX_ANNUALIZATION_FACTOR: float = 1.0 

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

    if pnl_series.empty:
        pf: float = 1.0
    else:
        gains: float = float(pnl_series[pnl_series > 0].sum())
        losses: float = abs(float(pnl_series[pnl_series < 0].sum()))
        pf = float(gains / losses) if losses > 0 else (5.0 if gains > 0 else 1.0)

    return calc_romad_from_metrics(
        ret_pct=ret_pct,
        mdd_pct=mdd_pct,
        n_trades=n_trades,
        tf=tf,
        span_days=span_days,
        win_rate=win_rate,
        pf=pf,
        leverage=1.0,
    )

def calc_profit_factor_from_pnl(pnl_series: pd.Series) -> float:
    """Calculate Profit Factor from a pre-computed net PNL series (fee-deducted)."""
    if pnl_series.empty:
        return 1.0

    gross_profit: float = float(pnl_series[pnl_series > 0].sum())
    gross_loss: float = abs(float(pnl_series[pnl_series < 0].sum()))

    if gross_loss == 0.0:
        return 5.0 if gross_profit > 0 else 1.0

    return gross_profit / gross_loss


def calc_profit_factor(trades_df: pd.DataFrame) -> float:
    """Calculate Profit Factor (Gross Profit / Gross Loss) from raw trades_df."""
    if trades_df.empty:
        return 1.0
        
    gains: pd.Series = trades_df["pnl"][trades_df["pnl"] > 0]
    losses: pd.Series = trades_df["pnl"][trades_df["pnl"] < 0]
    
    gross_profit: float = float(gains.sum()) if not gains.empty else 0.0
    gross_loss: float = abs(float(losses.sum())) if not losses.empty else 0.0
    
    if gross_loss == 0.0:
        return 5.0 if gross_profit > 0 else 1.0
        
    return gross_profit / gross_loss

def calc_romad_from_metrics(
    ret_pct: float,
    mdd_pct: float,
    n_trades: int,
    tf: str,
    span_days: float,
    win_rate: float = 0.0,
    pf: float = 1.0,
    leverage: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Deprecated complex scoring. Returns Pure CAGR and MDD.
    """
    days: float = max(float(span_days), 1.0)
    total_ret_ratio: float = 1.0 + (ret_pct / 100.0)
    
    # Annualized Geometric Return (CAGR)
    if total_ret_ratio > 0:
        cagr = ((total_ret_ratio ** (365.0 / days)) - 1.0) * 100.0
    else:
        cagr = -100.0

    return float(cagr), ret_pct, float(abs(mdd_pct))

# ==============================================================================
# [NEW] Portfolio Aggregated Equity Curve Metrics 
# ==============================================================================

def calc_mdd_from_equity(equity_curve: np.ndarray) -> float:
    """Calculate Maximum Drawdown from an aggregated equity curve."""
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    running_max[running_max == 0] = 1e-9
    drawdown = (equity_curve - running_max) / running_max * 100.0
    return float(abs(np.min(np.nan_to_num(drawdown, nan=0.0))))

def calc_sortino_from_equity(equity_curve: np.ndarray, span_days: float) -> float:
    """Calculate Sortino Ratio from an aggregated equity curve."""
    if len(equity_curve) < 2 or span_days <= 0:
        return 0.0
        
    start_eq = equity_curve[0] if equity_curve[0] > 0 else 1e-9
    end_eq = equity_curve[-1]
    total_ret = (end_eq / start_eq) - 1.0
    
    # Log return is safer for compounding
    if total_ret > -0.99:
        log_ret = np.log(1.0 + total_ret)
    else:
        log_ret = np.log(0.01) # Max -99% floor
        
    annualized_return = (log_ret / span_days) * 365.0
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = np.clip(returns, -0.999, None)
    log_returns = np.log(1.0 + returns)
    
    downside_returns = log_returns[log_returns < 0]
    
    if len(downside_returns) == 0:
        return 999.0 if annualized_return > 0 else 0.0
        
    bars_per_day = len(log_returns) / span_days
    bars_per_year = bars_per_day * 365.0
    
    downside_dev = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(bars_per_year)
    
    if downside_dev == 0:
        return 999.0 if annualized_return > 0 else 0.0
        
    return float(annualized_return / downside_dev)
