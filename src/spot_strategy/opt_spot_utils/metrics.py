"""
백테스트 결과인 손익 데이터를 바탕으로 수익률, MDD, Sortino 등 정량적 성과 지표를 계산함.
전략의 우수성을 판단하기 위해 단순 수익률뿐만 아니라 리스크 대비 효율성을 점수화하는 역할을 수행함.
"""
import numpy as np
import pandas as pd

def calc_profit_factor_from_pnl(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 1.0

    gross_profit: float = float(pnl_series[pnl_series > 0].sum())
    gross_loss: float = abs(float(pnl_series[pnl_series < 0].sum()))

    if gross_loss == 0.0:
        return 5.0 if gross_profit > 0 else 1.0

    return gross_profit / gross_loss

def calc_mdd_from_equity(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    running_max[running_max == 0] = 1e-9
    drawdown = (equity_curve - running_max) / running_max * 100.0
    return float(abs(np.min(np.nan_to_num(drawdown, nan=0.0))))

def calc_sortino_from_equity(equity_curve: np.ndarray, span_days: float) -> float:
    if len(equity_curve) < 2 or span_days <= 0:
        return 0.0
        
    start_eq = equity_curve[0] if equity_curve[0] > 0 else 1e-9
    end_eq = equity_curve[-1]
    
    total_ret_ratio = max(end_eq / start_eq, 0.0001)
    cagr_decimal = (total_ret_ratio ** (365.0 / span_days)) - 1.0
    
    safe_curve = np.clip(equity_curve, 1e-9, None)
    step_log_returns = np.log(safe_curve[1:] / safe_curve[:-1])
    
    downside_log_returns = step_log_returns[step_log_returns < 0]
    
    if len(downside_log_returns) == 0:
        return 999.0 if cagr_decimal > 0 else 0.0
        
    step_downside_var = np.mean(downside_log_returns**2.0)
    bars_per_year = (len(equity_curve) / span_days) * 365.0
    annual_downside_dev = np.sqrt(step_downside_var * bars_per_year)
    
    if annual_downside_dev == 0.0:
        return 999.0 if cagr_decimal > 0 else 0.0
        
    sortino = cagr_decimal / annual_downside_dev
    return float(sortino)
