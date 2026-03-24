"""
Backtest metrics: profit factor, MDD, Sortino, portfolio CAGR, CVaR, PSR/DSR, underwater duration.
"""
from __future__ import annotations

import math
from typing import List, Sequence

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


def portfolio_cagr_pct_from_equity(equity_curve: np.ndarray, span_days: float) -> float:
    if equity_curve.size < 2 or span_days <= 1e-9:
        return -100.0
    start_eq = float(max(equity_curve[0], 1e-9))
    end_eq = float(max(equity_curve[-1], 1e-9))
    ratio = end_eq / start_eq
    return float((ratio ** (365.0 / span_days) - 1.0) * 100.0)


def max_underwater_bars_from_equity(equity_curve: np.ndarray) -> int:
    if equity_curve.size < 2:
        return 0
    peak = np.maximum.accumulate(equity_curve)
    underwater = (equity_curve < peak).astype(np.int8)
    padded = np.concatenate(([0], underwater, [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    if starts.size == 0:
        return 0
    return int(np.max(ends - starts))


def calc_tail_ratio_from_equity(equity_curve: np.ndarray) -> float:
    """95th / |5th| percentile of step log-returns (asymmetry proxy)."""
    if equity_curve.size < 2:
        return 1.0
    safe = np.clip(equity_curve.astype(np.float64, copy=False), 1e-9, None)
    log_step = np.log(safe[1:] / safe[:-1])
    if log_step.size < 2:
        return 1.0
    p95 = float(np.percentile(log_step, 95))
    p5 = float(np.percentile(log_step, 5))
    if abs(p5) < 1e-12:
        return 999.0 if p95 > 0 else 0.0
    return float(p95 / abs(p5))


def cvar_loss_pct_from_simple_returns(equity_curve: np.ndarray, tail_frac: float = 0.05) -> float:
    """Mean loss magnitude (%) of worst tail_frac bar returns (positive number = worse tail)."""
    if equity_curve.size < 2:
        return 0.0
    safe = np.clip(equity_curve, 1e-9, None)
    r = np.diff(safe) / safe[:-1]
    if r.size == 0:
        return 0.0
    sorted_r = np.sort(r)
    k = max(1, int(tail_frac * len(sorted_r)))
    tail = sorted_r[:k]
    return float(abs(float(np.mean(tail))) * 100.0)


def mean_of_worst_quartile(values: Sequence[float]) -> float:
    arr = np.asarray(sorted(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    k = max(1, int(math.ceil(0.25 * arr.size)))
    return float(np.mean(arr[:k]))


def probabilistic_sharpe_ratio(
    sharpe_estimate: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bailey & Lopez de Prado (2012) PSR approximation."""
    if n_obs < 2:
        return 0.0
    n = float(n_obs)
    skewness = float(skew)
    excess_kurt = float(kurtosis) - 3.0
    denom = max(
        1e-12,
        math.sqrt(1.0 - skewness * sharpe_estimate + (excess_kurt + 3.0 - 3.0) / 4.0 * sharpe_estimate**2),
    )
    z = sharpe_estimate * math.sqrt(n - 1.0) / denom
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def compute_dsr_from_path_values(path_values: Sequence[float], n_independent_trials: int) -> float:
    """
    Deflated Sharpe on path-level scalars (e.g. log terminal wealth per CPCV path).
    Uses variance of Sharpe estimator and expected max SR under multiple testing (rough).
    """
    x = np.asarray(path_values, dtype=np.float64)
    if x.size < 2:
        return 0.0
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if sigma < 1e-12:
        return 0.0
    sr_hat = mu / sigma
    n = float(max(1, int(n_independent_trials)))
    # Expected max of n independent ~N(0,1)
    sr_star = math.sqrt(2.0 * math.log(max(n * math.pi / 2.0, 1.0)))
    var_sr = (1.0 + 0.5 * sr_hat**2) / max(float(x.size - 1), 1.0)
    return float((sr_hat - sr_star * math.sqrt(var_sr)) / (math.sqrt(var_sr) + 1e-12))


def compute_dsr_from_path_sortinos(path_values: Sequence[float], n_independent_trials: int) -> float:
    return compute_dsr_from_path_values(path_values, n_independent_trials)
