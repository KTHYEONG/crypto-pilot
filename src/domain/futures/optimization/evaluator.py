"""
Performance Evaluation and Metrics for Optimization.
Combines Alpha IC evaluation, OOS Portfolio Backtesting, and statistical metrics.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import (
    FUTURES_INITIAL_BALANCE,
    SLIPPAGE_RATE,
    TRADING_FEE_RATE,
)
from src.domain.futures.backtest_engine import (
    MultiSymbolEngine as PortfolioBacktestEngineFast,
    SingleSymbolEngine as BacktestEngineFast,
)

_logger: logging.Logger = logging.getLogger("opt_futures")

# --- Alpha Evaluation (from alpha_evaluator.py) ---

def compute_vol_adj_forward_returns(
    df: pd.DataFrame, 
    horizons: List[int] = [2, 6, 12],
    market_returns: Optional[Dict[int, np.ndarray]] = None
) -> Dict[int, np.ndarray]:
    """
    여러 호흡(Horizons)에 대해 변동성으로 정규화된 미래 수익률을 계산함.
    """
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    
    n = len(close)
    if n < 20:
        return {h: np.full(n, np.nan) for h in horizons}

    # 14-period ATR for normalization
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().to_numpy()
    atr = np.maximum(atr, 1e-9)
    
    results = {}
    for h in horizons:
        fwd_ret = np.full(n, np.nan)
        if n > h:
            fwd_ret[:-h] = (close[h:] - close[:-h]) / close[:-h]
        
        vol_adj_ret = fwd_ret / atr
        
        if market_returns and h in market_returns:
            m_ret = market_returns[h]
            mask = ~np.isnan(vol_adj_ret) & ~np.isnan(m_ret)
            if np.sum(mask) > 50:
                beta = np.cov(vol_adj_ret[mask], m_ret[mask])[0, 1] / (np.var(m_ret[mask]) + 1e-9)
                vol_adj_ret = vol_adj_ret - beta * m_ret
                
        results[h] = vol_adj_ret
        
    return results

def calculate_spearman_ic(signal_scores: np.ndarray, target_returns: np.ndarray) -> float:
    """Spearman Rank Correlation (IC) 계산."""
    if len(signal_scores) != len(target_returns):
        return 0.0
        
    mask = ~np.isnan(signal_scores) & ~np.isnan(target_returns)
    if np.sum(mask) < 50:
        return 0.0
    
    if np.unique(signal_scores[mask]).size < 2:
        return 0.0
        
    ic, _ = spearmanr(signal_scores[mask], target_returns[mask])
    return float(ic) if not np.isnan(ic) else 0.0

def calculate_residual_score(candidate_scores: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
    """Candidate 시그널에서 Base 시그널의 잔차 점수 계산."""
    mask = ~np.isnan(candidate_scores) & ~np.isnan(base_scores)
    if np.sum(mask) < 50:
        return candidate_scores
        
    x = base_scores[mask]
    y = candidate_scores[mask]
    beta = np.cov(x, y)[0, 1] / (np.var(x) + 1e-9)
    residual = candidate_scores - beta * base_scores
    return residual

def calculate_conditional_ic(
    signal_scores: np.ndarray, 
    target_returns: np.ndarray, 
    regime_mask: np.ndarray
) -> Tuple[float, float]:
    """특정 Regime 구간에서의 조건부 IC 및 커버리지 계산."""
    if len(signal_scores) != len(regime_mask):
        return 0.0, 0.0
        
    active_mask = (regime_mask > 0.5)
    active_scores = signal_scores[active_mask]
    active_returns = target_returns[active_mask]
    
    ic = calculate_spearman_ic(active_scores, active_returns)
    coverage = float(np.mean(active_mask))
    return ic, coverage

# --- Performance Metrics (from metrics.py) ---

def calc_profit_factor_from_pnl(pnl_series: pd.Series | np.ndarray | Sequence[float]) -> float:
    """Calculate Profit Factor from a pre-computed net PNL series."""
    pnl_arr = np.asarray(pnl_series)
    if pnl_arr.size == 0:
        return 1.0
    gross_profit: float = float(pnl_arr[pnl_arr > 0].sum())
    gross_loss: float = abs(float(pnl_arr[pnl_arr < 0].sum()))
    if gross_loss == 0.0:
        return 5.0 if gross_profit > 0 else 1.0
    return gross_profit / gross_loss

def calc_profit_factor(trades_df: pd.DataFrame) -> float:
    """Calculate Profit Factor from raw trades_df."""
    if trades_df.empty:
        return 1.0
    gains: pd.Series = trades_df["pnl"][trades_df["pnl"] > 0]
    losses: pd.Series = trades_df["pnl"][trades_df["pnl"] < 0]
    gross_profit: float = float(gains.sum()) if not gains.empty else 0.0
    gross_loss: float = abs(float(losses.sum())) if not losses.empty else 0.0
    if gross_loss == 0.0:
        return 5.0 if gross_profit > 0 else 1.0
    return gross_profit / gross_loss

def calc_mdd_from_equity(equity_curve: np.ndarray) -> float:
    """Calculate Maximum Drawdown from an aggregated equity curve."""
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    running_max[running_max == 0] = 1e-9
    drawdown = (equity_curve - running_max) / running_max * 100.0
    return float(abs(np.min(np.nan_to_num(drawdown, nan=0.0))))

def calc_sortino_from_equity(equity_curve: np.ndarray, span_days: float) -> float:
    """Compute annualized Sortino ratio from equity curve."""
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
    return float(cagr_decimal / annual_downside_dev)

def compute_pbo_from_cpcv_paths(
    is_path_scores: Sequence[float],
    oos_path_scores: Sequence[float],
    *,
    min_paths: int = 6,
) -> tuple[float, float]:
    """Return PBO proxy from CPCV path IS vs OOS score ranks."""
    is_arr = np.asarray(list(is_path_scores), dtype=np.float64)
    oos_arr = np.asarray(list(oos_path_scores), dtype=np.float64)
    if is_arr.size != oos_arr.size or is_arr.size < int(min_paths):
        return (1.0, 0.0)
    ri = pd.Series(is_arr).rank(method="average").to_numpy(dtype=np.float64)
    ro = pd.Series(oos_arr).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(ri) < 1e-12 or np.std(ro) < 1e-12:
        return (0.5, 0.0)
    rho = float(np.corrcoef(ri, ro)[0, 1])
    if not np.isfinite(rho):
        rho = 0.0
    pbo = float(np.clip(0.5 * (1.0 - rho), 0.0, 1.0))
    return (pbo, rho)

def calc_cvar5_loss_pct_from_equity(equity_curve: np.ndarray) -> float:
    """Portfolio CVaR(5%) as positive loss %."""
    if equity_curve.size < 2:
        return 0.0
    eq = np.asarray(equity_curve, dtype=np.float64)
    r = np.diff(eq) / np.clip(eq[:-1], 1e-9, None)
    if r.size == 0:
        return 0.0
    sorted_r = np.sort(r)
    k = max(1, int(len(sorted_r) * 0.05))
    worst = sorted_r[:k]
    return float(-np.mean(worst) * 100.0)

def calc_max_underwater_days_from_equity(equity_curve: np.ndarray, hours_per_bar: float) -> float:
    """Longest stretch below running peak, converted to days."""
    if equity_curve.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    underwater = equity_curve < peak
    max_run = 0
    cur = 0
    for u in underwater:
        if u:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return float(max_run * hours_per_bar / 24.0)

def calc_ulcer_index_from_equity(equity_curve: np.ndarray) -> float:
    """Ulcer Index: RMS of percentage drawdown."""
    if equity_curve.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd_pct = (equity_curve - peak) / np.clip(peak, 1e-9, None) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2.0)))

def calc_tail_ratio_from_equity(equity: np.ndarray) -> float:
    """95th percentile return / abs(5th percentile return)."""
    if equity.size < 2:
        return 1.0
    r = np.diff(equity) / np.clip(equity[:-1], 1e-12, None)
    if r.size < 5:
        return 1.0
    val95 = float(np.percentile(r, 95.0))
    val5 = float(np.percentile(r, 5.0))
    if abs(val5) < 1e-12:
        return 5.0 if val95 > 0 else 1.0
    return float(val95 / abs(val5))

def _log_tw_from_ret_pct(ret_pct: float) -> float:
    r = 1.0 + float(ret_pct) / 100.0
    if r <= 0.0 or not math.isfinite(r):
        return -10.0
    return float(math.log(max(r, 1e-9)))

def calc_gate1_dsr_from_path_log_tw(
    path_arr: np.ndarray,
    tf: str,
    stat_ref_len: float,
    n_trials_opt: float,
) -> float:
    """Compute Deflated Sharpe on CPCV path log-TWR samples."""
    if path_arr.size < 2:
        return 0.0
    m_pt = float(np.mean(path_arr))
    s_pt = float(np.std(path_arr, ddof=1))
    if not math.isfinite(s_pt) or s_pt < 1e-6:
        return -1.0
    sharpe = m_pt / (s_pt + 1e-12)
    hrs = int(tf.replace("h", "")) if tf.endswith("h") else 4
    t_samples = float(stat_ref_len) / (24.0 / float(hrs))
    sk = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 3))
    ex_kurt = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 4)) - 3.0
    sr_var_denom = max(
        1.0 - sk * sharpe + ((ex_kurt + 2.0) / 4.0) * sharpe**2,
        1e-12,
    )
    sr_bench = math.sqrt(2.0 * math.log(max(n_trials_opt, 2.0)))
    z_dsr = (sharpe - sr_bench) * math.sqrt(max(t_samples - 1.0, 1.0)) / math.sqrt(sr_var_denom)
    dsr_val = float(0.5 * (1.0 + math.erf(z_dsr / math.sqrt(2.0))))
    return float(min(0.99, max(0.0, dsr_val)))

def calc_time_to_target_wealth(
    path_log_returns: np.ndarray,
    target_multiplier: float,
    bars_per_year: float,
) -> tuple[float, float]:
    """Return probability-weighted time to reach target asset multiplier."""
    if path_log_returns.size < 2:
        return 999.0, 999.0
    mu = float(np.mean(path_log_returns)) * bars_per_year
    sigma = float(np.std(path_log_returns, ddof=1)) * math.sqrt(bars_per_year)
    if mu <= 1e-6:
        return 999.0, 999.0
    log_target = math.log(max(1.0001, target_multiplier))
    expected_years = log_target / mu
    z = 1.645
    n_years = float(len(path_log_returns)) / bars_per_year
    drift_ci_lower = mu - z * sigma / math.sqrt(max(1.0, n_years))
    ci_upper_years = log_target / drift_ci_lower if drift_ci_lower > 1e-6 else 999.0
    return float(expected_years), float(ci_upper_years)

def calc_net_alpha_with_friction(
    equity_curve: np.ndarray,
    benchmark_cagr: float,
    bars_per_year: float,
    avg_funding_rate_bps: float = 1.0,
    avg_slippage_bps: float = 2.0,
    turnover_per_bar: float = 0.1,
) -> float:
    """Compute net alpha accounting for liquidity-depth slippage and funding-rate friction."""
    if len(equity_curve) < 2:
        return 0.0
    start_eq = equity_curve[0] if equity_curve[0] > 0 else 1e-9
    end_eq = equity_curve[-1]
    total_ret_ratio = max(end_eq / start_eq, 0.0001)
    years = len(equity_curve) / max(1.0, bars_per_year)
    if years <= 0:
        return 0.0
    cagr_decimal = (total_ret_ratio ** (1.0 / years)) - 1.0
    annual_turnover = turnover_per_bar * bars_per_year
    annual_slippage_cost = annual_turnover * (avg_slippage_bps / 10000.0)
    annual_funding_cost = bars_per_year * (avg_funding_rate_bps / 10000.0) * 0.5
    return float(cagr_decimal - annual_slippage_cost - annual_funding_cost - benchmark_cagr)

def stationary_bootstrap_spa(
    leg_log_tw: np.ndarray,
    n_bootstrap: int = 2000,
    block_length: int | None = None,
    seed: int = 42,
) -> float:
    """Sign-permutation / circular-block bootstrap p-value for H0: E[leg_log_tw] <= 0."""
    arr = np.asarray(leg_log_tw, dtype=np.float64)
    k = arr.size
    if k < 2:
        return 0.5
    obs_mean = float(np.mean(arr))
    if obs_mean <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    blk = max(1, int(block_length or max(1, k // 2)))
    arr_dm = arr - obs_mean
    bootstrap_means: np.ndarray = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sample = np.empty(k, dtype=np.float64)
        idx = 0
        while idx < k:
            start = int(rng.integers(0, k))
            take = min(blk, k - idx)
            for j in range(take):
                sample[idx] = arr_dm[(start + j) % k]
                idx += 1
        bootstrap_means[b] = float(np.mean(sample))
    return float(np.mean(bootstrap_means >= obs_mean))

# --- OOS Evaluation (from oos_evaluator.py) ---

def run_oos_margin_shared_portfolio(
    symbols: List[str],
    tf: str,
    params: Dict[str, Any],
    oos_data_maps: Dict[str, Dict[str, Any]],
    cache_root: Optional[Path] = None,
    return_signal_dfs: bool = False,
    oos_end_idx: Optional[int] = None,
    oos_start_idx: Optional[int] = None,
) -> Dict[str, Any]:
    # Import locally to avoid circular dependencies if any
    from src.domain.futures.strategy_ml import FuturesMLStrategy
    from .data_aligner import align_data_for_2d_engine, _segment_with_context

    strategy = FuturesMLStrategy(name="OOS_Portfolio", params=params)
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    seg_dfs: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        full_df = oos_data_maps[sym][tf]
        oos_start = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
        if oos_start_idx is not None:
            oos_start = int(oos_start_idx)
        
        # In the refactored ML-focused structure, we might not use tiered signals anymore
        # or it might be renamed. For now, we assume strategy.generate_all_signals or similar.
        full_sig = strategy.generate_all_signals(full_df)
        
        end_cap = int(oos_end_idx) if oos_end_idx is not None else len(full_df)
        seg, _ = _segment_with_context(full_sig, oos_start, end_cap)
        full_signal_dfs[sym] = full_sig
        seg_dfs[sym] = seg

    aligned_data, _ = align_data_for_2d_engine(seg_dfs, symbols)
    if not aligned_data:
        return {
            "cagr_pct": -100.0, "mdd_pct": 100.0, "profit_factor": 0.0,
            "total_trades": 0, "moic": 0.0, "terminal_wealth_ratio": 0.0,
            "cvar_pct": 100.0, "hw_recovery_days": 999.0, "win_rate_pct": 0.0,
            "oos_long_short_minority_pct": 0.0, "long_trades": 0, "short_trades": 0,
            "equity_curve": np.array([FUTURES_INITIAL_BALANCE]),
            "bt_dust_skip_cnt": 0, "bt_margin_fail_cnt": 0,
        }

    engine = PortfolioBacktestEngineFast(
        aligned_data=aligned_data,
        symbol_names=symbols,
        strategy_params=params,
        initial_balance=float(FUTURES_INITIAL_BALANCE),
        fee_rate=TRADING_FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    trades_df, equity_curve, final_balance, bt_diag = engine.run()

    # Metrics calculation
    hours_per_bar = int(tf.replace("h", "")) if tf.endswith("h") else 4
    n_days = (len(equity_curve) * hours_per_bar) / 24.0
    cagr = ((final_balance / FUTURES_INITIAL_BALANCE) ** (365.0 / max(n_days, 1.0)) - 1.0) * 100.0
    mdd = calc_mdd_from_equity(equity_curve)
    moic = final_balance / FUTURES_INITIAL_BALANCE
    cvar = calc_cvar5_loss_pct_from_equity(equity_curve)
    hw_days = calc_max_underwater_days_from_equity(equity_curve, float(hours_per_bar))
    ulcer = calc_ulcer_index_from_equity(equity_curve)

    if not trades_df.empty:
        gains = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        losses = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        pf_val = min(gains / max(losses, 1e-9), 5.0)
        wr = (trades_df["pnl"] > 0).mean() * 100.0
        lt = int((trades_df["side"] == "LONG").sum())
        st = int((trades_df["side"] == "SHORT").sum())
        minority_pct = float(min(lt, st) / max(lt + st, 1) * 100.0)
        avg_net_pnl = float(trades_df["pnl"].mean())
        avg_abs_pnl = float(trades_df["pnl"].abs().mean())
        rt_cost = TRADING_FEE_RATE * 2.0 + SLIPPAGE_RATE * 2.0
        ev_ratio = avg_net_pnl / max(avg_abs_pnl * rt_cost, 1e-9)
    else:
        pf_val, wr, lt, st, minority_pct, ev_ratio = 1.0, 0.0, 0, 0, 0.0, 0.0

    out = {
        "cagr_pct": cagr, "mdd_pct": mdd, "profit_factor": pf_val, "total_trades": len(trades_df),
        "moic": moic, "terminal_wealth_ratio": moic, "win_rate_pct": wr, "long_trades": lt,
        "short_trades": st, "equity_curve": equity_curve, "cvar_pct": cvar,
        "hw_recovery_days": hw_days, "oos_long_short_minority_pct": minority_pct,
        "calmar_ratio": cagr / abs(mdd) if abs(mdd) > 1e-6 else 0.0,
        "ulcer_index": ulcer, "ev_cost_ratio": ev_ratio,
        "avg_trade_pnl_pct": float(trades_df["pnl"].mean() / max(FUTURES_INITIAL_BALANCE, 1.0) * 100.0) if not trades_df.empty else 0.0,
        "trades_df": trades_df,
    }
    if return_signal_dfs:
        out["full_signal_dfs"] = full_signal_dfs
    return out
