"""Performance Evaluation and Metrics for Optimization.

Combines Alpha IC evaluation, OOS Portfolio Backtesting, and statistical metrics.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.core.settings import (
    FUTURES_INITIAL_BALANCE,
    MAKER_FEE_RATE,
    SLIPPAGE_RATE,
    SMART_ORDER_OFFSET,
    TAKER_FEE_RATE,
)
from src.domain.futures.backtest.engine import (
    PortfolioBacktestEngine as PortfolioBacktestEngineFast,
)

_logger: logging.Logger = logging.getLogger("opt_futures")

# --- Alpha Evaluation (from alpha_evaluator.py) ---

def compute_vol_adj_forward_returns(
    df: pd.DataFrame, 
    horizons: list[int] | None = None,
    market_returns: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """여러 호흡(Horizons)에 대해 변동성으로 정규화된 미래 수익률을 계산함."""
    if horizons is None:
        horizons = [2, 6, 12]
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
) -> tuple[float, float]:
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
    drawdowns = (equity_curve - running_max) / np.maximum(running_max, 1e-9)
    return float(abs(np.min(drawdowns)) * 100.0)


def calc_mdd_duration(equity_curve: np.ndarray) -> int:
    """Calculate Maximum Drawdown Duration in bars from an equity curve."""
    if len(equity_curve) < 2:
        return 0
    running_max = np.maximum.accumulate(equity_curve)
    is_underwater = equity_curve < running_max
    
    max_duration = 0
    current_duration = 0
    for underwater in is_underwater:
        if underwater:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0
    return max_duration


def calc_sortino_ratio(
    equity_curve: np.ndarray, 
    ann_factor: float, 
    risk_free_rate: float = 0.0
) -> float:
    """Calculate Sortino Ratio (Downside-only risk) from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-9)
    if returns.size == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / ann_factor)
    downside_returns = excess_returns[excess_returns < 0]
    
    if downside_returns.size < 2:
        # If no downside, Sortino is technically infinite; cap at 10.0 for stability
        return 10.0 if np.mean(excess_returns) > 0 else 0.0
        
    downside_std = np.std(downside_returns) * np.sqrt(ann_factor)
    if downside_std < 1e-9:
        return 10.0 if np.mean(excess_returns) > 0 else 0.0
        
    return float(np.mean(excess_returns) * ann_factor) / downside_std

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
    """Compute Deflated Sharpe on AWF leg log-TW samples."""
    if path_arr.size < 2:
        return 0.0
    m_pt = float(np.mean(path_arr))
    s_pt = float(np.std(path_arr, ddof=1))
    if not math.isfinite(s_pt) or s_pt < 1e-6:
        return -1.0
    sharpe = m_pt / (s_pt + 1e-12)
    # t_samples: Sharpe의 표준오차는 path_arr 관측치 수(k_chunks)로 계산해야 함.
    # stat_ref_len(IS bars→days)을 쓰면 t≈333이 되어 sqrt(t)≈18 배율로 z를 폭발시킴.
    t_samples = float(max(path_arr.size, 2))
    sk = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 3))
    ex_kurt = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 4)) - 3.0
    sr_var_denom = max(
        1.0 - sk * sharpe + ((ex_kurt + 2.0) / 4.0) * sharpe**2,
        1e-12,
    )
    # 과도한 sr_bench 방지: 최대 50개 유효 독립 검정으로 한정
    effective_trials = min(float(n_trials_opt), 50.0)
    sr_bench = math.sqrt(2.0 * math.log(max(effective_trials, 2.0)))
    z_dsr = (sharpe - sr_bench) * math.sqrt(max(t_samples - 1.0, 1.0)) / math.sqrt(sr_var_denom)
    dsr_val = float(0.5 * (1.0 + math.erf(z_dsr / math.sqrt(2.0))))
    return float(min(0.99, max(0.0, dsr_val)))


def median_absolute_deviation_1d(samples: Sequence[float]) -> float:
    """MAD around the median (1.4826-scaled std equivalent for normals not applied)."""
    arr = np.asarray(list(samples), dtype=np.float64)
    if arr.size < 2:
        return 0.0
    med = float(median(arr.tolist()))
    dev = np.abs(arr - med)
    return float(median(dev.tolist()))


def compute_awf_robust_objective_score(
    leg_log_tw: np.ndarray,
    max_mdd_pct: float,
    *,
    lambda_mad: float = 1.0,
    psi_dd: float = 0.5,
) -> float:
    """Kelly compound growth: mean(log_TW) - semi_deviation_penalty - DD_term.

    mean(leg_log_tw) is the unbiased estimator of compound growth rate (Kelly criterion).
    Semi-deviation penalizes only downside variability, preserving upside asymmetry.
    """
    arr = np.asarray(leg_log_tw, dtype=np.float64)
    if arr.size == 0:
        return float(-10.0 - psi_dd * max(float(max_mdd_pct), 0.0) / 100.0)
    mu = float(np.mean(arr))
    # 하방 semi-deviation: 평균 이하 구간만 패널티 (MAD는 상방도 패널티 → 복리에 불리)
    downside = arr[arr < mu]
    semi_dev = float(np.sqrt(np.mean((downside - mu) ** 2))) if downside.size > 0 else 0.0
    dd_term = psi_dd * max(float(max_mdd_pct), 0.0) / 100.0
    return float(mu - lambda_mad * semi_dev - dd_term)


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
    symbols: list[str],
    tf: str,
    params: dict[str, Any],
    oos_data_maps: dict[str, dict[str, Any]],
    cache_root: Path | None = None,
    return_signal_dfs: bool = False,
    oos_end_idx: int | None = None,
    oos_start_idx: int | None = None,
) -> dict[str, Any]:
    import gc

    from src.domain.futures.strategy_runtime.bridge import FuturesMLStrategy

    from .data_aligner import _segment_with_context, align_data_for_2d_engine

    strategy = FuturesMLStrategy(name="OOS_Portfolio", params=params)
    full_signal_dfs: dict[str, pd.DataFrame] = {}
    seg_dfs: dict[str, pd.DataFrame] = {}

    required_backtest_cols = {
        "open", "high", "low", "close", "volume", "atr",
        "entry_upper", "entry_lower", "trend_direction", "strength_filter",
        "garch_kelly_f", "funding_rate_sum", "kill_signal", "membership_kill_signal",
        "entry_block_mask", "slot_rank_score", "ml_calib_prob", "dyn_leverage",
        "xs_score_long", "xs_score_short", "alpha_long", "alpha_short",
        "hmm_prob_crisis", "hmm_hard_state", "hmm_modulator_long", "hmm_modulator_short",
        "datetime"
    }

    for sym in symbols:
        full_df = oos_data_maps[sym][tf]
        oos_start = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
        if oos_start_idx is not None:
            oos_start = int(oos_start_idx)
        
        end_cap = int(oos_end_idx) if oos_end_idx is not None else len(full_df)
        
        # Warmup bars slicing optimization
        warmup_margin = int(params.get("WARMUP_BARS", 120))
        slice_start = max(0, oos_start - warmup_margin)
        sliced_df = full_df.iloc[slice_start:end_cap].copy(deep=False)
        
        full_sig = strategy.generate_signals(sliced_df)
        
        # Drop heavy non-essential columns to free RAM immediately
        cols_to_drop = [c for c in full_sig.columns if c not in required_backtest_cols]
        if cols_to_drop:
            full_sig.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        
        adjusted_oos_start = oos_start - slice_start
        adjusted_end_cap = end_cap - slice_start
        seg, _ = _segment_with_context(full_sig, adjusted_oos_start, adjusted_end_cap)
        
        if return_signal_dfs:
            full_signal_dfs[sym] = full_sig
        seg_dfs[sym] = seg
        del sliced_df

    aligned_data, master_index = align_data_for_2d_engine(seg_dfs, symbols)
    del seg_dfs
    gc.collect()

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
        maker_fee=MAKER_FEE_RATE,
        taker_fee=TAKER_FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
        smart_offset=SMART_ORDER_OFFSET,
    )
    trades_df, equity_curve, final_balance, _bt_diag = engine.run()

    # Metrics calculation
    hours_per_bar = int(tf.replace("h", "")) if tf.endswith("h") else 4
    n_days = (len(equity_curve) * hours_per_bar) / 24.0
    moic = final_balance / max(FUTURES_INITIAL_BALANCE, 1e-9)
    
    # Safe CAGR calculation using log space to prevent OverflowError
    try:
        exponent = 365.0 / max(n_days, 1e-3)
        log_moic = math.log(max(moic, 1e-9))
        log_cagr_plus_1 = exponent * log_moic
        if log_cagr_plus_1 > 15.0: # Cap at extremely high value (exp(15) ~ 3.2M)
            cagr = 1e8
        elif log_cagr_plus_1 < -15.0:
            cagr = -100.0
        else:
            cagr = (math.exp(log_cagr_plus_1) - 1.0) * 100.0
    except (OverflowError, ValueError):
        cagr = 1e8 if moic > 1.0 else -100.0

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
        # Realistic: One entry (Maker 0.02%), one exit (Taker 0.05% + Slippage 0.02%)
        rt_cost = MAKER_FEE_RATE + TAKER_FEE_RATE + SLIPPAGE_RATE
        ev_ratio = avg_net_pnl / max(avg_abs_pnl * rt_cost, 1e-9)

        long_df = trades_df[trades_df["side"] == "LONG"]
        short_df = trades_df[trades_df["side"] == "SHORT"]
        long_gains = float(long_df[long_df["pnl"] > 0]["pnl"].sum())
        long_losses = float(abs(long_df[long_df["pnl"] < 0]["pnl"].sum()))
        short_gains = float(short_df[short_df["pnl"] > 0]["pnl"].sum())
        short_losses = float(abs(short_df[short_df["pnl"] < 0]["pnl"].sum()))
        long_pf = (
            long_gains / max(long_losses, 1e-9) 
            if long_losses > 0 else (1.5 if long_gains > 0 else 1.0)
        )
        short_pf = (
            short_gains / max(short_losses, 1e-9) 
            if short_losses > 0 else (1.5 if short_gains > 0 else 1.0)
        )
    else:
        pf_val, wr, lt, st, minority_pct, ev_ratio = 1.0, 0.0, 0, 0, 0.0, 0.0
        long_pf, short_pf = 1.0, 1.0

    _n_trades = len(trades_df)
    out = {
        "cagr_pct": cagr, "mdd_pct": mdd, "profit_factor": pf_val, "total_trades": _n_trades,
        "trade_count": _n_trades, "n_trades": _n_trades, "oos_trade_count": _n_trades,
        "moic": moic, "terminal_wealth_ratio": moic, "win_rate_pct": wr, "long_trades": lt,
        "short_trades": st, "equity_curve": equity_curve, "cvar_pct": cvar,
        "hw_recovery_days": hw_days, "oos_long_short_minority_pct": minority_pct,
        "calmar_ratio": cagr / abs(mdd) if abs(mdd) > 1e-6 else 0.0,
        "ulcer_index": ulcer, "ev_cost_ratio": ev_ratio,
        "avg_trade_pnl_pct": (
            float(trades_df["pnl"].mean() / max(FUTURES_INITIAL_BALANCE, 1.0) * 100.0)
            if not trades_df.empty else 0.0
        ),
        "long_pf": float(long_pf),
        "short_pf": float(short_pf),
        "long_profit_factor": float(long_pf),
        "short_profit_factor": float(short_pf),
        "trades_df": trades_df,
    }
    if return_signal_dfs:
        out["full_signal_dfs"] = full_signal_dfs
        out["aligned_master_index"] = master_index
    return out

def perform_online_capital_allocation(
    ensemble_curves: list[np.ndarray],
    initial_balance: float,
    window_size: int = 24,
    eta: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an online weighting process using Exponentiated Gradient (EG).
    
    Args:
        ensemble_curves: List of equity curves (numpy arrays).
        initial_balance: Starting balance.
        window_size: Period (in bars) to update weights.
        eta: Learning rate for EG update.
        
    Returns:
        meta_equity: The resulting meta-equity curve.
        weight_history: History of weights over time.

    """
    if not ensemble_curves:
        return np.array([initial_balance]), np.array([])
        
    n_members = len(ensemble_curves)
    n_bars = len(ensemble_curves[0])
    
    # Initialize equal weights (1/N)
    weights = np.ones(n_members) / n_members
    weight_history = np.zeros((n_bars, n_members))
    meta_equity = np.zeros(n_bars)
    meta_equity[0] = initial_balance
    
    # Member returns (bar-by-bar)
    member_returns = []
    for curve in ensemble_curves:
        # Avoid division by zero
        safe_curve = np.clip(curve, 1e-9, None)
        rets = np.zeros(n_bars)
        rets[1:] = (safe_curve[1:] - safe_curve[:-1]) / safe_curve[:-1]
        member_returns.append(rets)
    member_returns = np.array(member_returns) # (N, T)
    
    for t in range(1, n_bars):
        # Apply current weights to calculate meta-return for this bar
        meta_ret = np.dot(weights, member_returns[:, t])
        meta_equity[t] = meta_equity[t-1] * (1.0 + meta_ret)
        weight_history[t] = weights
        
        # Every window_size, update weights based on previous window performance
        if t % window_size == 0:
            # Calculate cumulative relative returns over the previous window
            # R_{i, window} = (Curve[t] / Curve[t - window_size]) - 1
            # Or use sum of log returns for stability, but standard EG uses simple returns
            window_returns = []
            for i in range(n_members):
                c = ensemble_curves[i]
                start_val = max(c[t - window_size], 1e-9)
                end_val = c[t]
                win_ret = (end_val - start_val) / start_val
                window_returns.append(win_ret)
            
            window_returns = np.array(window_returns)
            
            # EG update rule: w_{i, t+1} = w_{i, t} * exp(eta * R_{i, window})
            # Normalize to sum to 1
            new_weights = weights * np.exp(eta * window_returns)
            # Clip for numerical stability if needed
            new_weights = np.clip(new_weights, 1e-10, 1e10)
            weights = new_weights / np.sum(new_weights)
            
    return meta_equity, weight_history


# ---------------------------------------------------------------------------
# Phase 2: v3.0 Score 공식 및 DSR Entropy Effective Rank
# ---------------------------------------------------------------------------

_V3_LAMBDA_DOWN: float = 0.50
_V3_LAMBDA_MDD: float = 1.00
_V3_LAMBDA_CVAR: float = 0.30
_V3_LAMBDA_TURNOVER: float = 0.20
_V3_LAMBDA_FUNDING: float = 0.50
_V3_LAMBDA_CAPACITY: float = 0.40


def compute_v3_score(
    leg_log_tw: np.ndarray,
    worst_mdd: float,
    cvar_5: float,
    excess_turnover: float,
    funding_drag: float,
    aum_impact_penalty: float,
) -> float:
    """v3.0 고정 λ 기반 6항 score 공식.

    Args:
        leg_log_tw: shape [K], 각 leg의 log Terminal Wealth.
        worst_mdd: 0~1 scale 최대 낙폭.
        cvar_5: 0~1 scale CVaR 5%.
        excess_turnover: 정규화된 초과 회전율.
        funding_drag: 0~1 scale 펀딩 비용 비율.
        aum_impact_penalty: 0~1 scale AUM 충격 패널티.

    Returns:
        score = mean(log_tw)
                - λ_down * semidev
                - λ_mdd * worst_mdd
                - λ_cvar * cvar_5
                - λ_turnover * excess_turnover
                - λ_funding * funding_drag
                - λ_capacity * aum_impact_penalty

    """
    arr = np.asarray(leg_log_tw, dtype=np.float64)
    if arr.size == 0:
        return -10.0

    mu = float(np.mean(arr))
    downside = arr[arr < 0.0]
    semidev = float(np.std(downside, ddof=0)) if downside.size > 1 else 0.0

    return float(
        mu
        - _V3_LAMBDA_DOWN * semidev
        - _V3_LAMBDA_MDD * float(worst_mdd)
        - _V3_LAMBDA_CVAR * float(cvar_5)
        - _V3_LAMBDA_TURNOVER * float(excess_turnover)
        - _V3_LAMBDA_FUNDING * float(funding_drag)
        - _V3_LAMBDA_CAPACITY * float(aum_impact_penalty)
    )


def calc_n_trials_eff_entropy(
    signatures: np.ndarray,
    weights: np.ndarray,
) -> float:
    """DSR entropy effective rank.

    Args:
        signatures: shape [n_trials, 11] (K=8 log_tw + 3 stats).
        weights: shape [n_trials] (completed_legs / K; pruned < 1.0).

    Returns:
        exp(-Σ p_i * log(p_i)) — eigenvalue entropy of weighted correlation matrix.

    Algorithm:
        1. 가중 상관행렬 C = weighted_corr(signatures, weights)
        2. λ_i = eigenvalues(C)
        3. p_i = λ_i / Σλ_i
        4. return exp(-Σ p_i * log(p_i + ε))

    """
    sig = np.asarray(signatures, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    n_trials, n_features = sig.shape
    if n_trials < 2:
        return 1.0

    # weight 정규화
    w = np.maximum(w, 0.0)
    w_sum = float(np.sum(w))
    if w_sum < 1e-12:
        return 1.0
    w_norm = w / w_sum  # shape [n_trials]

    # 가중 평균 및 가중 공분산 계산
    mu_w = (w_norm[:, None] * sig).sum(axis=0)  # shape [n_features]
    sig_centered = sig - mu_w[None, :]  # shape [n_trials, n_features]

    # 가중 공분산 행렬 (편향 추정기)
    cov_w = (w_norm[:, None] * sig_centered).T @ sig_centered  # [n_features, n_features]

    # 가중 상관행렬로 변환
    std_w = np.sqrt(np.diag(cov_w))
    std_w = np.where(std_w < 1e-12, 1e-12, std_w)
    corr_w = cov_w / np.outer(std_w, std_w)
    corr_w = np.clip(corr_w, -1.0, 1.0)

    # 고유값 분해
    eigenvalues = np.linalg.eigvalsh(corr_w)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigval_sum = float(np.sum(eigenvalues))
    if eigval_sum < 1e-12:
        return 1.0

    p = eigenvalues / eigval_sum  # 확률 분포
    p = np.where(p < 1e-15, 1e-15, p)
    entropy = -float(np.sum(p * np.log(p)))
    return float(math.exp(entropy))
