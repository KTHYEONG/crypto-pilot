"""
전략의 파라미터를 입력받아 실제 백테스트 엔진을 구동하고, 최적화 타겟인 목적 함수(Objective Score)를 산출함.
심볼별/폴드별 성과를 종합하여 전략의 견고함(Robustness)을 평가하는 핵심 연산 로직을 포함함.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from config.settings import FUTURES_INITIAL_BALANCE
from config.opt_config import WARMUP_PERIODS
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.engine_fast_futures import BacktestEngineFast
from src.futures_strategy.opt_futures_utils.metrics import (
    calc_profit_factor_from_pnl,
    calc_mdd_from_equity,
    calc_sortino_from_equity
)
from src.futures_strategy.opt_futures_utils.cv_utils import build_purged_walk_forward_folds
from src.futures_strategy.opt_futures_utils.opt_params import suggest_params_futures

_logger: logging.Logger = logging.getLogger("opt_futures")

SymbolFoldResult = Tuple[str, float, float, float, float, float, float, np.ndarray, float]

_MAX_SYMBOL_WORKERS: int = max(1, int(os.getenv("OPT_FUTURES_SYMBOL_WORKERS", "1")))

def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: Dict[str, int] = {"1h": 24, "4h": 6}
    ratio_map: Dict[str, float] = {"1h": 0.08, "4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))

EMBARGO_BARS: Dict[str, int] = {
    "1h": compute_embargo_bars("1h"),
    "4h": compute_embargo_bars("4h"),
}

SIGNAL_CACHE_PARAM_KEYS: frozenset[str] = frozenset([
    "MACRO_EMA_PERIOD",
    "KC_MULT",
    "MOMENTUM_PERIOD",
    "ATR_PERIOD"
])

# (signal_params, sym, tf, data_len)
_SignalCacheKey = Tuple[Tuple[Tuple[str, Any], ...], str, str, int]

_SIGNAL_CACHE_MAXSIZE: int = 64
_cache_lock: threading.Lock = threading.Lock()
_signal_cache: OrderedDict[_SignalCacheKey, pd.DataFrame] = OrderedDict()


def _segment_with_context(
    full_signal_df: pd.DataFrame,
    exec_start_idx: int,
    exec_end_idx: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Include one prior bar as context so the first tradable bar can legally read prev_i.
    Only the absolute dataset first bar is sacrificed when no prior bar exists.
    """
    slice_start = max(0, int(exec_start_idx) - 1)
    slice_end = max(slice_start, int(exec_end_idx))
    segment = full_signal_df.iloc[slice_start:slice_end].copy()
    execution_start_idx = int(exec_start_idx) - slice_start
    if execution_start_idx == 0 and len(segment) > 1:
        execution_start_idx = 1
    return segment, execution_start_idx


def _evaluate_symbol_for_fold_parallel(
    sym: str,
    *,
    params: Dict[str, Any],
    strategy: UltimateStrategy,
    tf: str,
    data_maps: Dict[str, Dict[str, Any]],
    full_signal_dfs: Dict[str, pd.DataFrame],
    test_start: int,
    test_end: int,
) -> Optional[SymbolFoldResult]:
    target_df: Optional[pd.DataFrame] = data_maps.get(sym, {}).get(tf)
    daily_df: Optional[pd.DataFrame] = data_maps.get(sym, {}).get("1d")
    full_merge_idx: Optional[np.ndarray] = data_maps.get(sym, {}).get(f"merge_idx_{tf}")
    if target_df is None or daily_df is None or full_merge_idx is None:
        return None

    is_start_idx: int = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
    adj_test_start: int = test_start + is_start_idx
    adj_test_end: int = test_end + is_start_idx

    if sym not in full_signal_dfs:
        _logger.warning("Signal df missing for %s, skipping.", sym)
        return None

    segment, execution_start_idx = _segment_with_context(
        full_signal_dfs[sym], adj_test_start, adj_test_end
    )

    try:
        score: float
        ret_pct: float
        mdd_pct: float
        trades_count: int
        win_rate: float
        pf: float
        long_count: int
        short_count: int
        equity_curve: np.ndarray
        (
            score,
            ret_pct,
            mdd_pct,
            trades_count,
            win_rate,
            pf,
            long_count,
            short_count,
            equity_curve,
        ) = evaluate_symbol_fold(
            strategy,
            params,
            sym,
            tf,
            target_df,
            daily_df,
            full_merge_idx,
            None,
            adj_test_start,
            adj_test_end,
            precomputed_signal_df=segment,
            execution_start_idx=execution_start_idx,
        )
    except Exception as exc:
        _logger.warning("Symbol-level evaluation error for %s: %s", sym, exc, exc_info=True)
        return None

    if equity_curve.size == 0:
        span_days_sym: float = 0.0
    elif "datetime" in segment.columns and len(segment) > execution_start_idx:
        span_seconds: float = float(
            (
                segment["datetime"].iloc[-1]
                - segment["datetime"].iloc[execution_start_idx]
            ).total_seconds()
        )
        span_days_sym = max(span_seconds / 86400.0, 1.0)
    else:
        span_days_sym = 0.0

    return (
        sym,
        score,
        ret_pct,
        mdd_pct,
        float(trades_count),
        win_rate,
        pf,
        equity_curve,
        span_days_sym,
    )

def _build_signal_cache_key(params: Dict[str, Any], sym: str, tf: str, data_len: int) -> _SignalCacheKey:
    """Build cache key from signal-affecting params AND data length to isolate walk-forward segments."""
    signal_items: List[Tuple[str, Any]] = sorted(
        (k, params[k]) for k in SIGNAL_CACHE_PARAM_KEYS if k in params
    )
    return (tuple(signal_items), sym, tf, data_len)


def get_or_compute_signals(
    cache_key: _SignalCacheKey,
    target_df: pd.DataFrame,
    strategy: UltimateStrategy,
) -> pd.DataFrame:
    """Lazy memoization: compute signals once per (signal_params, sym, tf); LRU eviction. Key uses signal-only params."""
    with _cache_lock:
        if cache_key in _signal_cache:
            _signal_cache.move_to_end(cache_key)
            return _signal_cache[cache_key]
    full_df: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
    with _cache_lock:
        if cache_key in _signal_cache:
            _signal_cache.move_to_end(cache_key)
            return _signal_cache[cache_key]
        while len(_signal_cache) >= _SIGNAL_CACHE_MAXSIZE:
            _signal_cache.popitem(last=False)
        _signal_cache[cache_key] = full_df
        return full_df


def evaluate_symbol_fold(
    strategy: UltimateStrategy,
    params: Dict[str, Any],
    symbol: str,
    tf: str,
    target_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    full_merge_idx: np.ndarray,
    precomputed_daily_df: pd.DataFrame,
    test_start: int,
    test_end: int,
    precomputed_signal_df: Optional[pd.DataFrame] = None,
    execution_start_idx: int = 0,
) -> Tuple[float, float, float, int, float, float, int, int, np.ndarray]:
    if precomputed_signal_df is not None:
        sig_oos: pd.DataFrame = precomputed_signal_df
    else:
        full_signal: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
        sig_oos, execution_start_idx = _segment_with_context(full_signal, test_start, test_end)

    warmup_bars: int = 0
    sig_oos.attrs = {"warmup_bars": warmup_bars}
    
    # Parse timeframe hours for proper funding fee calculation
    tf_hours = 1.0
    if tf.endswith("h"):
        try: tf_hours = float(tf.replace("h", ""))
        except: pass
    elif tf.endswith("d"):
        try: tf_hours = float(tf.replace("d", "")) * 24.0
        except: pass

    # Single Pass: Fixed Risk Compounding 
    engine: BacktestEngineFast = BacktestEngineFast(
        hourly_df=sig_oos,
        daily_df=daily_df,
        strategy=strategy,
        initial_balance=FUTURES_INITIAL_BALANCE,
        merge_index_map=None,
        precomputed_daily_df=None,
        warmup_bars=warmup_bars,
        execution_start_idx=execution_start_idx,
    )
    engine.leverage = params.get("LEVERAGE", 1)
    engine.risk_per_trade = params.get("RISK_PER_TRADE", 0.01)
    # Crypto generic standard: Numba engine triggers exactly at UTC 0, 8, 16.
    # Do not override with fractional values (tf_hours / 8.0) as the engine handles exact timestamp matches.
    engine.funding_events_per_bar = 1.0

    params_fixed = params.copy()
    params_fixed["USE_COMPOUNDING"] = True
    engine.strategy = type("MockStrategy", (object,), {"params": params_fixed, "name": getattr(strategy, "name", "Mock")})
    
    try:
        result: Dict[str, Any] = engine.run()
        trades_df: pd.DataFrame = result.get("trades_df", pd.DataFrame())
    except Exception as e:
        _logger.warning("Backtest engine error: %s", e, exc_info=True)
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0, np.array([])
        
    if trades_df is None or trades_df.empty:
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0, np.array([])
        
    long_count: int = len(trades_df[trades_df["side"] == "LONG"])
    short_count: int = len(trades_df[trades_df["side"] == "SHORT"])

    equity_curve = result.get("equity_curve", np.array([]))
    mdd_pct: float = abs(float(result.get("mdd_pct", 0.0)))
    ret_pct: float = float(result.get("total_return_pct", 0.0))

    span_days: float = (
        float(
            (
                sig_oos["datetime"].iloc[-1]
                - sig_oos["datetime"].iloc[min(execution_start_idx, len(sig_oos) - 1)]
            ).total_seconds() / 86400.0
        )
        if "datetime" in sig_oos.columns and not sig_oos.empty
        else 1.0
    )
    span_days = max(span_days, 1.0)
    
    true_pnl = trades_df["pnl"] - trades_df["entry_fee"]
    win_rate: float = float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100) if len(trades_df) > 0 else 0.0
    pf = calc_profit_factor_from_pnl(true_pnl)
    
    # Calculate geometric mean return directly for single symbol
    total_ret_ratio = 1.0 + (ret_pct / 100.0)
    cagr = ((total_ret_ratio ** (365.0 / span_days)) - 1.0) * 100.0 if total_ret_ratio > 0 else -100.0

    return cagr, ret_pct, mdd_pct, len(trades_df), win_rate, pf, long_count, short_count, equity_curve


def objective_futures(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf_target: str,
    *,
    space: Dict[str, Dict[str, Any]],
    mode: str = "single",
    project_root: Optional[str] = None,
) -> float:
    params: Dict[str, Any] = suggest_params_futures(trial, space, tf_target)
    if params.pop("_INVALID_CONSTRAINT", False):
        raise optuna.TrialPruned()

    tf: str = tf_target
    strategy: UltimateStrategy = UltimateStrategy(name="OptFutures", params=params)

    # Use first available symbol length as base for walk-forward folds
    ref_sym = symbols[0]
    ref_len = len(data_maps[ref_sym][tf]) - data_maps[ref_sym].get(f"is_start_idx_{tf}", 0)
    base_df_for_folds: pd.DataFrame = pd.DataFrame(index=range(ref_len))
    
    cv_folds, holdout_fold = build_purged_walk_forward_folds(
        base_df_for_folds, 
        n_folds=4, 
        holdout_ratio=0.20, 
        embargo=EMBARGO_BARS.get(tf, 0)
    )

    if not cv_folds:
        raise optuna.TrialPruned()

    all_folds = cv_folds + ([holdout_fold] if holdout_fold[2] > holdout_fold[1] else [])

    cagr_penalty: float = 0.0
    mdd_penalty: float = 0.0

    fold_scores: List[float] = []
    fold_rets: List[float] = []
    fold_mdds: List[float] = []
    fold_trades: List[float] = []
    fold_wins: List[float] = []
    fold_pfs: List[float] = []
    
    sym_total_rets: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_mdds: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_scores: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_trades: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_wins: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_pfs: Dict[str, List[float]] = {sym: [] for sym in symbols}

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_full: Optional[pd.DataFrame] = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty: continue
        cache_key: _SignalCacheKey = _build_signal_cache_key(params, sym, tf, len(target_df_full))
        full_signal_dfs[sym] = get_or_compute_signals(cache_key, target_df_full, strategy)

    if not full_signal_dfs:
        raise optuna.TrialPruned()
    
    for f_idx, fold_info in enumerate(all_folds):
        if len(fold_info) == 4: train_start, train_end, test_start, test_end = fold_info
        else: train_start = 0; train_end, test_start, test_end = fold_info

        target_symbols: List[str] = symbols
        fold_agg_eq: Optional[np.ndarray] = None
        fold_span_days: float = 0.0

        max_workers: int = min(len(target_symbols), _MAX_SYMBOL_WORKERS)
        symbol_results: List[SymbolFoldResult] = []

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_evaluate_symbol_for_fold_parallel, sym, params=params, strategy=strategy, tf=tf, 
                                   data_maps=data_maps, full_signal_dfs=full_signal_dfs, test_start=test_start, test_end=test_end)
                    for sym in target_symbols
                ]
                for future in as_completed(futures):
                    result: Optional[SymbolFoldResult] = future.result()
                    if result: symbol_results.append(result)
        else:
            for sym in target_symbols:
                result = _evaluate_symbol_for_fold_parallel(sym, params=params, strategy=strategy, tf=tf, 
                                                            data_maps=data_maps, full_signal_dfs=full_signal_dfs, test_start=test_start, test_end=test_end)
                if result: symbol_results.append(result)

        if not symbol_results: raise optuna.TrialPruned()

        fold_sym_cagrs: List[float] = []
        fold_sym_rets: List[float] = []
        fold_sym_mdds: List[float] = []
        fold_sym_trades: List[float] = []

        for (sym, s, r, m, t, w, pf, eq, span_days_sym) in symbol_results:
            fold_sym_cagrs.append(s)
            fold_sym_rets.append(r)
            fold_sym_mdds.append(m)
            fold_sym_trades.append(t)
            
            sym_total_scores[sym].append(s)
            sym_total_rets[sym].append(r)
            sym_total_mdds[sym].append(m)
            sym_total_trades[sym].append(t)
            sym_total_wins[sym].append(w)
            sym_total_pfs[sym].append(pf)

            if eq.size > 0:
                eq_norm = eq / eq[0]
                if fold_agg_eq is None: fold_agg_eq = eq_norm.copy()
                else: 
                    min_l = min(len(fold_agg_eq), len(eq_norm))
                    fold_agg_eq = fold_agg_eq[:min_l] + eq_norm[:min_l]
            fold_span_days = max(fold_span_days, span_days_sym)

        if fold_agg_eq is None: raise optuna.TrialPruned()
        
        # [NEW] Anti-Ruin Protection: If any single symbol went bust (< -99%), prune the trial.
        for r in fold_sym_rets:
            if r <= -99.0:
                return -100.0, 100.0 # Instant ruin penalty (CAGR, MDD)
        
        # Normalized Portfolio Equity (Equal Risk/Weight per symbol)
        port_eq = fold_agg_eq / len(symbol_results)
        port_mdd = float(calc_mdd_from_equity(port_eq))
        
        # Kelly Proxy for Portfolio
        safe_eq = np.clip(port_eq, 1e-9, None)
        log_rets = np.log(safe_eq[1:] / safe_eq[:-1])
        bars_per_year = (len(port_eq) / max(fold_span_days, 1.0)) * 365.0
        ann_log_ret = np.mean(log_rets) * bars_per_year
        ann_var = np.var(log_rets) * bars_per_year
        port_kelly = ann_log_ret - (ann_var * 0.5)

        fold_scores.append(port_kelly)
        fold_rets.append(float(np.mean(fold_sym_cagrs)))
        fold_mdds.append(port_mdd)
        fold_trades.append(float(np.mean(fold_sym_trades)))
        fold_wins.append(float(np.mean([res[5] for res in symbol_results])))
        fold_pfs.append(float(np.mean([res[6] for res in symbol_results])))

        # --- [NEW] Rolling Robustness (Time-series Consistency) ---
        # Instead of chopping the backtest, we run continuously and evaluate consistency of the resulting equity curve.
        n_segments = 4
        if len(port_eq) > n_segments * 10:
            seg_len = len(port_eq) // n_segments
            seg_rets = []
            for s_i in range(n_segments):
                segment = port_eq[s_i * seg_len : (s_i + 1) * seg_len]
                if segment[0] > 0:
                    seg_ret_ratio = segment[-1] / segment[0]
                    seg_rets.append(max(0.0, seg_ret_ratio - 1.0))
                else:
                    seg_rets.append(0.0)
            
            # Penalty if any segment is deeply negative or if variance is extremely high
            if len(seg_rets) > 1:
                seg_std = float(np.std(seg_rets))
                seg_avg = float(np.mean(seg_rets))
                # Coefficient of Variation penalty: higher CV means less consistency
                cv_penalty = (seg_std / (seg_avg + 1e-6)) * 10.0
                cagr_penalty += cv_penalty
                mdd_penalty += cv_penalty * 2.0

    # --- [MULTI-OBJECTIVE: NSGA-II] ---
    has_holdout = len(all_folds) > len(cv_folds)
    avg_cagr = float(np.mean(fold_rets))
    cv_mdd = float(np.mean(fold_mdds[:-1] if has_holdout else fold_mdds))
    avg_pf = float(np.mean(fold_pfs))
    avg_win_rate = float(np.mean(fold_wins))

    # [UPGRADED] Soft Penalty (Gradient-Preserving Constraint)
    # NSGA-II needs a continuous slope to learn. A hard gate (-100) destroys the fitness landscape.
    # We apply a penalty proportional to how far the strategy is from our minimum viable targets.
    
    # Target PF: 1.2 / Target WR: 35%
    if avg_pf < 1.2:
        cagr_penalty += (1.2 - avg_pf) * 50.0  
        mdd_penalty += (1.2 - avg_pf) * 20.0
        
    if avg_win_rate < 35.0:
        cagr_penalty += (35.0 - avg_win_rate) * 2.0
        mdd_penalty += (35.0 - avg_win_rate) * 1.0

    # [NEW] Consistency Penalty (Robustness Check)
    # 1. Fold-level variance penalty
    if len(fold_rets) > 1:
        std_cagr = float(np.std(fold_rets))
        std_mdd = float(np.std(fold_mdds))
        cagr_penalty += std_cagr * 1.5
        mdd_penalty += std_mdd * 1.0

    # 2. [RESTORED] Symbol-level Worst-Case Penalty (Avoid Single-Asset Curve Fitting)
    # If we don't penalize the worst asset, Optuna will curve-fit SOL and let ETH die.
    if len(symbols) > 1:
        sym_pfs = [float(np.mean(sym_total_pfs[s])) for s in symbols]
        min_sym_pf = float(np.min(sym_pfs))
        if min_sym_pf < 1.3: # Increased baseline requirement for all assets
            cagr_penalty += (1.3 - min_sym_pf) * 40.0 
            mdd_penalty += (1.3 - min_sym_pf) * 10.0

    # 3. [NEW] Trade Count Penalty (Avoid Statistical Flukes)
    # Penalize if total trades over 2 years (per symbol) is too low to be trusted.
    avg_trades_per_sym = float(np.mean(fold_trades)) / max(1, len(symbols))
    if avg_trades_per_sym < 35.0:  # Relaxed from 50.0 to 35.0 (approx 1.5 trade/month on 4H)
        cagr_penalty += (35.0 - avg_trades_per_sym) * 3.0
        mdd_penalty += (35.0 - avg_trades_per_sym) * 1.0
    adjusted_cagr = avg_cagr - cagr_penalty
    adjusted_mdd = cv_mdd + mdd_penalty

    # Store Attributes (Logging - unpenalized raw values for transparency)
    trial.set_user_attr("avg_cagr", avg_cagr)
    trial.set_user_attr("avg_mdd", cv_mdd)
    trial.set_user_attr("avg_trades", float(np.mean(fold_trades)))
    trial.set_user_attr("avg_pf", avg_pf)
    
    for sym in symbols:
        scrs_sym = sym_total_scores[sym][:len(cv_folds)]
        trial.set_user_attr(f"{sym}_cv_cagr", float(np.mean(sym_total_scores[sym])) if sym_total_scores[sym] else -100.0)
        trial.set_user_attr(f"{sym}_mdd", float(np.max(sym_total_mdds[sym])) if sym_total_mdds[sym] else 0.0)

    # Return Penalized Tuple for NSGA-II
    return adjusted_cagr, adjusted_mdd
