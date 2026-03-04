"""
전략의 파라미터를 입력받아 실제 백테스트 엔진을 구동하고, 최적화 타겟인 목적 함수(Objective Score)를 산출함.
심볼별/폴드별 성과를 종합하여 전략의 견고함(Robustness)을 평가하는 핵심 연산 로직을 포함함.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from config.settings import FUTURES_INITIAL_BALANCE
from config.opt_config import WARMUP_PERIOD
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.engine_fast_futures import BacktestEngineFast
from src.futures_strategy.opt_v2_utils.metrics import (
    calc_romad_from_metrics, 
    calc_profit_factor_from_pnl,
    calc_mdd_from_equity,
    calc_sortino_from_equity
)
from src.futures_strategy.opt_v2_utils.cv_utils import build_purged_walk_forward_folds
from src.futures_strategy.opt_v2_utils.opt_params import suggest_params_v2

_logger: logging.Logger = logging.getLogger("opt_v2")

SymbolFoldResult = Tuple[str, float, float, float, float, float, float, np.ndarray, float]

_MAX_SYMBOL_WORKERS: int = 8

def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: Dict[str, int] = {"1h": 24, "4h": 6}
    ratio_map: Dict[str, float] = {"1h": 0.08, "4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))

EMBARGO_BARS: Dict[str, int] = {
    "1h": compute_embargo_bars("1h"),
    "4h": compute_embargo_bars("4h"),
}

# Params that affect strategy.generate_signals() (ultimate.py). Only keys actually suggested
# in opt_config / suggest_params_v2; trials differing only in LEVERAGE/RISK reuse cache.
SIGNAL_CACHE_PARAM_KEYS: frozenset[str] = frozenset({
    "TSMOM_ENTRY_THRESHOLD",
    "TSMOM_WEIGHT_DECAY",
    "ATR_WINDOW",
    "VELOCITY_K",
    "ATR_PRC_WINDOW",
})

_SIGNAL_CACHE_MAXSIZE: int = 64
_cache_lock: threading.Lock = threading.Lock()
_signal_cache: OrderedDict[Tuple[Tuple[Tuple[str, Any], ...], str, str], pd.DataFrame] = OrderedDict()


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

    segment: pd.DataFrame = full_signal_dfs[sym].iloc[adj_test_start:adj_test_end]

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
        )
    except Exception as exc:
        _logger.warning("Symbol-level evaluation error for %s: %s", sym, exc, exc_info=True)
        return None

    if equity_curve.size == 0:
        span_days_sym: float = 0.0
    elif "datetime" in segment.columns and not segment.empty:
        span_seconds: float = float(
            (segment["datetime"].iloc[-1] - segment["datetime"].iloc[0]).total_seconds()
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

# ... (skip to objective_v2)

def _build_signal_cache_key(params: Dict[str, Any], sym: str, tf: str) -> Tuple[Tuple[Tuple[str, Any], ...], str, str]:
    """Build cache key from signal-affecting params only so exit/risk/leverage differences still hit cache."""
    signal_items: List[Tuple[str, Any]] = sorted(
        (k, params[k]) for k in SIGNAL_CACHE_PARAM_KEYS if k in params
    )
    return (tuple(signal_items), sym, tf)


def get_or_compute_signals(
    cache_key: Tuple[Tuple[Tuple[str, Any], ...], str, str],
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
) -> Tuple[float, float, float, int, float, float, int, int, np.ndarray]:
    if precomputed_signal_df is not None:
        sig_oos: pd.DataFrame = precomputed_signal_df
    else:
        full_signal: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
        sig_oos = full_signal.iloc[test_start:test_end].copy()

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
    )
    engine.leverage = params.get("LEVERAGE", 1)
    engine.risk_per_trade = params.get("RISK_PER_TRADE", 0.01)
    # Crypto generic standard: funding fee charged every 8 hours
    engine.funding_events_per_bar = tf_hours / 8.0

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

    span_days: float = float((sig_oos["datetime"].iloc[-1] - sig_oos["datetime"].iloc[0]).total_seconds() / 86400.0) if "datetime" in sig_oos.columns else 1.0
    span_days = max(span_days, 1.0)
    
    true_pnl = trades_df["pnl"] - trades_df["entry_fee"]
    win_rate: float = float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100) if len(trades_df) > 0 else 0.0
    pf = calc_profit_factor_from_pnl(true_pnl)
    
    # Calculate geometric mean return directly for single symbol
    total_ret_ratio = 1.0 + (ret_pct / 100.0)
    cagr = ((total_ret_ratio ** (365.0 / span_days)) - 1.0) * 100.0 if total_ret_ratio > 0 else -100.0

    return cagr, ret_pct, mdd_pct, len(trades_df), win_rate, pf, long_count, short_count, equity_curve


def objective_v2(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf_target: str,
    *,
    space: Dict[str, Dict[str, Any]],
    project_root: Optional[str] = None,
) -> Tuple[float, float]:
    params: Dict[str, Any] = suggest_params_v2(trial, space, tf_target)
    if params.pop("_INVALID_CONSTRAINT", False):
        raise optuna.TrialPruned()

    tf: str = tf_target
    strategy: UltimateStrategy = UltimateStrategy(name="FuturesV2", params=params)

    lengths: List[int] = [len(data_maps[sym][tf]) - data_maps[sym].get(f"is_start_idx_{tf}", 0) for sym in symbols if tf in data_maps.get(sym, {})]
    if not lengths:
        raise optuna.TrialPruned()
    min_len: int = min(lengths)
    base_df_for_folds: pd.DataFrame = pd.DataFrame(index=range(min_len))
    
    # [INSTITUTIONAL] 4H 매크로 추세의 사이클을 온전히 담아내기 위해 테스트 폴드를 분기(3개월) 단위로 확대.
    # 진성 OOS가 6개월 확보되어 있으므로 내부 Holdout은 15%로 축소하여 훈련/CV 데이터를 극대화.
    cv_folds, holdout_fold = build_purged_walk_forward_folds(
        base_df_for_folds, 
        n_folds=3, 
        holdout_ratio=0.15, 
        embargo=EMBARGO_BARS.get(tf, 0)
    )

    if not cv_folds:
        raise optuna.TrialPruned()

    all_folds = cv_folds + ([holdout_fold] if holdout_fold[2] > holdout_fold[1] else [])

    # Pre-fetch full signal DataFrame per (params, sym, tf) for ALL symbols (cache reuse across folds).
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_pre: Optional[pd.DataFrame] = data_maps[sym].get(tf)
        if target_df_pre is None:
            continue
        cache_key: Tuple[Tuple[Tuple[str, Any], ...], str, str] = _build_signal_cache_key(params, sym, tf)
        full_signal_dfs[sym] = get_or_compute_signals(cache_key, target_df_pre, strategy)

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
    
    for f_idx, fold_info in enumerate(all_folds):
        if len(fold_info) == 4:
            train_start, train_end, test_start, test_end = fold_info
        else:
            train_start = 0
            train_end, test_start, test_end = fold_info
            
        sym_scores: List[float] = []
        sym_rets: List[float] = []
        sym_mdds: List[float] = []
        sym_trades_fold: List[float] = []
        sym_wins_fold: List[float] = []
        sym_pfs_fold: List[float] = []
        
        # Array to accumulate aggregated portfolio equity curve for the current fold
        target_symbols: List[str] = symbols
        fold_agg_eq: Optional[np.ndarray] = None
        fold_span_days: float = 0.0

        max_workers: int = min(len(target_symbols), _MAX_SYMBOL_WORKERS)
        symbol_results: List[SymbolFoldResult] = []

        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _evaluate_symbol_for_fold_parallel,
                        sym,
                        params=params,
                        strategy=strategy,
                        tf=tf,
                        data_maps=data_maps,
                        full_signal_dfs=full_signal_dfs,
                        test_start=test_start,
                        test_end=test_end,
                    )
                    for sym in target_symbols
                ]
                for future in as_completed(futures):
                    result: Optional[SymbolFoldResult] = future.result()
                    if result is None:
                        continue
                    symbol_results.append(result)

        for (
            sym,
            s,
            r,
            m,
            t,
            w,
            pf,
            eq,
            span_days_sym,
        ) in symbol_results:
            sym_scores.append(s)
            sym_rets.append(r)
            sym_mdds.append(m)
            sym_trades_fold.append(float(t))
            sym_wins_fold.append(w)
            sym_pfs_fold.append(pf)

            sym_total_scores[sym].append(s)
            sym_total_rets[sym].append(r)
            sym_total_mdds[sym].append(m)
            sym_total_trades[sym].append(float(t))
            sym_total_wins[sym].append(w)
            sym_total_pfs[sym].append(pf)

            # Aggregate Equity Curve
            if eq.size > 0:
                eq_start: float = float(eq[0]) if eq[0] > 0 else 1e-9
                eq_norm: np.ndarray = eq / eq_start
                if fold_agg_eq is None:
                    fold_agg_eq = eq_norm.copy()
                else:
                    # Sum correctly by maintaining the base capital accurately
                    min_len_eq: int = min(len(fold_agg_eq), len(eq_norm))
                    fold_agg_eq = fold_agg_eq[:min_len_eq] + eq_norm[:min_len_eq]

            fold_span_days = max(fold_span_days, span_days_sym)

        if not sym_scores or fold_agg_eq is None:
            raise optuna.TrialPruned()

        # Pure Portfolio Metrics: CAGR and MDD
        start_eq = fold_agg_eq[0] if fold_agg_eq[0] > 0 else 1e-9
        end_eq = fold_agg_eq[-1]
        total_ret_ratio = max(end_eq / start_eq, 0.0001)
        
        # Portfolio CAGR (kept for logging/attributes if needed)
        port_cagr = ((total_ret_ratio ** (365.0 / max(fold_span_days, 1.0))) - 1.0) * 100.0
        port_mdd = float(calc_mdd_from_equity(fold_agg_eq))
        
        # No artificial penalties.
        mean_trades_fold: float = float(np.mean(sym_trades_fold))
        port_trades: float = float(np.sum(sym_trades_fold))
        
        # Prune if zero trades to prevent optimization failure
        if port_trades == 0.0:
            raise optuna.TrialPruned()
            
        # Use Sortino instead of CAGR (cap at 5.0 to prevent overfitting to low trade count)
        port_sortino: float = min(float(calc_sortino_from_equity(fold_agg_eq, fold_span_days)), 5.0)
        
        # Hybrid Score: Scale absolute returns by their risk-adjusted quality
        hybrid_score: float = port_cagr * max(port_sortino, 0.0)

        fold_scores.append(hybrid_score)
        fold_rets.append(float(np.mean(sym_rets)))
        fold_mdds.append(port_mdd)
        fold_trades.append(mean_trades_fold)
        fold_wins.append(float(np.mean(sym_wins_fold)))
        fold_pfs.append(float(np.mean(sym_pfs_fold)))

    # --- [NEW] Aggregated Equity Curve Calculation ---
    # We will build a single continuous aggregated equity curve across all folds and symbols
    # to evaluate Portfolio Sortino and Portfolio Maximum Drawdown.
    # Since we didn't store all `eq` arrays in the loop to save memory, we can approximate 
    # cross-sectional robustness by using average of Sortino/MDD, OR we can collect them.
    # Actually, the user specifically requested an aggregated portfolio equity curve per trial.
    # To do this cleanly, we'd need to sum eq arrays.
    
    has_holdout = len(all_folds) > len(cv_folds)
    cv_scores = fold_scores[:-1] if has_holdout else fold_scores
    holdout_score = fold_scores[-1] if has_holdout else 0.0

    if not cv_scores:
        raise optuna.TrialPruned()
    cv_mean_score: float = float(np.mean(cv_scores))
    cv_mdd: float = float(np.mean(fold_mdds[:-1] if has_holdout else fold_mdds))
    cv_std_score: float = float(np.std(cv_scores)) if len(cv_scores) > 1 else 0.0
    # Final consistency-adjusted objective
    cv_final_obj: float = cv_mean_score - 0.5 * cv_std_score

    # User Attributes
    trial.set_user_attr("avg_cagr", cv_mean_score)
    trial.set_user_attr("avg_mdd", cv_mdd)
    trial.set_user_attr("avg_ret", float(np.mean(fold_rets)))
    trial.set_user_attr("avg_trades", float(np.mean(fold_trades)))
    trial.set_user_attr("avg_win_rate", float(np.mean(fold_wins)))
    trial.set_user_attr("avg_pf", float(np.mean(fold_pfs)))
    trial.set_user_attr("holdout_cagr", holdout_score)
    trial.set_user_attr("cv_scores", cv_scores)
    
    for sym in symbols:
        if len(sym_total_scores[sym]) > 0:
            cv_scrs_sym: List[float] = sym_total_scores[sym][:len(cv_folds)]
            ho_scr_sym: float = sym_total_scores[sym][-1] if has_holdout else 0.0
            s_cv_mean: float = float(np.mean(cv_scrs_sym)) if cv_scrs_sym else -100.0
            s_cv_std: float = float(np.std(cv_scrs_sym)) if len(cv_scrs_sym) > 1 else 0.0
            s_cv_min: float = float(np.min(cv_scrs_sym)) if cv_scrs_sym else -100.0
            s_pass_fold: int = sum(1 for sc in cv_scrs_sym if sc > 0)
            s_fold_cnt: int = len(cv_scrs_sym)
            
            s_mdd: float = float(np.max(sym_total_mdds[sym]))
            s_trades: float = float(np.sum(sym_total_trades[sym]))
            s_ret_sum: float = float(np.sum(sym_total_rets[sym]))
            s_pf_mean: float = float(np.mean(sym_total_pfs[sym])) if sym_total_pfs[sym] else 1.0
            s_win: float = float(np.mean(sym_total_wins[sym])) if sym_total_wins[sym] else 0.0
        else:
            s_cv_mean, s_cv_std, s_cv_min, s_pass_fold, s_fold_cnt = -100.0, 0.0, -100.0, 0, 0
            ho_scr_sym, s_mdd, s_trades, s_ret_sum, s_pf_mean, s_win = -100.0, 0.0, 0.0, 0.0, 1.0, 0.0
            
        trial.set_user_attr(f"{sym}_cv_mean", s_cv_mean)
        trial.set_user_attr(f"{sym}_cv_std", s_cv_std)
        trial.set_user_attr(f"{sym}_cv_min", s_cv_min)
        trial.set_user_attr(f"{sym}_pass_ratio", f"{s_pass_fold}/{s_fold_cnt}")
        trial.set_user_attr(f"{sym}_ho_score", ho_scr_sym)
        trial.set_user_attr(f"{sym}_mdd", s_mdd)
        trial.set_user_attr(f"{sym}_trades", s_trades)
        trial.set_user_attr(f"{sym}_ret_sum", s_ret_sum)
        trial.set_user_attr(f"{sym}_pf", s_pf_mean)
        trial.set_user_attr(f"{sym}_win", s_win)

    # Final Dual-Objective return: Maximize Adjusted Hybrid Score, Minimize MDD
    return cv_final_obj, cv_mdd
