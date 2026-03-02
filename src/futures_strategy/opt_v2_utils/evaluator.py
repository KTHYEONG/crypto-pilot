"""
전략의 파라미터를 입력받아 실제 백테스트 엔진을 구동하고, 최적화 타겟인 목적 함수(Objective Score)를 산출함.
심볼별/폴드별 성과를 종합하여 전략의 견고함(Robustness)을 평가하는 핵심 연산 로직을 포함함.
"""
from __future__ import annotations

import logging
import threading
import optuna
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from config.settings import FUTURES_INITIAL_BALANCE
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.engine_fast_futures import BacktestEngineFast
from src.futures_strategy.opt_v2_utils.metrics import calc_romad_from_metrics, calc_profit_factor_from_pnl
from src.futures_strategy.opt_v2_utils.cv_utils import build_purged_walk_forward_folds
from src.futures_strategy.opt_v2_utils.opt_params import suggest_params_v2

_logger: logging.Logger = logging.getLogger("opt_v2")

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
# in opt_config / suggest_params_v2; trials differing only in LEVERAGE/STOP_LOSS/TP reuse cache.
SIGNAL_CACHE_PARAM_KEYS: frozenset[str] = frozenset({
    "ATR_PERIOD", "ENTRY_PERIOD", "BB_STD", "KC_MULT",
    "VOL_WINDOW", "VOL_MULTIPLIER",
    "W_BREAKOUT", "W_TREND", "W_VOLUME", "W_MEAN_REVERSION", 
    "THRESHOLD_LOOKBACK", "THRESHOLD_QUANTILE", # [NEW] Priority 1 Adaptive Thresholds
    "EXIT_TYPE", "PSAR_STEP", "PSAR_MAX",
    "VWAP_STD_MULT", "STOCH_RSI_PERIOD", "STOCH_RSI_EXTREME", "CMF_PERIOD",
    "MACRO_SMA_PERIOD",
})

_SIGNAL_CACHE_MAXSIZE: int = 64
_cache_lock: threading.Lock = threading.Lock()
_signal_cache: OrderedDict[Tuple[Tuple[Tuple[str, Any], ...], str, str], pd.DataFrame] = OrderedDict()

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
) -> Tuple[float, float, float, int, float, float, int, int]:
    if precomputed_signal_df is not None:
        sig_oos: pd.DataFrame = precomputed_signal_df.copy()
    else:
        # True OOS: compute indicators on full history so warmup is correct, then slice test range
        full_signal: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
        sig_oos = full_signal.iloc[test_start:test_end].copy()
    sig_oos.attrs = {"warmup_bars": 0}
    
    # Intraday native signals don't require the merge index mapping anymore
    # merge_oos: np.ndarray = full_merge_idx[test_start:test_end]

    engine: BacktestEngineFast = BacktestEngineFast(
        hourly_df=sig_oos,
        daily_df=daily_df, # Kept for API signature, but ignored internally
        strategy=strategy,
        initial_balance=FUTURES_INITIAL_BALANCE,
        merge_index_map=None, # Disabled
        precomputed_daily_df=None, # Disabled, calculation happens natively inside engine
        warmup_bars=0,
    )
    engine.leverage = params.get("LEVERAGE", 1)
    engine.risk_per_trade = params.get("RISK_PER_TRADE", 0.02)
    # V2.6: Intraday timeframes (30m, 1h, 4h) naturally trigger funding events per bar via timestamp checks.
    # We fix this to 1 to prevent duplicated fees on lower timeframes.
    engine.funding_events_per_bar = 1

    try:
        result: Dict[str, Any] = engine.run()
        trades_df: pd.DataFrame = result.get("trades_df")
    except Exception as e:
        _logger.warning("Backtest engine error: %s", e, exc_info=True)
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0

    if trades_df is None or trades_df.empty:
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0

    long_count: int = len(trades_df[trades_df["side"] == "LONG"])
    short_count: int = len(trades_df[trades_df["side"] == "SHORT"])

    mdd_pct: float = abs(float(result.get("mdd_pct", 0.0)))
    ret_pct: float = float(result.get("total_return_pct", 0.0))

    span_days: float = float((sig_oos["datetime"].iloc[-1] - sig_oos["datetime"].iloc[0]).total_seconds() / 86400.0) if "datetime" in sig_oos.columns else 1.0
    span_days = max(span_days, 1.0)

    true_pnl: pd.Series = trades_df["pnl"] - trades_df["entry_fee"]
    win_rate: float = float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100) if len(trades_df) > 0 else 0.0
    
    # V2.2: PF calculated from fee-deducted net PNL to prevent inflated selection bias.
    pf: float = calc_profit_factor_from_pnl(true_pnl)
    
    score: float
    ret_pct, mdd_pct = float(ret_pct), float(mdd_pct)
    score, ret_pct, mdd_pct = calc_romad_from_metrics(
        ret_pct, mdd_pct, len(trades_df), tf, span_days, win_rate=win_rate, pf=pf, leverage=float(engine.leverage)
    )
    
    return score, ret_pct, mdd_pct, len(trades_df), win_rate, pf, long_count, short_count

def objective_v2(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf_target: str,
    *,
    space: Dict[str, Dict[str, Any]],
    project_root: Optional[str] = None,
) -> float:
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

    # Pre-fetch full signal DataFrame per (params, sym, tf) for cache reuse across folds.
    target_symbols_pre: List[str] = [symbols[0]] if symbols else []
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in target_symbols_pre:
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
        
        # V2.1: Coin-Specific Optimization
        # Optimize only for the primary symbol (symbols[0]) to prevent sub-optimal parameter compromise
        target_symbols = [symbols[0]] if symbols else []
        for sym in target_symbols:
            target_df: pd.DataFrame = data_maps[sym].get(tf)
            daily_df: pd.DataFrame = data_maps[sym].get("1d")
            full_merge_idx: np.ndarray = data_maps[sym].get(f"merge_idx_{tf}")
            if target_df is None or daily_df is None or full_merge_idx is None:
                continue

            is_start_idx = data_maps[sym].get(f"is_start_idx_{tf}", 0)
            adj_test_start = test_start + is_start_idx
            adj_test_end = test_end + is_start_idx

            tf_idx_test_end_minus_1: int = adj_test_end - 1
            daily_end_idx: int = int(full_merge_idx[tf_idx_test_end_minus_1]) if tf_idx_test_end_minus_1 < len(full_merge_idx) else len(daily_df) - 1
            daily_df_trunc: pd.DataFrame = daily_df.iloc[:daily_end_idx + 1].copy()

            segment: pd.DataFrame = full_signal_dfs[sym].iloc[adj_test_start:adj_test_end].copy()
            s: float
            r: float
            m: float
            t: int
            w: float
            pf: float
            lc: int
            sc: int
            s, r, m, t, w, pf, lc, sc = evaluate_symbol_fold(
                strategy, params, sym, tf, target_df, daily_df_trunc,
                full_merge_idx, None, adj_test_start, adj_test_end,
                precomputed_signal_df=segment,
            )
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

        if not sym_scores:
            raise optuna.TrialPruned()

        mean_score: float = float(np.mean(sym_scores))
        
        # Soft penalty for fold high variance (removed to rely on global Mean-Variance utility)
        avg_fold_sym_score: float = mean_score
        
        mean_trades_fold: float = float(np.mean(sym_trades_fold))
        
        # Trade count soft penalty is removed. Relying purely on sigmoid credibility in metrics.py

        fold_scores.append(avg_fold_sym_score)
        fold_rets.append(float(np.mean(sym_rets)))
        # Risk management: Use max MDD across symbols to identify the worst-case bottleneck in the fold
        fold_mdds.append(float(np.max(sym_mdds)) if sym_mdds else 0.0)
        fold_trades.append(mean_trades_fold)
        fold_wins.append(float(np.mean(sym_wins_fold)))
        fold_pfs.append(float(np.mean(sym_pfs_fold)))

    has_holdout = len(all_folds) > len(cv_folds)
    cv_scores = fold_scores[:-1] if has_holdout else fold_scores
    holdout_score = fold_scores[-1] if has_holdout else 0.0

    if not cv_scores:
        raise optuna.TrialPruned()
    cv_array: np.ndarray = np.array(cv_scores)
    cv_mean: float = float(np.mean(cv_array))
    cv_std: float = float(np.std(cv_array)) if len(cv_array) > 1 else 0.0
    cv_min: float = float(np.min(cv_array))

    # 1. 기관급 평균-분산 효용 함수 (Institutional Mean-Variance Utility)
    # 단일 자산의 수익률 잠재력(Fat-Tail)을 해방하기 위해 편차 페널티를 1.5에서 1.0(표준 샤프 비율)으로 완화.
    base_score: float = cv_mean - (1.0 * cv_std)
    
    # 2. 치명적 실패 방어 (Minimax Penalty)
    # 단 하나의 폴드라도 심각한 손실(Score < -0.5)이 발생하면 TPE 그래디언트에 강력한 하방 페널티를 줌.
    if cv_min < -0.5:
        base_score -= abs(cv_min + 0.5) * 2.0
        
    final_score: float = base_score

    # User Attributes
    trial.set_user_attr("avg_score", float(np.mean(fold_scores)))
    trial.set_user_attr("avg_ret", float(np.mean(fold_rets)))
    trial.set_user_attr("avg_mdd", float(np.mean(fold_mdds)))
    trial.set_user_attr("avg_trades", float(np.mean(fold_trades)))
    trial.set_user_attr("avg_win_rate", float(np.mean(fold_wins)))
    trial.set_user_attr("avg_pf", float(np.mean(fold_pfs)))
    trial.set_user_attr("holdout_score", holdout_score)
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

    return float(final_score)
