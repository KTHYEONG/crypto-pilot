"""
전략의 파라미터를 입력받아 실제 백테스트 엔진을 구동하고, 최적화 타겟인 목적 함수(Objective Score)를 산출함.
심볼별/폴드별 성과를 종합하여 전략의 견고함(Robustness)을 평가하는 핵심 연산 로직을 포함함.
"""
import logging
import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from config.settings import FUTURES_INITIAL_BALANCE
from config.opt_config import SEARCH_SPACE_V2
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.engine_fast_futures import BacktestEngineFast
from src.futures_strategy.opt_v2_utils.metrics import calc_romad_from_metrics
from src.futures_strategy.opt_v2_utils.cv_utils import build_anchored_folds
from src.futures_strategy.opt_v2_utils.opt_params import suggest_params_v2

_logger: logging.Logger = logging.getLogger("opt_v2")
EMBARGO_BARS: Dict[str, int] = {"4h": 6, "1d": 2}

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
) -> Tuple[float, float, float, int, float]:
    sig_oos: pd.DataFrame = target_df.iloc[test_start:test_end].copy()
    sig_oos.attrs = {"warmup_bars": 0}
    merge_oos: np.ndarray = full_merge_idx[test_start:test_end]

    engine: BacktestEngineFast = BacktestEngineFast(
        hourly_df=sig_oos,
        daily_df=daily_df,
        strategy=strategy,
        initial_balance=FUTURES_INITIAL_BALANCE,
        merge_index_map=merge_oos,
        precomputed_daily_df=precomputed_daily_df,
    )
    engine.leverage = params.get("LEVERAGE", 1)
    engine.risk_per_trade = params.get("RISK_PER_TRADE", 0.02)
    engine.funding_events_per_bar = 3 if params.get("TIMEFRAME") == "1d" else 1

    try:
        result: Dict[str, Any] = engine.run()
        trades_df: pd.DataFrame = result.get("trades_df")
    except Exception as e:
        _logger.warning("Backtest engine error: %s", e)
        return -100.0, 0.0, 0.0, 0, 0.0

    if trades_df is None or trades_df.empty:
        return -100.0, 0.0, 0.0, 0, 0.0

    long_count: int = len(trades_df[trades_df["side"] == "LONG"])
    short_count: int = len(trades_df[trades_df["side"] == "SHORT"])

    mdd_pct: float = abs(float(result.get("mdd_pct", 0.0)))
    ret_pct: float = float(result.get("total_return_pct", 0.0))

    span_days: float = float((sig_oos["datetime"].iloc[-1] - sig_oos["datetime"].iloc[0]).total_seconds() / 86400.0) if "datetime" in sig_oos.columns else 1.0
    span_days = max(span_days, 1.0)

    true_pnl: pd.Series = trades_df["pnl"] - trades_df["entry_fee"]
    win_rate: float = float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100) if len(trades_df) > 0 else 0.0
    
    score: float
    ret_pct, mdd_pct = float(ret_pct), float(mdd_pct)
    score, ret_pct, mdd_pct = calc_romad_from_metrics(ret_pct, mdd_pct, len(trades_df), tf, span_days)

    if long_count == 0 or short_count == 0:
        score -= 5.0

    return score, ret_pct, mdd_pct, len(trades_df), win_rate

def objective_v2(trial: optuna.Trial, data_maps: Dict[str, Dict[str, Any]], symbols: List[str]) -> float:
    params: Dict[str, Any] = suggest_params_v2(trial, SEARCH_SPACE_V2)
    if params.pop("_INVALID_CONSTRAINT", False):
        return -100.0

    tf: str = str(params["TIMEFRAME"])
    strategy: UltimateStrategy = UltimateStrategy(name="FuturesV2", params=params)

    lengths: List[int] = [len(data_maps[sym][tf]) for sym in symbols if tf in data_maps.get(sym, {})]
    if not lengths:
        return -10000.0
    min_len: int = min(lengths)
    base_df_for_folds: pd.DataFrame = pd.DataFrame(index=range(min_len))
    cv_folds, holdout_fold = build_anchored_folds(base_df_for_folds, n_folds=3, holdout_ratio=0.25, embargo=EMBARGO_BARS.get(tf, 0))

    if not cv_folds:
        return -10000.0

    all_folds = cv_folds + ([holdout_fold] if holdout_fold[2] > holdout_fold[1] else [])

    fold_scores: List[float] = []
    fold_rets: List[float] = []
    fold_mdds: List[float] = []
    fold_trades: List[float] = []
    fold_wins: List[float] = []
    
    sym_total_rets: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_mdds: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_scores: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_trades: Dict[str, List[float]] = {sym: [] for sym in symbols}
    sym_total_wins: Dict[str, List[float]] = {sym: [] for sym in symbols}
    
    for f_idx, (train_end, test_start, test_end) in enumerate(all_folds):
        sym_scores: List[float] = []
        sym_rets: List[float] = []
        sym_mdds: List[float] = []
        sym_trades_fold: List[float] = []
        sym_wins_fold: List[float] = []
        
        for sym in symbols:
            target_df: pd.DataFrame = data_maps[sym].get(tf)
            daily_df: pd.DataFrame = data_maps[sym].get("1d")
            full_merge_idx: np.ndarray = data_maps[sym].get(f"merge_idx_{tf}")
            if target_df is None or daily_df is None or full_merge_idx is None:
                continue

            tf_idx_test_end_minus_1: int = test_end - 1
            daily_end_idx: int = int(full_merge_idx[tf_idx_test_end_minus_1]) if tf_idx_test_end_minus_1 < len(full_merge_idx) else len(daily_df) - 1
            daily_df_trunc: pd.DataFrame = daily_df.iloc[:daily_end_idx + 1].copy()

            try:
                precomputed_daily_df: pd.DataFrame = strategy.generate_signals(daily_df_trunc)
            except Exception as e:
                continue
            
            s: float; r: float; m: float; t: int; w: float
            s, r, m, t, w = evaluate_symbol_fold(
                strategy, params, sym, tf, target_df, daily_df_trunc, 
                full_merge_idx, precomputed_daily_df, test_start, test_end
            )
            sym_scores.append(s)
            sym_rets.append(r)
            sym_mdds.append(m)
            sym_trades_fold.append(float(t))
            sym_wins_fold.append(w)
            
            sym_total_scores[sym].append(s)
            sym_total_rets[sym].append(r)
            sym_total_mdds[sym].append(m)
            sym_total_trades[sym].append(float(t))
            sym_total_wins[sym].append(w)

        if not sym_scores:
            return -10000.0

        mean_score: float = float(np.mean(sym_scores))
        min_score: float = float(np.min(sym_scores))
        penalty: float = max(0.0, 0.0 - min_score) * 1.5
        avg_fold_sym_score: float = mean_score - penalty
        
        mean_trades_fold: float = float(np.mean(sym_trades_fold))
        min_trades_per_fold: float = 50.0 if tf == "4h" else 15.0
        
        if mean_trades_fold < min_trades_per_fold:
            trade_fold_penalty_factor: float = (mean_trades_fold / min_trades_per_fold) ** 2.0
            if avg_fold_sym_score > 0.0:
                avg_fold_sym_score *= trade_fold_penalty_factor
            else:
                avg_fold_sym_score /= max(trade_fold_penalty_factor, 0.01)

        fold_scores.append(avg_fold_sym_score)
        fold_rets.append(float(np.mean(sym_rets)))
        fold_mdds.append(float(np.mean(sym_mdds)))
        fold_trades.append(mean_trades_fold)
        fold_wins.append(float(np.mean(sym_wins_fold)))

    has_holdout = len(all_folds) > len(cv_folds)
    cv_scores = fold_scores[:-1] if has_holdout else fold_scores
    holdout_score = fold_scores[-1] if has_holdout else 0.0

    shifted: List[float] = [s + 10.0 for s in cv_scores]
    final_score: float
    if any(s <= 0 for s in shifted):
        final_score = float(np.mean(cv_scores)) - 20.0
    else:
        hm: float = len(shifted) / sum(1.0 / s for s in shifted)
        final_score = hm - 10.0

    if has_holdout:
        if holdout_score < 0.0:
            final_score -= abs(holdout_score) * 2.0
        else:
            final_score = (final_score * 0.7) + (holdout_score * 0.3)

    trial.set_user_attr("avg_score", float(np.mean(fold_scores)))
    trial.set_user_attr("avg_ret", float(np.mean(fold_rets)))
    trial.set_user_attr("avg_mdd", float(np.mean(fold_mdds)))
    trial.set_user_attr("avg_trades", float(np.mean(fold_trades)))
    trial.set_user_attr("avg_win_rate", float(np.mean(fold_wins)))
    
    for sym in symbols:
        if len(sym_total_scores[sym]) > 0:
            s_score: float = float(np.mean(sym_total_scores[sym]))
            s_ret: float = float(np.mean(sym_total_rets[sym]))
            s_mdd: float = float(np.max(sym_total_mdds[sym]))
            s_trades: float = float(np.sum(sym_total_trades[sym]))
            s_wins: float = float(np.mean(sym_total_wins[sym]))
        else:
            s_score, s_ret, s_mdd, s_trades, s_wins = -100.0, 0.0, 0.0, 0.0, 0.0
            
        trial.set_user_attr(f"{sym}_score", s_score)
        trial.set_user_attr(f"{sym}_ret", s_ret)
        trial.set_user_attr(f"{sym}_mdd", s_mdd)
        trial.set_user_attr(f"{sym}_trades", s_trades)
        trial.set_user_attr(f"{sym}_win_rate", s_wins)
        
    return float(final_score)
