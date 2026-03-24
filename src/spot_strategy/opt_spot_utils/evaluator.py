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

from config.opt_config import OPT_SPOT_CONFIG, get_spot_effective_independent_trials
from config.settings import SPOT_INITIAL_BALANCE
from src.spot_strategy.strategies_spot import UltimateSpotStrategy
from src.spot_strategy.engine_spot import BacktestEngineFastSpot
from src.spot_strategy.portfolio_shared_cash import run_shared_cash_multi_symbol
from src.spot_strategy.opt_spot_utils.metrics import (
    calc_profit_factor_from_pnl,
    calc_mdd_from_equity,
    calc_sortino_from_equity,
    cvar_loss_pct_from_simple_returns,
    compute_dsr_from_path_values,
    max_underwater_bars_from_equity,
    mean_of_worst_quartile,
    portfolio_cagr_pct_from_equity,
    probabilistic_sharpe_ratio,
)
from src.spot_strategy.opt_spot_utils.cv_utils import build_cpcv_test_paths_with_fallback
from src.spot_strategy.opt_spot_utils.opt_params import suggest_params_spot

_logger: logging.Logger = logging.getLogger("opt_spot")

SymbolFoldResult = Tuple[str, float, float, float, float, float, float, np.ndarray, float]

_MAX_SYMBOL_WORKERS: int = max(1, int(os.getenv("OPT_SPOT_SYMBOL_WORKERS", "1")))

def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: Dict[str, int] = {"4h": 6}
    ratio_map: Dict[str, float] = {"4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))

EMBARGO_BARS: Dict[str, int] = {
    "4h": compute_embargo_bars("4h"),
}

SIGNAL_CACHE_PARAM_KEYS: frozenset[str] = frozenset([
    "MACRO_EMA_PERIOD",
    "ADX_PERIOD",
    "ADX_THRESHOLD",
    "MOMENTUM_PERIOD",
    "ATR_PERIOD",
    "VOL_Z_THRESHOLD",
    "BB_WINDOW",
    "VOL_Z_WINDOW",
    "BTC_REGIME_SMA_PERIOD",
    "VOL_CONFIRM_OR_MODE",
])

_SignalCacheKey = Tuple[Tuple[Tuple[str, Any], ...], str, str, int]
_SIGNAL_CACHE_MAXSIZE: int = 64
_cache_lock: threading.Lock = threading.Lock()
_signal_cache: OrderedDict[_SignalCacheKey, pd.DataFrame] = OrderedDict()

def _segment_with_context(
    full_signal_df: pd.DataFrame,
    exec_start_idx: int,
    exec_end_idx: int,
) -> Tuple[pd.DataFrame, int]:
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
    strategy: UltimateSpotStrategy,
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
        return None

    segment, execution_start_idx = _segment_with_context(
        full_signal_dfs[sym], adj_test_start, adj_test_end
    )

    try:
        (
            score,
            ret_pct,
            mdd_pct,
            trades_count,
            win_rate,
            pf,
            long_count,
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
    signal_items: List[Tuple[str, Any]] = sorted(
        (k, params[k]) for k in SIGNAL_CACHE_PARAM_KEYS if k in params
    )
    return (tuple(signal_items), sym, tf, data_len)

def get_or_compute_signals(
    cache_key: _SignalCacheKey,
    target_df: pd.DataFrame,
    strategy: UltimateSpotStrategy,
) -> pd.DataFrame:
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
    strategy: UltimateSpotStrategy,
    params: Dict[str, Any],
    symbol: str,
    tf: str,
    target_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    full_merge_idx: np.ndarray,
    precomputed_daily_df: Optional[pd.DataFrame],
    test_start: int,
    test_end: int,
    precomputed_signal_df: Optional[pd.DataFrame] = None,
    execution_start_idx: int = 0,
) -> Tuple[float, float, float, int, float, float, int, np.ndarray]:
    if precomputed_signal_df is not None:
        sig_oos: pd.DataFrame = precomputed_signal_df
    else:
        full_signal: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
        sig_oos, execution_start_idx = _segment_with_context(full_signal, test_start, test_end)

    warmup_bars: int = 0
    sig_oos.attrs = {"warmup_bars": warmup_bars}
    
    engine: BacktestEngineFastSpot = BacktestEngineFastSpot(
        hourly_df=sig_oos,
        daily_df=daily_df,
        strategy=strategy,
        initial_balance=SPOT_INITIAL_BALANCE,
        merge_index_map=None,
        precomputed_daily_df=None,
        warmup_bars=warmup_bars,
        execution_start_idx=execution_start_idx,
    )

    params_fixed = params.copy()
    params_fixed["USE_COMPOUNDING"] = True
    engine.strategy = type("MockStrategy", (object,), {"params": params_fixed, "name": getattr(strategy, "name", "Mock")})
    
    try:
        result: Dict[str, Any] = engine.run()
        trades_df: pd.DataFrame = result.get("trades_df", pd.DataFrame())
    except Exception as e:
        _logger.warning("Backtest engine error: %s", e, exc_info=True)
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, np.array([])
        
    if trades_df is None or trades_df.empty:
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, np.array([])
        
    long_count: int = len(trades_df[trades_df["side"] == "LONG"])

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
    
    total_ret_ratio = 1.0 + (ret_pct / 100.0)
    cagr = ((total_ret_ratio ** (365.0 / span_days)) - 1.0) * 100.0 if total_ret_ratio > 0 else -100.0

    return cagr, ret_pct, mdd_pct, len(trades_df), win_rate, pf, long_count, equity_curve

def _merge_spot_fixed_signal_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    out.setdefault("BB_WINDOW", int(OPT_SPOT_CONFIG.get("BB_WINDOW", 20)))
    out.setdefault("VOL_Z_WINDOW", int(OPT_SPOT_CONFIG.get("VOL_Z_WINDOW", 20)))
    out.setdefault("VOL_EXPANSION_MULT", float(OPT_SPOT_CONFIG.get("VOL_EXPANSION_MULT", 1.05)))
    macro = int(out.get("MACRO_EMA_PERIOD", 200))
    out["BTC_REGIME_SMA_PERIOD"] = int(out.get("BTC_REGIME_SMA_PERIOD", macro))
    return out


def _dataframe_to_symbol_arrays(sig_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    required = ("open", "high", "low", "close", "atr", "long_entry_signal", "entry_upper")
    for c in required:
        if c not in sig_df.columns:
            raise ValueError(f"Missing column {c} for shared-cash segment.")
    out: Dict[str, np.ndarray] = {}
    for c in required:
        out[c] = sig_df[c].to_numpy(dtype=np.float64)
    if "regime_risk_mult" in sig_df.columns:
        out["regime_risk_mult"] = sig_df["regime_risk_mult"].to_numpy(dtype=np.float64)
    return out


def _segment_span_days(sig_df: pd.DataFrame, execution_start_idx: int) -> float:
    if "datetime" not in sig_df.columns or sig_df.empty:
        return 1.0
    i0 = min(max(0, int(execution_start_idx)), len(sig_df) - 1)
    span_seconds = float(
        (sig_df["datetime"].iloc[-1] - sig_df["datetime"].iloc[i0]).total_seconds()
    )
    return max(span_seconds / 86400.0, 1.0)


def objective_spot(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf_target: str,
    *,
    space: Dict[str, Dict[str, Any]],
    mode: str = "single",
    project_root: Optional[str] = None,
) -> float:
    params: Dict[str, Any] = _merge_spot_fixed_signal_params(suggest_params_spot(trial, space, tf_target))
    tf: str = tf_target
    strategy: UltimateSpotStrategy = UltimateSpotStrategy(name="OptSpot", params=params)

    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    if ref_df is None or ref_df.empty:
        raise optuna.TrialPruned()
    ref_len = len(ref_df) - is_off
    if ref_len < 200:
        raise optuna.TrialPruned()

    embargo = int(EMBARGO_BARS.get(tf, 0))
    cpcv_paths, nb_cpcv, k_cpcv = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)
    if not cpcv_paths:
        raise optuna.TrialPruned()
    n_independent_paths = max(2, nb_cpcv // k_cpcv)

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_full: Optional[pd.DataFrame] = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            continue
        cache_key: _SignalCacheKey = _build_signal_cache_key(params, sym, tf, len(target_df_full))
        full_signal_dfs[sym] = get_or_compute_signals(cache_key, target_df_full, strategy)

    if len(full_signal_dfs) != len(symbols):
        raise optuna.TrialPruned()

    max_slots = int(OPT_SPOT_CONFIG.get("SPOT_MAX_CONCURRENT_POSITIONS", 3))
    min_seg_trades = int(OPT_SPOT_CONFIG.get("SPOT_MIN_TRADES_PER_CPCV_SEGMENT", 4))
    # Signals are pre-computed on full IS; per-segment re-warmup would skip most of each test block.
    warmup_bars = 0

    cfg = OPT_SPOT_CONFIG
    std_pen_mult = float(cfg.get("SPOT_PATH_CROSS_PATH_LOG_TW_STD_PENALTY_MULT", 2.0))
    min_path_pen_mult = float(cfg.get("SPOT_MIN_PATH_CAGR_PENALTY_MULT", 1.5))
    seg_fail_pen = float(cfg.get("SPOT_SEGMENT_TRADE_FAIL_PENALTY", 2.0))
    mdd_thr = float(cfg.get("SPOT_MDD_PENALTY_THRESHOLD_PCT", 35.0))
    cvar_thr = float(cfg.get("SPOT_OBJECTIVE_CVAR_PENALTY_THRESHOLD", 15.0))
    cvar25_w = float(cfg.get("SPOT_OBJECTIVE_CVAR25_LOG_TW_WEIGHT", 0.20))
    dd_bars_thr = int(cfg.get("SPOT_DD_DURATION_BARS_THRESHOLD", 100))
    dd_bar_pen = float(cfg.get("SPOT_DD_DURATION_PENALTY_PER_BAR", 0.001))

    path_compound_log_tw: List[float] = []
    path_compound_tw_ratio: List[float] = []
    path_worst_mdd: List[float] = []
    path_max_cvar: List[float] = []
    for path_idx, path in enumerate(cpcv_paths):
        seg_log_tw: List[float] = []
        seg_tw_ratio: List[float] = []
        seg_mdds: List[float] = []
        seg_cvars: List[float] = []
        running_balance = float(SPOT_INITIAL_BALANCE)
        for test_start, test_end in path:
            abs_start = is_off + int(test_start)
            abs_end = is_off + int(test_end)
            slice_start = max(0, abs_start - 1)
            slice_end = min(len(ref_df), abs_end)
            if slice_end - slice_start < 5:
                continue
            symbol_arrays: Dict[str, Dict[str, np.ndarray]] = {}
            rank_scores: Dict[str, np.ndarray] = {}
            for sym in symbols:
                full_df = full_signal_dfs[sym]
                seg = full_df.iloc[slice_start:slice_end].copy()
                symbol_arrays[sym] = _dataframe_to_symbol_arrays(seg)
                if "slot_rank_score" in seg.columns:
                    rank_scores[sym] = seg["slot_rank_score"].to_numpy(dtype=np.float64)

            execution_start_idx = max(1, abs_start - slice_start)
            segment_initial = max(running_balance, 1e-9)
            result = run_shared_cash_multi_symbol(
                symbol_arrays,
                symbols,
                params,
                initial_balance=segment_initial,
                max_concurrent_positions=max_slots,
                rank_scores=rank_scores if rank_scores else None,
                warmup_bars=warmup_bars,
                execution_start_idx=execution_start_idx,
            )
            eq = result.equity_curve
            if eq.size == 0:
                twr = 1.0
            else:
                twr = max(float(result.final_balance / segment_initial), 1e-9)
            log_tw = float(np.log(twr))
            if eq.size > 1:
                seg_mdds.append(float(calc_mdd_from_equity(eq)))
                seg_cvars.append(float(cvar_loss_pct_from_simple_returns(eq)))
                uw = max_underwater_bars_from_equity(eq)
                if uw > dd_bars_thr:
                    log_tw -= float(uw - dd_bars_thr) * dd_bar_pen
            else:
                seg_mdds.append(0.0)
                seg_cvars.append(0.0)

            if int(result.total_trades) < min_seg_trades:
                log_tw -= seg_fail_pen

            seg_log_tw.append(log_tw)
            seg_tw_ratio.append(twr)
            running_balance = max(float(result.final_balance), 1e-9)

        if not seg_log_tw:
            raise optuna.TrialPruned()
        path_compound_log_tw.append(float(np.sum(seg_log_tw)))
        path_compound_tw_ratio.append(float(np.prod(seg_tw_ratio)) if seg_tw_ratio else 1.0)
        path_worst_mdd.append(float(np.max(seg_mdds)) if seg_mdds else 0.0)
        path_max_cvar.append(float(np.max(seg_cvars)) if seg_cvars else 0.0)

        interm = float(np.mean(path_compound_log_tw))
        trial.report(interm, step=path_idx)

    mean_log_tw = float(np.mean(path_compound_log_tw))
    cvar25_log = float(mean_of_worst_quartile(path_compound_log_tw))
    base_growth = 100.0 * (mean_log_tw + cvar25_w * cvar25_log)

    penalty = 0.0
    if len(path_compound_log_tw) > 1:
        penalty += std_pen_mult * float(np.std(path_compound_log_tw, ddof=1))
    min_log_tw = float(np.min(path_compound_log_tw))
    if min_log_tw < 0.0:
        penalty += min_path_pen_mult * abs(min_log_tw)

    worst_mdd = float(np.max(path_worst_mdd)) if path_worst_mdd else 0.0
    if worst_mdd > mdd_thr:
        penalty += (worst_mdd - mdd_thr) * 0.08

    max_cvar = float(np.max(path_max_cvar)) if path_max_cvar else 0.0
    if max_cvar > cvar_thr:
        penalty += (max_cvar - cvar_thr) * 0.15

    growth_score = base_growth - penalty

    min_path_tw = float(np.min(path_compound_tw_ratio)) if path_compound_tw_ratio else 1.0

    trial.set_user_attr("mean_path_terminal_wealth_ratio", float(np.mean(path_compound_tw_ratio)))
    trial.set_user_attr("min_path_terminal_wealth_ratio", min_path_tw)
    trial.set_user_attr("mean_log_terminal_wealth", mean_log_tw)
    trial.set_user_attr("cvar25_log_tw", cvar25_log)
    trial.set_user_attr(
        "path_mean_log_tw_std",
        float(np.std(path_compound_log_tw, ddof=1)) if len(path_compound_log_tw) > 1 else 0.0,
    )
    trial.set_user_attr("growth_score", growth_score)
    trial.set_user_attr("cpcv_embargo_bars", embargo)
    trial.set_user_attr("cpcv_n_independent_paths", int(n_independent_paths))

    n_done = trial.number + 1
    n_startup = int(OPT_SPOT_CONFIG.get("tpe_n_startup_trials", 96))
    n_eff = get_spot_effective_independent_trials(n_done, n_startup)
    psr_val = probabilistic_sharpe_ratio(
        float(np.mean(path_compound_log_tw) / (np.std(path_compound_log_tw, ddof=1) + 1e-12)),
        n_independent_paths,
    )
    trial.set_user_attr("psr_paths", psr_val)
    trial.set_user_attr("n_effective_independent_trials", n_eff)

    path_vals = [float(x) for x in path_compound_log_tw]
    trial.set_user_attr(
        "dsr_paths",
        float(compute_dsr_from_path_values(path_vals, n_eff)),
    )

    return float(growth_score)


def run_holdout_shared_cash_portfolio(
    params: Dict[str, Any],
    symbols: List[str],
    tf: str,
    oos_data_maps: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    OOS holdout: single shared-cash run from oos_start_idx to end for all symbols.
    """
    p = _merge_spot_fixed_signal_params(dict(params))
    strategy: UltimateSpotStrategy = UltimateSpotStrategy(name="HoldoutSpot", params=p)
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df_full: Optional[pd.DataFrame] = oos_data_maps.get(sym, {}).get(tf)
        if df_full is None or df_full.empty:
            continue
        cache_key: _SignalCacheKey = _build_signal_cache_key(p, sym, tf, len(df_full))
        full_signal_dfs[sym] = get_or_compute_signals(cache_key, df_full, strategy)
    if len(full_signal_dfs) != len(symbols):
        return {
            "portfolio_cagr_pct": -100.0,
            "mdd_pct": 100.0,
            "cvar_pct": 100.0,
            "pf": 0.0,
            "long_trades": 0.0,
            "min_path_tw": 0.0,
            "dd_bars": 0.0,
        }

    ref_sym = symbols[0]
    oos_start = int(oos_data_maps[ref_sym].get(f"oos_start_idx_{tf}", 0))
    ref_df = full_signal_dfs[ref_sym]
    slice_start = max(0, oos_start - 1)
    slice_end = len(ref_df)
    if slice_end - slice_start < 5:
        return {
            "portfolio_cagr_pct": -100.0,
            "mdd_pct": 100.0,
            "cvar_pct": 100.0,
            "pf": 0.0,
            "long_trades": 0.0,
            "min_path_tw": 0.0,
            "dd_bars": 0.0,
        }

    symbol_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    rank_scores: Dict[str, np.ndarray] = {}
    for sym in symbols:
        seg = full_signal_dfs[sym].iloc[slice_start:slice_end].copy()
        symbol_arrays[sym] = _dataframe_to_symbol_arrays(seg)
        if "slot_rank_score" in seg.columns:
            rank_scores[sym] = seg["slot_rank_score"].to_numpy(dtype=np.float64)

    execution_start_idx = max(1, oos_start - slice_start)
    holdout_warmup_bars = 0
    initial_balance = float(SPOT_INITIAL_BALANCE)
    max_slots = int(OPT_SPOT_CONFIG.get("SPOT_MAX_CONCURRENT_POSITIONS", 3))
    res = run_shared_cash_multi_symbol(
        symbol_arrays,
        symbols,
        p,
        initial_balance=initial_balance,
        max_concurrent_positions=max_slots,
        rank_scores=rank_scores if rank_scores else None,
        warmup_bars=holdout_warmup_bars,
        execution_start_idx=execution_start_idx,
    )
    eq = res.equity_curve
    span_days = _segment_span_days(
        full_signal_dfs[ref_sym].iloc[slice_start:slice_end],
        max(holdout_warmup_bars, execution_start_idx),
    )
    cagr = float(portfolio_cagr_pct_from_equity(eq, span_days)) if eq.size > 1 else -100.0
    mdd = float(calc_mdd_from_equity(eq)) if eq.size > 1 else 100.0
    cvar_pct = float(cvar_loss_pct_from_simple_returns(eq)) if eq.size > 1 else 100.0
    twr = max(float(res.final_balance / initial_balance), 1e-9)
    pf_est = float((1.0 + max(cagr, 0.0) / 100.0) / (1.0 + abs(mdd) / 100.0))
    dd_bars = float(max_underwater_bars_from_equity(eq)) if eq.size > 1 else 0.0
    return {
        "portfolio_cagr_pct": cagr,
        "mdd_pct": mdd,
        "cvar_pct": cvar_pct,
        "pf": pf_est,
        "long_trades": float(res.total_trades),
        "min_path_tw": twr,
        "dd_bars": dd_bars,
    }
