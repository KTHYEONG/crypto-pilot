"""
Futures Optuna objective: CPCV paths, Kelly-CVaR scalar, disk+memory signal cache.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

from config.settings import (
    FUTURES_INITIAL_BALANCE,
    SLIPPAGE_RATE,
    TRADING_FEE_RATE,
)
from src.domain.futures.engine_futures import BacktestEngineFast
from src.domain.futures.engine_portfolio_futures import PortfolioBacktestEngineFast
from src.domain.futures.opt_futures_utils.cv_utils import (
    CPCVPath,
    cpcv_complement_segments,
)
from src.domain.futures.opt_futures_utils.metrics import (
    calc_cvar5_loss_pct_from_equity,
    calc_max_underwater_days_from_equity,
    calc_mdd_from_equity,
    calc_profit_factor_from_pnl,
    compute_pbo_from_cpcv_paths,
)
from src.domain.futures.strategies_futures import UltimateStrategy

from .data_utils import (
    _build_aligned_2d_from_prebuilt,
    _dataframe_to_symbol_arrays,
    _segment_with_context,
    align_data_for_2d_engine,
)
from .objective import _log_tw_from_ret_pct, calc_tail_ratio_from_equity
from .signal_cache import (
    _ARRAYS_CACHE_MAXSIZE,
    _arrays_cache,
    _build_signal_cache_key,
    _cache_lock,
    _dataset_fingerprint_from_df,
    _SignalCacheKey,
    get_or_compute_signals,
)

_logger: logging.Logger = logging.getLogger("opt_futures")









































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
) -> Tuple[float, float, float, int, float, float, int, int, np.ndarray, float, float]:
    if precomputed_signal_df is not None:
        sig_oos: pd.DataFrame = precomputed_signal_df
    else:
        full_signal: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
        sig_oos, execution_start_idx = _segment_with_context(full_signal, test_start, test_end)

    warmup_bars: int = 0
    sig_oos.attrs = {"warmup_bars": warmup_bars}

    tf_hours = 1.0
    if tf.endswith("h"):
        try:
            tf_hours = float(tf.replace("h", ""))
        except ValueError:
            tf_hours = 1.0
    elif tf.endswith("d"):
        try:
            tf_hours = float(tf.replace("d", "")) * 24.0
        except ValueError:
            tf_hours = 24.0

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
    engine.leverage = float(params.get("LEVERAGE", 1))
    engine.risk_per_trade = float(params.get("RISK_PER_TRADE", 0.01))
    engine.funding_events_per_bar = 1.0

    params_fixed = params.copy()
    params_fixed["USE_COMPOUNDING"] = True
    engine.strategy = type(
        "MockStrategy",
        (object,),
        {"params": params_fixed, "name": getattr(strategy, "name", "Mock")},
    )

    try:
        result: Dict[str, Any] = engine.run()
        trades_df: pd.DataFrame = result.get("trades_df", pd.DataFrame())
    except Exception as exc:
        _logger.warning("Backtest engine error: %s", exc, exc_info=True)
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0, np.array([]), 0.0, 0.0

    if trades_df is None or trades_df.empty:
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0, np.array([]), 0.0, 0.0

    long_count: int = int(len(trades_df[trades_df["side"] == "LONG"]))
    short_count: int = int(len(trades_df[trades_df["side"] == "SHORT"]))

    equity_curve = result.get("equity_curve", np.array([]))
    mdd_pct: float = abs(float(result.get("mdd_pct", 0.0)))
    ret_pct: float = float(result.get("total_return_pct", 0.0))
    fund_paid = float(result.get("total_funding_paid", 0.0))
    gross_abs = float(result.get("gross_pnl_abs", 0.0))

    span_days: float = (
        float(
            (
                sig_oos["datetime"].iloc[-1]
                - sig_oos["datetime"].iloc[min(execution_start_idx, len(sig_oos) - 1)]
            ).total_seconds()
            / 86400.0
        )
        if "datetime" in sig_oos.columns and not sig_oos.empty
        else 1.0
    )
    span_days = max(span_days, 1.0)

    true_pnl = trades_df["pnl"] - trades_df["entry_fee"]
    win_rate: float = (
        float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100) if len(trades_df) > 0 else 0.0
    )
    pf = calc_profit_factor_from_pnl(true_pnl)

    total_ret_ratio = 1.0 + (ret_pct / 100.0)
    cagr = ((total_ret_ratio ** (365.0 / span_days)) - 1.0) * 100.0 if total_ret_ratio > 0 else -100.0

    return (
        cagr,
        ret_pct,
        mdd_pct,
        len(trades_df),
        win_rate,
        pf,
        long_count,
        short_count,
        equity_curve,
        fund_paid,
        gross_abs,
    )








def run_oos_margin_shared_portfolio(
    symbols: List[str],
    tf: str,
    params: Dict[str, Any],
    oos_data_maps: Dict[str, Dict[str, Any]],
    *,
    cache_root: Optional[Path] = None,
    oos_end_idx: Optional[int] = None,
    return_signal_dfs: bool = False,
) -> Dict[str, Any]:
    """
    OOS slice only: aligned multi-symbol portfolio engine (same as CPCV multi path).
    """
    strategy: UltimateStrategy = UltimateStrategy(name="OOS_Portfolio", params=params)
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    seg_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        full_df = oos_data_maps[sym][tf]
        oos_start = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
        fp = _dataset_fingerprint_from_df(full_df)
        cache_key = _build_signal_cache_key(params, sym, tf, len(full_df), fp)
        full_sig = get_or_compute_signals(cache_key, full_df, strategy, disk_cache_root=cache_root)
        end_cap = int(oos_end_idx) if oos_end_idx is not None else len(full_df)
        seg, _ = _segment_with_context(full_sig, oos_start, end_cap)
        full_signal_dfs[sym] = full_sig
        seg_dfs[sym] = seg

    aligned_data, master_dt = align_data_for_2d_engine(seg_dfs, symbols)
    if not aligned_data or master_dt.empty:
        return {"ok": False}

    engine = PortfolioBacktestEngineFast(
        aligned_data=aligned_data,
        symbol_names=symbols,
        strategy_params=params,
        initial_balance=FUTURES_INITIAL_BALANCE,
        fee_rate=TRADING_FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    trades_df, equity_curve, final_balance = engine.run()

    tf_hours = 4.0
    if tf.endswith("h"):
        try:
            tf_hours = float(tf.replace("h", ""))
        except ValueError:
            tf_hours = 4.0

    span_days = float(len(master_dt)) * (tf_hours / 24.0)
    span_days = max(span_days, 1.0)

    mdd_pct = float(calc_mdd_from_equity(equity_curve))
    total_ret_pct = float((final_balance / FUTURES_INITIAL_BALANCE - 1.0) * 100.0)
    total_ret_ratio = max(1.0 + total_ret_pct / 100.0, 1e-9)
    cagr_pct = float((total_ret_ratio ** (365.0 / span_days) - 1.0) * 100.0)
    calmar = float(cagr_pct / max(mdd_pct, 1e-6)) if mdd_pct > 1e-6 else 0.0
    cvar_pct = float(calc_cvar5_loss_pct_from_equity(equity_curve))
    hw_days = float(calc_max_underwater_days_from_equity(equity_curve, tf_hours))

    moic = float(final_balance / FUTURES_INITIAL_BALANCE)
    eq_np = np.asarray(equity_curve, dtype=np.float64).ravel()
    min_eq_ratio = (
        float(np.min(eq_np) / float(FUTURES_INITIAL_BALANCE)) if eq_np.size > 0 else moic
    )
    tw_ratio = float(min(moic, min_eq_ratio))

    long_c = int(len(trades_df[trades_df["side"] == "LONG"])) if not trades_df.empty else 0
    short_c = int(len(trades_df[trades_df["side"] == "SHORT"])) if not trades_df.empty else 0
    tot_t = long_c + short_c
    minority = float(min(long_c, short_c)) / float(max(tot_t, 1)) * 100.0

    true_pnl = trades_df["pnl"] - trades_df["entry_fee"] if not trades_df.empty else pd.Series(dtype=float)
    win_rate = (
        float((len(true_pnl[true_pnl > 0]) / len(trades_df)) * 100.0) if tot_t > 0 else 0.0
    )
    pf = float(calc_profit_factor_from_pnl(true_pnl)) if tot_t > 0 else 1.0

    gross_abs = float(trades_df["pnl"].abs().sum()) if not trades_df.empty else 0.0

    res = {
        "ok": True,
        "trades_df": trades_df,
        "equity_curve": equity_curve,
        "final_balance": float(final_balance),
        "mdd_pct": mdd_pct,
        "cagr_pct": cagr_pct,
        "total_return_pct": total_ret_pct,
        "calmar_ratio": calmar,
        "cvar_pct": cvar_pct,
        "hw_recovery_days": hw_days,
        "moic": moic,
        "min_equity_wealth_ratio": min_eq_ratio,
        "terminal_wealth_ratio": tw_ratio,
        "long_trades": long_c,
        "short_trades": short_c,
        "total_trades": tot_t,
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "oos_long_short_minority_pct": minority,
        "gross_pnl_abs": gross_abs,
        "span_days": span_days,
        "tail_ratio": float(calc_tail_ratio_from_equity(equity_curve)),
    }
    if return_signal_dfs:
        res["full_signal_dfs"] = full_signal_dfs
    return res


def run_multi_window_oos_holdout(
    params: Dict[str, Any],
    symbols: List[str],
    tf: str,
    oos_data_maps: Dict[str, Dict[str, Any]],
    n_sub_windows: int = 2,
    *,
    cache_root: Optional[Path] = None,
    full_holdout_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Anchored expanding OOS windows for futures.
    """
    ref_sym = symbols[0]
    oos_start = int(oos_data_maps[ref_sym].get(f"oos_start_idx_{tf}", 0))
    full_end = len(oos_data_maps[ref_sym][tf])
    # 4H data: ~6 bars/day. 4 mo: ~120 days * 6 = 720 bars.
    bars_per_sub = 720 if tf == "4h" else 2880
    
    ends_raw: List[int] = []
    for i in range(1, n_sub_windows + 1):
        cap = oos_start + int(i * bars_per_sub)
        ends_raw.append(min(cap, full_end))
    ends_raw.append(full_end)

    ordered: List[int] = []
    seen = set()
    for e in ends_raw:
        if e > oos_start + 100 and e not in seen:
            seen.add(e)
            ordered.append(int(e))

    if full_holdout_result is not None:
        full_res = full_holdout_result
    else:
        full_res = run_oos_margin_shared_portfolio(
            symbols, tf, params, oos_data_maps, cache_root=cache_root
        )

    if not ordered:
        return {
            "windows": [],
            "median_cagr_pct": float(full_res.get("cagr_pct", -100.0)),
            "worst_mdd_pct": float(full_res.get("mdd_pct", 100.0)),
            "positive_windows": 0,
            "total_windows": 0,
            "full_window_result": full_res,
        }

    windows: List[Dict[str, Any]] = []
    cagrs: List[float] = []
    for end in ordered:
        if end >= full_end:
            r = full_res
        else:
            # Expanding window: reuse the same OOS evaluator with capped end index.
            r = run_oos_margin_shared_portfolio(
                symbols,
                tf,
                params,
                oos_data_maps,
                cache_root=cache_root,
                oos_end_idx=end,
            )
            
        cagr_w = float(r.get("cagr_pct", -100.0))
        cagrs.append(cagr_w)
        windows.append({
            "end_idx": int(end),
            "cagr_pct": cagr_w,
            "mdd_pct": float(r.get("mdd_pct", 100.0)),
            "pf": float(r.get("profit_factor", 1.0)),
            "trades": float(r.get("total_trades", 0)),
        })

    med = float(np.median(cagrs)) if cagrs else -100.0
    worst_mdd = float(max((float(w["mdd_pct"]) for w in windows), default=100.0))
    pos = int(sum(1 for c in cagrs if c > 0.0))

    return {
        "windows": windows,
        "median_cagr_pct": med,
        "worst_mdd_pct": worst_mdd,
        "positive_windows": pos,
        "total_windows": len(windows),
        "full_window_result": full_res,
    }


def _regime_stress_label(mult: float) -> str:
    if mult > 0.5: return "risk_on"
    if mult > 0.0: return "cautious"
    return "stress"


def compute_regime_conditional_oos_metrics(
    full_signal_dfs: Dict[str, pd.DataFrame],
    portfolio_equity_curve: np.ndarray,
    oos_start_idx: int,
    symbols: List[str],
) -> Dict[str, Dict[str, float]]:
    ref = symbols[0]
    if ref not in full_signal_dfs: return {}
    sig = full_signal_dfs[ref]
    if "regime_risk_mult" not in sig.columns: return {}
    
    eq = np.asarray(portfolio_equity_curve, dtype=np.float64).ravel()
    rrm = sig["regime_risk_mult"].to_numpy(dtype=np.float64)
    start = int(oos_start_idx)
    n = min(len(rrm) - start, len(eq))
    if n < 2: return {}
    
    rrm_slice = rrm[start : start + n]
    eq_slice = eq[:n]
    
    labels = [_regime_stress_label(float(rrm_slice[i])) for i in range(n)]
    log_ret = np.diff(np.log(np.maximum(eq_slice, 1e-12)))
    
    keys = ("risk_on", "cautious", "stress")
    sum_log = {k: 0.0 for k in keys}
    bar_ct = {k: 0.0 for k in keys}
    for j in range(n):
        bar_ct[labels[j]] += 1.0
    for i in range(1, n):
        lab = labels[i]
        sum_log[lab] += float(log_ret[i-1])
        
    out = {}
    for lab in keys:
        bc = bar_ct[lab]
        slr = sum_log[lab]
        ret_pct = float((np.exp(slr) - 1.0) * 100.0) if bc > 0 else 0.0
        idx = [j for j in range(n) if labels[j] == lab]
        mdd_c = 0.0
        if len(idx) >= 2:
            sub_eq = eq_slice[np.asarray(idx, dtype=np.int64)]
            mdd_c = float(calc_mdd_from_equity(sub_eq))
        
        avg_br = float((np.exp(slr / max(bc, 1.0)) - 1.0) * 100.0) if bc > 0 else 0.0
        out[lab] = {
            "bar_count": bc,
            "return_pct": ret_pct,
            "mdd_pct": mdd_c,
            "avg_bar_return": avg_br,
        }
    return out


def run_cpcv_complement_evaluation(
    params: Dict[str, Any],
    symbols: List[str],
    tf: str,
    data_maps: Dict[str, Dict[str, Any]],
    cpcv_paths: List[CPCVPath],
    all_block_ranges: List[Tuple[int, int]],
    *,
    oos_path_scores: Sequence[float],
    signal_disk_cache_root: Optional[Path] = None,
    project_root: Optional[str] = None,
    concurrency_penalty_scale: float = 1.0,
) -> Tuple[float, float]:
    """
    Evaluate CPCV complement (train) segments for each path on fixed params; compare to stored OOS path scores.
    Returns (pbo, spearman_rho).
    """
    oos_list = [float(x) for x in oos_path_scores]
    if len(cpcv_paths) != len(oos_list) or not cpcv_paths or not all_block_ranges:
        return (0.5, 0.0)

    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    if ref_df is None or ref_df.empty:
        return (0.5, 0.0)

    p = dict(params)
    strategy: UltimateStrategy = UltimateStrategy(name="PBOComplement", params=p)

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_full: Optional[pd.DataFrame] = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            continue
        fp = _dataset_fingerprint_from_df(target_df_full)
        cache_key: _SignalCacheKey = _build_signal_cache_key(p, sym, tf, len(target_df_full), fp)
        full_signal_dfs[sym] = get_or_compute_signals(
            cache_key, target_df_full, strategy, disk_cache_root=signal_disk_cache_root
        )

    if len(full_signal_dfs) != len(symbols):
        return (0.5, 0.0)

    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in symbols:
        target_df_full = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            return (0.5, 0.0)
        fp = _dataset_fingerprint_from_df(target_df_full)
        # Use string key for dictionary
        sig_key = str(_build_signal_cache_key(p, sym, tf, len(target_df_full), fp))
        with _cache_lock:
            if sig_key in _arrays_cache:
                _arrays_cache.move_to_end(sig_key)
                prebuilt_full_arrays[sym] = _arrays_cache[sig_key]
                continue
        arrs = _dataframe_to_symbol_arrays(full_signal_dfs[sym])
        with _cache_lock:
            while len(_arrays_cache) >= _ARRAYS_CACHE_MAXSIZE:
                _arrays_cache.popitem(last=False)
            _arrays_cache[sig_key] = arrs
        prebuilt_full_arrays[sym] = arrs

    liq_mdd_thr = float(p.get("LIQUIDATION_MDD_THRESHOLD", 20.0))
    is_scores: List[float] = []

    for path in cpcv_paths:
        comp = cpcv_complement_segments(path, all_block_ranges)
        seg_raw_logs: List[float] = []
        running_balance = float(FUTURES_INITIAL_BALANCE)
        for test_start, test_end in comp:
            adj_s = test_start + is_off
            adj_e = test_end + is_off

            aligned_data = _build_aligned_2d_from_prebuilt(
                prebuilt_full_arrays, symbols, adj_s, adj_e
            )
            if aligned_data is None:
                seg_raw_logs.append(-10.0)
                continue

            segment_initial = max(running_balance, 1e-9)
            engine = PortfolioBacktestEngineFast(
                aligned_data=aligned_data,
                symbol_names=symbols,
                strategy_params=p,
                initial_balance=segment_initial,
                fee_rate=TRADING_FEE_RATE,
                slippage_rate=SLIPPAGE_RATE,
            )
            _, equity_curve, final_balance = engine.run()
            running_balance = max(float(final_balance), 1e-9)

            ret_pct = float((final_balance / segment_initial - 1.0) * 100.0)
            raw_log = _log_tw_from_ret_pct(ret_pct)
            mdd_seg = (
                float(calc_mdd_from_equity(equity_curve)) if equity_curve.size > 0 else 0.0
            )
            if mdd_seg >= liq_mdd_thr:
                raw_log -= 1e9
            seg_raw_logs.append(raw_log)

        is_scores.append(float(np.sum(seg_raw_logs)) if seg_raw_logs else -10.0)

    if any(not np.isfinite(x) for x in is_scores):
        return (0.5, 0.0)
    return compute_pbo_from_cpcv_paths(is_scores, oos_list)
