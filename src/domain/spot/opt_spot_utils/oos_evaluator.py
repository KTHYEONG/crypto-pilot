from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .data_utils import (
    _dataframe_to_symbol_arrays,
    _dataframe_to_symbol_arrays_extended,
    _segment_span_days,
    _segment_with_context,
)
from .objective import _cpcv_path_compound_raw_log_tw
from .signal_cache import (
    _ARRAYS_CACHE_MAXSIZE,
    _arrays_cache,
    _build_signal_cache_key,
    _cache_lock,
    _dataset_fingerprint_from_df,
    _SignalCacheKey,
    get_or_compute_signals,
)

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]

from config.opt_config import OPT_SPOT_CONFIG
from config.settings import SPOT_INITIAL_BALANCE
from src.domain.spot.engine_spot import BacktestEngineFastSpot
from src.domain.spot.opt_spot_utils.cv_utils import (
    CPCVPath,
    cpcv_complement_segments,
)
from src.domain.spot.opt_spot_utils.metrics import (
    calc_mdd_from_equity,
    calc_profit_factor_from_pnl,
    calc_tail_ratio_from_equity,
    calc_tail_ratio_from_trades,
    compute_pbo_from_cpcv_paths,
    cvar_loss_pct_from_simple_returns,
    max_underwater_bars_from_equity,
    portfolio_cagr_pct_from_equity,
)
from src.domain.spot.portfolio_shared_cash import run_shared_cash_multi_symbol
from src.domain.spot.strategies_spot import UltimateSpotStrategy

_logger: logging.Logger = logging.getLogger("opt_spot")

# Optuna TPE `constraints_func`: each value <= 0 means satisfied (Gardner-style soft constraints).

SymbolFoldResult = Tuple[str, float, float, float, float, float, float, float, np.ndarray, float]


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
) -> Tuple[float, float, float, int, float, float, int, np.ndarray, float]:
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
    engine.strategy.params = params_fixed

    try:
        result: Dict[str, Any] = engine.run()
        trades_df: pd.DataFrame = result.get("trades_df", pd.DataFrame())
    except Exception as e:
        _logger.warning("Backtest engine error: %s", e, exc_info=True)
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, np.array([]), 0.0

    if trades_df is None or trades_df.empty:
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, np.array([]), 0.0

    long_count: int = len(trades_df[trades_df["side"] == "LONG"])

    equity_curve = result.get("equity_curve", np.array([]))
    mdd_pct: float = abs(float(result.get("mdd_pct", 0.0)))
    ret_pct: float = float(result.get("total_return_pct", 0.0))

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

    true_pnl = trades_df["pnl"]
    win_rate: float = (
        float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100) if len(trades_df) > 0 else 0.0
    )
    pf = calc_profit_factor_from_pnl(true_pnl)

    total_ret_ratio = 1.0 + (ret_pct / 100.0)
    cagr = (
        ((total_ret_ratio ** (365.0 / span_days)) - 1.0) * 100.0 if total_ret_ratio > 0 else -100.0
    )

    tail_ratio = calc_tail_ratio_from_equity(equity_curve) if equity_curve.size > 1 else 0.0

    return (
        cagr,
        ret_pct,
        mdd_pct,
        len(trades_df),
        win_rate,
        pf,
        long_count,
        equity_curve,
        tail_ratio,
    )


def _compute_signal_stats(sig_df: pd.DataFrame) -> Dict[str, float]:
    """OOS holdout: quantify signal vs regime gating on the execution window."""
    n = len(sig_df)
    if n == 0:
        return {}
    if "long_entry_signal" not in sig_df.columns:
        return {}
    les = sig_df["long_entry_signal"]
    signal_rate = float((les > 0).sum() / n)
    if "regime_risk_mult" in sig_df.columns:
        rrm = sig_df["regime_risk_mult"]
        regime_rate = float((rrm > 0.0).sum() / n)
        joint_rate = float(((les > 0) & (rrm > 0.0)).sum() / n)
    else:
        regime_rate = float("nan")
        joint_rate = float("nan")
    return {
        "signal_fire_rate": signal_rate,
        "regime_on_rate": regime_rate,
        "joint_entry_eligible_rate": joint_rate,
    }


def run_holdout_shared_cash_portfolio(
    params: Dict[str, Any],
    symbols: List[str],
    tf: str,
    oos_data_maps: Dict[str, Dict[str, Any]],
    *,
    signal_disk_cache_root: Optional[Path] = None,
    return_signal_dfs: bool = False,
    concurrency_penalty_scale: float = 1.0,
    oos_end_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    OOS holdout: single shared-cash run from oos_start_idx to end for all symbols.
    If oos_end_idx is set, evaluation ends at that absolute bar index (exclusive upper bound on OHLCV index).
    """
    p = dict(params)
    strategy: UltimateSpotStrategy = UltimateSpotStrategy(name="HoldoutSpot", params=p)
    strategy._portfolio_eval_ctx = {"data_maps": oos_data_maps, "symbols": list(symbols), "tf": tf}
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    try:
        for sym in symbols:
            df_full: Optional[pd.DataFrame] = oos_data_maps.get(sym, {}).get(tf)
            if df_full is None or df_full.empty:
                continue
            fp = _dataset_fingerprint_from_df(df_full)
            cache_key: _SignalCacheKey = _build_signal_cache_key(p, sym, tf, len(df_full), fp)
            full_signal_dfs[sym] = get_or_compute_signals(
                cache_key,
                df_full,
                strategy,
                disk_cache_root=signal_disk_cache_root,
            )
    finally:
        strategy._portfolio_eval_ctx = None
    if len(full_signal_dfs) != len(symbols):
        failed: Dict[str, Any] = {
            "portfolio_cagr_pct": -100.0,
            "mdd_pct": 100.0,
            "cvar_pct": 100.0,
            "tail_ratio": 0.0,
            "long_trades": 0.0,
            "min_path_tw": 0.0,
            "dd_bars": 0.0,
            "final_balance": 0.0,
            "moic": 0.0,
            "equity_curve": np.array([]),
            "oos_signal_stats_by_symbol": {},
        }
        if return_signal_dfs:
            failed["full_signal_dfs"] = {}
        return failed

    ref_sym = symbols[0]
    oos_start = int(oos_data_maps[ref_sym].get(f"oos_start_idx_{tf}", 0))
    ref_df = full_signal_dfs[ref_sym]
    slice_start = max(0, oos_start - 1)
    slice_end = len(ref_df)
    if oos_end_idx is not None:
        slice_end = min(int(oos_end_idx), len(ref_df))
    _logger.info(
        "Holdout OOS debug: oos_start=%d, slice_start=%d, slice_end=%d, "
        "exec_start=%d, seg_len=%d, n_symbols=%d",
        oos_start,
        slice_start,
        slice_end,
        max(1, oos_start - slice_start),
        slice_end - slice_start,
        len(symbols),
    )
    if slice_end - slice_start < 5:
        _logger.warning("Holdout OOS segment too short (len < 5). Returning FAIL.")
        failed: Dict[str, Any] = {
            "portfolio_cagr_pct": -100.0,
            "mdd_pct": 100.0,
            "cvar_pct": 100.0,
            "tail_ratio": 0.0,
            "long_trades": 0.0,
            "min_path_tw": 0.0,
            "dd_bars": 0.0,
            "final_balance": 0.0,
            "moic": 0.0,
            "equity_curve": np.array([]),
            "oos_signal_stats_by_symbol": {},
        }
        if return_signal_dfs:
            failed["full_signal_dfs"] = full_signal_dfs
        return failed

    symbol_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    rank_scores: Dict[str, np.ndarray] = {}
    for sym in symbols:
        seg = full_signal_dfs[sym].iloc[slice_start:slice_end]
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
        allow_python_fallback=False,
        concurrency_penalty_scale=float(concurrency_penalty_scale),
    )
    eq = res.equity_curve
    _logger.info(
        "Holdout OOS result: final_balance=%.2f, total_trades=%d, eq_first=%.2f, eq_last=%.2f",
        res.final_balance,
        res.total_trades,
        float(eq[0]) if eq.size > 0 else -1.0,
        float(eq[-1]) if eq.size > 0 else -1.0,
    )
    span_days = _segment_span_days(
        full_signal_dfs[ref_sym].iloc[slice_start:slice_end],
        max(holdout_warmup_bars, execution_start_idx),
    )
    cagr = float(portfolio_cagr_pct_from_equity(eq, span_days)) if eq.size > 1 else -100.0
    mdd = float(calc_mdd_from_equity(eq)) if eq.size > 1 else 100.0
    cvar_pct = float(cvar_loss_pct_from_simple_returns(eq)) if eq.size > 1 else 100.0
    pnl = res.pnl_array
    pos_pnl = float(np.sum(pnl[pnl > 0.0]))
    neg_pnl = float(np.abs(np.sum(pnl[pnl < 0.0])))
    pf = pos_pnl / neg_pnl if neg_pnl > 1e-12 else 10.0
    calmar = (cagr / abs(mdd)) if abs(mdd) > 1e-6 else 0.0
    win_rate = float(np.sum(pnl > 0.0) / len(pnl)) * 100.0 if len(pnl) > 0 else 0.0

    twr = max(float(res.final_balance / initial_balance), 1e-9)
    pnl_for_tail = np.asarray(res.pnl_array, dtype=np.float64)
    if pnl_for_tail.size >= 10:
        tail_r = float(calc_tail_ratio_from_trades(pnl_for_tail))
    else:
        tail_r = 1.0
    eq_tail_r = float(calc_tail_ratio_from_equity(eq)) if eq.size > 1 else 0.0
    _logger.info(
        "Holdout tail ratio: trade-based=%.4f, equity-curve (reference)=%.4f, n_trades=%d",
        tail_r,
        eq_tail_r,
        int(pnl_for_tail.size),
    )
    dd_bars = float(max_underwater_bars_from_equity(eq)) if eq.size > 1 else 0.0
    final_bal = float(res.final_balance)
    moic = final_bal / initial_balance if initial_balance > 0 else 0.0
    oos_signal_stats_by_symbol: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        seg = full_signal_dfs[sym].iloc[slice_start:slice_end]
        diag = seg.iloc[execution_start_idx:]
        oos_signal_stats_by_symbol[sym] = _compute_signal_stats(diag)

    out: Dict[str, Any] = {
        "portfolio_cagr_pct": cagr,
        "mdd_pct": mdd,
        "cvar_pct": cvar_pct,
        "tail_ratio": tail_r,
        "long_trades": float(res.total_trades),
        "min_path_tw": twr,
        "dd_bars": dd_bars,
        "final_balance": final_bal,
        "moic": float(moic),
        "equity_curve": eq,
        "oos_signal_stats_by_symbol": oos_signal_stats_by_symbol,
        "profit_factor": pf,
        "calmar_ratio": calmar,
        "win_rate_pct": win_rate,
        "span_days": span_days,
        "equity_tail_ratio": eq_tail_r,
        "per_symbol_trades": res.per_symbol_trades,
        "per_symbol_wins": res.per_symbol_wins,
        "per_symbol_pnl": res.per_symbol_pnl,
    }
    if return_signal_dfs:
        out["full_signal_dfs"] = full_signal_dfs
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
    ref_len = len(ref_df) - is_off
    if ref_len < 200:
        return (0.5, 0.0)

    p = dict(params)
    strategy: UltimateSpotStrategy = UltimateSpotStrategy(name="PBOComplement", params=p)
    cache_root: Optional[Path] = signal_disk_cache_root
    if cache_root is None and project_root is not None:
        cache_root = Path(project_root) / ".spot_signal_cache"

    strategy._portfolio_eval_ctx = {"data_maps": data_maps, "symbols": list(symbols), "tf": tf}
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    try:
        for sym in symbols:
            target_df_full: Optional[pd.DataFrame] = data_maps.get(sym, {}).get(tf)
            if target_df_full is None or target_df_full.empty:
                continue
            fp = _dataset_fingerprint_from_df(target_df_full)
            cache_key: _SignalCacheKey = _build_signal_cache_key(
                p, sym, tf, len(target_df_full), fp
            )
            full_signal_dfs[sym] = get_or_compute_signals(
                cache_key, target_df_full, strategy, disk_cache_root=cache_root
            )
    finally:
        strategy._portfolio_eval_ctx = None

    if len(full_signal_dfs) != len(symbols):
        return (0.5, 0.0)

    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in symbols:
        target_df_full = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            return (0.5, 0.0)
        fp = _dataset_fingerprint_from_df(target_df_full)
        sig_key = _build_signal_cache_key(p, sym, tf, len(target_df_full), fp)
        with _cache_lock:
            if sig_key in _arrays_cache:
                _arrays_cache.move_to_end(sig_key)
                prebuilt_full_arrays[sym] = _arrays_cache[sig_key]
                continue
        arrs = _dataframe_to_symbol_arrays_extended(full_signal_dfs[sym])
        with _cache_lock:
            while len(_arrays_cache) >= _ARRAYS_CACHE_MAXSIZE:
                _arrays_cache.popitem(last=False)
            _arrays_cache[sig_key] = arrs
        prebuilt_full_arrays[sym] = arrs

    max_slots = int(OPT_SPOT_CONFIG.get("SPOT_MAX_CONCURRENT_POSITIONS", 3))
    is_scores: List[float] = []
    for path in cpcv_paths:
        comp = cpcv_complement_segments(path, all_block_ranges)
        raw = _cpcv_path_compound_raw_log_tw(
            comp,
            prebuilt_full_arrays=prebuilt_full_arrays,
            symbols=symbols,
            params=p,
            is_off=is_off,
            ref_df=ref_df,
            max_slots=max_slots,
            warmup_bars=0,
            concurrency_penalty_scale=concurrency_penalty_scale,
        )
        is_scores.append(raw)
    if any(not np.isfinite(x) for x in is_scores):
        return (0.5, 0.0)
    return compute_pbo_from_cpcv_paths(is_scores, oos_list)


def run_multi_window_oos_holdout(
    params: Dict[str, Any],
    symbols: List[str],
    tf: str,
    oos_data_maps: Dict[str, Dict[str, Any]],
    n_sub_windows: int = 2,
    *,
    signal_disk_cache_root: Optional[Path] = None,
    concurrency_penalty_scale: float = 1.0,
    full_holdout_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Anchored expanding OOS windows (4mo, 8mo, ... + full). Reuses full-window holdout once for the last window.
    Pass full_holdout_result to avoid a second full-OOS shared-cash run when already computed.
    """
    ref_sym = symbols[0]
    oos_start = int(oos_data_maps[ref_sym].get(f"oos_start_idx_{tf}", 0))
    full_end = len(oos_data_maps[ref_sym][tf])
    bars_pm = float(OPT_SPOT_CONFIG.get("SPOT_MULTI_WINDOW_BARS_PER_MONTH", 180.0))
    ends_raw: List[int] = []
    for i in range(1, n_sub_windows + 1):
        cap = oos_start + int(i * 4 * bars_pm)
        ends_raw.append(min(cap, full_end))
    ends_raw.append(full_end)

    ordered: List[int] = []
    seen: Set[int] = set()
    for e in ends_raw:
        if e > oos_start and e not in seen:
            seen.add(e)
            ordered.append(int(e))

    if full_holdout_result is not None:
        full_res = full_holdout_result
    else:
        full_res = run_holdout_shared_cash_portfolio(
            params,
            symbols,
            tf,
            oos_data_maps,
            signal_disk_cache_root=signal_disk_cache_root,
            return_signal_dfs=False,
            concurrency_penalty_scale=concurrency_penalty_scale,
            oos_end_idx=None,
        )

    if not ordered:
        return {
            "windows": [],
            "median_cagr_pct": float(full_res.get("portfolio_cagr_pct", -100.0)),
            "worst_mdd_pct": float(full_res.get("mdd_pct", 100.0)),
            "positive_windows": 0,
            "total_windows": 0,
            "cagr_dispersion": 0.0,
            "full_window_result": full_res,
        }

    windows: List[Dict[str, Any]] = []
    cagrs: List[float] = []
    for end in ordered:
        if end >= full_end:
            r = full_res
        else:
            r = run_holdout_shared_cash_portfolio(
                params,
                symbols,
                tf,
                oos_data_maps,
                signal_disk_cache_root=signal_disk_cache_root,
                return_signal_dfs=False,
                concurrency_penalty_scale=concurrency_penalty_scale,
                oos_end_idx=end,
            )
        cagr_w = float(r["portfolio_cagr_pct"])
        cagrs.append(cagr_w)
        windows.append(
            {
                "end_idx": int(end),
                "cagr_pct": cagr_w,
                "mdd_pct": float(r["mdd_pct"]),
                "pf": float(r["profit_factor"]),
                "trades": float(r["long_trades"]),
                "calmar": float(r["calmar_ratio"]),
                "tail_ratio": float(r["tail_ratio"]),
            }
        )

    mean_c = float(np.mean(cagrs)) if cagrs else 0.0
    std_c = float(np.std(cagrs, ddof=1)) if len(cagrs) > 1 else 0.0
    disp = float(std_c / max(abs(mean_c), 1e-6))
    pos = int(sum(1 for c in cagrs if c > 0.0))
    med = float(np.median(cagrs)) if cagrs else -100.0
    worst_mdd = float(max((float(w["mdd_pct"]) for w in windows), default=100.0))

    return {
        "windows": windows,
        "median_cagr_pct": med,
        "worst_mdd_pct": worst_mdd,
        "positive_windows": pos,
        "total_windows": len(windows),
        "cagr_dispersion": disp,
        "full_window_result": full_res,
    }


def _regime_stress_label(mult: float) -> str:
    if mult > 0.5:
        return "risk_on"
    if mult > 0.0:
        return "cautious"
    return "stress"


def compute_regime_conditional_oos_metrics(
    full_signal_dfs: Dict[str, pd.DataFrame],
    portfolio_equity_curve: np.ndarray,
    oos_start_idx: int,
    symbols: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    OOS bars classified by reference symbol regime_risk_mult; per-regime return and MDD (diagnostic).
    """
    ref = symbols[0]
    if ref not in full_signal_dfs:
        return {}
    sig = full_signal_dfs[ref]
    if "regime_risk_mult" not in sig.columns:
        return {}
    eq = np.asarray(portfolio_equity_curve, dtype=np.float64).ravel()
    rrm = sig["regime_risk_mult"].to_numpy(dtype=np.float64)
    start = int(oos_start_idx)
    n_sig = max(0, len(rrm) - start)
    n_eq = len(eq)
    n = min(n_sig, n_eq)
    if n < 2:
        return {}
    rrm = rrm[start : start + n]
    eq = eq[:n]

    labels = [_regime_stress_label(float(rrm[i])) for i in range(n)]
    log_ret = np.diff(np.log(np.maximum(eq, 1e-12)))
    keys = ("risk_on", "cautious", "stress")
    sum_log: Dict[str, float] = {k: 0.0 for k in keys}
    bar_ct: Dict[str, float] = {k: 0.0 for k in keys}
    for j in range(n):
        bar_ct[labels[j]] += 1.0
    for i in range(1, n):
        lab = labels[i]
        lr = float(log_ret[i - 1])
        sum_log[lab] += lr

    out: Dict[str, Dict[str, float]] = {}
    for lab in keys:
        slr = sum_log[lab]
        bc = bar_ct[lab]
        ret_pct = float((np.exp(slr) - 1.0) * 100.0) if bc > 0 else 0.0
        idx = [j for j in range(n) if labels[j] == lab]
        if len(idx) >= 2:
            sub_eq = eq[np.asarray(idx, dtype=np.int64)]
            mdd_c = float(calc_mdd_from_equity(sub_eq))
        else:
            mdd_c = 0.0
        avg_br = float((np.exp(slr / max(bc, 1.0)) - 1.0) * 100.0) if bc > 0 else 0.0
        out[lab] = {
            "bar_count": bc,
            "return_pct": ret_pct,
            "mdd_pct": mdd_c,
            "avg_bar_return": avg_br,
        }
    return out
