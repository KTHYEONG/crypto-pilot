from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]

from config.opt_config import OPT_SPOT_CONFIG, get_spot_effective_independent_trials
from config.settings import SPOT_INITIAL_BALANCE
from src.spot_strategy.strategies_spot import UltimateSpotStrategy
from src.spot_strategy.engine_spot import BacktestEngineFastSpot
from src.spot_strategy.portfolio_shared_cash import run_shared_cash_multi_symbol
from src.spot_strategy.opt_spot_utils.metrics import (
    calc_profit_factor_from_pnl,
    calc_mdd_from_equity,
    calc_sortino_from_equity,
    calc_tail_ratio_from_equity,
    cvar_loss_pct_from_simple_returns,
    compute_dsr_from_path_values,
    max_underwater_bars_from_equity,
    mean_of_worst_quartile,
    portfolio_cagr_pct_from_equity,
    probabilistic_sharpe_ratio,
)
from src.spot_strategy.opt_spot_utils.cv_utils import (
    CPCVPath,
    build_cpcv_test_paths_with_fallback,
)
from src.spot_strategy.opt_spot_utils.opt_params import suggest_params_spot

_logger: logging.Logger = logging.getLogger("opt_spot")

SymbolFoldResult = Tuple[str, float, float, float, float, float, float, float, np.ndarray, float]

_MAX_SYMBOL_WORKERS: int = max(1, int(os.getenv("OPT_SPOT_SYMBOL_WORKERS", "1")))

def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: Dict[str, int] = {"4h": 24}
    ratio_map: Dict[str, float] = {"4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))

EMBARGO_BARS: Dict[str, int] = {
    "4h": compute_embargo_bars("4h"),
}

SIGNAL_CACHE_PARAM_KEYS: frozenset[str] = frozenset([
    "ATR_PERIOD",
    "HMM_TRAIN_WINDOW",
    "HMM_RETRAIN_FREQ",
    "FRAMA_PERIOD",
    "FRAMA_MIN_SLOPE",
    "EVR_WINDOW",
    "EVR_THRESHOLD",
    "GARCH_WINDOW",
    "GARCH_RETRAIN_FREQ",
    "GARCH_NU_FALLBACK",
])

_SIGNAL_CACHE_SCHEMA_VERSION: int = 2

_SignalCacheKey = Tuple[Tuple[Tuple[str, Any], ...], str, str, int, int, int]
_SIGNAL_CACHE_MAXSIZE: int = 512
_ARRAYS_CACHE_MAXSIZE: int = 256
_cache_lock: threading.Lock = threading.Lock()
_signal_cache: OrderedDict[_SignalCacheKey, pd.DataFrame] = OrderedDict()
_arrays_cache: OrderedDict[_SignalCacheKey, Dict[str, np.ndarray]] = OrderedDict()
_numpy_cache_warning_emitted: bool = False

_DISK_CACHE_READ_EXCEPTIONS: tuple[type[Exception], ...] = (
    OSError,
    EOFError,
    ValueError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    pickle.UnpicklingError,
)

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
            tail_ratio,
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
        tail_ratio,
        equity_curve,
        span_days_sym,
    )

def _dataset_fingerprint_from_df(df: pd.DataFrame) -> int:
    """Lightweight cache invalidation when OHLCV rows change but length stays equal."""
    if "close" not in df.columns or df.empty:
        return 0
    c = df["close"].to_numpy(dtype=np.float64)
    n = len(c)
    head = c[: min(5, n)]
    tail = c[max(0, n - 5) :]
    h = hash((tuple(head.tolist()), tuple(tail.tolist()), n))
    return int(h & ((1 << 63) - 1))


def _signal_disk_cache_path(cache_key: _SignalCacheKey, root: Path) -> Path:
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()
    return root / f"spot_sig_{digest}.pkl"


def _build_signal_cache_key(
    params: Dict[str, Any],
    sym: str,
    tf: str,
    data_len: int,
    fingerprint: int,
) -> _SignalCacheKey:
    signal_items: List[Tuple[str, Any]] = sorted(
        (k, params[k]) for k in SIGNAL_CACHE_PARAM_KEYS if k in params
    )
    return (
        tuple(signal_items),
        sym,
        tf,
        data_len,
        int(fingerprint),
        _SIGNAL_CACHE_SCHEMA_VERSION,
    )

def get_or_compute_signals(
    cache_key: _SignalCacheKey,
    target_df: pd.DataFrame,
    strategy: UltimateSpotStrategy,
    *,
    disk_cache_root: Optional[Path] = None,
) -> pd.DataFrame:
    with _cache_lock:
        if cache_key in _signal_cache:
            _signal_cache.move_to_end(cache_key)
            return _signal_cache[cache_key]

    root = disk_cache_root
    if root is None:
        env = os.getenv("OPT_SPOT_SIGNAL_CACHE_DIR")
        if env:
            root = Path(env)
    if root is not None and FileLock is not None:
        root.mkdir(parents=True, exist_ok=True)
        p = _signal_disk_cache_path(cache_key, root)
        lock_path = str(p) + ".lock"
        with FileLock(lock_path):
            if p.is_file():
                try:
                    full_df_disk = pd.read_pickle(p)
                except _DISK_CACHE_READ_EXCEPTIONS as exc:
                    global _numpy_cache_warning_emitted
                    is_numpy_module_miss = isinstance(exc, ModuleNotFoundError) and (
                        "numpy._core.numeric" in str(exc)
                    )
                    if is_numpy_module_miss:
                        if not _numpy_cache_warning_emitted:
                            _numpy_cache_warning_emitted = True
                            _logger.warning(
                                "Legacy numpy signal-cache detected. Auto-rebuilding corrupted cache files."
                            )
                    else:
                        _logger.warning(
                            "Signal cache load failed. Rebuilding cache. path=%s err=%s",
                            p,
                            exc,
                        )
                    try:
                        p.unlink(missing_ok=True)
                    except OSError as unlink_exc:
                        _logger.warning(
                            "Failed to remove corrupted signal cache file. path=%s err=%s",
                            p,
                            unlink_exc,
                        )
                else:
                    with _cache_lock:
                        if cache_key in _signal_cache:
                            _signal_cache.move_to_end(cache_key)
                            return _signal_cache[cache_key]
                        while len(_signal_cache) >= _SIGNAL_CACHE_MAXSIZE:
                            _signal_cache.popitem(last=False)
                        _signal_cache[cache_key] = full_df_disk
                        return full_df_disk

    full_df: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
    with _cache_lock:
        if cache_key in _signal_cache:
            _signal_cache.move_to_end(cache_key)
            return _signal_cache[cache_key]
        while len(_signal_cache) >= _SIGNAL_CACHE_MAXSIZE:
            _signal_cache.popitem(last=False)
        _signal_cache[cache_key] = full_df
        out = full_df

    if root is not None and FileLock is not None:
        p = _signal_disk_cache_path(cache_key, root)
        lock_path = str(p) + ".lock"
        with FileLock(lock_path):
            if not p.is_file():
                out.to_pickle(p)
    return out

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
            ).total_seconds() / 86400.0
        )
        if "datetime" in sig_oos.columns and not sig_oos.empty
        else 1.0
    )
    span_days = max(span_days, 1.0)
    
    true_pnl = trades_df["pnl"]
    win_rate: float = float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100) if len(trades_df) > 0 else 0.0
    pf = calc_profit_factor_from_pnl(true_pnl)
    
    total_ret_ratio = 1.0 + (ret_pct / 100.0)
    cagr = ((total_ret_ratio ** (365.0 / span_days)) - 1.0) * 100.0 if total_ret_ratio > 0 else -100.0

    tail_ratio = calc_tail_ratio_from_equity(equity_curve) if equity_curve.size > 1 else 0.0

    return cagr, ret_pct, mdd_pct, len(trades_df), win_rate, pf, long_count, equity_curve, tail_ratio

def _merge_spot_fixed_signal_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Student-t HMM + FRAMA/EvR are fixed in UltimateSpotStrategy; no Optuna toggles."""
    return dict(params)


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
    if "garch_kelly_f" in sig_df.columns:
        out["garch_kelly_f"] = sig_df["garch_kelly_f"].to_numpy(dtype=np.float64)
    if "kill_signal" in sig_df.columns:
        out["kill_signal"] = sig_df["kill_signal"].to_numpy(dtype=np.float64)
    return out


def _dataframe_to_symbol_arrays_extended(sig_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Full IS arrays for shared-cash + optional slot_rank_score (views used in CPCV slices)."""
    base = _dataframe_to_symbol_arrays(sig_df)
    if "slot_rank_score" in sig_df.columns:
        base["slot_rank_score"] = sig_df["slot_rank_score"].to_numpy(dtype=np.float64)
    return base


def _slice_symbol_arrays_view(
    full: Dict[str, np.ndarray],
    slice_start: int,
    slice_end: int,
) -> Dict[str, np.ndarray]:
    return {k: v[slice_start:slice_end] for k, v in full.items()}


def _segment_span_days(sig_df: pd.DataFrame, execution_start_idx: int) -> float:
    if "datetime" not in sig_df.columns or sig_df.empty:
        return 1.0
    i0 = min(max(0, int(execution_start_idx)), len(sig_df) - 1)
    span_seconds = float(
        (sig_df["datetime"].iloc[-1] - sig_df["datetime"].iloc[i0]).total_seconds()
    )
    return max(span_seconds / 86400.0, 1.0)


def _span_days_ref_slice(ref_df: pd.DataFrame, abs_start: int, abs_end: int) -> float:
    """Calendar span (days) for CPCV segment [abs_start, abs_end) on aligned reference OHLCV."""
    if ref_df.empty:
        return 1.0
    if "datetime" not in ref_df.columns:
        return max(float(abs_end - abs_start), 1.0) * (4.0 / 24.0)
    i0 = int(np.clip(abs_start, 0, len(ref_df) - 1))
    i1 = int(np.clip(abs_end - 1, 0, len(ref_df) - 1))
    if i1 < i0:
        return 1.0 / 24.0
    dt0 = ref_df["datetime"].iloc[i0]
    dt1 = ref_df["datetime"].iloc[i1]
    sec = float((dt1 - dt0).total_seconds())
    return max(sec / 86400.0, 1.0 / 24.0)


def objective_spot(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf_target: str,
    *,
    space: Dict[str, Dict[str, Any]],
    mode: str = "single",
    project_root: Optional[str] = None,
    prebuilt_cpcv_bundle: Optional[Tuple[List[CPCVPath], int, int]] = None,
    signal_disk_cache_root: Optional[Path] = None,
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
    if prebuilt_cpcv_bundle is not None:
        cpcv_paths, nb_cpcv, k_cpcv = prebuilt_cpcv_bundle
    else:
        cpcv_paths, nb_cpcv, k_cpcv = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)
    if not cpcv_paths:
        raise optuna.TrialPruned()
    n_independent_paths = max(2, nb_cpcv // k_cpcv)

    cache_root: Optional[Path] = signal_disk_cache_root
    if cache_root is None and project_root is not None:
        cache_root = Path(project_root) / ".spot_signal_cache"

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_full: Optional[pd.DataFrame] = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            continue
        fp = _dataset_fingerprint_from_df(target_df_full)
        cache_key: _SignalCacheKey = _build_signal_cache_key(
            params, sym, tf, len(target_df_full), fp
        )
        full_signal_dfs[sym] = get_or_compute_signals(
            cache_key, target_df_full, strategy, disk_cache_root=cache_root
        )

    if len(full_signal_dfs) != len(symbols):
        raise optuna.TrialPruned()

    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in symbols:
        target_df_full = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            raise optuna.TrialPruned()
        fp = _dataset_fingerprint_from_df(target_df_full)
        sig_key = _build_signal_cache_key(params, sym, tf, len(target_df_full), fp)
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
    min_seg_trades = int(OPT_SPOT_CONFIG.get("SPOT_MIN_TRADES_PER_CPCV_SEGMENT", 4))
    # Signals are pre-computed on full IS; per-segment re-warmup would skip most of each test block.
    warmup_bars = 0

    cfg = OPT_SPOT_CONFIG
    seg_fail_pen = float(cfg.get("SPOT_SEGMENT_TRADE_FAIL_PENALTY", 2.0))
    sortino_eps = float(cfg.get("SPOT_OBJECTIVE_SORTINO_EPS", 1e-6))
    sortino_ratio_cap = float(cfg.get("SPOT_OBJECTIVE_SORTINO_RATIO_CAP", 1.0e6))
    path_sortino_clip = float(cfg.get("SPOT_OBJECTIVE_PATH_SORTINO_CLIP", 500.0))

    path_compound_log_tw: List[float] = []
    path_compound_raw_log_tw: List[float] = []
    path_compound_tw_ratio: List[float] = []
    path_sortino_vals: List[float] = []
    path_worst_mdd: List[float] = []
    path_max_cvar: List[float] = []
    path_trades: List[int] = []
    path_tail_ratios: List[float] = []

    for path_idx, path in enumerate(cpcv_paths):
        seg_log_tw: List[float] = []
        seg_raw_log_tw: List[float] = []
        seg_tw_ratio: List[float] = []
        seg_mdds: List[float] = []
        seg_cvars: List[float] = []
        seg_tails: List[float] = []
        path_total_trades = 0
        running_balance = float(SPOT_INITIAL_BALANCE)
        path_eq_chunks: List[np.ndarray] = []
        span_path_days = 0.0
        for test_start, test_end in path:
            abs_start = is_off + int(test_start)
            abs_end = is_off + int(test_end)
            slice_start = max(0, abs_start - 1)
            slice_end = min(len(ref_df), abs_end)
            if slice_end - slice_start < 5:
                continue
            span_path_days += _span_days_ref_slice(ref_df, abs_start, abs_end)
            symbol_arrays: Dict[str, Dict[str, np.ndarray]] = {}
            rank_scores: Dict[str, np.ndarray] = {}
            for sym in symbols:
                symbol_arrays[sym] = _slice_symbol_arrays_view(
                    prebuilt_full_arrays[sym], slice_start, slice_end
                )
                rs = symbol_arrays[sym].get("slot_rank_score")
                if rs is not None:
                    rank_scores[sym] = rs

            execution_start_idx = max(1, abs_start - slice_start)
            segment_initial = max(running_balance, 1e-9)
            try:
                result = run_shared_cash_multi_symbol(
                    symbol_arrays,
                    symbols,
                    params,
                    initial_balance=segment_initial,
                    max_concurrent_positions=max_slots,
                    rank_scores=rank_scores if rank_scores else None,
                    warmup_bars=warmup_bars,
                    execution_start_idx=execution_start_idx,
                    allow_python_fallback=False,
                )
            except Exception as exc:
                _logger.warning(
                    "Shared-cash Numba path failed (allow_python_fallback=False): %s",
                    exc,
                    exc_info=True,
                )
                raise
            eq = result.equity_curve
            path_total_trades += int(result.total_trades)
            if eq.size == 0:
                twr = 1.0
            else:
                twr = max(float(result.final_balance / segment_initial), 1e-9)
            raw_log_tw = float(np.log(twr))
            log_tw = raw_log_tw
            if eq.size > 0:
                eq0 = float(eq[0]) if float(eq[0]) > 1e-12 else segment_initial
                scale = float(segment_initial) / eq0
                scaled_eq = eq.astype(np.float64, copy=False) * scale
                if not path_eq_chunks:
                    path_eq_chunks.append(scaled_eq)
                else:
                    path_eq_chunks.append(scaled_eq[1:])
            if eq.size > 1:
                mdd_seg = float(calc_mdd_from_equity(eq))
                seg_mdds.append(mdd_seg)
                seg_cvars.append(float(cvar_loss_pct_from_simple_returns(eq)))
                seg_tails.append(float(calc_tail_ratio_from_equity(eq)))
            else:
                seg_mdds.append(0.0)
                seg_cvars.append(0.0)
                seg_tails.append(1.0)

            if int(result.total_trades) < min_seg_trades:
                log_tw -= seg_fail_pen

            seg_raw_log_tw.append(raw_log_tw)
            seg_log_tw.append(log_tw)
            seg_tw_ratio.append(twr)
            running_balance = max(float(result.final_balance), 1e-9)

        if not seg_log_tw:
            raise optuna.TrialPruned()
        path_compound_log_tw.append(float(np.sum(seg_log_tw)))
        path_compound_raw_log_tw.append(float(np.sum(seg_raw_log_tw)))
        path_compound_tw_ratio.append(float(np.prod(seg_tw_ratio)) if seg_tw_ratio else 1.0)
        path_worst_mdd.append(float(np.max(seg_mdds)) if seg_mdds else 0.0)
        path_max_cvar.append(float(np.max(seg_cvars)) if seg_cvars else 0.0)
        path_trades.append(path_total_trades)
        path_tail_ratios.append(float(np.mean(seg_tails)) if seg_tails else 1.0)

        path_eq = np.concatenate(path_eq_chunks) if path_eq_chunks else np.array([], dtype=np.float64)
        span_for_sortino = max(span_path_days, 1.0)
        raw_ps = float(calc_sortino_from_equity(path_eq, span_for_sortino)) if path_eq.size >= 2 else 0.0
        if not np.isfinite(raw_ps):
            raw_ps = 0.0
        path_sortino = float(np.clip(raw_ps, -path_sortino_clip, path_sortino_clip))
        path_sortino_vals.append(path_sortino)

        partial_worst_tw = (
            float(np.percentile(np.asarray(path_compound_tw_ratio, dtype=np.float64), 10.0))
            if path_compound_tw_ratio
            else 1.0
        )
        partial_mean_s = float(np.mean(path_sortino_vals)) if path_sortino_vals else 0.0
        partial_std_s = (
            float(np.std(path_sortino_vals, ddof=1)) if len(path_sortino_vals) > 1 else 0.0
        )
        partial_ratio = partial_mean_s / (partial_std_s + sortino_eps)
        partial_ratio = float(np.clip(partial_ratio, -sortino_ratio_cap, sortino_ratio_cap))
        interm = partial_worst_tw * partial_ratio
        trial.report(interm, step=path_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_log_tw = float(np.mean(path_compound_raw_log_tw))
    mean_penalized_log_tw = float(np.mean(path_compound_log_tw))
    cvar25_log = float(mean_of_worst_quartile(path_compound_raw_log_tw))

    worst_path_tw = (
        float(np.percentile(np.asarray(path_compound_tw_ratio, dtype=np.float64), 10.0))
        if path_compound_tw_ratio
        else 1.0
    )
    mean_path_sortino = float(np.mean(path_sortino_vals)) if path_sortino_vals else 0.0
    std_path_sortino = (
        float(np.std(path_sortino_vals, ddof=1)) if len(path_sortino_vals) > 1 else 0.0
    )
    sortino_ratio = mean_path_sortino / (std_path_sortino + sortino_eps)
    sortino_ratio = float(np.clip(sortino_ratio, -sortino_ratio_cap, sortino_ratio_cap))
    growth_score = float(worst_path_tw * sortino_ratio)

    mini_oos_ratio = 1.0
    mini_window = max(24, int(ref_len * 0.20))
    mini_start = max(is_off + 1, is_off + ref_len - mini_window)
    mini_end = is_off + ref_len
    if mini_end - mini_start >= 5:
        mini_slice_start = max(0, mini_start - 1)
        mini_slice_end = min(len(ref_df), mini_end)
        mini_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        mini_rank_scores: Dict[str, np.ndarray] = {}
        for sym in symbols:
            mini_arr = _slice_symbol_arrays_view(prebuilt_full_arrays[sym], mini_slice_start, mini_slice_end)
            mini_arrays[sym] = mini_arr
            rs = mini_arr.get("slot_rank_score")
            if rs is not None:
                mini_rank_scores[sym] = rs
        mini_execution_start_idx = max(1, mini_start - mini_slice_start)
        try:
            mini_res = run_shared_cash_multi_symbol(
                mini_arrays,
                symbols,
                params,
                initial_balance=float(SPOT_INITIAL_BALANCE),
                max_concurrent_positions=max_slots,
                rank_scores=mini_rank_scores if mini_rank_scores else None,
                warmup_bars=warmup_bars,
                execution_start_idx=mini_execution_start_idx,
                allow_python_fallback=False,
            )
            mini_oos_ratio = max(float(mini_res.final_balance / float(SPOT_INITIAL_BALANCE)), 1e-9)
        except Exception as exc:
            _logger.warning("Mini-holdout evaluation failed: %s", exc, exc_info=True)
            mini_oos_ratio = 1.0
    mean_path_tw = float(np.mean(path_compound_tw_ratio)) if path_compound_tw_ratio else 1.0
    alpha_ratio = mini_oos_ratio / max(abs(mean_path_tw), 1e-6)
    decay_penalty = max(0.0, 1.0 - alpha_ratio)
    objective_final = float(growth_score * (1.0 - 0.3 * decay_penalty))

    trial.set_user_attr("mean_path_terminal_wealth_ratio", float(np.mean(path_compound_tw_ratio)))
    trial.set_user_attr("min_path_terminal_wealth_ratio", worst_path_tw)
    trial.set_user_attr("mean_log_terminal_wealth", mean_log_tw)
    trial.set_user_attr("mean_penalized_log_terminal_wealth", mean_penalized_log_tw)
    trial.set_user_attr("cvar25_log_tw", cvar25_log)
    trial.set_user_attr(
        "path_mean_log_tw_std",
        float(np.std(path_compound_raw_log_tw, ddof=1)) if len(path_compound_raw_log_tw) > 1 else 0.0,
    )
    trial.set_user_attr("growth_score", growth_score)
    trial.set_user_attr("mini_oos_terminal_wealth_ratio", mini_oos_ratio)
    trial.set_user_attr("alpha_ratio_proxy", float(alpha_ratio))
    trial.set_user_attr("alpha_decay_penalty", float(decay_penalty))
    trial.set_user_attr("objective_final", objective_final)
    trial.set_user_attr("cpcv_mean_path_sortino", mean_path_sortino)
    trial.set_user_attr("cpcv_std_path_sortino", std_path_sortino)
    trial.set_user_attr("composite_sortino_ratio", float(sortino_ratio))
    trial.set_user_attr("cpcv_embargo_bars", embargo)
    trial.set_user_attr("cpcv_n_independent_paths", int(n_independent_paths))

    n_done = trial.number + 1
    n_startup = int(OPT_SPOT_CONFIG.get("tpe_n_startup_trials", 96))
    n_eff = get_spot_effective_independent_trials(n_done, n_startup)
    psr_val = probabilistic_sharpe_ratio(
        float(
            np.mean(path_compound_raw_log_tw)
            / (np.std(path_compound_raw_log_tw, ddof=1) + 1e-12)
        ),
        n_independent_paths,
    )
    trial.set_user_attr("psr_paths", psr_val)
    trial.set_user_attr("n_effective_independent_trials", n_eff)

    path_vals = [float(x) for x in path_compound_raw_log_tw]
    trial.set_user_attr(
        "dsr_paths",
        float(compute_dsr_from_path_values(path_vals, n_eff)),
    )

    path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
    n_paths_ct = int(path_arr.size)
    if n_paths_ct >= 2:
        mu_paths = float(np.mean(path_arr))
        sd_paths = float(np.std(path_arr, ddof=1))
        MAX_RATIO_FALLBACK: float = 99.0
        gate1_sqn = math.sqrt(float(n_paths_ct)) * mu_paths / sd_paths if sd_paths > 1e-12 else 0.0
        pr95 = float(np.percentile(path_arr, 95))
        pr5 = float(np.percentile(path_arr, 5))
        cpcv_path_tail_ratio = pr95 / abs(pr5) if abs(pr5) > 1e-12 else (MAX_RATIO_FALLBACK if pr95 > 0.0 else 0.0)
    else:
        gate1_sqn = 0.0
        cpcv_path_tail_ratio = 1.0
    gate1_path_sortino = float(sortino_ratio)

    worst_seg_mdd_pct = float(np.max(path_worst_mdd)) if path_worst_mdd else 0.0
    cpcv_mean_path_return_pct = (
        float(np.mean(path_compound_tw_ratio) - 1.0) * 100.0 if path_compound_tw_ratio else 0.0
    )

    trial.set_user_attr("gate1_sqn", float(gate1_sqn))
    trial.set_user_attr("gate1_path_sortino", float(gate1_path_sortino))
    trial.set_user_attr("cpcv_path_tail_ratio", float(cpcv_path_tail_ratio))
    trial.set_user_attr("cpcv_worst_segment_mdd_pct", worst_seg_mdd_pct)
    trial.set_user_attr("cpcv_mean_path_return_pct", cpcv_mean_path_return_pct)

    return float(objective_final)


def run_holdout_shared_cash_portfolio(
    params: Dict[str, Any],
    symbols: List[str],
    tf: str,
    oos_data_maps: Dict[str, Dict[str, Any]],
    *,
    signal_disk_cache_root: Optional[Path] = None,
    return_signal_dfs: bool = False,
) -> Dict[str, Any]:
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
        fp = _dataset_fingerprint_from_df(df_full)
        cache_key: _SignalCacheKey = _build_signal_cache_key(p, sym, tf, len(df_full), fp)
        full_signal_dfs[sym] = get_or_compute_signals(
            cache_key,
            df_full,
            strategy,
            disk_cache_root=signal_disk_cache_root,
        )
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
        }
        if return_signal_dfs:
            failed["full_signal_dfs"] = {}
        return failed

    ref_sym = symbols[0]
    oos_start = int(oos_data_maps[ref_sym].get(f"oos_start_idx_{tf}", 0))
    ref_df = full_signal_dfs[ref_sym]
    slice_start = max(0, oos_start - 1)
    slice_end = len(ref_df)
    if slice_end - slice_start < 5:
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
    tail_r = float(calc_tail_ratio_from_equity(eq)) if eq.size > 1 else 0.0
    dd_bars = float(max_underwater_bars_from_equity(eq)) if eq.size > 1 else 0.0
    final_bal = float(res.final_balance)
    moic = final_bal / initial_balance if initial_balance > 0 else 0.0
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
    }
    if return_signal_dfs:
        out["full_signal_dfs"] = full_signal_dfs
    return out
