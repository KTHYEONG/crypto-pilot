"""
Futures Optuna objective: CPCV paths, Kelly-CVaR scalar, disk+memory signal cache.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_CACHE_DIR, FUTURES_INITIAL_BALANCE, SLIPPAGE_RATE, TRADING_FEE_RATE
from src.futures_strategy.engine_futures import BacktestEngineFast
from src.futures_strategy.engine_portfolio_futures import PortfolioBacktestEngineFast
from src.futures_strategy.opt_futures_utils.cv_utils import (
    CPCVPath,
    build_cpcv_test_paths_with_fallback,
    cpcv_complement_segments,
    list_cpcv_block_ranges,
)
from src.futures_strategy.opt_futures_utils.metrics import (
    calc_cvar5_loss_pct_from_equity,
    calc_max_underwater_days_from_equity,
    calc_mdd_from_equity,
    calc_profit_factor_from_pnl,
    calc_sortino_from_equity,
    compute_pbo_from_cpcv_paths,
)
from src.futures_strategy.opt_futures_utils.opt_params import suggest_params_futures
from src.futures_strategy.strategies_futures import UltimateStrategy

_logger: logging.Logger = logging.getLogger("opt_futures")

_FUTURES_SIGNAL_CACHE_KEYS: frozenset[str] | None = None


def _signal_cache_param_keys_futures() -> frozenset[str]:
    global _FUTURES_SIGNAL_CACHE_KEYS
    if _FUTURES_SIGNAL_CACHE_KEYS is None:
        from src.futures_strategy.opt_futures_utils.opt_params import build_full_discovery_space_futures

        _FUTURES_SIGNAL_CACHE_KEYS = frozenset(build_full_discovery_space_futures().keys())
    return _FUTURES_SIGNAL_CACHE_KEYS

_SIGNAL_CACHE_SCHEMA_VERSION: int = 1
_cache_lock: threading.Lock = threading.Lock()
_signal_cache: OrderedDict["_SignalCacheKey", pd.DataFrame] = OrderedDict()
_SIGNAL_CACHE_MAXSIZE: int = max(16, int(os.getenv("OPT_FUTURES_SIGNAL_MEM_CACHE_MAX", "48")))
_arrays_cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
_ARRAYS_CACHE_MAXSIZE: int = max(8, int(os.getenv("OPT_FUTURES_ARRAYS_MEM_CACHE_MAX", "32")))
_CACHE_CLEANUP_DONE: bool = False
_CACHE_LAST_CLEANUP_TS: float = 0.0
_STRATEGY_LOGIC_HASH: Optional[str] = None


def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: Dict[str, int] = {"1h": 24, "4h": 6}
    ratio_map: Dict[str, float] = {"1h": 0.08, "4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))


EMBARGO_BARS: Dict[str, int] = {
    "1h": compute_embargo_bars("1h"),
    "4h": compute_embargo_bars("4h"),
}

_SignalCacheKey = Tuple[Tuple[Tuple[str, Any], ...], str, str, int, int, int, str]


def _get_logic_hash() -> str:
    global _STRATEGY_LOGIC_HASH
    if _STRATEGY_LOGIC_HASH is None:
        try:
            src = inspect.getsource(UltimateStrategy.generate_signals)
            _STRATEGY_LOGIC_HASH = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
        except Exception:
            _STRATEGY_LOGIC_HASH = "legacy"
    return _STRATEGY_LOGIC_HASH


def _dataset_fingerprint_from_df(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    n = len(df)
    d0 = str(df["datetime"].iloc[0]) if "datetime" in df.columns else ""
    d1 = str(df["datetime"].iloc[-1]) if "datetime" in df.columns else ""
    if "close" in df.columns:
        c = df["close"].to_numpy(dtype=np.float64)
        head = c[: min(5, n)]
        tail = c[max(0, n - 5) :]
        fp = hash((d0, d1, n, tuple(head.tolist()), tuple(tail.tolist())))
    else:
        fp = hash((d0, d1, n))
    return int(fp & ((1 << 63) - 1))


def _build_signal_cache_key(
    params: Dict[str, Any], sym: str, tf: str, data_len: int, fingerprint: int
) -> _SignalCacheKey:
    signal_items: Tuple[Tuple[str, Any], ...] = tuple(
        sorted((k, params[k]) for k in _signal_cache_param_keys_futures() if k in params)
    )
    return (
        signal_items,
        sym,
        tf,
        data_len,
        int(fingerprint),
        _SIGNAL_CACHE_SCHEMA_VERSION,
        _get_logic_hash(),
    )


def _disk_cache_path(cache_key: _SignalCacheKey, root: Path) -> Path:
    payload = repr(cache_key).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    sym, tf = cache_key[1], cache_key[2]
    folder = root / sym / tf
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.joblib"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _signal_cache_max_bytes() -> int:
    max_gb = max(0, _env_int("OPT_FUTURES_SIGNAL_CACHE_MAX_GB", 4))
    return max_gb * 1024 * 1024 * 1024


def _signal_cache_cleanup_interval_sec() -> int:
    return max(0, _env_int("OPT_FUTURES_SIGNAL_CACHE_CLEANUP_INTERVAL_SEC", 300))


def _cleanup_disk_cache_lru(root: Path) -> None:
    max_bytes = _signal_cache_max_bytes()
    if max_bytes <= 0 or not root.exists():
        return
    
    # Gather file paths and their stats safely to handle race conditions
    file_info: List[Tuple[Path, float, int]] = []
    for p in root.rglob("*.joblib"):
        if not p.is_file():
            continue
        try:
            stat = p.stat()
            file_info.append((p, stat.st_mtime, stat.st_size))
        except OSError:
            # File might have been deleted by another process/thread
            continue

    if not file_info:
        return

    # Sort by mtime descending (newest first)
    file_info.sort(key=lambda x: x[1], reverse=True)
    
    total = sum(info[2] for info in file_info)
    target = int(max_bytes * 0.8)
    if total <= target:
        return

    # Remove oldest files until total size is below target
    for p, _, sz in reversed(file_info):
        if total <= target:
            break
        try:
            if p.exists():
                p.unlink(missing_ok=True)
                total -= sz
        except OSError:
            continue


def _maybe_cleanup_disk_cache(root: Path, *, force: bool = False) -> None:
    global _CACHE_LAST_CLEANUP_TS
    now = time.time()
    interval = _signal_cache_cleanup_interval_sec()
    if not force and interval > 0 and (now - _CACHE_LAST_CLEANUP_TS) < interval:
        return
    _cleanup_disk_cache_lru(root)
    _CACHE_LAST_CLEANUP_TS = now


def get_or_compute_signals(
    cache_key: _SignalCacheKey,
    target_df: pd.DataFrame,
    strategy: UltimateStrategy,
    *,
    disk_cache_root: Optional[Path] = None,
) -> pd.DataFrame:
    global _CACHE_CLEANUP_DONE
    with _cache_lock:
        if cache_key in _signal_cache:
            _signal_cache.move_to_end(cache_key)
            return _signal_cache[cache_key]

    if disk_cache_root is None:
        disk_cache_root = FUTURES_CACHE_DIR

    if not _CACHE_CLEANUP_DONE:
        _maybe_cleanup_disk_cache(disk_cache_root, force=True)
        _CACHE_CLEANUP_DONE = True

    path = _disk_cache_path(cache_key, disk_cache_root)
    if path.exists():
        try:
            full_df: pd.DataFrame = joblib.load(path)
            with _cache_lock:
                while len(_signal_cache) >= _SIGNAL_CACHE_MAXSIZE:
                    _signal_cache.popitem(last=False)
                _signal_cache[cache_key] = full_df
            return full_df
        except Exception:
            pass

    full_df = strategy.generate_signals(target_df.copy(deep=True))
    try:
        tmp_p = path.with_suffix(f".tmp.{os.getpid()}")
        joblib.dump(full_df, tmp_p, compress=3)
        tmp_p.replace(path)
    except Exception as exc:
        _logger.warning("Futures signal disk cache write failed: %s", exc)
    _maybe_cleanup_disk_cache(disk_cache_root, force=False)

    with _cache_lock:
        while len(_signal_cache) >= _SIGNAL_CACHE_MAXSIZE:
            _signal_cache.popitem(last=False)
        _signal_cache[cache_key] = full_df
        return full_df


_FUTURES_2D_REQUIRED_COLS: Tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "entry_upper",
    "entry_lower",
    "trend_direction",
    "strength_filter",
    "atr",
    "garch_kelly_f",
    "funding_rate_sum",
    "slot_rank_score",
)


def _dataframe_to_symbol_arrays(sig_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    for c in _FUTURES_2D_REQUIRED_COLS:
        if c not in sig_df.columns:
            if c in ("garch_kelly_f", "funding_rate_sum", "slot_rank_score"):
                sig_df[c] = 0.0
            else:
                raise ValueError(f"Missing required column {c} for futures 2D engine.")
    out: Dict[str, np.ndarray] = {}
    out["open"] = sig_df["open"].to_numpy(dtype=np.float64, copy=False)
    out["high"] = sig_df["high"].to_numpy(dtype=np.float64, copy=False)
    out["low"] = sig_df["low"].to_numpy(dtype=np.float64, copy=False)
    out["close"] = sig_df["close"].to_numpy(dtype=np.float64, copy=False)
    out["atr"] = sig_df["atr"].ffill().to_numpy(dtype=np.float64, copy=False)
    out["strength_filter"] = sig_df["strength_filter"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    out["trend_direction"] = sig_df["trend_direction"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    out["entry_upper"] = sig_df["entry_upper"].fillna(999999.0).to_numpy(dtype=np.float64, copy=False)
    out["entry_lower"] = sig_df["entry_lower"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    out["garch_kelly_f"] = sig_df["garch_kelly_f"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    out["funding_rate_sum"] = sig_df["funding_rate_sum"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    out["slot_rank_score"] = sig_df["slot_rank_score"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    return out


def _build_aligned_2d_from_prebuilt(
    prebuilt_arrays: Dict[str, Dict[str, np.ndarray]],
    symbols: List[str],
    slice_start: int,
    slice_end: int,
) -> Optional[Dict[str, np.ndarray]]:
    if slice_end - slice_start < 2:
        return None
    aligned_data: Dict[str, np.ndarray] = {}
    for col in _FUTURES_2D_REQUIRED_COLS:
        col_views: List[np.ndarray] = []
        for sym in symbols:
            sym_arrs = prebuilt_arrays.get(sym)
            if sym_arrs is None:
                return None
            arr = sym_arrs.get(col)
            if arr is None or slice_end > int(arr.shape[0]):
                return None
            col_views.append(arr[slice_start:slice_end])
        try:
            merged = np.column_stack(col_views).astype(np.float64, copy=False)
        except ValueError:
            return None
        aligned_data[col] = np.ascontiguousarray(merged)
    return aligned_data


def align_data_for_2d_engine(
    signal_dfs: Dict[str, pd.DataFrame],
    symbols: List[str],
) -> Tuple[Dict[str, np.ndarray], pd.Series]:
    all_dates: List[pd.Series] = []
    for sym in symbols:
        df = signal_dfs.get(sym)
        if df is not None and "datetime" in df.columns:
            all_dates.append(df["datetime"])
    if not all_dates:
        empty: Dict[str, np.ndarray] = {}
        return empty, pd.Series(dtype="datetime64[ns]")

    master_index = pd.concat(all_dates, ignore_index=True).drop_duplicates().sort_values().reset_index(drop=True)
    master_df = pd.DataFrame({"datetime": master_index})
    n_bars = len(master_index)
    n_syms = len(symbols)

    target_cols = [
        "open",
        "high",
        "low",
        "close",
        "entry_upper",
        "entry_lower",
        "trend_direction",
        "strength_filter",
        "atr",
        "garch_kelly_f",
        "funding_rate_sum",
        "slot_rank_score",
    ]
    aligned_data: Dict[str, np.ndarray] = {
        col: np.full((n_bars, n_syms), np.nan, dtype=np.float64) for col in target_cols
    }

    for s_idx, sym in enumerate(symbols):
        df = signal_dfs.get(sym)
        if df is None:
            continue
        merged = pd.merge(master_df, df, on="datetime", how="left")
        for col in ["open", "high", "low", "close", "atr"]:
            if col in merged.columns:
                aligned_data[col][:, s_idx] = merged[col].ffill().values
        for col in ["strength_filter", "trend_direction", "garch_kelly_f", "funding_rate_sum", "slot_rank_score"]:
            if col in merged.columns:
                aligned_data[col][:, s_idx] = merged[col].fillna(0).values
        for col in ["entry_upper", "entry_lower"]:
            if col in merged.columns:
                default_val = 999999.0 if col == "entry_upper" else 0.0
                aligned_data[col][:, s_idx] = merged[col].fillna(default_val).values

    return aligned_data, master_index


def _segment_with_context(
    full_signal_df: pd.DataFrame,
    exec_start_idx: int,
    exec_end_idx: int,
) -> Tuple[pd.DataFrame, int]:
    slice_start = max(0, int(exec_start_idx) - 1)
    slice_end = max(slice_start, int(exec_end_idx))
    segment = full_signal_df.iloc[slice_start:slice_end].copy(deep=False)
    execution_start_idx = int(exec_start_idx) - slice_start
    if execution_start_idx == 0 and len(segment) > 1:
        execution_start_idx = 1
    return segment, execution_start_idx


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


def objective_futures(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf_target: str,
    *,
    space: Dict[str, Dict[str, Any]],
    mode: str = "single",
    project_root: Optional[str] = None,
    prebuilt_cpcv_bundle: Optional[Tuple[List[List[Tuple[int, int]]], int, int]] = None,
    signal_disk_cache_root: Optional[Path] = None,
) -> float:
    params: Dict[str, Any] = suggest_params_futures(trial, space, tf_target)
    tf: str = tf_target
    strategy: UltimateStrategy = UltimateStrategy(name="OptFutures", params=params)

    ref_sym = symbols[0]
    ref_df: Optional[pd.DataFrame] = data_maps.get(ref_sym, {}).get(tf)
    if ref_df is None or ref_df.empty:
        raise optuna.TrialPruned()

    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_len = len(ref_df) - is_off
    if ref_len < 200:
        raise optuna.TrialPruned()

    embargo = int(EMBARGO_BARS.get(tf, 0))
    if prebuilt_cpcv_bundle is not None:
        cpcv_paths, _, _ = prebuilt_cpcv_bundle
    else:
        cpcv_paths, _, _ = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)
    if not cpcv_paths:
        raise optuna.TrialPruned()

    cfg: Dict[str, Any] = OPT_FUTURES_CONFIG
    min_seg_trades = int(cfg.get("FUTURES_MIN_TRADES_PER_CPCV_SEGMENT", 5))
    min_pf_trades_dynamic = max(40, len(symbols) * 8)
    
    w_mean = float(np.clip(float(cfg.get("FUTURES_OBJECTIVE_W_MEAN_LOG_TW", 0.7)), 0.0, 1.0))
    w_p10 = 1.0 - w_mean
    cvar_alpha = float(cfg.get("FUTURES_CPCV_CVAR_ALPHA", 0.10))
    cvar_thr = float(cfg.get("FUTURES_CPCV_CVAR_THRESHOLD", 0.05))
    cvar_weight = float(cfg.get("FUTURES_CPCV_CVAR_WEIGHT", 0.80))
    temporal_lambda = float(cfg.get("FUTURES_CPCV_TEMPORAL_LAMBDA", 1.5))

    cache_root = signal_disk_cache_root
    if cache_root is None and project_root is not None:
        cache_root = Path(project_root) / "cache_futures"

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_full = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            continue
        fp = _dataset_fingerprint_from_df(target_df_full)
        cache_key = _build_signal_cache_key(params, sym, tf, len(target_df_full), fp)
        full_signal_dfs[sym] = get_or_compute_signals(
            cache_key, target_df_full, strategy, disk_cache_root=cache_root
        )

    if len(full_signal_dfs) != len(symbols):
        raise optuna.TrialPruned()

    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    if mode == "multi":
        for sym in symbols:
            prebuilt_full_arrays[sym] = _dataframe_to_symbol_arrays(full_signal_dfs[sym])

    leverage = float(params.get("LEVERAGE", 20.0))
    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))

    path_compound_raw_log_tw: List[float] = []
    path_pfs: List[float] = []
    path_wins: List[float] = []
    path_trades: List[float] = []
    path_mdds: List[float] = []
    all_seg_long: int = 0
    all_seg_short: int = 0
    funding_ratios: List[float] = []
    sym_pf_accum: Dict[str, List[float]] = {s: [] for s in symbols}
    path_terminal_wealth_ratios: List[float] = []

    for path_idx, path in enumerate(cpcv_paths):
        seg_raw_logs: List[float] = []
        seg_mdds: List[float] = []
        seg_pfs: List[float] = []
        seg_wins: List[float] = []
        seg_trades: List[float] = []
        seg_fund_ratio: List[float] = []
        
        running_balance = float(FUTURES_INITIAL_BALANCE)

        for test_start, test_end in path:
            if mode == "multi":
                abs_start = is_off + int(test_start)
                abs_end = is_off + int(test_end)
                slice_start = max(0, abs_start - 1)
                slice_end = min(len(ref_df), abs_end)
                if slice_end - slice_start < 2:
                    continue
                aligned_data = _build_aligned_2d_from_prebuilt(
                    prebuilt_full_arrays,
                    symbols,
                    slice_start,
                    slice_end,
                )
                if not aligned_data:
                    raise optuna.TrialPruned()
                    
                segment_initial = max(running_balance, 1e-9)
                try:
                    engine = PortfolioBacktestEngineFast(
                        aligned_data=aligned_data,
                        symbol_names=symbols,
                        strategy_params=params,
                        initial_balance=segment_initial,
                        fee_rate=TRADING_FEE_RATE,
                        slippage_rate=SLIPPAGE_RATE,
                    )
                    trades_df, equity_curve, final_balance = engine.run()
                except Exception as exc:
                    _logger.warning("Portfolio engine CPCV error: %s", exc, exc_info=True)
                    raise optuna.TrialPruned()

                if trades_df is None or trades_df.empty:
                    raise optuna.TrialPruned()

                ret_pct = float((final_balance / segment_initial - 1.0) * 100.0)
                raw_log = _log_tw_from_ret_pct(ret_pct)
                running_balance = max(float(final_balance), 1e-9)
                mdd_seg = float(calc_mdd_from_equity(equity_curve)) if equity_curve.size > 0 else 0.0
                true_pnl = trades_df["pnl"] - trades_df["entry_fee"]
                pf_s = calc_profit_factor_from_pnl(true_pnl)
                win_rate_s = (
                    float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100)
                    if len(trades_df) > 0
                    else 0.0
                )
                n_tr = int(len(trades_df))
                lc = int(len(trades_df[trades_df["side"] == "LONG"]))
                sc = int(len(trades_df[trades_df["side"] == "SHORT"]))
                all_seg_long += lc
                all_seg_short += sc
                seg_raw_logs.append(raw_log)
                seg_mdds.append(mdd_seg)
                seg_pfs.append(pf_s)
                seg_wins.append(win_rate_s)
                seg_trades.append(float(n_tr))
                _fp = float(trades_df["funding_fee"].sum()) if "funding_fee" in trades_df.columns else 0.0
                _gr = float(trades_df["pnl"].abs().sum())
                funding_ratios.append(_fp / max(_gr, 1e-9))

                if mdd_seg >= liq_mdd_thr:
                    raise optuna.TrialPruned()

                if n_tr < min_seg_trades:
                    raise optuna.TrialPruned()

                continue

            sym_logs: List[float] = []
            sym_mdds: List[float] = []
            sym_pfs: List[float] = []
            sym_wins: List[float] = []
            sym_trades: List[float] = []
            sym_fund_r: List[float] = []

            for sym in symbols:
                target_df = data_maps[sym][tf]
                daily_df = data_maps[sym]["1d"]
                merge_idx = data_maps[sym][f"merge_idx_{tf}"]
                is_start_idx = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
                adj_s = test_start + is_start_idx
                adj_e = test_end + is_start_idx
                seg, ex_idx = _segment_with_context(full_signal_dfs[sym], adj_s, adj_e)
                _cagr, ret_pct, mdd_pct, ntr, wr, pf, lc, sc, _eq, fpaid, gross = evaluate_symbol_fold(
                    strategy,
                    params,
                    sym,
                    tf,
                    target_df,
                    daily_df,
                    merge_idx,
                    None,
                    adj_s,
                    adj_e,
                    precomputed_signal_df=seg,
                    execution_start_idx=ex_idx,
                )
                lg = _log_tw_from_ret_pct(ret_pct)
                if mdd_pct >= liq_mdd_thr:
                    lg -= 1e9
                denom = max(abs(gross), 1e-9)
                sym_fund_r.append(float(fpaid) / denom)
                sym_logs.append(lg)
                sym_mdds.append(mdd_pct)
                sym_pfs.append(pf)
                sym_wins.append(wr)
                sym_trades.append(float(ntr))
                all_seg_long += lc
                all_seg_short += sc
                sym_pf_accum[sym].append(pf)

            n_sym = max(len(sym_logs), 1)
            total_trades_seg = float(np.sum(sym_trades))
            if total_trades_seg < float(min_seg_trades):
                raise optuna.TrialPruned()

            seg_raw_logs.append(float(np.mean(sym_logs)) if sym_logs else -10.0)
            seg_mdds.append(float(np.mean(sym_mdds)) if sym_mdds else 0.0)
            seg_pfs.append(float(np.mean(sym_pfs)) if sym_pfs else 1.0)
            seg_wins.append(float(np.mean(sym_wins)) if sym_wins else 0.0)
            seg_trades.append(total_trades_seg / n_sym)
            seg_fund_ratio.append(float(np.mean(sym_fund_r)) if sym_fund_r else 0.0)

        if mode == "multi":
            path_terminal_wealth_ratios.append(
                float(running_balance / float(FUTURES_INITIAL_BALANCE))
            )

        if not seg_raw_logs:
            raise optuna.TrialPruned()

        path_compound_raw_log_tw.append(float(np.sum(seg_raw_logs)))
        path_pfs.append(float(np.mean(seg_pfs)))
        path_wins.append(float(np.mean(seg_wins)))
        path_trades.append(float(np.mean(seg_trades)))
        path_mdds.append(float(np.max(seg_mdds)) if seg_mdds else 0.0)
        funding_ratios.extend(seg_fund_ratio)

        if path_idx >= 4 and path_compound_raw_log_tw:
            trial.report(float(np.mean(path_compound_raw_log_tw)), step=path_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)

    mean_log_tw_k = float(np.mean(path_arr)) if path_arr.size > 0 else 0.0
    p10_log_tw_path = (
        float(np.percentile(path_arr, 10.0)) if path_arr.size >= 10 else mean_log_tw_k
    )
    kelly_obj = w_mean * mean_log_tw_k + w_p10 * p10_log_tw_path

    sorted_rtns = np.sort(path_arr)
    n_paths_log = int(sorted_rtns.size)
    if n_paths_log > 0:
        k_worst = max(2, int(n_paths_log * cvar_alpha))
        k_worst = min(k_worst, n_paths_log)
        cvar_val = float(-np.mean(sorted_rtns[:k_worst]))
    else:
        cvar_val = 0.0
    cvar_pen = max(0.0, cvar_val - cvar_thr) * cvar_weight

    n_p_raw = len(path_compound_raw_log_tw)
    temporal_decay_pen = 0.0
    if n_p_raw >= 4 and temporal_lambda > 0.0:
        tw_raw_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
        k_recent = max(2, n_p_raw // 4)
        all_mean_tw = float(np.mean(tw_raw_arr))
        recent_mean_tw = float(np.mean(tw_raw_arr[-k_recent:]))
        temporal_decay_pen = max(0.0, all_mean_tw - recent_mean_tw) * temporal_lambda

    mu_paths = float(np.mean(path_arr)) if path_arr.size > 0 else -10.0
    sd_paths = float(np.std(path_arr, ddof=1)) if path_arr.size > 1 else 10.0
    cv_paths = sd_paths / (abs(mu_paths) + 1e-6) if path_arr.size > 1 else 10.0
    concentration_pen = max(0.0, cv_paths - 1.5) * 0.15

    mean_fund_r = float(np.mean(funding_ratios)) if funding_ratios else 0.0
    # Penalize both excessive funding cost (>15%) and excessive funding income (< -15%, strong penalty to prevent yield-farming overfit)
    funding_drag_pen = max(0.0, mean_fund_r - 0.15) * 2.0 + max(0.0, -mean_fund_r - 0.15) * 4.0

    min_reg_cfg = float(cfg.get("FUTURES_MIN_REGIME_ON_RATE", 0.0))
    w_reg_cfg = float(cfg.get("FUTURES_REGIME_ON_PENALTY_WEIGHT", 0.0))
    regime_on_rate = 0.5
    ref_sig_df = full_signal_dfs.get(ref_sym)
    if ref_sig_df is not None and "regime_risk_mult" in ref_sig_df.columns:
        rrm = ref_sig_df["regime_risk_mult"].to_numpy(dtype=np.float64)
        if rrm.size > is_off:
            rrm_is = rrm[is_off:]
            regime_on_rate = float(np.mean(rrm_is > 0.5)) if rrm_is.size else 0.5

    avg_trades = float(np.mean(path_trades)) if path_trades else 0.0
    
    trade_penalty = 0.0
    if avg_trades < min_pf_trades_dynamic:
        trade_penalty = ((min_pf_trades_dynamic - avg_trades) / min_pf_trades_dynamic) ** 2 * 5.0
        
    sortino_bonus = 0.0
    if path_arr.size >= 2:
        m_pt = float(np.mean(path_arr))
        s_pt = float(np.std(path_arr, ddof=1))
        neg_only = path_arr[path_arr < 0]
        ddev = float(np.std(neg_only, ddof=1)) if neg_only.size > 1 else s_pt
        psort = float(m_pt / (ddev + 1e-12)) if ddev > 0 else 0.0
        pos_only = path_arr[path_arr > 0]
        neg_mean = float(np.mean(neg_only)) if neg_only.size else -1e-9
        tr_val = 0.0
        if pos_only.size and neg_only.size:
             tr_val = float(np.mean(pos_only) / (abs(neg_mean) + 1e-12))
        elif pos_only.size and not neg_only.size:
             tr_val = 5.0
             
        if psort > 1.5:
             sortino_bonus += min(0.5, (psort - 1.5) * 0.1)
        if tr_val > 1.5:
             sortino_bonus += min(0.5, (tr_val - 1.5) * 0.1)

    objective_final = float(
        kelly_obj
        - cvar_pen
        - funding_drag_pen
        - concentration_pen
        - temporal_decay_pen
        - trade_penalty
        + sortino_bonus
    )
    if min_reg_cfg > 0.0 and w_reg_cfg > 0.0:
        objective_final -= max(0.0, min_reg_cfg - regime_on_rate) * w_reg_cfg

    _opt_meta = data_maps.get(ref_sym, {}).get("_futures_opt_meta")
    if isinstance(_opt_meta, dict):
        _obj_floor = _opt_meta.get("objective_floor_strict")
        if _obj_floor is not None and objective_final < float(_obj_floor):
            raise optuna.TrialPruned()

    avg_pf = float(np.mean(path_pfs)) if path_pfs else 1.0
    avg_win_rate = float(np.mean(path_wins)) if path_wins else 0.0
    avg_mdd = float(np.mean(path_mdds)) if path_mdds else 0.0

    if mode == "multi":
        min_sym_pf = avg_pf
    else:
        per_sym_mean_pf = [
            float(np.mean(sym_pf_accum[s])) if sym_pf_accum.get(s) else avg_pf for s in symbols
        ]
        min_sym_pf = float(np.min(per_sym_mean_pf)) if per_sym_mean_pf else avg_pf

    tot_l = float(all_seg_long)
    tot_s = float(all_seg_short)
    minority = float(min(tot_l, tot_s))
    majority = float(max(tot_l + tot_s, 1.0))
    ls_ratio = minority / majority

    n_cpcv_paths = int(len(cpcv_paths))
    worst_seg_mdd_pct = float(np.max(path_mdds)) if path_mdds else 0.0
    mean_path_ret_pct = (
        float(np.mean(np.expm1(path_arr) * 100.0)) if path_arr.size else 0.0
    )

    if path_arr.size >= 2:
        m_pt = float(np.mean(path_arr))
        s_pt = float(np.std(path_arr, ddof=1))
        sharpe_paths = m_pt / (s_pt + 1e-12)
        gate1_sqn = float(math.sqrt(float(path_arr.size)) * sharpe_paths)
        neg_only = path_arr[path_arr < 0]
        ddev = float(np.std(neg_only, ddof=1)) if neg_only.size > 1 else s_pt
        gate1_path_sortino = float(m_pt / (ddev + 1e-12)) if ddev > 0 else 999.0
        pos_only = path_arr[path_arr > 0]
        neg_mean = float(np.mean(neg_only)) if neg_only.size else -1e-9
        if pos_only.size and neg_only.size:
            gate1_tail_ratio = float(np.mean(pos_only) / (abs(neg_mean) + 1e-12))
        elif pos_only.size and not neg_only.size:
            gate1_tail_ratio = 10.0
        else:
            gate1_tail_ratio = 0.0
        gate1_psr = float(min(0.99, max(0.0, 0.5 + 0.12 * sharpe_paths)))
        gate1_dsr = float(
            min(
                0.99,
                max(
                    0.0,
                    gate1_psr * (1.0 - 0.08 * min(1.0, 1.0 / float(path_arr.size))),
                ),
            )
        )
    else:
        gate1_sqn = 0.0
        gate1_path_sortino = 0.0
        gate1_tail_ratio = 0.0
        gate1_psr = 0.0
        gate1_dsr = 0.0

    gate1_p10_gmgr = float(p10_log_tw_path)
    funding_drag_pct_stat = float(mean_fund_r * 100.0)

    if mode == "multi":
        min_path_terminal_wealth_ratio = (
            float(np.min(np.asarray(path_terminal_wealth_ratios, dtype=np.float64)))
            if path_terminal_wealth_ratios
            else 0.0
        )
        mean_path_terminal_wealth_ratio = (
            float(np.mean(np.asarray(path_terminal_wealth_ratios, dtype=np.float64)))
            if path_terminal_wealth_ratios
            else 0.0
        )
    else:
        tw_path_arr = np.asarray(
            [float(np.exp(np.clip(x, -50.0, 50.0))) for x in path_compound_raw_log_tw],
            dtype=np.float64,
        )
        min_path_terminal_wealth_ratio = (
            float(np.min(tw_path_arr)) if tw_path_arr.size else 0.0
        )
        mean_path_terminal_wealth_ratio = (
            float(np.mean(tw_path_arr)) if tw_path_arr.size else 0.0
        )

    trial.set_user_attr("avg_cagr", float(np.mean(path_compound_raw_log_tw)))
    trial.set_user_attr("avg_mdd", avg_mdd)
    trial.set_user_attr("avg_trades", avg_trades)
    trial.set_user_attr("avg_pf", avg_pf)
    trial.set_user_attr("avg_win_rate", avg_win_rate)
    trial.set_user_attr("min_sym_pf", min_sym_pf)
    trial.set_user_attr("long_short_ratio", ls_ratio)
    trial.set_user_attr("objective_final", objective_final)
    trial.set_user_attr("kelly_score_pct", float(mean_log_tw_k * 100.0))
    trial.set_user_attr("gate1_sqn", gate1_sqn)
    trial.set_user_attr("gate1_path_sortino", gate1_path_sortino)
    trial.set_user_attr("gate1_tail_ratio", gate1_tail_ratio)
    trial.set_user_attr("gate1_psr", gate1_psr)
    trial.set_user_attr("regime_on_rate", float(regime_on_rate))
    trial.set_user_attr("gate1_dsr", gate1_dsr)
    trial.set_user_attr("gate1_p10_gmgr", gate1_p10_gmgr)
    trial.set_user_attr("cpcv_mean_path_return_pct", mean_path_ret_pct)
    trial.set_user_attr("cpcv_worst_segment_mdd_pct", worst_seg_mdd_pct)
    trial.set_user_attr("n_cpcv_paths", n_cpcv_paths)
    trial.set_user_attr("mean_funding_drag_ratio_pct", funding_drag_pct_stat)
    trial.set_user_attr("cpcv_path_oos_log_tw", path_compound_raw_log_tw)
    trial.set_user_attr("min_path_terminal_wealth_ratio", float(min_path_terminal_wealth_ratio))
    trial.set_user_attr("mean_path_terminal_wealth_ratio", float(mean_path_terminal_wealth_ratio))
    trial.set_user_attr("psr_paths", float(gate1_psr))
    trial.set_user_attr("growth_score", float(objective_final))

    return objective_final


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
