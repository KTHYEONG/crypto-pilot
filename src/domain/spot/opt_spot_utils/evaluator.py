from __future__ import annotations

import joblib
import inspect
import time
import hashlib
import logging
import math
import os
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]

from config.opt_config import OPT_SPOT_CONFIG, get_spot_effective_independent_trials
from config.settings import SPOT_INITIAL_BALANCE
from src.domain.spot.strategies_spot import UltimateSpotStrategy
from src.domain.spot.engine_spot import BacktestEngineFastSpot
from src.domain.spot.portfolio_shared_cash import run_shared_cash_multi_symbol
from src.domain.spot.opt_spot_utils.metrics import (
    calc_profit_factor_from_pnl,
    calc_mdd_from_equity,
    portfolio_cagr_pct_from_equity,
    calc_sortino_from_equity,
    calc_tail_ratio_from_equity,
    calc_tail_ratio_from_trades,
    cvar_loss_pct_from_simple_returns,
    compute_dsr_from_path_values,
    compute_pbo_from_cpcv_paths,
    max_underwater_bars_from_equity,
    mean_of_worst_quartile,
    probabilistic_sharpe_ratio,
)
from src.domain.spot.opt_spot_utils.cv_utils import (
    CPCVPath,
    build_cpcv_test_paths_with_fallback,
    cpcv_complement_segments,
    list_cpcv_block_ranges,
)
from src.domain.spot.opt_spot_utils.exit_family_prior import exit_family_prior_penalty
from src.domain.spot.opt_spot_utils.opt_params import suggest_params_spot

_logger: logging.Logger = logging.getLogger("opt_spot")

# Optuna TPE `constraints_func`: each value <= 0 means satisfied (Gardner-style soft constraints).
SPOT_OBJECTIVE_CONSTRAINT_DIM: int = 9


def spot_frozen_trial_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    raw = trial.user_attrs.get("spot_constraint_values")
    if isinstance(raw, (list, tuple)) and len(raw) == SPOT_OBJECTIVE_CONSTRAINT_DIM:
        return tuple(float(x) for x in raw)
    return tuple(1.0 for _ in range(SPOT_OBJECTIVE_CONSTRAINT_DIM))

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
    "SIGNAL_TYPE",
    "REGIME_TYPE",
    "EXIT_FAMILY",
    "SIZING_METHOD",
    "ATR_PERIOD",
    "EMA_FAST_PERIOD",
    "EMA_SLOW_PERIOD",
    "RSI_PERIOD",
    "RSI_LOW_THRESH",
    "KC_PERIOD",
    "KC_MULT",
    "MOMENTUM_PERIOD",
    "TP_MEAN_PERIOD",
    "KELLY_WINDOW",
    "W_SIGNAL",
    "K_ACCEL",
    "MB_FLOOR",
    "C_HYST",
    "EPSILON_MIN",
    "K_COOL_DOWN",
    "EMA_ATR_REGIME_SLOW",
    "ATR_REGIME_PERIOD",
    "VOL_PCT_WINDOW",
    "VOL_QUANTILE",
    "VOV_WINDOW",
    "ST_ATR_PERIOD",
    "ST_MULT",
    "TQ_EMA_FAST",
    "TQ_EMA_SLOW",
    "TQ_ADX_PERIOD",
    "TQ_ADX_THRESHOLD",
    "PFK_WINDOW",
    "PFK_MIN_F",
])

_SIGNAL_CACHE_SCHEMA_VERSION: int = 15

_SPOT_OBJECTIVE_CAGR_WEIGHT: float = 0.0  # Abandoning additive CAGR
_SPOT_OBJECTIVE_MIN_TRADES_HARD: float = 10.0 # Min trades per path to even consider
_SPOT_OBJECTIVE_MIN_TRADES_SOFT: float = 40.0  # Target trades for statistical robustness
_SPOT_OBJECTIVE_LOG_TWR_WEIGHT: float = 1.0
_SPOT_OBJECTIVE_PATH_CV_PENALTY: float = 0.75  # Path CV penalty (Generalized Kelly λ≈0.75)
_STRATEGY_LOGIC_HASH: Optional[str] = None
_CACHE_CLEANUP_DONE: bool = False
_CACHE_LAST_CLEANUP_TS: float = 0.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _signal_cache_max_days() -> int:
    return max(0, _env_int("OPT_SPOT_SIGNAL_CACHE_MAX_DAYS", 7))


def _signal_cache_max_bytes() -> int:
    max_gb = max(0, _env_int("OPT_SPOT_SIGNAL_CACHE_MAX_GB", 0))
    return max_gb * 1024 * 1024 * 1024


def _signal_cache_target_bytes(max_bytes: int) -> int:
    target_gb = max(0, _env_int("OPT_SPOT_SIGNAL_CACHE_TARGET_GB", 0))
    if target_gb > 0:
        return target_gb * 1024 * 1024 * 1024
    if max_bytes <= 0:
        return 0
    return int(max_bytes * 0.8)


def _signal_cache_cleanup_interval_sec() -> int:
    return max(0, _env_int("OPT_SPOT_SIGNAL_CACHE_CLEANUP_INTERVAL_SEC", 300))


def _signal_mem_cache_maxsize() -> int:
    return max(16, _env_int("OPT_SPOT_SIGNAL_MEM_CACHE_MAX", 96))


def _arrays_mem_cache_maxsize() -> int:
    return max(16, _env_int("OPT_SPOT_ARRAYS_MEM_CACHE_MAX", 96))

def _get_logic_hash() -> str:
    """Strategy logic fingerprinting via source code hashing."""
    global _STRATEGY_LOGIC_HASH
    if _STRATEGY_LOGIC_HASH is None:
        try:
            # Avoid circular import by local import
            from src.domain.spot.strategies_spot import UltimateSpotStrategy
            src = inspect.getsource(UltimateSpotStrategy.generate_signals)
            _STRATEGY_LOGIC_HASH = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
        except Exception:
            _STRATEGY_LOGIC_HASH = "legacy"
    return _STRATEGY_LOGIC_HASH

def _cleanup_old_cache(root: Path, *, force: bool = False) -> None:
    """Deletes expired cache files and trims total cache size with LRU-like mtime policy."""
    global _CACHE_LAST_CLEANUP_TS
    if not root.exists():
        return

    now = time.time()
    interval_sec = _signal_cache_cleanup_interval_sec()
    if not force and interval_sec > 0 and (now - _CACHE_LAST_CLEANUP_TS) < interval_sec:
        return

    max_days = _signal_cache_max_days()
    max_bytes = _signal_cache_max_bytes()
    target_bytes = _signal_cache_target_bytes(max_bytes)
    lock_path = root / ".cache_cleanup.lock"

    if FileLock is not None:
        try:
            with FileLock(str(lock_path), timeout=0.1):
                _cleanup_old_cache_impl(root, now, max_days=max_days, max_bytes=max_bytes, target_bytes=target_bytes)
            _CACHE_LAST_CLEANUP_TS = now
            return
        except Exception:
            pass

    _cleanup_old_cache_impl(root, now, max_days=max_days, max_bytes=max_bytes, target_bytes=target_bytes)
    _CACHE_LAST_CLEANUP_TS = now


def _cleanup_old_cache_impl(
    root: Path,
    now: float,
    *,
    max_days: int,
    max_bytes: int,
    target_bytes: int,
) -> None:
    now = time.time()
    max_sec = max_days * 86400
    expired_count = 0
    expired_bytes = 0
    kept_files: List[Tuple[float, int, Path]] = []
    total_bytes = 0
    for f in root.rglob("*.joblib"):
        try:
            stat = f.stat()
            size = int(stat.st_size)
            mtime = float(stat.st_mtime)
            if max_days > 0 and now - mtime > max_sec:
                f.unlink()
                expired_count += 1
                expired_bytes += size
                continue
            kept_files.append((mtime, size, f))
            total_bytes += size
        except OSError:
            pass
    if expired_count > 0:
        _logger.info(
            "Auto-cleaned %d expired cache files (>%d days, %.2f GB).",
            expired_count,
            max_days,
            expired_bytes / (1024 ** 3),
        )

    _cleanup_stale_signal_type_dirs(root)

    if max_bytes > 0 and total_bytes > max_bytes:
        trim_target = min(target_bytes if target_bytes > 0 else max_bytes, max_bytes)
        trim_target = max(0, trim_target)
        removed_count = 0
        removed_bytes = 0
        for _, size, path in sorted(kept_files, key=lambda item: item[0]):
            if total_bytes <= trim_target:
                break
            try:
                path.unlink()
                total_bytes -= size
                removed_count += 1
                removed_bytes += size
            except OSError:
                pass
        if removed_count > 0:
            _logger.info(
                "Trimmed signal cache by %d files (%.2f GB) to stay within %.2f GB target.",
                removed_count,
                removed_bytes / (1024 ** 3),
                trim_target / (1024 ** 3),
            )


def _cleanup_stale_signal_type_dirs(root: Path) -> None:
    try:
        from src.domain.spot.signals import SIGNAL_REGISTRY
        valid = {str(k).lower() for k in SIGNAL_REGISTRY.keys()}
    except Exception:
        return
    for sym_dir in root.iterdir():
        if not sym_dir.is_dir():
            continue
        for tf_dir in sym_dir.iterdir():
            if not tf_dir.is_dir():
                continue
            for sig_dir in tf_dir.iterdir():
                if not sig_dir.is_dir():
                    continue
                if sig_dir.name in valid:
                    continue
                try:
                    shutil.rmtree(sig_dir, ignore_errors=True)
                except OSError:
                    pass


def _touch_cache_file(path: Path) -> None:
    try:
        path.touch(exist_ok=True)
    except OSError:
        pass

_SignalCacheKey = Tuple[Tuple[Tuple[str, Any], ...], str, str, int, int, int, str]
_SignalComponentCacheKey = Tuple[str, Tuple[Tuple[str, Any], ...], str, str, int, int, int, str]
_SIGNAL_CACHE_MAXSIZE: int = _signal_mem_cache_maxsize()
_ARRAYS_CACHE_MAXSIZE: int = _arrays_mem_cache_maxsize()
_cache_lock: threading.Lock = threading.Lock()
_signal_cache: OrderedDict[_SignalCacheKey, pd.DataFrame] = OrderedDict()
_arrays_cache: OrderedDict[_SignalCacheKey, Dict[str, np.ndarray]] = OrderedDict()
_numpy_cache_warning_emitted: bool = False

_CACHE_PARAM_EXCLUDE: frozenset[str] = frozenset({"LEVERAGE", "USE_COMPOUNDING"})
_CACHE_COMPONENTS: Tuple[str, ...] = ("signal", "regime", "sizing", "exit")
_BASE_OHLCV_COLS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
_RUNTIME_REQUIRED_SIGNAL_COLS: Tuple[str, ...] = (
    "long_entry_signal",
    "entry_upper",
    "trend_direction",
    "strength_filter",
    "atr",
)
_COMPONENT_HINTS: Dict[str, Tuple[str, ...]] = {
    "regime": ("regime_",),
    "sizing": ("garch_", "kelly", "position_size", "size_"),
    "exit": ("bb_", "trail_", "exit_"),
}

_DISK_CACHE_READ_EXCEPTIONS: tuple[type[Exception], ...] = (
    OSError,
    EOFError,
    ValueError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
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


def _normalize_key_value(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return tuple(_normalize_key_value(x) for x in v)
    return v


def _cache_key_to_params(cache_key: _SignalCacheKey) -> Dict[str, Any]:
    return {k: v for k, v in cache_key[0]}


def _params_for_component(params: Dict[str, Any], component: str) -> Set[str]:
    signal_keys: Set[str] = set()
    regime_keys: Set[str] = set()
    sizing_keys: Set[str] = set()

    try:
        from src.domain.spot.signals import SIGNAL_REGISTRY
        st = str(params.get("SIGNAL_TYPE", "ADX_BREAKOUT")).upper()
        signal_keys.update({"SIGNAL_TYPE", "ATR_PERIOD"})
        if st in SIGNAL_REGISTRY:
            signal_keys.update(SIGNAL_REGISTRY[st].param_space.keys())
    except Exception:
        signal_keys.update({"SIGNAL_TYPE", "ATR_PERIOD"})

    try:
        from src.domain.spot.regimes import REGIME_REGISTRY
        rt = str(params.get("REGIME_TYPE", "MARKET_BREADTH")).upper()
        regime_keys.add("REGIME_TYPE")
        if rt in REGIME_REGISTRY:
            regime_keys.update(REGIME_REGISTRY[rt].param_space.keys())
    except Exception:
        regime_keys.add("REGIME_TYPE")

    try:
        from src.domain.spot.sizing import SIZING_REGISTRY
        sm = str(params.get("SIZING_METHOD", "vol_target")).lower()
        sizing_keys.add("SIZING_METHOD")
        if sm in SIZING_REGISTRY:
            sizing_keys.update(SIZING_REGISTRY[sm].param_space.keys())
    except Exception:
        sizing_keys.add("SIZING_METHOD")

    keys: Set[str] = set()
    if component == "signal":
        keys.update(signal_keys)
    elif component == "regime":
        keys.update(regime_keys)
    elif component == "sizing":
        keys.update(sizing_keys)
    elif component == "exit":
        known = signal_keys | regime_keys | sizing_keys | _CACHE_PARAM_EXCLUDE
        keys.update({k for k in params.keys() if k not in known})
    keys.add("TIMEFRAME")
    keys.difference_update(_CACHE_PARAM_EXCLUDE)
    return keys


def _component_for_column(col: str) -> str:
    c = str(col).lower()
    for comp, hints in _COMPONENT_HINTS.items():
        if any(h in c for h in hints):
            return comp
    return "signal"


def _disk_dtype_for_array(arr: np.ndarray) -> str:
    dt = arr.dtype
    if np.issubdtype(dt, np.bool_):
        return "int8"
    if np.issubdtype(dt, np.integer):
        return "int8" if dt.itemsize <= 2 else "int32"
    if np.issubdtype(dt, np.floating):
        return "float32"
    return "float32"


def _runtime_dtype_for_disk_dtype(disk_dtype: str) -> str:
    if disk_dtype.startswith("int"):
        return "int32"
    return "float64"


def _collect_component_specs(
    full_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Dict[str, List[Tuple[str, str, str]]]:
    base_cols: Set[str] = set(target_df.columns)
    dynamic_cols = [c for c in full_df.columns if c not in base_cols]
    specs: Dict[str, List[Tuple[str, str, str]]] = {c: [] for c in _CACHE_COMPONENTS}
    for col in dynamic_cols:
        s = full_df[col]
        if s.empty:
            continue
        if not (
            np.issubdtype(s.dtype, np.number)
            or np.issubdtype(s.dtype, np.bool_)
        ):
            continue
        arr = s.to_numpy(copy=False)
        disk_dtype = _disk_dtype_for_array(arr)
        runtime_dtype = _runtime_dtype_for_disk_dtype(disk_dtype)
        comp = _component_for_column(col)
        specs[comp].append((str(col), disk_dtype, runtime_dtype))
    for req in _RUNTIME_REQUIRED_SIGNAL_COLS:
        if req in full_df.columns and not any(req == n for n, _, _ in specs["signal"]):
            s = full_df[req]
            arr = s.to_numpy(copy=False)
            disk_dtype = _disk_dtype_for_array(arr)
            runtime_dtype = _runtime_dtype_for_disk_dtype(disk_dtype)
            specs["signal"].append((req, disk_dtype, runtime_dtype))
    return specs


def _build_component_cache_key(cache_key: _SignalCacheKey, params: Dict[str, Any], component: str) -> _SignalComponentCacheKey:
    _, sym, tf, data_len, fingerprint, version, logic_hash = cache_key
    names = _params_for_component(params, component)
    items: List[Tuple[str, Any]] = sorted(
        (k, _normalize_key_value(params[k])) for k in names if k in params
    )
    return (component, tuple(items), sym, tf, data_len, fingerprint, version, logic_hash)


def _signal_component_cache_path(component_key: _SignalComponentCacheKey, root: Path) -> Path:
    component, params_tuple, sym, tf, data_len, fingerprint, version, logic_hash = component_key
    sig_type = str(dict(params_tuple).get("SIGNAL_TYPE", "default")).lower()
    hash_payload = (component, params_tuple, data_len, fingerprint, version, logic_hash)
    digest = hashlib.sha256(repr(hash_payload).encode("utf-8")).hexdigest()
    folder = root / sym / tf / sig_type / component
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.joblib"


def _extract_component_arrays(
    full_df: pd.DataFrame,
    component: str,
    col_specs: List[Tuple[str, str, str]],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for col, disk_dtype, _ in col_specs:
        if col not in full_df.columns:
            raise KeyError(f"Missing required component column: {col}")
        out[col] = full_df[col].to_numpy(dtype=np.dtype(disk_dtype), copy=True)
    return out


def _write_component_cache(
    cache_key: _SignalCacheKey,
    params: Dict[str, Any],
    full_df: pd.DataFrame,
    target_df: pd.DataFrame,
    root: Path,
) -> None:
    component_specs = _collect_component_specs(full_df, target_df)
    for component in _CACHE_COMPONENTS:
        col_specs = component_specs.get(component, [])
        if not col_specs:
            continue
        comp_key = _build_component_cache_key(cache_key, params, component)
        cache_path = _signal_component_cache_path(comp_key, root)
        payload: Dict[str, Any] = {
            "schema": _SIGNAL_CACHE_SCHEMA_VERSION,
            "component": component,
            "n": int(len(full_df)),
            "columns": col_specs,
            "arrays": _extract_component_arrays(full_df, component, col_specs),
        }
        tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
        joblib.dump(payload, tmp_path, compress=3)
        tmp_path.replace(cache_path)


def _load_component_cache(
    cache_key: _SignalCacheKey,
    params: Dict[str, Any],
    root: Path,
) -> Optional[Dict[str, np.ndarray]]:
    all_cols: Dict[str, np.ndarray] = {}
    expected_n = int(cache_key[3])
    for component in _CACHE_COMPONENTS:
        comp_key = _build_component_cache_key(cache_key, params, component)
        cache_path = _signal_component_cache_path(comp_key, root)
        if not cache_path.exists():
            continue
        try:
            payload = joblib.load(cache_path)
            if not isinstance(payload, dict):
                return None
            arrays = payload.get("arrays")
            col_specs = payload.get("columns")
            n = int(payload.get("n", -1))
            if not isinstance(arrays, dict) or not isinstance(col_specs, list) or n != expected_n:
                return None
            for spec in col_specs:
                if not isinstance(spec, (list, tuple)) or len(spec) != 3:
                    return None
                col, _, runtime_dtype = spec
                arr = arrays.get(col)
                if arr is None:
                    return None
                all_cols[col] = np.asarray(arr, dtype=np.dtype(runtime_dtype))
            _touch_cache_file(cache_path)
        except _DISK_CACHE_READ_EXCEPTIONS:
            return None
    if not all_cols:
        return None
    if not all(k in all_cols for k in _RUNTIME_REQUIRED_SIGNAL_COLS):
        return None
    return all_cols


def _rebuild_full_df_from_components(
    target_df: pd.DataFrame,
    component_cols: Dict[str, np.ndarray],
) -> pd.DataFrame:
    full_df = target_df.copy(deep=True)
    for col in _BASE_OHLCV_COLS:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype(np.float64)
    if "btc_close" not in full_df.columns and "close" in full_df.columns:
        full_df["btc_close"] = full_df["close"].astype(np.float64)
    for col, arr in component_cols.items():
        full_df[str(col)] = arr
    for req in _RUNTIME_REQUIRED_SIGNAL_COLS:
        if req not in full_df.columns:
            raise KeyError(f"Missing required cached runtime column: {req}")
    return full_df


def _build_signal_cache_key(
    params: Dict[str, Any],
    sym: str,
    tf: str,
    data_len: int,
    fingerprint: int,
) -> _SignalCacheKey:
    signal_items: List[Tuple[str, Any]] = sorted(
        (k, _normalize_key_value(v))
        for k, v in params.items()
        if k not in _CACHE_PARAM_EXCLUDE
    )
    return (
        tuple(signal_items),
        sym,
        tf,
        data_len,
        int(fingerprint),
        _SIGNAL_CACHE_SCHEMA_VERSION,
        _get_logic_hash(),
    )


def get_or_compute_signals(
    cache_key: _SignalCacheKey,
    target_df: pd.DataFrame,
    strategy: UltimateSpotStrategy,
    *,
    disk_cache_root: Optional[Path] = None,
) -> pd.DataFrame:
    global _CACHE_CLEANUP_DONE
    params = _cache_key_to_params(cache_key)
    if disk_cache_root is not None and not _CACHE_CLEANUP_DONE:
        # 최초 실행 시 1회 정리 후, 이후에는 주기적으로 재정리한다.
        _cleanup_old_cache(disk_cache_root, force=True)
        _CACHE_CLEANUP_DONE = True

    with _cache_lock:
        if cache_key in _signal_cache:
            _signal_cache.move_to_end(cache_key)
            return _signal_cache[cache_key]

    # 1. 디스크 캐시 확인
    if disk_cache_root is not None:
        cached_components = _load_component_cache(cache_key, params, disk_cache_root)
        if cached_components is not None:
            try:
                full_df = _rebuild_full_df_from_components(target_df, cached_components)
                with _cache_lock:
                    while len(_signal_cache) >= _SIGNAL_CACHE_MAXSIZE:
                        _signal_cache.popitem(last=False)
                    _signal_cache[cache_key] = full_df
                return full_df
            except Exception:
                pass

    # 2. 신규 계산
    full_df: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
    
    # 3. 디스크 캐시 저장
    if disk_cache_root is not None:
        try:
            _write_component_cache(cache_key, params, full_df, target_df, disk_cache_root)
            _cleanup_old_cache(disk_cache_root, force=False)
        except Exception as e:
            _logger.warning("Failed to write split signal cache: %s", e)

    # 4. 메모리 캐시 저장
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
    # volume: required for shared-cash Numba ADV anchor (concurrency slippage scaling).
    required = ("open", "high", "low", "close", "volume", "atr", "long_entry_signal", "entry_upper")
    for c in required:
        if c not in sig_df.columns:
            raise ValueError(f"Missing column {c} for shared-cash segment.")
    out: Dict[str, np.ndarray] = {}
    for c in required:
        out[c] = sig_df[c].to_numpy(dtype=np.float64)
    if "regime_risk_mult" in sig_df.columns:
        out["regime_risk_mult"] = sig_df["regime_risk_mult"].to_numpy(dtype=np.float64)
    if "regime_entry_gate" in sig_df.columns:
        out["regime_entry_gate"] = sig_df["regime_entry_gate"].to_numpy(dtype=np.float64)
    if "regime_state" in sig_df.columns:
        out["regime_state"] = sig_df["regime_state"].to_numpy(dtype=np.float64)
    if "garch_kelly_f" in sig_df.columns:
        out["garch_kelly_f"] = sig_df["garch_kelly_f"].to_numpy(dtype=np.float64)
    if "kill_signal" in sig_df.columns:
        out["kill_signal"] = sig_df["kill_signal"].to_numpy(dtype=np.float64)
    if "bb_upper" in sig_df.columns:
        out["bb_upper"] = sig_df["bb_upper"].to_numpy(dtype=np.float64)
    if "trail_tighten_flag" in sig_df.columns:
        out["trail_tighten_flag"] = sig_df["trail_tighten_flag"].to_numpy(dtype=np.float64)
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


def _compute_path_gmgr_high_moments(equity: np.ndarray) -> float:
    """
    Geometric mean growth proxy with high-moment correction on log returns:
    r_mean - σ²/2 + S·σ³/6 - K·σ⁴/24. Returns winsorized at [p1, p99].
    """
    eq = np.asarray(equity, dtype=np.float64).ravel()
    if eq.size < 3:
        return -1.0
    eq = np.maximum(eq, 1e-12)
    r = np.diff(np.log(eq))
    r = r[np.isfinite(r)]
    if r.size < 5:
        return float(np.mean(r)) if r.size > 0 else -1.0
    lo, hi = np.percentile(r, [1.0, 99.0])
    rw = np.clip(r, lo, hi)
    mu = float(np.mean(rw))
    sd = float(np.std(rw, ddof=1))
    if sd < 1e-12:
        return mu
    m3 = float(np.mean((rw - mu) ** 3))
    m4 = float(np.mean((rw - mu) ** 4))
    skew = m3 / (sd**3)
    ex_kurt = m4 / (sd**4) - 3.0
    return float(mu - (sd**2) / 2.0 + skew * (sd**3) / 6.0 - ex_kurt * (sd**4) / 24.0)


def _compute_ulcer_index(equity: np.ndarray) -> float:
    """Ulcer Index: sqrt(mean(D_i^2)), D_i = (peak_i - eq_i) / peak_i * 100."""
    eq = np.asarray(equity, dtype=np.float64).ravel()
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    safe = np.where(peak > 1e-12, peak, 1.0)
    drawdown_pct = (peak - eq) / safe * 100.0
    return float(np.sqrt(np.mean(drawdown_pct * drawdown_pct)))


def _cpcv_path_compound_raw_log_tw(
    segments: List[Tuple[int, int]],
    *,
    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]],
    symbols: List[str],
    params: Dict[str, Any],
    is_off: int,
    ref_df: pd.DataFrame,
    max_slots: int,
    warmup_bars: int,
    concurrency_penalty_scale: float = 1.0,
) -> float:
    """Sum of raw log terminal-wealth ratios per segment (matches objective_spot CPCV path metric)."""
    seg_raw_log_tw: List[float] = []
    running_balance = float(SPOT_INITIAL_BALANCE)
    for test_start, test_end in segments:
        abs_start = is_off + int(test_start)
        abs_end = is_off + int(test_end)
        slice_start = max(0, abs_start - 1)
        slice_end = min(len(ref_df), abs_end)
        if slice_end - slice_start < 5:
            continue
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
            concurrency_penalty_scale=float(concurrency_penalty_scale),
        )
        eq = result.equity_curve
        if eq.size == 0:
            twr = 1.0
        else:
            twr = max(float(result.final_balance / segment_initial), 1e-9)
        raw_log_tw = float(np.log(twr))
        seg_raw_log_tw.append(raw_log_tw)
        running_balance = max(float(result.final_balance), 1e-9)
    if not seg_raw_log_tw:
        return float("nan")
    return float(np.sum(seg_raw_log_tw))


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
    params: Dict[str, Any] = suggest_params_spot(trial, space, tf_target)
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

    strategy._portfolio_eval_ctx = {"data_maps": data_maps, "symbols": list(symbols), "tf": tf}
    try:
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
        path_pfs: List[float] = []
        path_gmgr: List[float] = []
        path_ui: List[float] = []
        path_calmars: List[float] = []
        path_cagrs: List[float] = []
        path_regime_rates: List[float] = []
        total_sym_pnl = np.zeros(len(symbols), dtype=np.float64)

        for path_idx, path in enumerate(cpcv_paths):
            seg_log_tw: List[float] = []
            seg_raw_log_tw: List[float] = []
            seg_tw_ratio: List[float] = []
            seg_mdds: List[float] = []
            seg_cvars: List[float] = []
            seg_pfs: List[float] = []
            path_total_trades = 0
            path_regime_on = 0
            path_regime_len = 0
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

                ref_seg = symbol_arrays.get(ref_sym)
                if ref_seg is not None:
                    rs_arr = ref_seg.get("regime_state")
                    if rs_arr is not None and rs_arr.size > 0:
                        path_regime_on += int(np.sum(rs_arr > 0.1))
                        path_regime_len += int(rs_arr.size)

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
                        concurrency_penalty_scale=1.0,
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
                if hasattr(result, "per_symbol_pnl"):
                    total_sym_pnl += result.per_symbol_pnl
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
                else:
                    seg_mdds.append(0.0)
                    seg_cvars.append(0.0)
                    
                pnl = result.pnl_array
                pos_pnl = float(np.sum(pnl[pnl > 0.0]))
                neg_pnl = float(np.abs(np.sum(pnl[pnl < 0.0])))
                # Asymptotic Cap for zero-loss singularity to avoid penalizing perfect segments
                if neg_pnl > 1e-12:
                    seg_pf = pos_pnl / neg_pnl
                elif pos_pnl > 1e-12:
                    seg_pf = 3.0  # Theoretical cap for stable alpha
                else:
                    seg_pf = 1.0  # Neutral for zero trades
                seg_pfs.append(seg_pf)

                if int(result.total_trades) < min_seg_trades:
                    log_tw -= seg_fail_pen

                seg_raw_log_tw.append(raw_log_tw)
                seg_log_tw.append(log_tw)
                seg_tw_ratio.append(twr)
                running_balance = max(float(result.final_balance), 1e-9)

            if not seg_log_tw:
                raise optuna.TrialPruned()
            if path_regime_len > 0:
                path_regime_rates.append(path_regime_on / float(path_regime_len))
            path_compound_log_tw.append(float(np.sum(seg_log_tw)))
            path_compound_raw_log_tw.append(float(np.sum(seg_raw_log_tw)))
            path_compound_tw_ratio.append(float(np.prod(seg_tw_ratio)) if seg_tw_ratio else 1.0)
            path_worst_mdd.append(float(np.max(seg_mdds)) if seg_mdds else 0.0)
            
            # HARD HURDLE: If any path exceeds MDD limit, prune immediately.
            path_mdd_limit = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MDD_LIMIT_PCT", 20.0))
            if path_worst_mdd[-1] > path_mdd_limit:
                 raise optuna.TrialPruned()
            
            # HARD HURDLE: Min trades per path for statistical relevance.
            if path_total_trades < _SPOT_OBJECTIVE_MIN_TRADES_HARD:
                 raise optuna.TrialPruned()

            path_trades.append(path_total_trades)
            path_pfs.append(float(np.mean(seg_pfs)) if seg_pfs else 1.0)

            path_eq = np.concatenate(path_eq_chunks) if path_eq_chunks else np.array([], dtype=np.float64)
            path_level_tail = float(calc_tail_ratio_from_equity(path_eq)) if path_eq.size >= 2 else 1.0
            path_tail_ratios.append(path_level_tail)
            span_for_sortino = max(span_path_days, 1.0)
            raw_ps = float(calc_sortino_from_equity(path_eq, span_for_sortino)) if path_eq.size >= 2 else 0.0
            if not np.isfinite(raw_ps):
                raw_ps = 0.0
            path_sortino = float(np.clip(raw_ps, -path_sortino_clip, path_sortino_clip))
            path_sortino_vals.append(path_sortino)

            if path_eq.size >= 2:
                path_gmgr.append(_compute_path_gmgr_high_moments(path_eq))
                path_ui.append(_compute_ulcer_index(path_eq))
                span_c = max(span_path_days, 1.0)
                pc_cagr = float(portfolio_cagr_pct_from_equity(path_eq, span_c))
                pc_mdd = float(calc_mdd_from_equity(path_eq))
                path_calmars.append(pc_cagr / max(pc_mdd, 0.01))
                path_cagrs.append(pc_cagr)
            else:
                path_gmgr.append(-1.0)
                path_ui.append(0.0)
                path_calmars.append(0.0)
                path_cagrs.append(0.0)

            # Intermediate pruning: mean CPCV path log-TWR so far (enables MedianPruner / PatientPruner).
            if path_idx >= 4 and path_compound_raw_log_tw:
                intermediate = float(np.mean(path_compound_raw_log_tw))
                trial.report(intermediate, step=path_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_regime_on_rate = float(np.mean(path_regime_rates)) if path_regime_rates else 1.0
        cpcv_mean_path_cagr_pct = float(np.mean(path_cagrs)) if path_cagrs else 0.0
        trial.set_user_attr("cpcv_mean_path_cagr_pct", float(cpcv_mean_path_cagr_pct))

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

        gmgr_arr = np.asarray(path_gmgr, dtype=np.float64)
        ui_arr = np.asarray(path_ui, dtype=np.float64)
        p10_gmgr = float(np.percentile(gmgr_arr, 10.0)) if gmgr_arr.size else -1.0
        cagr_arr = np.asarray(path_cagrs, dtype=np.float64)
        p10_cagr = float(np.percentile(cagr_arr, 10.0)) if cagr_arr.size else -100.0
        max_ui = float(np.max(ui_arr)) if ui_arr.size else 0.0
        n_trades_mean = float(np.mean(path_trades)) if path_trades else 0.0

        path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
        n_paths_ct = int(path_arr.size)
        mu_paths = float(np.mean(path_arr)) if path_arr.size > 0 else -10.0
        sd_paths = float(np.std(path_arr, ddof=1)) if path_arr.size > 1 else 10.0
        if n_paths_ct >= 2:
            gate1_sqn = math.sqrt(float(n_paths_ct)) * mu_paths / sd_paths if sd_paths > 1e-12 else 0.0
            cv_paths = sd_paths / (abs(mu_paths) + 1e-6)
        else:
            gate1_sqn = 0.0
            cv_paths = 10.0

        # DSR (Deflated Sharpe Ratio) for path log-TWR stability
        n_done = trial.number + 1
        n_startup = int(OPT_SPOT_CONFIG.get("tpe_n_startup_trials", 100))
        n_eff = get_spot_effective_independent_trials(n_done, n_startup)
        path_vals = [float(x) for x in path_compound_raw_log_tw]
        dsr_val = float(compute_dsr_from_path_values(path_vals, n_eff))

        if n_paths_ct >= 2 and sd_paths > 1e-12:
            sr_est = mu_paths / sd_paths
            psr_val = float(probabilistic_sharpe_ratio(sr_est, n_obs=n_paths_ct))
        else:
            psr_val = 0.0

        if n_paths_ct >= 4 and dsr_val < -1.0:
            raise optuna.TrialPruned()

        min_path_tw_ratio = (
            float(np.min(np.asarray(path_compound_tw_ratio, dtype=np.float64)))
            if path_compound_tw_ratio
            else 0.0
        )
        psr_floor = float(cfg.get("SPOT_CONSTRAINT_PSR_FLOOR", 0.02))
        dsr_floor = max(
            float(cfg.get("SPOT_CONSTRAINT_DSR_FLOOR", 0.0)),
            float(cfg.get("SPOT_DISCOVERY_DSR_MIN", 0.35)),
        )
        min_tw_req = float(cfg.get("SPOT_CONSTRAINT_MIN_PATH_TW_RATIO", 0.88))
        min_mean_tr_req = float(cfg.get("SPOT_CONSTRAINT_MIN_MEAN_TRADES", 12.0))
        mean_path_pf_gate = float(np.mean(path_pfs)) if path_pfs else 1.0
        mean_tail_gate = float(np.mean(path_tail_ratios)) if path_tail_ratios else 0.0
        worst25_cal_gate = (
            float(np.percentile(np.asarray(path_calmars, dtype=np.float64), 25.0))
            if path_calmars
            else 0.0
        )
        min_mean_pf_req = float(cfg.get("SPOT_CONSTRAINT_MIN_MEAN_PF", 1.0))
        min_mean_tail_req = float(cfg.get("SPOT_CONSTRAINT_MIN_MEAN_PATH_TAIL", 1.0))
        min_w25_cal_req = float(cfg.get("SPOT_CONSTRAINT_MIN_WORST25_CALMAR", 0.0))
        min_p10_cagr_req = float(cfg.get("SPOT_MIN_P10_GMGR_CAGR_PCT", 5.0))
        min_regime_floor = float(cfg.get("SPOT_MIN_REGIME_ON_RATE", 0.15))
        if n_paths_ct >= 3:
            c_w25 = (
                float(min_w25_cal_req - worst25_cal_gate)
                if min_w25_cal_req > 1e-12
                else 0.0
            )
            c_reg = (
                float(min_regime_floor - mean_regime_on_rate)
                if path_regime_rates
                else 0.0
            )
            constraint_vec = (
                float(psr_floor - psr_val),
                float(dsr_floor - dsr_val),
                float(min_p10_cagr_req - p10_cagr),
                float(min_tw_req - min_path_tw_ratio),
                float(min_mean_tr_req - n_trades_mean),
                float(min_mean_pf_req - mean_path_pf_gate),
                float(min_mean_tail_req - mean_tail_gate),
                c_w25,
                c_reg,
            )
        else:
            constraint_vec = tuple(0.0 for _ in range(SPOT_OBJECTIVE_CONSTRAINT_DIM))

        trial.set_user_attr("spot_constraint_values", list(constraint_vec))
        spot_feasible = all(c <= 0.0 for c in constraint_vec)
        trial.set_user_attr("spot_constraints_feasible", spot_feasible)

        infeasible_pen = float(cfg.get("SPOT_OBJECTIVE_INFEASIBLE_RETURN", -1e9))
        if n_paths_ct >= 3 and not spot_feasible:
            trial.set_user_attr("objective_final", float(infeasible_pen))
            return float(infeasible_pen)

        # Kelly-CVaR: E[log W] with coherent tail penalty (replaces CRRA CEQ + z_conf LPM).
        path_log_tw_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
        path_returns = path_log_tw_arr

        mean_log_tw_k = float(np.mean(path_log_tw_arr)) if path_log_tw_arr.size > 0 else 0.0
        w_mean = float(cfg.get("SPOT_OBJECTIVE_W_MEAN_LOG_TW", 0.7))
        w_mean = float(np.clip(w_mean, 0.0, 1.0))
        w_p10 = 1.0 - w_mean
        p10_log_tw_path = (
            float(np.percentile(path_log_tw_arr, 10.0))
            if path_log_tw_arr.size >= 10
            else mean_log_tw_k
        )
        kelly_obj = w_mean * mean_log_tw_k + w_p10 * p10_log_tw_path

        cvar_alpha = float(cfg.get("SPOT_CPCV_CVAR_ALPHA", 0.10))
        cvar_thr = float(cfg.get("SPOT_CPCV_CVAR_THRESHOLD", 0.05))
        cvar_weight = float(cfg.get("SPOT_CPCV_CVAR_WEIGHT", 0.80))
        sorted_rtns = np.sort(path_log_tw_arr)
        n_paths_log = int(sorted_rtns.size)
        if n_paths_log > 0:
            k_worst = max(2, int(n_paths_log * cvar_alpha))
            k_worst = min(k_worst, n_paths_log)
            cvar_val = float(-np.mean(sorted_rtns[:k_worst]))
        else:
            cvar_val = 0.0
        cvar_pen = max(0.0, cvar_val - cvar_thr) * cvar_weight

        stat_lcb = kelly_obj - cvar_pen

        # Refined Penalties (Minimal & Targetted)
        concentration_pen = max(0.0, cv_paths - 1.5) * 0.15

        total_abs_pnl = float(np.sum(np.abs(total_sym_pnl))) + 1e-9
        sym_shares = total_sym_pnl / total_abs_pnl
        hhi = float(np.sum(sym_shares**2))
        n_sym = max(1, len(total_sym_pnl))
        hhi_equal = 1.0 / float(n_sym)
        hhi_penalty = max(0.0, hhi - 3.0 * hhi_equal) * 0.50

        trade_count_pen = max(0.0, _SPOT_OBJECTIVE_MIN_TRADES_SOFT - n_trades_mean) * 0.15

        _tail_ratio_target = float(
            cfg.get("SPOT_OBJECTIVE_TAIL_RATIO_TARGET", cfg.get("SPOT_GATE1_TAIL_RATIO_MIN", 1.1))
        )
        w_tail_obj = float(cfg.get("SPOT_OBJECTIVE_W_TAIL_RATIO", 0.60))
        tail_shortfall = max(0.0, _tail_ratio_target - mean_tail_gate)
        tail_bonus = math.log(max(mean_tail_gate, 0.5)) * w_tail_obj
        tail_penalty = tail_shortfall * 0.80
        tail_ratio_reward = tail_bonus - tail_penalty
        
        # Metrics for Optuna attributes and transparency
        p25_log = float(np.percentile(path_returns, 25.0)) if path_returns.size else -10.0
        p10_log = float(np.percentile(path_returns, 10.0)) if path_returns.size else -10.0
        total_soft_penalty = trade_count_pen + hhi_penalty

        temporal_decay_pen = 0.0
        n_p_raw = len(path_compound_raw_log_tw)
        temporal_decay_weight = float(
            cfg.get("SPOT_CPCV_TEMPORAL_LAMBDA", cfg.get("SPOT_TEMPORAL_DECAY_PEN_WEIGHT", 0.20))
        )
        if n_p_raw >= 4 and temporal_decay_weight > 0.0:
            k_recent = max(2, n_p_raw // 4)
            tw_raw_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
            all_mean_tw = float(np.mean(tw_raw_arr))
            recent_mean_tw = float(np.mean(tw_raw_arr[-k_recent:]))
            temporal_decay = max(0.0, all_mean_tw - recent_mean_tw)
            temporal_decay_pen = temporal_decay * temporal_decay_weight

        objective_final = float(
            stat_lcb             # Core Risk-Adjusted Growth
            + tail_ratio_reward  # CPCV path tail ratio (soft reward toward gate)
            - concentration_pen  # Path stability
            - hhi_penalty        # Asset diversity
            - trade_count_pen    # Statistical significance
            - temporal_decay_pen  # Recent CPCV paths must not collapse vs earlier paths
        )

        fwd_weight = float(cfg.get("SPOT_RECENT_IS_GATE_WEIGHT", 0.25))
        fwd_pen = 0.0
        recent_is_log_tw = float("nan")
        if fwd_weight > 0.0:
            n_is_bars = int(ref_len)
            recent_start = is_off + int(n_is_bars * 0.75)
            recent_end = is_off + n_is_bars
            slice_rs = max(0, recent_start - 1)
            slice_re = min(len(ref_df), recent_end)
            if slice_re - slice_rs >= 5:
                symbol_arrays_fwd: Dict[str, Dict[str, np.ndarray]] = {}
                rank_scores_fwd: Dict[str, np.ndarray] = {}
                for sym_fwd in symbols:
                    symbol_arrays_fwd[sym_fwd] = _slice_symbol_arrays_view(
                        prebuilt_full_arrays[sym_fwd], slice_rs, slice_re
                    )
                    rs_f = symbol_arrays_fwd[sym_fwd].get("slot_rank_score")
                    if rs_f is not None:
                        rank_scores_fwd[sym_fwd] = rs_f
                exec_fwd = max(1, recent_start - slice_rs)
                result_fwd = run_shared_cash_multi_symbol(
                    symbol_arrays_fwd,
                    symbols,
                    params,
                    initial_balance=float(SPOT_INITIAL_BALANCE),
                    max_concurrent_positions=max_slots,
                    rank_scores=rank_scores_fwd if rank_scores_fwd else None,
                    warmup_bars=warmup_bars,
                    execution_start_idx=exec_fwd,
                    allow_python_fallback=False,
                    concurrency_penalty_scale=1.0,
                )
                recent_is_log_tw = float(
                    np.log(max(result_fwd.final_balance / float(SPOT_INITIAL_BALANCE), 1e-9))
                )
                fwd_pen = max(0.0, -recent_is_log_tw) * fwd_weight
                objective_final = float(objective_final - fwd_pen)

        prior_scale = float(cfg.get("SPOT_EXIT_FAMILY_PRIOR_SCALE", 1.0))
        prior_pen = float(
            exit_family_prior_penalty(
                str(params.get("SIGNAL_TYPE", "ADX_BREAKOUT")),
                str(params.get("EXIT_FAMILY", "BALANCED")),
            )
        ) * prior_scale
        objective_final = float(objective_final - prior_pen)

        worst_seg_mdd = float(np.max(path_worst_mdd)) if path_worst_mdd else 0.0
        mean_path_return_pct = (math.exp(mean_log_tw) - 1.0) * 100.0
        mean_path_calmar = float(np.mean(path_calmars)) if path_calmars else 0.0

        trial.set_user_attr("objective_final", objective_final)
        trial.set_user_attr("growth_score", float(objective_final))
        trial.set_user_attr("temporal_decay_pen", float(temporal_decay_pen))
        trial.set_user_attr("exit_family_prior_penalty", float(prior_pen))
        trial.set_user_attr("p10_gmgr", float(p10_gmgr))
        trial.set_user_attr("cpcv_p25_log_tw", float(p25_log))
        trial.set_user_attr("cpcv_path_cv", float(cv_paths))
        trial.set_user_attr("kelly_obj", float(kelly_obj))
        trial.set_user_attr("cvar_val", float(cvar_val))
        trial.set_user_attr("objective_ceq_growth", float(kelly_obj))
        trial.set_user_attr("recent_is_log_tw", float(recent_is_log_tw))
        trial.set_user_attr("recent_is_gate_penalty", float(fwd_pen))
        trial.set_user_attr("objective_geometric_mean_log", float(mu_paths))
        trial.set_user_attr("objective_mean_path_calmar", float(mean_path_calmar))
        trial.set_user_attr("max_ulcer_index", float(max_ui))
        trial.set_user_attr("objective_soft_penalty", float(total_soft_penalty))
        trial.set_user_attr("dsr_paths", dsr_val)
        trial.set_user_attr("gate1_sqn", float(gate1_sqn))
        trial.set_user_attr("psr_paths", float(psr_val))
        trial.set_user_attr("gate1_path_sortino", float(mean_path_sortino))
        trial.set_user_attr("cpcv_path_tail_ratio", float(mean_tail_gate))
        trial.set_user_attr("cpcv_mean_path_return_pct", float(mean_path_return_pct))
        trial.set_user_attr("cpcv_worst_segment_mdd_pct", float(worst_seg_mdd))
        trial.set_user_attr(
            "cpcv_path_oos_log_tw",
            [float(x) for x in path_compound_raw_log_tw],
        )

        return float(objective_final)
    finally:
        strategy._portfolio_eval_ctx = None


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
        oos_start, slice_start, slice_end, max(1, oos_start - slice_start),
        slice_end - slice_start, len(symbols)
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
        "Holdout OOS result: final_balance=%.2f, total_trades=%d, "
        "eq_first=%.2f, eq_last=%.2f",
        res.final_balance, res.total_trades,
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
            cache_key: _SignalCacheKey = _build_signal_cache_key(p, sym, tf, len(target_df_full), fp)
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
        "total_windows": int(len(windows)),
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
