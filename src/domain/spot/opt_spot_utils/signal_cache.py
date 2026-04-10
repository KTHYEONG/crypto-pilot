from __future__ import annotations

import hashlib
import inspect
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]

from src.domain.spot.strategies_spot import UltimateSpotStrategy

_logger: logging.Logger = logging.getLogger("opt_spot")

# Optuna TPE `constraints_func`: each value <= 0 means satisfied (Gardner-style soft constraints).
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
































