"""
Futures Optuna objective: CPCV paths, Kelly-CVaR scalar, disk+memory signal cache.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    FUTURES_CACHE_DIR,
)
from src.domain.futures.strategies_futures import UltimateStrategy

_logger: logging.Logger = logging.getLogger("opt_futures")

_FUTURES_SIGNAL_CACHE_KEYS: frozenset[str] | None = None


def _signal_cache_param_keys_futures() -> frozenset[str]:
    global _FUTURES_SIGNAL_CACHE_KEYS
    if _FUTURES_SIGNAL_CACHE_KEYS is None:
        from src.domain.futures.opt_futures_utils.opt_params import (
            build_full_discovery_space_futures,
        )

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
    params: Dict[str, Any], sym: str, tf: str, data_len: int, fingerprint: int,
    whitelist: Optional[frozenset[str]] = None
) -> _SignalCacheKey:
    """
    Builds a cache key. If whitelist is provided, only those keys are used from params.
    Otherwise, all known discovery keys are used. 
    Using a whitelist is CRITICAL for optimization to prevent unrelated params from breaking the cache.
    """
    if whitelist is not None:
        target_keys = whitelist
    else:
        target_keys = _signal_cache_param_keys_futures()

    signal_items: Tuple[Tuple[str, Any], ...] = tuple(
        sorted((k, params[k]) for k in target_keys if k in params)
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
            ...

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
