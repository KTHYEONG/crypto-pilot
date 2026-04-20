"""
Tiered Signal Cache: Decoupled caching for Indicators, Signals/Regime, and Sizing.
Uses /dev/shm (if available) for high-speed IPC and Numpy-specific serialization.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config.settings import FUTURES_CACHE_DIR
from src.domain.futures.strategies_futures import UltimateStrategy

_logger: logging.Logger = logging.getLogger("opt_futures")

# --- Tiered Cache Configuration ---
_MEM_CACHE: Dict[str, Any] = {}
_MEM_CACHE_LOCK = threading.Lock()
_MEM_CACHE_MAX_ENTRIES = 200

# Try to use /dev/shm for ultra-fast disk cache (Shared Memory effectively)
SHM_PATH = Path("/dev/shm/my_coin_traider_cache")  # noqa: S108
if os.access("/dev/shm", os.W_OK):  # noqa: S108
    DISK_CACHE_ROOT = SHM_PATH
else:
    DISK_CACHE_ROOT = FUTURES_CACHE_DIR / "tiered_cache"

DISK_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _get_param_hash(params: Dict[str, Any], whitelist: frozenset[str]) -> str:
    """Helper to hash only relevant parameters."""
    items = tuple(sorted((k, params[k]) for k in whitelist if k in params))
    return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:16]


def _save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    """Fast compressed numpy save with directory safety."""
    try:
        # Ensure parent directory exists (handle potential deletion between initial check and write)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}.npz")
        np.savez_compressed(tmp_path, **data)  # type: ignore
        
        if tmp_path.exists():
            tmp_path.replace(path)
    except Exception as e:
        _logger.warning("Cache write failed for %s: %s", path.name, e)
        # Fallback: if /dev/shm is problematic, don't crash, just proceed without disk cache


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Fast numpy load."""
    try:
        with np.load(path) as data:
            return {k: v for k, v in data.items()}
    except Exception:
        return None


def get_tiered_signals(
    params: Dict[str, Any],
    symbol: str,
    tf: str,
    df_raw: pd.DataFrame,
    strategy: UltimateStrategy,
) -> pd.DataFrame:
    """
    Tiered caching entry point.
    1. Base Indicators (Tier 1)
    2. Signals/Regime (Tier 2) - Depends on SIGNAL_TYPE, REGIME_TYPE
    3. Sizing (Tier 3) - Depends on SIZING_METHOD
    """
    df = df_raw.copy(deep=False)
    n_bars = len(df)
    clean_sym = symbol.replace("/", "_")

    # --- Tier 1: Base Indicators (Static) ---
    t1_whitelist = frozenset(["ATR_PERIOD", "MACRO_EMA_PERIOD"])
    t1_hash = _get_param_hash(params, t1_whitelist)
    t1_key = f"t1_{clean_sym}_{tf}_{n_bars}_{t1_hash}"

    t1_path = DISK_CACHE_ROOT / f"{t1_key}.npz"
    t1_data = None

    with _MEM_CACHE_LOCK:
        t1_data = _MEM_CACHE.get(t1_key)

    if t1_data is None and t1_path.exists():
        t1_data = _load_npz(t1_path)

    if t1_data is None:
        df = strategy.generate_base_indicators(df)
        t1_data = {
            "atr": df["atr"].to_numpy(dtype=np.float64),
            "macro_ema": df["macro_ema"].to_numpy(dtype=np.float64),
        }
        if "btc_ema" in df.columns:
            t1_data["btc_ema"] = df["btc_ema"].to_numpy(dtype=np.float64)
        _save_npz(t1_path, t1_data)
        with _MEM_CACHE_LOCK:
            _MEM_CACHE[t1_key] = t1_data
    else:
        for k, v in t1_data.items():
            df[k] = v

    # --- Tier 2: Signal & Regime ---
    from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
    from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY

    st_key = str(params.get("SIGNAL_TYPE", "RSM_VT")).upper()
    rt_key = str(params.get("REGIME_TYPE", "EMA_ATR")).upper()

    t2_keys = ["SIGNAL_TYPE", "REGIME_TYPE", "CSM_RANK_COL"]
    if st_key in FUTURES_SIGNAL_REGISTRY:
        t2_keys.extend(FUTURES_SIGNAL_REGISTRY[st_key].param_space.keys())
    if rt_key in FUTURES_REGIME_REGISTRY:
        t2_keys.extend(FUTURES_REGIME_REGISTRY[rt_key].param_space.keys())

    t2_hash = _get_param_hash(params, frozenset(t2_keys))
    t2_key = f"t2_{clean_sym}_{tf}_{n_bars}_{t1_hash}_{t2_hash}"

    t2_path = DISK_CACHE_ROOT / f"{t2_key}.npz"
    t2_data = None

    with _MEM_CACHE_LOCK:
        t2_data = _MEM_CACHE.get(t2_key)

    if t2_data is None and t2_path.exists():
        t2_data = _load_npz(t2_path)

    if t2_data is None:
        df = strategy.compute_signal_regime_component(df)
        cols = [
            "trend_direction", "entry_upper", "entry_lower",
            "kill_signal", "slot_rank_score", "strength_filter", "regime_risk_mult",
            "xs_score_long", "xs_score_short", "hmm_prob_crisis",
            "hmm_modulator_long", "hmm_modulator_short",
        ]
        t2_data = {c: df[c].to_numpy(dtype=np.float64) for c in cols if c in df.columns}
        _save_npz(t2_path, t2_data)
        with _MEM_CACHE_LOCK:
            _MEM_CACHE[t2_key] = t2_data
    else:
        for k, v in t2_data.items():
            df[k] = v

    # --- Tier 3: Sizing ---
    from src.domain.futures.sizing import FUTURES_SIZING_REGISTRY
    sm_key = str(params.get("SIZING_METHOD", "vol_target")).lower()
    t3_keys = ["SIZING_METHOD"]
    if sm_key in FUTURES_SIZING_REGISTRY:
        t3_keys.extend(FUTURES_SIZING_REGISTRY[sm_key].param_space.keys())

    t3_hash = _get_param_hash(params, frozenset(t3_keys))
    t3_key = f"t3_{clean_sym}_{tf}_{n_bars}_{t1_hash}_{t3_hash}"

    t3_path = DISK_CACHE_ROOT / f"{t3_key}.npz"
    t3_data = None

    with _MEM_CACHE_LOCK:
        t3_data = _MEM_CACHE.get(t3_key)

    if t3_data is None and t3_path.exists():
        t3_data = _load_npz(t3_path)

    if t3_data is None:
        df["garch_kelly_f"] = strategy.compute_sizing_component(df)
        t3_data = {"garch_kelly_f": df["garch_kelly_f"].to_numpy(dtype=np.float64)}
        _save_npz(t3_path, t3_data)
        with _MEM_CACHE_LOCK:
            _MEM_CACHE[t3_key] = t3_data
    else:
        df["garch_kelly_f"] = t3_data["garch_kelly_f"]

    if len(_MEM_CACHE) > _MEM_CACHE_MAX_ENTRIES:
        with _MEM_CACHE_LOCK:
            _MEM_CACHE.clear()

    return df


def cleanup_old_cache_files(max_age_days: int = 1) -> None:
    """Non-blocking cleanup of old cache files."""
    now = time.time()
    max_age_sec = max_age_days * 86400
    for p in DISK_CACHE_ROOT.glob("*.npz"):
        try:
            if now - p.stat().st_mtime > max_age_sec:
                p.unlink(missing_ok=True)
        except Exception:  # noqa: S112
            continue


def _dataset_fingerprint_from_df(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    n = len(df)
    d0 = str(df["datetime"].iloc[0]) if "datetime" in df.columns else ""
    d1 = str(df["datetime"].iloc[-1]) if "datetime" in df.columns else ""
    if "close" in df.columns:
        c = df["close"].to_numpy(dtype=np.float64)
        head = c[: min(5, n)]
        tail = c[max(0, n - 5):]
        fp = hash((d0, d1, n, tuple(head.tolist()), tuple(tail.tolist())))
    else:
        fp = hash((d0, d1, n))
    return int(fp & ((1 << 63) - 1))
