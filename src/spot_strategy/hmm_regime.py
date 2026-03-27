"""Walk-forward 3-state Gaussian HMM regime engine with downside-risk relabeling."""

from __future__ import annotations

import logging
import threading
import inspect
import os
import copy
from collections import OrderedDict
from typing import Any, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

_logger = logging.getLogger(__name__)

# Suppress noisy hmmlearn convergence warnings.
_HMMLEARN_LOGGER_NAMES: tuple[str, ...] = ("hmmlearn", "hmmlearn.base")
for _name in _HMMLEARN_LOGGER_NAMES:
    logging.getLogger(_name).setLevel(logging.ERROR)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None  # type: ignore[misc, assignment]

_HMM_SUPPORTS_WARM_START: bool = False
if GaussianHMM is not None:
    try:
        _HMM_SUPPORTS_WARM_START = "warm_start" in inspect.signature(GaussianHMM.__init__).parameters
    except (TypeError, ValueError):
        _HMM_SUPPORTS_WARM_START = False
_hmm_env_logged: bool = False

_DEFAULT_MIN_COVAR: float = 1e-4
_DEFAULT_STALE_MODEL_MAX_BARS: int = 48
_RELABEL_BAD_SCORE: float = -1e9

_HMM_N_ITER_COLD: int = 50
_HMM_N_ITER_WARM: int = 20
_HMM_TOL: float = 1e-3
_EWM_FEATURE_SPAN: int = 50

_HMM_IMPL_ID: int = 7  # Extended causal prefix: train_window history for predict_proba

# Increase cache size for 32GB RAM environment (still very safe)
_HMM_CACHE_MAXSIZE: int = 1024
_hmm_cache_lock: threading.Lock = threading.Lock()
_hmm_cache: OrderedDict[
    tuple[int, int, float, int, int, int, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = OrderedDict()

# Feature matrix memory cache to avoid recomputing indicators across trials
_FEATURE_CACHE_MAXSIZE: int = 64
_feature_cache_lock: threading.Lock = threading.Lock()
_feature_cache: OrderedDict[int, Tuple[np.ndarray, np.ndarray]] = OrderedDict()

def _get_disk_cache_dir() -> Path:
    """Uses environment variable or project root for persistent disk cache."""
    root = os.environ.get("OPT_SPOT_SIGNAL_CACHE_DIR", ".spot_signal_cache")
    path = Path(root) / "hmm_disk_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _hmm_data_fingerprint(df: pd.DataFrame) -> int:
    """Lightweight invalidation: close + volume + hurst_40 head/tail + length."""
    if "close" not in df.columns or df.empty:
        return 0
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(close)
    head_c = tuple(close[: min(5, n)].tolist())
    tail_c = tuple(close[max(0, n - 5) :].tolist())
    
    vol = df["volume"].to_numpy(dtype=np.float64) if "volume" in df.columns else None
    head_v = tuple(vol[: min(5, n)].tolist()) if vol is not None else ("_nv_",)
    tail_v = tuple(vol[max(0, n - 5) :].tolist()) if vol is not None else ("_nv_",)
    
    hurst = df["hurst_40"].to_numpy(dtype=np.float64) if "hurst_40" in df.columns else None
    head_h = tuple(hurst[: min(5, n)].tolist()) if hurst is not None else ("_nh_",)
    tail_h = tuple(hurst[max(0, n - 5) :].tolist()) if hurst is not None else ("_nh_",)
    
    h = hash((head_c, tail_c, head_v, tail_v, head_h, tail_h, n))
    return int(h & ((1 << 63) - 1))

def _winsorize_frame(x: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    if x.size == 0:
        return x
    lo = np.nanpercentile(x, low_q)
    hi = np.nanpercentile(x, high_q)
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), lo, hi)

def _causal_ewm_standardize_matrix(X: np.ndarray, span: int) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float64)
    df = pd.DataFrame(X.astype(np.float64))
    mu = df.ewm(span=span, min_periods=1, adjust=False).mean()
    sig = df.ewm(span=span, min_periods=1, adjust=False).std()
    sig = sig.replace(0.0, 1e-12)
    out = (df - mu) / sig
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

def _apply_hmm_t1_shift(
    viterbi_label: np.ndarray,
    p_bull: np.ndarray,
    p_side: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.roll(viterbi_label, 1)
    v[0] = 1
    pb = np.roll(p_bull, 1)
    pb[0] = 1.0 / 3.0
    ps = np.roll(p_side, 1)
    ps[0] = 1.0 / 3.0
    return v.astype(np.int32, copy=False), pb.astype(np.float64, copy=False), ps.astype(np.float64, copy=False)

def _build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    close = df["close"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64) if "volume" in df.columns else np.ones(len(close))

    log_ret = np.zeros(len(close), dtype=np.float64)
    log_ret[1:] = np.log(np.maximum(close[1:], 1e-12) / np.maximum(close[:-1], 1e-12))

    log_ret_72 = pd.Series(log_ret).rolling(18, min_periods=18).sum().to_numpy(dtype=np.float64)

    rv14 = pd.Series(log_ret).rolling(14, min_periods=14).std().to_numpy(dtype=np.float64)
    rv14_safe = np.maximum(np.nan_to_num(rv14, nan=1e-12), 1e-12)
    momentum_sharpe = np.divide(
        np.nan_to_num(log_ret_72, nan=0.0),
        rv14_safe,
        out=np.zeros(len(close), dtype=np.float64),
        where=rv14_safe > 0.0,
    )
    momentum_sharpe = np.clip(momentum_sharpe, -20.0, 20.0)

    vma50 = pd.Series(vol).rolling(50, min_periods=50).mean().to_numpy(dtype=np.float64)
    vol_ratio = np.divide(
        pd.Series(vol).rolling(14, min_periods=14).mean().to_numpy(dtype=np.float64),
        np.maximum(vma50, 1e-12),
    )

    hurst_col = (
        df["hurst_40"].to_numpy(dtype=np.float64)
        if "hurst_40" in df.columns
        else np.full(len(df), 0.5, dtype=np.float64)
    )

    ret_ser = pd.Series(log_ret, copy=False)
    skew_20 = ret_ser.rolling(20, min_periods=20).skew().to_numpy(dtype=np.float64)
    up_std_20 = (
        ret_ser.clip(lower=0.0).rolling(20, min_periods=20).std(ddof=0).to_numpy(dtype=np.float64)
    )
    dn_std_20 = (
        (-ret_ser.clip(upper=0.0)).rolling(20, min_periods=20).std(ddof=0).to_numpy(dtype=np.float64)
    )
    asym_20 = np.divide(
        np.nan_to_num(up_std_20, nan=0.0),
        np.maximum(np.nan_to_num(dn_std_20, nan=0.0), 1e-12),
        out=np.ones(len(close), dtype=np.float64),
        where=np.maximum(np.nan_to_num(dn_std_20, nan=0.0), 1e-12) > 0.0,
    )

    X = np.column_stack(
        [
            log_ret,
            momentum_sharpe,
            np.nan_to_num(rv14, nan=0.0),
            np.nan_to_num(vol_ratio, nan=1.0),
            np.nan_to_num(hurst_col, nan=0.5),
            np.nan_to_num(skew_20, nan=0.0),
            np.nan_to_num(asym_20, nan=1.0),
        ]
    )
    return X, log_ret

def _compute_walk_forward_hmm_core(
    n: int,
    X_scaled_full: np.ndarray,
    log_ret: np.ndarray,
    *,
    train_window: int,
    retrain_freq: int,
    min_covar: float,
    random_state: int,
    stale_model_max_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    viterbi_label = np.ones(n, dtype=np.int32)
    p_bull = np.full(n, 1.0 / 3.0, dtype=np.float64)
    p_side = np.full(n, 1.0 / 3.0, dtype=np.float64)

    n_fit_ok = 0
    n_fit_fail = 0
    n_predict_fail = 0
    fit_fail_warned = False
    predict_fail_warned = False

    tw = max(120, int(train_window))
    rf = max(1, int(retrain_freq))
    stale_cap = max(1, int(stale_model_max_bars))

    last_model: object | None = None
    last_perm = np.array([0, 1, 2], dtype=np.int32)
    last_fit_t: int = -10**9

    t = tw
    while t < n:
        should_refit = (t - tw) % rf == 0 or last_model is None

        if should_refit:
            train_start = t - tw
            train_end = t
            Xw_scaled = X_scaled_full[train_start:train_end].copy()
            lr_win = log_ret[train_start:train_end]

            Xw = np.nan_to_num(Xw_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            for j in range(Xw.shape[1]):
                Xw[:, j] = _winsorize_frame(Xw[:, j])
            try:
                if GaussianHMM is None:
                    raise RuntimeError("GaussianHMM unavailable")
                
                def _is_invalid(m: Any) -> bool:
                    if (np.isnan(m.startprob_).any() or np.isnan(m.means_).any()
                            or np.isnan(m.covars_).any() or np.isnan(m.transmat_).any()):
                        return True
                    row_sums = m.transmat_.sum(axis=1)
                    return bool(np.any(np.abs(row_sums - 1.0) > 0.01))

                # Attempt 1: Warm start if possible, else Cold start
                is_warm_attempt = last_model is not None
                if is_warm_attempt:
                    candidate = copy.deepcopy(last_model)
                    candidate.n_iter = _HMM_N_ITER_WARM
                    candidate.init_params = ""
                else:
                    candidate = GaussianHMM(
                        n_components=3, covariance_type="diag",
                        n_iter=_HMM_N_ITER_COLD, tol=_HMM_TOL,
                        random_state=random_state, min_covar=min_covar,
                        init_params="stmc"
                    )

                candidate.fit(Xw)

                # Fallback: If warm fit failed or produced NaNs, try cold start once
                if (is_warm_attempt and _is_invalid(candidate)):
                    _logger.debug("HMM warm fit produced NaNs at t=%s; falling back to cold start.", t)
                    candidate = GaussianHMM(
                        n_components=3, covariance_type="diag",
                        n_iter=_HMM_N_ITER_COLD, tol=_HMM_TOL,
                        random_state=random_state, min_covar=min_covar,
                        init_params="stmc"
                    )
                    candidate.fit(Xw)

                if _is_invalid(candidate):
                    raise ValueError("HMM fit resulted in NaNs even after cold start.")

                hidden = candidate.predict(Xw)
                state_scores = np.zeros(3, dtype=np.float64)
                for k in range(3):
                    mask = (hidden == k)
                    if np.sum(mask) < 5:
                        state_scores[k] = _RELABEL_BAD_SCORE
                    else:
                        ret_k = lr_win[mask]
                        mean_ret = float(np.mean(ret_k))
                        downside = ret_k[ret_k < 0.0]
                        downside_std = float(np.std(downside) + 1e-12) if downside.size > 0 else 1e-12
                        state_scores[k] = mean_ret / downside_std

                if int(np.sum(state_scores <= -1e8)) < 2:
                    last_perm = np.argsort(state_scores).astype(np.int32)
                    last_model = candidate
                    last_fit_t = t
                    n_fit_ok += 1
            except Exception as exc:
                n_fit_fail += 1
                if not fit_fail_warned:
                    _logger.warning("HMM fit failed at t=%s: %s", t, exc)
                    fit_fail_warned = True

        if last_model is None:
            t += 1
            continue

        if (t - last_fit_t) > stale_cap:
            p_bull[t] = 1.0 / 3.0
            p_side[t] = 1.0 / 3.0
            viterbi_label[t] = 1
            t += 1
            continue

        t_end = min(n - 1, last_fit_t + rf - 1, last_fit_t + stale_cap)
        if t > t_end:
            t += 1
            continue

        try:
            lm = last_model
            # Causal: full-block predict_proba uses future bars inside the block; use growing prefixes only.
            prefix_lo = max(0, int(t) - int(tw))
            for tb in range(t, t_end + 1):
                X_prefix = X_scaled_full[prefix_lo : tb + 1]
                if X_prefix.shape[0] == 0:
                    continue
                if hasattr(lm, "predict_proba"):
                    proba_row = lm.predict_proba(X_prefix)[-1]
                else:
                    st = lm.predict(X_prefix)
                    proba_row = np.zeros(3, dtype=np.float64)
                    proba_row[int(st[-1])] = 1.0
                p_bull[tb] = float(proba_row[last_perm[2]])
                p_side[tb] = float(proba_row[last_perm[1]])
                vit = int(np.argmax(proba_row))
                vit_map = np.where(last_perm == vit)[0]
                viterbi_label[tb] = int(vit_map[0]) if vit_map.size > 0 else 1
        except Exception as exc:
            n_predict_fail += 1
            if not predict_fail_warned:
                _logger.warning("HMM predict failed at t=%s: %s", t, exc)
                predict_fail_warned = True

        t = t_end + 1

    _logger.info(
        "HMM walk-forward summary: bars=%d fit_ok=%d fit_fail=%d predict_fail=%d",
        n, n_fit_ok, n_fit_fail, n_predict_fail,
    )
    return _apply_hmm_t1_shift(viterbi_label, p_bull, p_side)

def compute_walk_forward_hmm(
    df: pd.DataFrame,
    *,
    train_window: int,
    retrain_freq: int,
    min_covar: float = _DEFAULT_MIN_COVAR,
    random_state: int = 42,
    stale_model_max_bars: int = _DEFAULT_STALE_MODEL_MAX_BARS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if GaussianHMM is None:
        n = len(df)
        return np.ones(n, dtype=np.int32), np.full(n, 1.0/3.0), np.full(n, 1.0/3.0)

    fp = _hmm_data_fingerprint(df)
    n = len(df)
    
    # 1. Feature Matrix Caching (Memory)
    with _feature_cache_lock:
        if fp in _feature_cache:
            _feature_cache.move_to_end(fp)
            X_scaled_full, log_ret = _feature_cache[fp]
        else:
            X_all, log_ret = _build_feature_matrix(df)
            X_scaled_full = _causal_ewm_standardize_matrix(X_all, _EWM_FEATURE_SPAN)
            while len(_feature_cache) >= _FEATURE_CACHE_MAXSIZE:
                _feature_cache.popitem(last=False)
            _feature_cache[fp] = (X_scaled_full, log_ret)

    cache_key = (tw:=int(train_window), rf:=int(retrain_freq), float(min_covar), int(random_state), int(stale_model_max_bars), n, fp, _HMM_IMPL_ID)

    # 2. LRU Cache (Memory)
    with _hmm_cache_lock:
        if cache_key in _hmm_cache:
            _hmm_cache.move_to_end(cache_key)
            res = _hmm_cache[cache_key]
            return res[0].copy(), res[1].copy(), res[2].copy()

    # 3. Disk Cache (Persistent)
    disk_cache_file = _get_disk_cache_dir() / f"hmm_{hash(cache_key) & 0xFFFFFFFFFFFFFFFF}.npz"
    if disk_cache_file.exists():
        try:
            with np.load(disk_cache_file) as data:
                out = (data["v"], data["pb"], data["ps"])
                with _hmm_cache_lock:
                    _hmm_cache[cache_key] = (out[0].copy(), out[1].copy(), out[2].copy())
                return out
        except Exception:
            _logger.debug("HMM disk cache load failed, recomputing...")

    # 4. Compute
    out = _compute_walk_forward_hmm_core(
        n, X_scaled_full, log_ret,
        train_window=tw, retrain_freq=rf, min_covar=min_covar,
        random_state=random_state, stale_model_max_bars=stale_model_max_bars
    )

    # 5. Save to Caches
    with _hmm_cache_lock:
        while len(_hmm_cache) >= _HMM_CACHE_MAXSIZE:
            _hmm_cache.popitem(last=False)
        _hmm_cache[cache_key] = (out[0].copy(), out[1].copy(), out[2].copy())
    
    try:
        np.savez_compressed(disk_cache_file, v=out[0], pb=out[1], ps=out[2])
    except Exception as exc:
        _logger.debug("Failed to save HMM disk cache: %s", exc)

    return out
