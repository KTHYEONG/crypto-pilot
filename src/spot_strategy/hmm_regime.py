"""Walk-forward 3-state Gaussian HMM regime engine with Sharpe-rank relabeling."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Tuple

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

_DEFAULT_MIN_COVAR: float = 1e-4
_DEFAULT_STALE_MODEL_MAX_BARS: int = 48
_RELABEL_BAD_SCORE: float = -1e9

_HMM_N_ITER_COLD: int = 80
_HMM_N_ITER_WARM: int = 20
_HMM_TOL: float = 1e-3
# Causal EWM span for per-column standardization (stable feature space across refits).
_EWM_FEATURE_SPAN: int = 50

_HMM_IMPL_ID: int = 1  # Bump when causal-EWM / fit logic changes (invalidates subcache keys).

_HMM_CACHE_MAXSIZE: int = 64
_hmm_cache_lock: threading.Lock = threading.Lock()
_hmm_cache: OrderedDict[
    tuple[int, int, float, int, int, int, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = OrderedDict()


def _hmm_data_fingerprint(df: pd.DataFrame) -> int:
    """Lightweight invalidation: close + volume + hurst_40 head/tail + length (aligned with signal cache)."""
    if "close" not in df.columns or df.empty:
        return 0
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(close)
    head_c = tuple(close[: min(5, n)].tolist())
    tail_c = tuple(close[max(0, n - 5) :].tolist())
    if "volume" in df.columns:
        vol = df["volume"].to_numpy(dtype=np.float64)
        head_v = tuple(vol[: min(5, n)].tolist())
        tail_v = tuple(vol[max(0, n - 5) :].tolist())
    else:
        head_v = ("__no_volume__",)
        tail_v = ("__no_volume__",)
    if "hurst_40" in df.columns:
        hurst = df["hurst_40"].to_numpy(dtype=np.float64)
        head_h = tuple(hurst[: min(5, n)].tolist())
        tail_h = tuple(hurst[max(0, n - 5) :].tolist())
    else:
        head_h = ("__default_hurst__",)
        tail_h = ("__default_hurst__",)
    h = hash((head_c, tail_c, head_v, tail_v, head_h, tail_h, n))
    return int(h & ((1 << 63) - 1))


def _winsorize_frame(x: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    if x.size == 0:
        return x
    lo = np.nanpercentile(x, low_q)
    hi = np.nanpercentile(x, high_q)
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), lo, hi)


def _causal_ewm_standardize_matrix(X: np.ndarray, span: int) -> np.ndarray:
    """Column-wise (x - ewm_mean) / ewm_std; causal EWM, no global future mean."""
    if X.size == 0:
        return X.astype(np.float64)
    df = pd.DataFrame(X.astype(np.float64))
    mu = df.ewm(span=span, min_periods=1, adjust=False).mean()
    sig = df.ewm(span=span, min_periods=1, adjust=False).std()
    sig = sig.replace(0.0, 1e-12)
    out = (df - mu) / sig
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


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

    X = np.column_stack(
        [
            log_ret,
            momentum_sharpe,
            np.nan_to_num(rv14, nan=0.0),
            np.nan_to_num(vol_ratio, nan=1.0),
            np.nan_to_num(hurst_col, nan=0.5),
        ]
    )
    return X, log_ret


def _compute_walk_forward_hmm_core(
    df: pd.DataFrame,
    *,
    train_window: int,
    retrain_freq: int,
    min_covar: float,
    random_state: int,
    stale_model_max_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core walk-forward HMM (no module-level cache)."""
    n = len(df)
    viterbi_label = np.zeros(n, dtype=np.int32)
    p_bull = np.full(n, 1.0 / 3.0, dtype=np.float64)
    p_side = np.full(n, 1.0 / 3.0, dtype=np.float64)

    tw = max(120, int(train_window))
    rf = max(1, int(retrain_freq))
    stale_cap = max(1, int(stale_model_max_bars))

    X_all, log_ret = _build_feature_matrix(df)
    X_scaled_full = _causal_ewm_standardize_matrix(X_all, _EWM_FEATURE_SPAN)

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
                if last_model is None:
                    candidate = GaussianHMM(
                        n_components=3,
                        covariance_type="diag",
                        n_iter=_HMM_N_ITER_COLD,
                        tol=_HMM_TOL,
                        random_state=random_state,
                        min_covar=min_covar,
                        warm_start=False,
                    )
                else:
                    candidate = clone(last_model)
                    candidate.n_iter = _HMM_N_ITER_WARM
                    candidate.warm_start = True
                candidate.fit(Xw)
                hidden = candidate.predict(Xw)
                sharpe_scores = np.zeros(3, dtype=np.float64)
                for k in range(3):
                    mask = hidden == k
                    if np.sum(mask) < 5:
                        sharpe_scores[k] = _RELABEL_BAD_SCORE
                    else:
                        m_lr = float(np.mean(lr_win[mask]))
                        s_lr = float(np.std(lr_win[mask]) + 1e-12)
                        sharpe_scores[k] = m_lr / s_lr

                bad_states = int(np.sum(sharpe_scores <= -1e8))
                if bad_states >= 2:
                    _logger.debug(
                        "HMM Sharpe relabel unreliable at t=%s (%d bad states); keeping previous model.",
                        t,
                        bad_states,
                    )
                else:
                    last_perm = np.argsort(sharpe_scores).astype(np.int32)
                    last_model = candidate
                    last_fit_t = t
            except Exception as exc:
                _logger.debug("HMM fit failed at t=%s: %s", t, exc)

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

        X_block = X_scaled_full[t - 1 : t_end]
        X_block = np.nan_to_num(X_block, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            lm = last_model
            if hasattr(lm, "predict_proba"):
                proba_mat = lm.predict_proba(X_block)
            else:
                st = lm.predict(X_block)
                proba_mat = np.zeros((X_block.shape[0], 3), dtype=np.float64)
                proba_mat[np.arange(X_block.shape[0]), st.astype(np.int32)] = 1.0

            for i, tb in enumerate(range(t, t_end + 1)):
                proba_row = proba_mat[i]
                p_sd = float(proba_row[last_perm[1]])
                p_bl = float(proba_row[last_perm[2]])
                p_bull[tb] = p_bl
                p_side[tb] = p_sd
                vit = int(np.argmax(proba_row))
                vit_map = np.where(last_perm == vit)[0]
                viterbi_label[tb] = int(vit_map[0]) if vit_map.size > 0 else 1
        except Exception as exc:
            _logger.debug("HMM predict failed at t=%s: %s", t, exc)

        t = t_end + 1

    return viterbi_label, p_bull, p_side


def compute_walk_forward_hmm(
    df: pd.DataFrame,
    *,
    train_window: int,
    retrain_freq: int,
    min_covar: float = _DEFAULT_MIN_COVAR,
    random_state: int = 42,
    stale_model_max_bars: int = _DEFAULT_STALE_MODEL_MAX_BARS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns per-bar:
      viterbi_label: int 0=Bear, 1=Side, 2=Bull (after Sharpe relabel)
      p_bull, p_side: posterior probabilities in relabeled order
    """
    n = len(df)
    viterbi_label = np.zeros(n, dtype=np.int32)
    p_bull = np.full(n, 1.0 / 3.0, dtype=np.float64)
    p_side = np.full(n, 1.0 / 3.0, dtype=np.float64)

    if GaussianHMM is None:
        _logger.warning("hmmlearn not installed; HMM regime fallback to neutral.")
        return viterbi_label, p_bull, p_side

    tw = max(120, int(train_window))
    rf = max(1, int(retrain_freq))
    stale_cap = max(1, int(stale_model_max_bars))
    fp = _hmm_data_fingerprint(df)
    cache_key: tuple[int, int, float, int, int, int, int, int] = (
        tw,
        rf,
        float(min_covar),
        int(random_state),
        stale_cap,
        n,
        fp,
        _HMM_IMPL_ID,
    )

    with _hmm_cache_lock:
        if cache_key in _hmm_cache:
            _hmm_cache.move_to_end(cache_key)
            v, pb, ps = _hmm_cache[cache_key]
            return v.copy(), pb.copy(), ps.copy()

    out = _compute_walk_forward_hmm_core(
        df,
        train_window=train_window,
        retrain_freq=retrain_freq,
        min_covar=min_covar,
        random_state=random_state,
        stale_model_max_bars=stale_model_max_bars,
    )

    with _hmm_cache_lock:
        while len(_hmm_cache) >= _HMM_CACHE_MAXSIZE:
            _hmm_cache.popitem(last=False)
        _hmm_cache[cache_key] = (
            out[0].copy(),
            out[1].copy(),
            out[2].copy(),
        )

    return out
