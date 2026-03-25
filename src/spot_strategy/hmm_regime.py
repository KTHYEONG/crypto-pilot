"""Walk-forward 3-state Gaussian HMM regime engine with Sharpe-rank relabeling."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# Suppress noisy hmmlearn convergence warnings.
# These are emitted via hmmlearn's logging (e.g., "Model is not converging..."),
# so filtering warnings alone may not help.
_HMMLEARN_LOGGER_NAMES: tuple[str, ...] = ("hmmlearn", "hmmlearn.base")
for _name in _HMMLEARN_LOGGER_NAMES:
    logging.getLogger(_name).setLevel(logging.ERROR)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None  # type: ignore[misc, assignment]

# After per-window Z-score, diagonal regularization in scaled space (variance ~1 per dim).
_DEFAULT_MIN_COVAR: float = 1e-4
# If no successful refit for this many bars, emit neutral posteriors (avoid stale HMM).
_DEFAULT_STALE_MODEL_MAX_BARS: int = 48
# Sharpe relabel uses -1e9 for "too few samples"; two or more bad states => discard fit.
_RELABEL_BAD_SCORE: float = -1e9


def _winsorize_frame(x: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    if x.size == 0:
        return x
    lo = np.nanpercentile(x, low_q)
    hi = np.nanpercentile(x, high_q)
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), lo, hi)


def _fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return mu.astype(np.float64), sigma.astype(np.float64)


def _apply_scaler(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


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

    X_all, log_ret = _build_feature_matrix(df)

    last_model: object | None = None
    last_perm = np.array([0, 1, 2], dtype=np.int32)
    last_mu: np.ndarray | None = None
    last_sigma: np.ndarray | None = None
    last_fit_t: int = -10**9

    for t in range(tw, n):
        train_start = t - tw
        train_end = t
        should_refit = (t - tw) % rf == 0 or last_model is None

        X_win = X_all[train_start:train_end].copy()
        lr_win = log_ret[train_start:train_end]

        if should_refit:
            Xw = np.nan_to_num(X_win, nan=0.0, posinf=0.0, neginf=0.0)
            for j in range(Xw.shape[1]):
                Xw[:, j] = _winsorize_frame(Xw[:, j])
            mu, sigma = _fit_scaler(Xw)
            Xw_scaled = _apply_scaler(Xw, mu, sigma)
            try:
                model = GaussianHMM(
                    n_components=3,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=random_state,
                    min_covar=min_covar,
                )
                model.fit(Xw_scaled)
                hidden = model.predict(Xw_scaled)
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
                    last_model = model
                    last_mu = mu
                    last_sigma = sigma
                    last_fit_t = t
            except Exception as exc:
                _logger.debug("HMM fit failed at t=%s: %s", t, exc)

        if last_model is None:
            continue

        if (t - last_fit_t) > stale_cap:
            p_bull[t] = 1.0 / 3.0
            p_side[t] = 1.0 / 3.0
            viterbi_label[t] = 1
            continue

        if last_mu is None or last_sigma is None:
            continue

        X_cur = X_all[t - 1 : t].copy()
        X_cur = np.nan_to_num(X_cur, nan=0.0, posinf=0.0, neginf=0.0)
        X_cur_scaled = _apply_scaler(X_cur, last_mu, last_sigma)

        try:
            if hasattr(last_model, "predict_proba"):
                proba_row = last_model.predict_proba(X_cur_scaled)[0]
            else:
                st = last_model.predict(X_cur_scaled)
                proba_row = np.zeros(3, dtype=np.float64)
                proba_row[int(st[0])] = 1.0

            p_sd = float(proba_row[last_perm[1]])
            p_bl = float(proba_row[last_perm[2]])
            p_bull[t] = p_bl
            p_side[t] = p_sd
            vit = int(np.argmax(proba_row))
            vit_map = np.where(last_perm == vit)[0]
            viterbi_label[t] = int(vit_map[0]) if vit_map.size > 0 else 1
        except Exception as exc:
            _logger.debug("HMM predict failed at t=%s: %s", t, exc)

    return viterbi_label, p_bull, p_side
