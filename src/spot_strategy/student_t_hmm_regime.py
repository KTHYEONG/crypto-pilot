"""
Walk-forward 3-state HMM with diagonal Student-t emissions and state anchoring.

Uses GaussianHMM for transition/start probabilities; fits per-state diagonal Student-t
on winsorized training features; posteriors via forward-backward on each causal prefix.
Cache namespace is independent from Gaussian `hmm_regime` (separate IMPL_ID + disk prefix).
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln, logsumexp

from src.spot_strategy.hmm_regime import (
    _FEATURE_CACHE_MAXSIZE,
    GaussianHMM,
    _EWM_FEATURE_SPAN,
    _apply_hmm_t1_shift,
    _build_feature_matrix,
    _causal_ewm_standardize_matrix,
    _feature_cache,
    _feature_cache_lock,
    _hmm_data_fingerprint,
    _winsorize_frame,
)

_logger = logging.getLogger(__name__)

_DEFAULT_MIN_COVAR: float = 1e-4
_DEFAULT_STALE_MODEL_MAX_BARS: int = 48
_RELABEL_BAD_SCORE: float = -1e9

_STUDENT_T_HMM_IMPL_ID: int = 2
_HMM_N_ITER_COLD: int = 80
_HMM_N_ITER_WARM: int = 30
_HMM_TOL: float = 1e-3

_STUDENT_T_HMM_CACHE_MAXSIZE: int = 1024
_student_t_hmm_cache_lock: threading.Lock = threading.Lock()
_student_t_hmm_cache: OrderedDict[
    tuple[int, int, float, int, int, int, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = OrderedDict()


def _get_student_t_disk_cache_dir() -> Path:
    root = os.environ.get("OPT_SPOT_SIGNAL_CACHE_DIR", ".spot_signal_cache")
    path = Path(root) / "student_t_hmm_disk_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cosine_cost(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    cos = float(np.dot(a, b) / (na * nb))
    return float(1.0 - np.clip(cos, -1.0, 1.0))


def _anchor_state_labels(
    new_means: np.ndarray,
    prev_means: np.ndarray | None,
    prev_last_perm: np.ndarray,
    state_ranks: np.ndarray,
) -> np.ndarray:
    """
    Returns last_perm[semantic_slot] = hmm_component_index (same convention as Gaussian HMM).

    If prev_means is None, use rank-based argsort (ascending: slot0 worst, slot2 best bull).
    """
    if prev_means is None:
        return state_ranks.astype(np.int32, copy=False)

    n_states = new_means.shape[0]
    cost = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for s in range(n_states):
            j = int(prev_last_perm[s])
            cost[i, s] = _cosine_cost(new_means[i], prev_means[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    last_perm = np.zeros(n_states, dtype=np.int32)
    for k in range(n_states):
        semantic_slot = int(col_ind[k])
        hmm_idx = int(row_ind[k])
        last_perm[semantic_slot] = hmm_idx
    return last_perm


def _fit_student_t_diag_emissions(
    Xw: np.ndarray,
    hidden: np.ndarray,
    *,
    n_states: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast moment-based diagonal Student-t parameter estimation (no iterative MLE)."""
    d = Xw.shape[1]
    nu = np.full((n_states, d), 8.0, dtype=np.float64)
    loc = np.zeros((n_states, d), dtype=np.float64)
    scale = np.full((n_states, d), 0.1, dtype=np.float64)

    for k in range(n_states):
        mask = hidden == k
        n_k = int(np.sum(mask))
        if n_k < 5:
            continue
        vals = np.ascontiguousarray(Xw[mask], dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, np.nan)
        valid_counts = np.sum(np.isfinite(vals), axis=0)
        if not np.any(valid_counts >= 5):
            continue

        loc_k = np.nanmedian(vals, axis=0)
        loc_k = np.where(np.isfinite(loc_k), loc_k, np.nanmean(vals, axis=0))
        loc_k = np.where(np.isfinite(loc_k), loc_k, 0.0)

        centered = vals - loc_k[None, :]
        m2 = np.nanmean(centered * centered, axis=0)
        m4 = np.nanmean(centered**4, axis=0)
        m2 = np.where(np.isfinite(m2), m2, 0.0)
        m4 = np.where(np.isfinite(m4), m4, 0.0)

        eps = 1e-12
        excess_kurt = (m4 / np.maximum(m2 * m2, eps)) - 3.0
        nu_k = np.where(excess_kurt > 0.0, 6.0 / np.maximum(excess_kurt, 1e-6) + 4.0, 80.0)
        nu_k = np.clip(nu_k, 2.1, 80.0)

        scale_k = np.sqrt(np.maximum(m2, eps) * (nu_k - 2.0) / nu_k)
        scale_k = np.maximum(scale_k, 1e-9)

        insufficient = valid_counts < 5
        nu_k = np.where(insufficient, 30.0, nu_k)
        scale_k = np.where(insufficient, 1e-3, scale_k)

        loc[k] = loc_k
        nu[k] = nu_k
        scale[k] = scale_k
    return nu, loc, scale


def _log_emit_diag_t_vec(
    x: np.ndarray,
    nu: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    x_row = np.asarray(x, dtype=np.float64)[None, :]
    nu_safe = np.maximum(np.asarray(nu, dtype=np.float64), 2.1)
    scale_safe = np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)
    z = (x_row - np.asarray(loc, dtype=np.float64)) / scale_safe
    log_pdf = (
        gammaln((nu_safe + 1.0) * 0.5)
        - gammaln(nu_safe * 0.5)
        - 0.5 * np.log(nu_safe * np.pi)
        - np.log(scale_safe)
        - ((nu_safe + 1.0) * 0.5) * np.log1p((z * z) / nu_safe)
    )
    log_pdf = np.where(np.isfinite(log_pdf), log_pdf, -1e6)
    return np.sum(log_pdf, axis=1, dtype=np.float64)


def _log_emit_matrix(
    x_mat: np.ndarray,
    nu: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Batch Student-t emission log-probabilities for X:(T,D) -> (T,3)."""
    x_arr = np.asarray(x_mat, dtype=np.float64)
    nu_safe = np.maximum(np.asarray(nu, dtype=np.float64), 2.1)
    scale_safe = np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)
    z = (x_arr[:, None, :] - np.asarray(loc, dtype=np.float64)[None, :, :]) / scale_safe[None, :, :]
    log_pdf = (
        gammaln((nu_safe + 1.0) * 0.5)[None, :, :]
        - gammaln(nu_safe * 0.5)[None, :, :]
        - 0.5 * np.log(nu_safe * np.pi)[None, :, :]
        - np.log(scale_safe)[None, :, :]
        - ((nu_safe + 1.0) * 0.5)[None, :, :] * np.log1p((z * z) / nu_safe[None, :, :])
    )
    log_pdf = np.where(np.isfinite(log_pdf), log_pdf, -1e6)
    return np.sum(log_pdf, axis=2, dtype=np.float64)


def _forward_filter_last(
    X_prefix: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
    nu: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward filter on prefix; return P(z_{T-1}|x) and log-alpha at T-1."""
    t_len = X_prefix.shape[0]
    if t_len == 0:
        return np.full(3, 1.0 / 3.0, dtype=np.float64), np.full(3, -np.log(3.0), dtype=np.float64)

    log_alpha = np.zeros((t_len, 3), dtype=np.float64)
    all_le = _log_emit_matrix(X_prefix, nu, loc, scale)
    for t in range(t_len):
        le = all_le[t]
        if t == 0:
            log_alpha[0] = log_start + le
        else:
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + le
    log_gamma = log_alpha[-1].copy()
    log_gamma -= logsumexp(log_gamma)
    return np.exp(np.clip(log_gamma, -80.0, 0.0)), log_alpha[-1].copy()


def _forward_filter_extend_one(
    prev_log_alpha: np.ndarray,
    x_t: np.ndarray,
    log_trans: np.ndarray,
    nu: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step forward extension from previous log-alpha state."""
    le = _log_emit_diag_t_vec(x_t, nu, loc, scale)
    next_log_alpha = logsumexp(prev_log_alpha[:, None] + log_trans, axis=0) + le
    log_gamma = next_log_alpha - logsumexp(next_log_alpha)
    return np.exp(np.clip(log_gamma, -80.0, 0.0)), next_log_alpha


def _logsumexp_axis0(mat: np.ndarray) -> np.ndarray:
    """Fast logsumexp over axis=0 for shape (3, 3)."""
    max_v = np.max(mat, axis=0)
    stabilized = np.exp(mat - max_v)
    return max_v + np.log(np.sum(stabilized, axis=0))


def _logsumexp_vec(vec: np.ndarray) -> float:
    """Fast logsumexp over 1D vector with numerical stability."""
    max_v = float(np.max(vec))
    return max_v + float(np.log(np.sum(np.exp(vec - max_v))))


def _compute_walk_forward_student_t_core(
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

    last_model: Any | None = None
    last_perm = np.array([0, 1, 2], dtype=np.int32)
    prev_means: np.ndarray | None = None
    prev_last_perm = np.array([0, 1, 2], dtype=np.int32)
    last_fit_t: int = -10**9
    last_nu: np.ndarray | None = None
    last_loc: np.ndarray | None = None
    last_scale: np.ndarray | None = None

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

                is_warm_attempt = last_model is not None
                if is_warm_attempt:
                    candidate = copy.deepcopy(last_model)
                    candidate.n_iter = _HMM_N_ITER_WARM
                    candidate.init_params = ""
                else:
                    candidate = GaussianHMM(
                        n_components=3,
                        covariance_type="diag",
                        n_iter=_HMM_N_ITER_COLD,
                        tol=_HMM_TOL,
                        random_state=random_state,
                        min_covar=min_covar,
                        init_params="stmc",
                    )

                candidate.fit(Xw)

                if is_warm_attempt and _is_invalid(candidate):
                    candidate = GaussianHMM(
                        n_components=3,
                        covariance_type="diag",
                        n_iter=_HMM_N_ITER_COLD,
                        tol=_HMM_TOL,
                        random_state=random_state,
                        min_covar=min_covar,
                        init_params="stmc",
                    )
                    candidate.fit(Xw)

                if _is_invalid(candidate):
                    raise ValueError("HMM fit resulted in NaNs even after cold start.")

                hidden = candidate.predict(Xw)
                state_scores = np.zeros(3, dtype=np.float64)
                for k in range(3):
                    mask = hidden == k
                    if np.sum(mask) < 5:
                        state_scores[k] = _RELABEL_BAD_SCORE
                    else:
                        ret_k = lr_win[mask]
                        mean_ret = float(np.mean(ret_k))
                        downside = ret_k[ret_k < 0.0]
                        downside_std = float(np.std(downside) + 1e-12) if downside.size > 0 else 1e-12
                        state_scores[k] = mean_ret / downside_std

                state_perm = (
                    np.argsort(state_scores).astype(np.int32)
                    if int(np.sum(state_scores <= -1e8)) < 2
                    else np.array([0, 1, 2], dtype=np.int32)
                )

                new_means = np.asarray(candidate.means_, dtype=np.float64)
                last_perm = _anchor_state_labels(new_means, prev_means, prev_last_perm, state_perm)

                nu, loc, scale = _fit_student_t_diag_emissions(Xw, hidden)
                last_model = candidate
                last_nu, last_loc, last_scale = nu, loc, scale
                prev_means = new_means.copy()
                prev_last_perm = last_perm.copy()
                last_fit_t = t
                n_fit_ok += 1
            except Exception as exc:
                n_fit_fail += 1
                if not fit_fail_warned:
                    _logger.warning("Student-t HMM fit failed at t=%s: %s", t, exc)
                    fit_fail_warned = True

        if last_model is None or last_nu is None or last_loc is None or last_scale is None:
            t += 1
            continue

        if (t - last_fit_t) > stale_cap:
            p_bull[t] = 1.0 / 3.0
            p_side[t] = 1.0 / 3.0
            viterbi_label[t] = 1
            t += 1
            continue

        t_end = min(n - 1, last_fit_t + rf - 1, last_fit_t + stale_cap)

        try:
            lm = last_model
            nu = last_nu
            loc = last_loc
            scale = last_scale
            log_start = np.log(np.asarray(lm.startprob_, dtype=np.float64) + 1e-300)
            log_trans = np.log(np.asarray(lm.transmat_, dtype=np.float64) + 1e-300)

            prefix_lo = max(0, int(t) - int(tw))
            X_prefix = np.nan_to_num(
                X_scaled_full[prefix_lo : t + 1],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if X_prefix.shape[0] == 0:
                t = t_end + 1
                continue
            proba_row, log_alpha_last = _forward_filter_last(X_prefix, log_start, log_trans, nu, loc, scale)
            block = np.nan_to_num(
                X_scaled_full[t : t_end + 1],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            all_le_block = _log_emit_matrix(block, nu, loc, scale)
            block_len = all_le_block.shape[0]
            for off in range(block_len):
                if off > 0:
                    lta = log_alpha_last[:, None] + log_trans
                    next_log_alpha = _logsumexp_axis0(lta) + all_le_block[off]
                    log_gamma = next_log_alpha - _logsumexp_vec(next_log_alpha)
                    proba_row = np.exp(np.clip(log_gamma, -80.0, 0.0))
                    log_alpha_last = next_log_alpha
                tb = t + off
                p_bull[tb] = float(proba_row[int(last_perm[2])])
                p_side[tb] = float(proba_row[int(last_perm[1])])
                vit_hmm = int(np.argmax(proba_row))
                vit_map = np.where(last_perm == vit_hmm)[0]
                viterbi_label[tb] = int(vit_map[0]) if vit_map.size > 0 else 1
        except Exception as exc:
            n_predict_fail += 1
            if not predict_fail_warned:
                _logger.warning("Student-t HMM predict failed at t=%s: %s", t, exc)
                predict_fail_warned = True

        t = t_end + 1

    _logger.info(
        "Student-t HMM walk-forward summary: bars=%d fit_ok=%d fit_fail=%d predict_fail=%d",
        n,
        n_fit_ok,
        n_fit_fail,
        n_predict_fail,
    )
    return _apply_hmm_t1_shift(viterbi_label, p_bull, p_side)


def compute_walk_forward_hmm_student_t(
    df: pd.DataFrame,
    *,
    train_window: int,
    retrain_freq: int,
    min_covar: float = _DEFAULT_MIN_COVAR,
    random_state: int = 42,
    stale_model_max_bars: int = _DEFAULT_STALE_MODEL_MAX_BARS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same contract as ``compute_walk_forward_hmm``; Student-t emission + anchoring."""
    if GaussianHMM is None:
        n = len(df)
        return (
            np.ones(n, dtype=np.int32),
            np.full(n, 1.0 / 3.0),
            np.full(n, 1.0 / 3.0),
        )

    fp = _hmm_data_fingerprint(df)
    n = len(df)

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

    tw = int(train_window)
    rf = int(retrain_freq)
    cache_key = (
        tw,
        rf,
        float(min_covar),
        int(random_state),
        int(stale_model_max_bars),
        n,
        fp,
        _STUDENT_T_HMM_IMPL_ID,
    )

    with _student_t_hmm_cache_lock:
        if cache_key in _student_t_hmm_cache:
            _student_t_hmm_cache.move_to_end(cache_key)
            res = _student_t_hmm_cache[cache_key]
            return res[0].copy(), res[1].copy(), res[2].copy()

    disk_cache_file = _get_student_t_disk_cache_dir() / f"student_t_hmm_{hash(cache_key) & 0xFFFFFFFFFFFFFFFF}.npz"
    if disk_cache_file.exists():
        try:
            with np.load(disk_cache_file) as data:
                out = (data["v"], data["pb"], data["ps"])
                with _student_t_hmm_cache_lock:
                    _student_t_hmm_cache[cache_key] = (
                        out[0].copy(),
                        out[1].copy(),
                        out[2].copy(),
                    )
                return out
        except OSError:
            _logger.debug("Student-t HMM disk cache load failed, recomputing...")

    out = _compute_walk_forward_student_t_core(
        n,
        X_scaled_full,
        log_ret,
        train_window=tw,
        retrain_freq=rf,
        min_covar=min_covar,
        random_state=random_state,
        stale_model_max_bars=stale_model_max_bars,
    )

    with _student_t_hmm_cache_lock:
        while len(_student_t_hmm_cache) >= _STUDENT_T_HMM_CACHE_MAXSIZE:
            _student_t_hmm_cache.popitem(last=False)
        _student_t_hmm_cache[cache_key] = (out[0].copy(), out[1].copy(), out[2].copy())

    try:
        np.savez_compressed(disk_cache_file, v=out[0], pb=out[1], ps=out[2])
    except OSError as exc:
        _logger.debug("Failed to save Student-t HMM disk cache: %s", exc)

    return out
