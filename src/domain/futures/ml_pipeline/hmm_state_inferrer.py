"""JAX-based Student-t HMM regime probabilities with SGD optimization and sticky priors."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

# Control JAX memory preallocation and log backend info
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_platform_name", "cpu")
_logger = logging.getLogger(__name__)

try:
    _logger.info("JAX Backend: %s", jax.default_backend())
    _logger.info("JAX Devices: %s", jax.devices())
except Exception as e:
    _logger.warning("JAX initialization info failed: %s", e)

import optax
import pandas as pd
from numba import njit
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_CACHE_DIR
from src.core.utils.cache_manager import CacheManager
from src.domain.futures.ml_pipeline.feature_engineering import (
    HMM_SEMANTIC_PROB_COLUMNS,
    SYSTEMIC_HMM_FEATURE_COLUMNS,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")

_logger = logging.getLogger(__name__)

_SEMANTIC_ORDER = list(HMM_SEMANTIC_PROB_COLUMNS)

# Phase 4: weaker BULL MSE guidance vs CRISIS so bull share is not collapsed (bear overhang).
_GUIDANCE_BULL_LL_WEIGHT = 550.0
_GUIDANCE_CRISIS_LL_WEIGHT = 2000.0
_GUIDANCE_BEAR_LL_WEIGHT = 1500.0


def _calibrate_crisis_logit_offset(
    sem_probs: np.ndarray,
    target_mean_crisis: float = 0.07,
    delta_hi_max: float = 80.0,
    bisect_rounds: int = 48,
) -> np.ndarray:
    """Cap series-mean P(CRISIS) without global column scaling: CRISIS gets exp(-delta) vs others (logit shift).

    Keeps relative weights among BULL/BEAR/CHOP unchanged per row; avoids mean-scaling overcorrection that
    flips idxmax vs BEAR on strong-crisis bars.
    """
    p = np.asarray(sem_probs, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    mean_c = float(np.mean(p[:, 3]))
    if mean_c <= target_mean_crisis + 1e-14:
        return p.copy()

    def apply_delta(delta: float) -> np.ndarray:
        w = np.ones_like(p)
        w[:, 3] = np.exp(-delta)
        q = p * w
        q /= q.sum(axis=1, keepdims=True)
        return q

    def crisis_mean(delta: float) -> float:
        q = apply_delta(delta)
        return float(np.mean(q[:, 3]))

    hi = 1.0
    while crisis_mean(hi) > target_mean_crisis and hi < delta_hi_max:
        hi = min(hi * 2.0, delta_hi_max)

    if crisis_mean(hi) > target_mean_crisis:
        _logger.warning(
            "Crisis logit-offset: delta capped at %s, mean(P_crisis)=%.4f > target %.4f",
            hi,
            crisis_mean(hi),
            target_mean_crisis,
        )
        return apply_delta(hi)

    lo = 0.0
    for _ in range(bisect_rounds):
        mid = 0.5 * (lo + hi)
        if crisis_mean(mid) > target_mean_crisis:
            lo = mid
        else:
            hi = mid
    return apply_delta(hi)


def _normalized_entropy_k4(probs_4: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs_4, dtype=np.float64), 1e-12, 1.0)
    return (-np.sum(p * np.log(p), axis=1) / np.log(4.0)).astype(np.float64)


# --- JAX Student-t Distribution Implementation ---

@jax.jit
def _student_t_log_pdf(
    x: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray, df: jnp.ndarray
) -> jnp.ndarray:
    """Multivariate Diagonal Student-t log-pdf."""
    # x: (D,), loc: (D,), scale: (D,), df: (1,)
    y = (x - loc) / scale
    log_c = (
        jax.scipy.special.gammaln((df + 1.0) / 2.0)
        - jax.scipy.special.gammaln(df / 2.0)
        - 0.5 * jnp.log(jnp.pi * df)
        - jnp.log(scale)
    )
    return log_c - (df + 1.0) / 2.0 * jnp.log1p(y**2 / df)


# --- JAX HMM Core Functions ---

MAX_HMM_WINDOW = 1500

# TVTP: cs_dispersion(4), oi_delta(5), funding_mom(6), liq_proxy(7), lsr_delta(8).
# macro_ret_1h (obs 9) excluded from transitions.
_TVTP_FEATURE_INDICES: tuple[int, ...] = (4, 5, 6, 7, 8)
_N_TVTP_FEATS: int = len(_TVTP_FEATURE_INDICES)

# macro_vol_24h index in SYSTEMIC_HMM_FEATURE_COLUMNS — matches locs[:, d] for σ ordering penalties.
_HMM_LOC_IDX_VOL: int = 2

@jax.jit
def _batched_student_t_log_pdf(
    x: jnp.ndarray, locs: jnp.ndarray, scales: jnp.ndarray, dfs: jnp.ndarray
) -> jnp.ndarray:
    """Vectorized Multivariate Student-t log-pdf for all states at once (Zero-Loop)."""
    # x: (D,), locs: (K, D), scales: (K, D), dfs: (K,)
    y = (x[None, :] - locs) / scales
    dfs_expanded = dfs[:, None]
    log_c = (
        jax.scipy.special.gammaln((dfs_expanded + 1.0) / 2.0)
        - jax.scipy.special.gammaln(dfs_expanded / 2.0)
        - 0.5 * jnp.log(jnp.pi * dfs_expanded)
        - jnp.log(scales)
    )
    return jnp.sum(log_c - (dfs_expanded + 1.0) / 2.0 * jnp.log1p(y**2 / dfs_expanded), axis=1)


@jax.jit
def _compute_tvtp_matrices(Z: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute time-varying transition matrices from exogenous features Z.
    
    Z: (T, D_z)
    W: (K, K, D_z)
    b: (K, K)
    Returns: (T, K, K) log-transition matrices.
    """
    logits = jnp.einsum("td,kjd->tkj", Z, W) + b[None, :, :]
    return jax.nn.log_softmax(logits, axis=2)


@jax.jit
def _forward_pass(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    log_init_probs: jnp.ndarray,
    log_trans_mats: jnp.ndarray,
    locs: jnp.ndarray,
    scales: jnp.ndarray,
    dfs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run Forward algorithm in log-space with static shape masking and dynamic transitions."""
    def scan_fn(
        log_alpha_prev: jnp.ndarray, val: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, m, log_trans_mat = val
        log_emissions = _batched_student_t_log_pdf(x, locs, scales, dfs)
        combined = log_alpha_prev[:, None] + log_trans_mat
        log_alpha_next_calc = jax.scipy.special.logsumexp(combined, axis=0) + log_emissions
        ll_t = jax.scipy.special.logsumexp(log_alpha_next_calc)
        log_alpha_next_normalized = log_alpha_next_calc - ll_t
        log_alpha_next = jnp.where(m, log_alpha_next_normalized, log_alpha_prev)
        incremental_ll = jnp.where(m, ll_t, 0.0)
        return log_alpha_next, (log_alpha_next, incremental_ll)

    initial_emissions = _batched_student_t_log_pdf(observations[0], locs, scales, dfs)
    log_alpha_0_calc = log_init_probs + initial_emissions
    ll_0 = jax.scipy.special.logsumexp(log_alpha_0_calc)
    log_alpha_0_normalized = log_alpha_0_calc - ll_0
    log_alpha_0 = jnp.where(mask[0], log_alpha_0_normalized, log_init_probs)
    incremental_ll_0 = jnp.where(mask[0], ll_0, 0.0)

    _, (log_alphas, incremental_lls) = jax.lax.scan(scan_fn, log_alpha_0, (observations[1:], mask[1:], log_trans_mats[1:]))
    log_alphas = jnp.concatenate([log_alpha_0[None, :], log_alphas], axis=0)
    incremental_lls = jnp.concatenate([incremental_ll_0[None], incremental_lls], axis=0)
    return log_alphas, incremental_lls


@jax.jit
def _viterbi_decode(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    log_init_probs: jnp.ndarray,
    log_trans_mats: jnp.ndarray,
    locs: jnp.ndarray,
    scales: jnp.ndarray,
    dfs: jnp.ndarray,
) -> jnp.ndarray:
    """Viterbi decoding in log-space to find the most likely hard state sequence."""
    K = log_init_probs.shape[0]

    def forward_step(prev_log_prob, val):
        x, m, log_trans_mat = val
        log_emissions = _batched_student_t_log_pdf(x, locs, scales, dfs)
        combined = prev_log_prob[:, None] + log_trans_mat
        max_log_prob = jnp.max(combined, axis=0) + log_emissions
        argmax_state = jnp.argmax(combined, axis=0)
        
        # Mask handling: if masked, keep previous probabilities and dummy argmax
        res_log_prob = jnp.where(m, max_log_prob, prev_log_prob)
        res_argmax = jnp.where(m, argmax_state, jnp.arange(K))
        return res_log_prob, res_argmax

    initial_emissions = _batched_student_t_log_pdf(observations[0], locs, scales, dfs)
    log_delta_0 = log_init_probs + initial_emissions
    
    final_log_probs, backpointers = jax.lax.scan(
        forward_step, log_delta_0, (observations[1:], mask[1:], log_trans_mats[1:])
    )
    
    def backward_step(best_next_state, bp):
        best_curr_state = bp[best_next_state]
        return best_curr_state, best_curr_state

    last_state = jnp.argmax(final_log_probs, axis=0)
    _, best_path_tail = jax.lax.scan(backward_step, last_state, backpointers, reverse=True)
    
    best_path = jnp.concatenate([best_path_tail, last_state[None]], axis=0)
    return jax.nn.one_hot(best_path, K)


@jax.jit
def _compute_nll(
    params: dict[str, Any],
    observations: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Pure Negative log-likelihood with TVTP and sticky prior."""
    tvtp_z = observations[:, jnp.array(_TVTP_FEATURE_INDICES)]
    log_trans_mats = _compute_tvtp_matrices(tvtp_z, params["tvtp_W"], params["tvtp_b"])
    log_alphas, incremental_lls = _forward_pass(
        observations,
        mask,
        jax.nn.log_softmax(params["initial"]),
        log_trans_mats,
        params["locs"],
        jnp.exp(params["log_scales"]),
        jnp.exp(params["log_dfs"]) + 2.0,
    )
    ll = jnp.sum(incremental_lls)
    
    # sticky_penalty: Encourage states to persist (diagonal of transition matrix)
    avg_trans_mat = jnp.exp(jax.scipy.special.logsumexp(log_trans_mats, axis=0) - jnp.log(log_trans_mats.shape[0]))
    sticky_penalty = -50.0 * jnp.sum(jnp.log(jnp.diag(avg_trans_mat) + 1e-6))

    return -ll + sticky_penalty


@partial(jax.jit, static_argnames=("n_iter", "optimizer"))
def _train_hmm_loop(
    X: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
    opt_state: Any,
    n_iter: int,
    tol: float,
    optimizer: optax.GradientTransformation,
) -> tuple[dict[str, Any], Any, jnp.ndarray, bool]:
    """Compiled HMM training loop with unsupervised NLL."""
    def cond_fn(state):
        i, _, _, loss, prev_loss, converged = state
        return (i < n_iter) & (~converged)

    def body_fun(state):
        i, p, opt_s, loss, _, _ = state
        current_loss, grads = jax.value_and_grad(_compute_nll)(p, X, mask)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        diff = jnp.abs(current_loss - loss)
        converged = (i > 5) & (diff < tol)
        return i + 1, new_p, new_opt_s, current_loss, loss, converged

    initial_loss = _compute_nll(params, X, mask)
    init_val = (0, params, opt_state, initial_loss, initial_loss + 1e6, False)
    final_val = jax.lax.while_loop(cond_fn, body_fun, init_val)
    return final_val[1], final_val[2], final_val[3], final_val[5]


@jax.jit
def _jax_posterior_and_viterbi(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-filter posterior (softmax(log_alpha)) + Viterbi one-hot for sticky/duration only."""
    tvtp_z = observations[:, jnp.array(_TVTP_FEATURE_INDICES)]
    log_trans_mats = _compute_tvtp_matrices(tvtp_z, params["tvtp_W"], params["tvtp_b"])
    log_init = jax.nn.log_softmax(params["initial"])
    locs = params["locs"]
    scales = jnp.exp(params["log_scales"])
    dfs = jnp.exp(params["log_dfs"]) + 2.0
    log_alphas, _ = _forward_pass(
        observations, mask, log_init, log_trans_mats, locs, scales, dfs
    )
    posterior = jax.nn.softmax(log_alphas, axis=1)
    viterbi_onehot = _viterbi_decode(
        observations, mask, log_init, log_trans_mats, locs, scales, dfs
    )
    return posterior, viterbi_onehot


# --- Utility Functions ---

@njit(cache=True, fastmath=True)
def _numba_ema_2d(data: np.ndarray, span: int) -> np.ndarray:
    """Vectorized EMA for 2D arrays (time, features)."""
    alpha = 2.0 / (span + 1.0)
    n, m = data.shape
    out = np.empty((n, m), dtype=np.float64)
    if n == 0:
        return out
    out[0] = data[0]
    for i in range(1, n):
        for j in range(m):
            out[i, j] = data[i, j] * alpha + out[i - 1, j] * (1.0 - alpha)
    return out


@njit(cache=True, fastmath=True)
def _numba_sliding_mean_2d(data: np.ndarray, window: int) -> np.ndarray:
    """Fast sliding mean for 2D arrays."""
    n, m = data.shape
    out = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        for j in range(m):
            s = 0.0
            for k in range(start, i + 1):
                s += data[k, j]
            out[i, j] = s / count
    return out


@njit(cache=True, fastmath=True)
def _numba_sticky_labels(labels: np.ndarray, min_duration: int) -> np.ndarray:
    """Min-duration constraint to reduce regime flip noise."""
    n = len(labels)
    if n < 2 or min_duration <= 1:
        return labels
    result = labels.copy()
    i = 0
    while i < n:
        curr = result[i]
        j = i + 1
        while j < n and result[j] == curr:
            j += 1
        run_len = j - i
        if run_len < min_duration and i > 0:
            prev = result[i - 1]
            for k in range(i, j):
                result[k] = prev
        i = j
    return result


@njit(cache=True, fastmath=True)
def _numba_current_duration(hard_states: np.ndarray) -> np.ndarray:
    """Calculate run length of states."""
    n = len(hard_states)
    dur_arr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return dur_arr
    c = 1.0
    dur_arr[0] = c
    for i in range(1, n):
        if hard_states[i] == hard_states[i - 1]:
            c += 1.0
        else:
            c = 1.0
        dur_arr[i] = c
    return dur_arr


def _quantile_scaling(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Pure RobustScaler based scaling to preserve fat-tails."""
    rs = RobustScaler()
    # Fit and transform all features
    X_scaled = rs.fit_transform(X)
    transformers = {"rs": rs}
    # Relaxed clipping to preserve extreme tail events while maintaining numerical stability
    return np.clip(X_scaled, -15.0, 15.0), transformers


def _quantile_transform(X: np.ndarray, transformers: dict) -> np.ndarray:
    """Apply pre-fitted RobustScaler transformation."""
    rs = transformers["rs"]
    X_out = rs.transform(X)
    return cast(np.ndarray, np.clip(X_out, -15.0, 15.0))


@dataclass
class HMMStateInferrer:
    """Guided 4-State TVTP-HMM; Phase 4 bull-guidance rebalance; Phase 5 logit-offset crisis cap + sticky durations."""

    n_states: int = 4
    max_states: int = 4
    n_iter: int = 2000
    predict_step: int = 24
    fit_step: int = 168
    _warmed_up: bool = False
    _last_params: dict[str, Any] | None = None
    _last_opt: Any | None = None

    def _map_semantic_states(self, params: dict[str, Any]) -> dict[str, int]:
        """Map unsupervised HMM states to BULL, BEAR, CHOP, CRISIS based on learned means."""
        locs = np.asarray(params["locs"])
        # Indices based on SYSTEMIC_HMM_FEATURE_COLUMNS
        # 0: macro_trend_168h, 2: macro_vol_24h, 9: macro_ret_1h
        ret_means = locs[:, 9]
        
        # Sort indices by return mean (ascending)
        # Typically: CRISIS < BEAR < CHOP < BULL (in terms of returns)
        sorted_by_ret = np.argsort(ret_means)
        
        crisis_idx = sorted_by_ret[0]
        bear_idx = sorted_by_ret[1]
        chop_idx = sorted_by_ret[2]
        bull_idx = sorted_by_ret[3]
        
        return {
            "bull": int(bull_idx),
            "bear": int(bear_idx),
            "chop": int(chop_idx),
            "crisis": int(crisis_idx)
        }

    def _warmup_jit(self, n_feats: int) -> None:
        """Warm up JAX HMM JIT cache."""
        if self._warmed_up: return
        _logger.info("Warming up JAX HMM JIT cache (4-states, TVTP, Unsupervised, FEATS=%d)...", n_feats)
        dummy_obs, dummy_mask = jnp.zeros((MAX_HMM_WINDOW, n_feats)), jnp.zeros(MAX_HMM_WINDOW)
        params = {"initial": jnp.zeros(4), "tvtp_W": jnp.zeros((4, 4, _N_TVTP_FEATS)), "tvtp_b": jnp.eye(4) * 1.0, "locs": jnp.zeros((4, n_feats)), "log_scales": jnp.zeros((4, n_feats)), "log_dfs": jnp.array([1.0, 1.0, 1.0, 0.5])}
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(params)
        dummy_tvtp_z = dummy_obs[:, jnp.array(_TVTP_FEATURE_INDICES)]
        log_trans_mats = _compute_tvtp_matrices(dummy_tvtp_z, params["tvtp_W"], params["tvtp_b"])
        _ = _forward_pass(dummy_obs, dummy_mask, jax.nn.log_softmax(params["initial"]), log_trans_mats, params["locs"], jnp.exp(params["log_scales"]), jnp.exp(params["log_dfs"]) + 2.0)
        _ = _compute_nll(params, dummy_obs, dummy_mask)
        _, _, _, _ = _train_hmm_loop(
            dummy_obs, dummy_mask, params, opt_state, 1, 1e-4, optimizer
        )
        _ = _jax_posterior_and_viterbi(dummy_obs, dummy_mask, params)
        self._warmed_up = True

    def fit_predict_systemic(self, features_df: pd.DataFrame, returns_ser: pd.Series, is_end_idx: int, symbol: str = "Market", tf: str = "1h") -> pd.DataFrame:
        """Expanding-window Unsupervised 4-State TVTP-HMM with Post-Mapping."""
        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        for req_col in ["macro_trend_168h", "macro_vol_24h", "macro_downside_vol_24h"]:
            assert req_col in feat_cols, f"Required HMM feature missing: {req_col}"

        cm = CacheManager(FUTURES_CACHE_DIR, max_files=15, max_size_mb=1000.0)
        deps = {"symbol": symbol, "tf": tf, "data_len": len(features_df), "ver": "v50_unsupervised_post_map", "feat_cols": sorted(feat_cols)}
        tag = cm.generate_hash(deps, source_files=[Path(__file__).resolve()])
        cache_path = cm.get_cache_path(f"HMM_v50_unsupervised_{symbol}_{tf}_len{len(features_df)}", ".parquet", tag)
        if cache_path.exists():
            try: return pd.read_parquet(cache_path)
            except Exception: pass

        X_frame = features_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(X_frame)
        if n < 200: return self._zeros_semantic(features_df)
        num_feats = len(feat_cols)
        self._warmup_jit(num_feats)
        X_raw = X_frame.to_numpy(dtype=np.float64)
        
        out_cols = [*_SEMANTIC_ORDER, "hmm_entropy", "hmm_expected_duration", "hmm_current_duration", "hmm_hard_state"]
        results, params, transformers, optimizer = np.full((n, len(out_cols)), 0.0, dtype=np.float64), None, None, optax.adam(learning_rate=0.02)
        viterbi_states_buf = np.zeros(n, dtype=np.int32)
        
        # Dynamic Index Mapping for Initialization
        idx_trend = feat_cols.index("macro_trend_168h")
        idx_vol = feat_cols.index("macro_vol_24h")

        for t in range(500, n, self.predict_step):
            if t % self.fit_step == 0 or params is None:
                win_start, win_end = max(0, t - MAX_HMM_WINDOW), max(1, t - 1)
                X_win = X_raw[win_start:win_end]
                L = len(X_win)
                X_train_raw, transformers = _quantile_scaling(X_win)
                X_pad, M_pad = np.zeros((MAX_HMM_WINDOW, num_feats)), np.zeros(MAX_HMM_WINDOW)
                X_pad[:L], M_pad[:L] = X_train_raw, 1.0
                
                if self._last_params:
                    curr_p = self._last_params
                    curr_o = optimizer.init(curr_p)
                    iters = 400
                else:
                    # Generic initialization for 4 states to encourage separation
                    locs = np.zeros((4, num_feats), dtype=np.float32)
                    locs[0, idx_trend], locs[0, idx_vol] = 2.0, -1.0  # State 0: Bull-ish
                    locs[1, idx_trend], locs[1, idx_vol] = -1.0, 1.0  # State 1: Bear-ish
                    locs[2, idx_trend], locs[2, idx_vol] = 0.0, -2.0  # State 2: Chop-ish
                    locs[3, idx_trend], locs[3, idx_vol] = -4.0, 4.0  # State 3: Crisis-ish

                    W_init = np.zeros((4, 4, _N_TVTP_FEATS), dtype=np.float32)
                    _tvtp_b_init = np.eye(4, dtype=np.float32) * 5.0  # Strong self-persistence
                    
                    curr_p = {
                        "initial": jnp.zeros(4),
                        "tvtp_W": jnp.array(W_init),
                        "tvtp_b": jnp.array(_tvtp_b_init),
                        "locs": jnp.array(locs),
                        "log_scales": jnp.zeros((4, num_feats)),
                        "log_dfs": jnp.array([1.5, 1.5, 1.5, 1.0]),
                    }
                    curr_o, iters = optimizer.init(curr_p), self.n_iter
                
                params, self._last_opt, _, _ = _train_hmm_loop(
                    jnp.array(X_pad),
                    jnp.array(M_pad),
                    curr_p,
                    curr_o,
                    iters,
                    1e-4,
                    optimizer,
                )
                if self._last_params is not None:
                    params = jax.tree_util.tree_map(
                        lambda new, old: 0.8 * new + 0.2 * old, params, self._last_params
                    )
                self._last_params = params
            
            try:
                inf_start = max(0, t - MAX_HMM_WINDOW)
                write_start = max(0, t - self.predict_step)
                X_inf_raw = _quantile_transform(X_raw[inf_start:t], transformers)
                L_inf = len(X_inf_raw)
                X_inf_pad, M_inf_pad = np.zeros((MAX_HMM_WINDOW, num_feats)), np.zeros(MAX_HMM_WINDOW)
                X_inf_pad[:L_inf], M_inf_pad[:L_inf] = X_inf_raw, 1.0
                
                post_pad, vit_pad = _jax_posterior_and_viterbi(
                    jnp.array(X_inf_pad), jnp.array(M_inf_pad), params
                )
                
                # Perform post-mapping to semantic labels
                mapping = self._map_semantic_states(params)
                
                post_np = np.asarray(post_pad)[:L_inf]
                vit_np = np.asarray(vit_pad)[:L_inf]
                
                # Reorder posterior probabilities to match (BULL, BEAR, CHOP, CRISIS)
                mapped_post = np.zeros_like(post_np)
                mapped_post[:, 0] = post_np[:, mapping["bull"]]
                mapped_post[:, 1] = post_np[:, mapping["bear"]]
                mapped_post[:, 2] = post_np[:, mapping["chop"]]
                mapped_post[:, 3] = post_np[:, mapping["crisis"]]
                
                # Update hard state using the mapping as well
                raw_hard = np.argmax(vit_np, axis=1)
                inv_mapping = {v: k for k, v in mapping.items()}
                semantic_to_idx = {"bull": 0, "bear": 1, "chop": 2, "crisis": 3}
                mapped_hard = np.array([semantic_to_idx[inv_mapping[s]] for s in raw_hard])
                
                write_offset = write_start - inf_start
                combined_probs = mapped_post[write_offset:]
                viterbi_write = mapped_hard[write_offset:]
                
                viterbi_states_buf[write_start:t] = viterbi_write
                p_clip = np.clip(combined_probs, 1e-12, 1.0)
                entropy = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(4)
                results[write_start:t, :4], results[write_start:t, 4] = combined_probs, entropy
                
            except Exception as e: 
                _logger.error("HMM Inference failed: %s", e)

        probs_df = pd.DataFrame(results, index=features_df.index, columns=out_cols).ffill().bfill().fillna(0.0)

        sem_cal = _calibrate_crisis_logit_offset(
            probs_df[list(_SEMANTIC_ORDER)].to_numpy(dtype=np.float64), target_mean_crisis=0.07
        )
        probs_df[list(_SEMANTIC_ORDER)] = sem_cal
        probs_df["hmm_entropy"] = _normalized_entropy_k4(sem_cal)

        sticky_hard_states = _numba_sticky_labels(viterbi_states_buf.astype(np.int32), min_duration=24)
        probs_df["hmm_hard_state"] = sticky_hard_states.astype(np.float64)
        probs_df["hmm_current_duration"] = _numba_current_duration(sticky_hard_states)

        if params is not None and transformers is not None:
            X_scaled_full = _quantile_transform(X_raw, transformers)
            Z_full = jnp.asarray(X_scaled_full[:, list(_TVTP_FEATURE_INDICES)])
            log_tm_full = _compute_tvtp_matrices(Z_full, params["tvtp_W"], params["tvtp_b"])
            P_full = np.asarray(jnp.exp(log_tm_full))
            n_row = len(features_df)
            sh = sticky_hard_states.astype(np.int64)
            diag_clipped = np.clip(np.diagonal(P_full, axis1=1, axis2=2), 0.0, 0.95)
            probs_df["hmm_expected_duration"] = (1.0 / (1.0 - diag_clipped[np.arange(n_row), sh])).astype(np.float64)
        else:
            probs_df["hmm_expected_duration"] = 0.0

        out = probs_df.reset_index().rename(columns={probs_df.index.name or "index": "datetime"})
        try: out.to_parquet(cache_path)
        except Exception: pass
        return out

    def _zeros_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        u = 1.0 / float(len(_SEMANTIC_ORDER))
        cols = [*_SEMANTIC_ORDER, "hmm_entropy", "hmm_expected_duration", "hmm_current_duration", "hmm_hard_state"]
        out = pd.DataFrame(np.zeros((len(df), len(cols))), index=df.index, columns=cols)
        for c in _SEMANTIC_ORDER: out[c] = u
        return out.reset_index().rename(columns={out.index.name or "index": "datetime"})
