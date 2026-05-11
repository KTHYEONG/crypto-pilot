"""JAX-based Student-t HMM regime probabilities with SGD optimization and sticky priors."""

from __future__ import annotations

import logging
import os
import warnings
import optax
import pandas as pd
import talib
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Tuple

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

from numba import njit
from config.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.ml_pipeline.features.engineering import (
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


# [v8.8.0] Removed _calibrate_crisis_logit_offset to restore mathematical integrity.

def _normalized_entropy_k(probs: np.ndarray, k: int) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return (-np.sum(p * np.log(p), axis=1) / np.log(float(k))).astype(np.float64)


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

# TVTP: macro_trend_168h(0), cs_dispersion(4), oi_delta(5), funding_mom(6),
# liq_proxy(7), lsr_delta(8).
# macro_ret_* observation dimensions (9-12) excluded from transition logits.
_TVTP_FEATURE_INDICES: tuple[int, ...] = (0, 4, 5, 6, 7, 8)
_N_TVTP_FEATS: int = len(_TVTP_FEATURE_INDICES)

# macro_vol_24h index in SYSTEMIC_HMM_FEATURE_COLUMNS matches locs[:, d] for ordering.
_HMM_LOC_IDX_VOL: int = 2

@jax.jit
def _batched_student_t_log_pdf(
    x: jnp.ndarray, locs: jnp.ndarray, scales: jnp.ndarray, dfs: jnp.ndarray
) -> jnp.ndarray:
    """Vectorized Multivariate Student-t log-pdf with Tiered Outcome Weighting (v8.6.0)."""
    # x: (D,), locs: (K, D), scales: (K, D), dfs: (K,)
    y = (x[None, :] - locs) / scales
    dfs_expanded = dfs[:, None]
    
    log_pdf_dims = (
        jax.scipy.special.gammaln((dfs_expanded + 1.0) / 2.0)
        - jax.scipy.special.gammaln(dfs_expanded / 2.0)
        - 0.5 * jnp.log(jnp.pi * dfs_expanded)
        - jnp.log(scales)
        - (dfs_expanded + 1.0) / 2.0 * jnp.log1p(y**2 / dfs_expanded)
    )
    
    return jnp.sum(log_pdf_dims, axis=1)


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

    _, (log_alphas, incremental_lls) = jax.lax.scan(
        scan_fn, log_alpha_0, (observations[1:], mask[1:], log_trans_mats[1:])
    )
    log_alphas = jnp.concatenate([log_alpha_0[None, :], log_alphas], axis=0)
    incremental_lls = jnp.concatenate([incremental_ll_0[None], incremental_lls], axis=0)
    return log_alphas, incremental_lls


def _forward_pass_incremental(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    log_alpha_init: jnp.ndarray,
    log_trans_mats: jnp.ndarray,
    locs: jnp.ndarray,
    scales: jnp.ndarray,
    dfs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Incremental forward pass starting from a preserved carry (log_alpha_init)."""
    def scan_fn(carry, inputs):
        log_alpha_prev, (obs, m, trans_mat) = carry, inputs
        log_pdf = _batched_student_t_log_pdf(obs, locs, scales, dfs)
        # Prediction step
        log_alpha_pred = jax.scipy.special.logsumexp(
            log_alpha_prev[:, None] + trans_mat, axis=0
        )
        # Update step
        log_alpha_curr_unnormalized = log_alpha_pred + log_pdf
        ll_curr = jax.scipy.special.logsumexp(log_alpha_curr_unnormalized)
        log_alpha_curr = log_alpha_curr_unnormalized - ll_curr
        
        # Mask handling
        log_alpha_out = jnp.where(m, log_alpha_curr, log_alpha_prev)
        ll_out = jnp.where(m, ll_curr, 0.0)
        return log_alpha_out, (log_alpha_out, ll_out)

    _, (log_alphas, incremental_lls) = jax.lax.scan(
        scan_fn, log_alpha_init, (observations, mask, log_trans_mats)
    )
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


@partial(jax.jit, static_argnames=("sticky_weight",))
def _compute_nll(
    params: dict[str, Any],
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    sticky_weight: float = 100.0,
) -> jnp.ndarray:
    """Pure Negative log-likelihood with TVTP and sticky prior."""
    tvtp_z = observations[:, jnp.array(_TVTP_FEATURE_INDICES)]
    log_trans_mats = _compute_tvtp_matrices(tvtp_z, params["tvtp_W"], params["tvtp_b"])
    _, incremental_lls = _forward_pass(
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
    avg_trans_mat = jnp.exp(
        jax.scipy.special.logsumexp(log_trans_mats, axis=0) - jnp.log(log_trans_mats.shape[0])
    )
    sticky_penalty = -sticky_weight * jnp.sum(jnp.log(jnp.diag(avg_trans_mat) + 1e-6))

    # [v8.8.0] Soft Bayesian Priors (L2) to enforce state identity
    # idx_ret = 9 (macro_ret_1h).
    mu = params["locs"][:, 9]
    n_obs = float(observations.shape[0])
    
    # MU Priors: Penalize deviations using L2 (square) when thresholds are crossed.
    mu_penalty = n_obs * (
        1000.0 * jnp.square(jnp.maximum(0.0, mu[4] + 0.002)) +    # CRISIS mu anchor: -0.002
        1000.0 * jnp.square(jnp.maximum(0.0, mu[2] + 0.0005)) +   # BEAR mu anchor: -0.0005
        1000.0 * jnp.square(jnp.maximum(0.0, 0.0005 - mu[0])) +   # BULL mu anchor: 0.0005
        500.0 * jnp.square(mu[3])                                 # CHOP centered at 0
    )

    # VOL Priors: Keep BULL (0) and CHOP (3) smaller than CRISIS (4)
    sig = jnp.exp(params["log_scales"][:, 9])
    vol_penalty = n_obs * (
        500.0 * jnp.square(jnp.maximum(0.0, sig[0] - sig[4])) +
        500.0 * jnp.square(jnp.maximum(0.0, sig[3] - sig[4]))
    )

    # State Separation Penalty: Prevent states from collapsing into the same location
    sep_penalty = n_obs * 5000.0 * (
        jnp.exp(-jnp.abs(mu[0] - mu[1]) * 1000.0) +
        jnp.exp(-jnp.abs(mu[2] - mu[4]) * 1000.0) +
        jnp.exp(-jnp.abs(mu[0] - mu[3]) * 1000.0)
    )

    # Transition Regularization: Penalty for excessive switching
    switching_penalty = 100.0 * (jnp.sum(avg_trans_mat) - jnp.sum(jnp.diag(avg_trans_mat)))

    return (
        -ll
        + sticky_penalty
        + mu_penalty
        + vol_penalty
        + sep_penalty
        + switching_penalty
    )


@partial(jax.jit, static_argnames=("n_iter", "optimizer", "sticky_weight"))
def _train_hmm_loop(
    X: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
    opt_state: Any,
    n_iter: int,
    tol: float,
    optimizer: optax.GradientTransformation,
    sticky_weight: float = 100.0,
) -> tuple[dict[str, Any], Any, jnp.ndarray, bool]:
    """Compiled HMM training loop with unsupervised NLL."""
    def cond_fn(state):
        i, _, _, _, _, converged = state
        return (i < n_iter) & (~converged)

    def body_fun(state):
        i, p, opt_s, loss, _, _ = state
        current_loss, grads = jax.value_and_grad(_compute_nll)(p, X, mask, sticky_weight)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        diff = jnp.abs(current_loss - loss)
        converged = (i > 5) & (diff < tol)
        return i + 1, new_p, new_opt_s, current_loss, loss, converged

    initial_loss = _compute_nll(params, X, mask, sticky_weight)
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


@jax.jit
def _jax_posterior_only(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> jnp.ndarray:
    """Forward-filter posterior only (softmax(log_alpha)) for faster inference."""
    tvtp_z = observations[:, jnp.array(_TVTP_FEATURE_INDICES)]
    log_trans_mats = _compute_tvtp_matrices(tvtp_z, params["tvtp_W"], params["tvtp_b"])
    log_init = jax.nn.log_softmax(params["initial"])
    locs = params["locs"]
    scales = jnp.exp(params["log_scales"])
    dfs = jnp.exp(params["log_dfs"]) + 2.0
    log_alphas, _ = _forward_pass(
        observations, mask, log_init, log_trans_mats, locs, scales, dfs
    )
    return jax.nn.softmax(log_alphas, axis=1)


@jax.jit
def _jax_posterior_incremental(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    log_alpha_init: jnp.ndarray,
    params: dict[str, Any],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Incremental posterior + returns the new carry for next step."""
    tvtp_z = observations[:, jnp.array(_TVTP_FEATURE_INDICES)]
    log_trans_mats = _compute_tvtp_matrices(tvtp_z, params["tvtp_W"], params["tvtp_b"])
    locs = params["locs"]
    scales = jnp.exp(params["log_scales"])
    dfs = jnp.exp(params["log_dfs"]) + 2.0
    
    log_alphas, _ = _forward_pass_incremental(
        observations, mask, log_alpha_init, log_trans_mats, locs, scales, dfs
    )
    return jax.nn.softmax(log_alphas, axis=1), log_alphas[-1]


@jax.jit
def _jax_posterior_full_with_carry(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Full forward pass that also returns the final log_alpha for carry-over."""
    tvtp_z = observations[:, jnp.array(_TVTP_FEATURE_INDICES)]
    log_trans_mats = _compute_tvtp_matrices(tvtp_z, params["tvtp_W"], params["tvtp_b"])
    log_init = jax.nn.log_softmax(params["initial"])
    locs = params["locs"]
    scales = jnp.exp(params["log_scales"])
    dfs = jnp.exp(params["log_dfs"]) + 2.0
    log_alphas, _ = _forward_pass(
        observations, mask, log_init, log_trans_mats, locs, scales, dfs
    )
    return jax.nn.softmax(log_alphas, axis=1), log_alphas[-1]


@jax.jit
def _viterbi_hard_onehot(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> jnp.ndarray:
    """Viterbi one-hot path only (for empirical semantic mapping on a segment)."""
    tvtp_z = observations[:, jnp.array(_TVTP_FEATURE_INDICES)]
    log_trans_mats = _compute_tvtp_matrices(tvtp_z, params["tvtp_W"], params["tvtp_b"])
    log_init = jax.nn.log_softmax(params["initial"])
    locs = params["locs"]
    scales = jnp.exp(params["log_scales"])
    dfs = jnp.exp(params["log_dfs"]) + 2.0
    return _viterbi_decode(
        observations, mask, log_init, log_trans_mats, locs, scales, dfs
    )


def _empirical_mu_sigma_sharpe(
    hard: np.ndarray, rets: np.ndarray, n_states: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emp_mu = np.zeros(n_states, dtype=np.float64)
    emp_sig = np.ones(n_states, dtype=np.float64)
    for k in range(n_states):
        m = hard == k
        if np.any(m):
            seg = rets[m].astype(np.float64)
            emp_mu[k] = float(np.mean(seg))
            emp_sig[k] = float(np.std(seg)) + 1e-12
        else:
            emp_mu[k] = 0.0
            emp_sig[k] = 1.0
    sharpe = emp_mu / emp_sig
    return emp_mu, emp_sig, sharpe


def _assign_semantic_five_state(
    emp_mu: np.ndarray, emp_sig: np.ndarray, sharpe: np.ndarray
) -> dict[str, int]:
    """v8.8.0: Static Semantic Mapping.
    Priors in _compute_nll now enforce state identity.
    """
    return {
        "bull_calm": 0,
        "bull_vol_up": 1,
        "bear": 2,
        "chop": 3,
        "crisis": 4,
    }


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
def _numba_sticky_labels(labels: np.ndarray, min_durations: np.ndarray) -> np.ndarray:
    """Asymmetric min-duration constraint to reduce regime flip noise (Phase 6)."""
    n = len(labels)
    if n < 2:
        return labels
    result = labels.copy()
    i = 0
    while i < n:
        curr = result[i]
        j = i + 1
        while j < n and result[j] == curr:
            j += 1
        run_len = j - i
        
        # Asymmetric lookup: different persistence required per state semantic
        m_dur = min_durations[int(curr)]
        
        if run_len < m_dur and i > 0:
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


def _fast_robust_scale(X: np.ndarray, non_ret_idx: list[int]) -> tuple[np.ndarray, dict]:
    """Fast NumPy-based RobustScaler implementation."""
    X_out = X.copy()
    tf = {}
    if non_ret_idx:
        sub = X[:, non_ret_idx]
        med = np.nanmedian(sub, axis=0)
        q75 = np.nanpercentile(sub, 75, axis=0)
        q25 = np.nanpercentile(sub, 25, axis=0)
        iqr = q75 - q25
        iqr = np.where(np.abs(iqr) < 1e-9, 1.0, iqr)
        X_out[:, non_ret_idx] = (sub - med) / iqr
        tf = {"med": med, "iqr": iqr, "non_ret_idx": non_ret_idx}
    return np.clip(X_out, -15.0, 15.0), tf


def _fast_robust_transform(X: np.ndarray, tf: dict) -> np.ndarray:
    """Apply fast RobustScaler transformation."""
    X_out = X.copy()
    if "non_ret_idx" in tf:
        idx = tf["non_ret_idx"]
        X_out[:, idx] = (X[:, idx] - tf["med"]) / tf["iqr"]
    return np.clip(X_out, -15.0, 15.0)


@dataclass
class HMMStateInferrer:
    """5-state TVTP-HMM (Phase 5): CALM_BULL / VOL_UP / BEAR / CHOP / CRISIS.
    Includes logit crisis cap and sticky durations.
    """

    n_states: int = 5
    max_states: int = 5
    n_iter: int = 2000
    tol: float = 1e-4
    predict_step: int = 24
    fit_step: int = 168
    _warmed_up: bool = False
    _last_params: dict[str, Any] | None = None
    _last_opt: Any | None = None
    _last_semantic_mapping: dict[str, int] | None = field(default=None, init=False, repr=False)
    _last_log_alpha: jnp.ndarray | None = field(default=None, init=False, repr=False)

    def _map_semantic_states_locs_fallback(
        self, params: dict[str, Any], idx_macro_ret: int, idx_vol: int
    ) -> dict[str, int]:
        """Fallback: Sort by locs (mu and sigma)."""
        locs = np.asarray(params["locs"])
        mu = locs[:, idx_macro_ret]
        sig = np.exp(np.asarray(params["log_scales"])[:, idx_macro_ret])
        sharpe = mu / sig
        return _assign_semantic_five_state(mu, sig, sharpe)

    def _map_semantic_states(
        self,
        params: dict[str, Any],
        returns_seg: np.ndarray,
        observations: jnp.ndarray,
        mask: jnp.ndarray,
        idx_macro_ret: int,
        idx_vol: int,
    ) -> dict[str, int]:
        """Viterbi + empirical mu/sigma on returns."""
        T = int(observations.shape[0])
        rs = np.asarray(returns_seg, dtype=np.float64).reshape(-1)
        if T < 32 or rs.shape[0] != T:
            return self._map_semantic_states_locs_fallback(params, idx_macro_ret, idx_vol)
        try:
            v_one = _viterbi_hard_onehot(observations, mask, params)
            hard = np.argmax(np.asarray(v_one), axis=1).astype(np.int32)
            emp_mu, emp_sig, sharpe = _empirical_mu_sigma_sharpe(hard, rs, self.n_states)
            return _assign_semantic_five_state(emp_mu, emp_sig, sharpe)
        except Exception:
            return self._map_semantic_states_locs_fallback(params, idx_macro_ret, idx_vol)

    def _warmup_jit(self, n_feats: int) -> None:
        """Warm up JAX HMM JIT cache."""
        if self._warmed_up:
            return
        k = self.n_states
        _logger.info(
            "Warming up JAX HMM JIT cache (K=%d states, TVTP, Unsupervised, FEATS=%d)...",
            k,
            n_feats,
        )
        dummy_obs, dummy_mask = jnp.zeros((MAX_HMM_WINDOW, n_feats)), jnp.zeros(MAX_HMM_WINDOW)
        params = {
            "initial": jnp.zeros(k),
            "tvtp_W": jnp.zeros((k, k, _N_TVTP_FEATS)),
            "tvtp_b": jnp.eye(k) * 1.0,
            "locs": jnp.zeros((k, n_feats)),
            "log_scales": jnp.zeros((k, n_feats)),
            "log_dfs": jnp.array([1.0] * (k - 1) + [0.5]) if k >= 1 else jnp.array([0.5]),
        }
        optimizer = optax.adamw(learning_rate=0.01, weight_decay=1e-4)
        opt_state = optimizer.init(params)
        dummy_tvtp_z = dummy_obs[:, jnp.array(_TVTP_FEATURE_INDICES)]
        log_trans_mats = _compute_tvtp_matrices(dummy_tvtp_z, params["tvtp_W"], params["tvtp_b"])
        _ = _forward_pass(
            dummy_obs,
            dummy_mask,
            jax.nn.log_softmax(params["initial"]),
            log_trans_mats,
            params["locs"],
            jnp.exp(params["log_scales"]),
            jnp.exp(params["log_dfs"]) + 2.0,
        )
        _ = _compute_nll(params, dummy_obs, dummy_mask, 100.0)
        _, _, _, _ = _train_hmm_loop(
            dummy_obs, dummy_mask, params, opt_state, 10, 1e-4, optimizer, 100.0
        )
        _ = _jax_posterior_and_viterbi(dummy_obs, dummy_mask, params)
        _ = _jax_posterior_only(dummy_obs, dummy_mask, params)
        _ = _jax_posterior_full_with_carry(dummy_obs, dummy_mask, params)
        _ = _jax_posterior_incremental(
            jnp.zeros((self.predict_step, n_feats)), 
            jnp.zeros(self.predict_step), 
            jnp.zeros(k), 
            params
        )
        _ = _viterbi_hard_onehot(dummy_obs, dummy_mask, params)
        self._warmed_up = True

    def fit_predict_systemic(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
        symbol: str = "Market",
        tf: str = "1h",
    ) -> pd.DataFrame:
        """Expanding-window unsupervised 5-state TVTP-HMM with empirical post-mapping."""
        # [v8.8.0] Dual-TF HMM: Resample 1h or faster to 4h for training stability
        is_dual_tf = tf in ["1h", "15m", "5m", "1m"]
        orig_features_df = features_df
        
        if is_dual_tf:
            _logger.info("HMM Dual-TF Mode: Resampling %s to 4h for training.", tf)
            features_df = features_df.resample("4h").last().ffill()
            # Pct returns resampled as cumulative
            returns_ser = returns_ser.resample("4h").apply(lambda x: (1 + x).prod() - 1)

        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        for req_col in ["macro_trend_168h", "macro_vol_24h", "macro_downside_vol_24h"]:
            assert req_col in feat_cols, f"Required HMM feature missing: {req_col}"

        X_frame = features_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(X_frame)
        if n < 200:
            return self._zeros_semantic(features_df)
        num_feats = len(feat_cols)
        self._warmup_jit(num_feats)
        
        sticky_weight = float(OPT_FUTURES_CONFIG.get("FUTURES_HMM_STICKY_PENALTY_WEIGHT", 100.0))
        dur_cfg = OPT_FUTURES_CONFIG.get("FUTURES_HMM_OUTPUT_STICKY_MIN_DURATION", [36, 12, 2, 12, 1])
        _logger.info("HMM Friction Reduction: weight=%.1f, durations=%s", sticky_weight, dur_cfg)

        X_raw = X_frame.to_numpy(dtype=np.float64)
        rs = np.nan_to_num(
            np.asarray(returns_ser.reindex(features_df.index), dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        out_cols = [
            *_SEMANTIC_ORDER,
            "hmm_entropy",
            "hmm_expected_duration",
            "hmm_current_duration",
            "hmm_hard_state",
        ]
        results = np.full((n, len(out_cols)), 0.0, dtype=np.float64)
        params, transformers = None, None
        optimizer = optax.adamw(learning_rate=0.02, weight_decay=1e-4)
        viterbi_states_buf = np.zeros(n, dtype=np.int32)

        # Dynamic Index Mapping for Initialization
        idx_trend = feat_cols.index("macro_trend_168h")
        idx_vol = feat_cols.index("macro_vol_24h")
        idx_macro_ret = feat_cols.index("macro_ret_1h")
        k_states = self.n_states

        # Pre-allocate inference buffers (Quick Win)
        X_inf_pad = np.zeros((MAX_HMM_WINDOW, num_feats), dtype=np.float32)
        M_inf_pad = np.zeros(MAX_HMM_WINDOW, dtype=np.float32)
        L_inf = 0  # Track actual content length in buffer

        for t in range(500, n, self.predict_step):
            inf_start = max(0, t - MAX_HMM_WINDOW)
            write_start = max(0, t - self.predict_step)

            if t % self.fit_step == 0 or params is None:
                win_start, win_end = max(0, t - MAX_HMM_WINDOW), max(1, t - 1)
                X_win = X_raw[win_start:win_end]
                L = len(X_win)
                # Quick Win: Fast Robust Scaling
                non_ret_idx = [i for i in range(num_feats) if i != 9]
                X_train_raw, transformers = _fast_robust_scale(X_win, non_ret_idx)
                
                X_pad, M_pad = np.zeros((MAX_HMM_WINDOW, num_feats)), np.zeros(MAX_HMM_WINDOW)
                X_pad[:L], M_pad[:L] = X_train_raw, 1.0

                locs = np.zeros((k_states, num_feats), dtype=np.float32)
                
                # Directional Bias Initialization (v5.0.0 Percentile-Based Anchors)
                ret_data = X_train_raw[:L, idx_macro_ret]
                vol_data = X_train_raw[:L, idx_vol]
                
                # Semantic Mu Anchors
                mu_exlow = np.percentile(ret_data, 5)
                mu_low = np.percentile(ret_data, 15)
                mu_high = np.percentile(ret_data, 85)
                mu_high = np.percentile(ret_data, 85)
                mu_vhigh = np.percentile(ret_data, 95)
                
                # Semantic Vol Anchors
                vol_low = np.percentile(vol_data, 20)
                vol_mid = np.percentile(vol_data, 50)
                vol_high = np.percentile(vol_data, 80)
                vol_vhigh = np.percentile(vol_data, 95)

                # BULL_CALM (0): High mu, Low sigma
                locs[0, idx_macro_ret], locs[0, idx_vol] = max(mu_high, 0.001), vol_low
                # BULL_VOL_UP (1): Very high mu, High sigma
                locs[1, idx_macro_ret], locs[1, idx_vol] = max(mu_vhigh, 0.002), max(vol_high, 0.02)
                # BEAR_TREND (2): Low mu, Mid sigma
                locs[2, idx_macro_ret], locs[2, idx_vol] = min(mu_low, -0.001), vol_mid
                # CHOP (3): Zero mu, Low sigma
                locs[3, idx_macro_ret], locs[3, idx_vol] = 0.0, min(vol_low, 0.01)
                # CRISIS (4): Extreme low mu, Very high sigma
                locs[4, idx_macro_ret] = min(mu_exlow, -0.005)
                locs[4, idx_vol] = max(vol_vhigh, 0.04)

                # Seed trend features for additional stability
                locs[0, idx_trend] = 2.0
                locs[1, idx_trend] = 1.5
                locs[2, idx_trend] = -1.0
                locs[3, idx_trend] = 0.0
                locs[4, idx_trend] = -4.0

                W_init = np.zeros((k_states, k_states, _N_TVTP_FEATS), dtype=np.float32)
                # v8.4.0: High persistence bias
                _tvtp_b_init = np.eye(k_states, dtype=np.float32) * 15.0

                new_initial: dict[str, Any] = {
                    "initial": jnp.zeros(k_states),
                    "tvtp_W": jnp.array(W_init),
                    "tvtp_b": jnp.array(_tvtp_b_init),
                    "locs": jnp.array(locs),
                    "log_scales": jnp.zeros((k_states, num_feats)),
                    "log_dfs": jnp.array([1.5] * (k_states - 1) + [1.0]) if k_states > 1 else jnp.array(
                        [1.0]
                    ),
                }

                if self._last_params is not None and self._last_opt is not None:
                    curr_p = jax.tree_util.tree_map(
                        lambda old, fresh: 0.8 * old + 0.2 * fresh,
                        self._last_params,
                        new_initial,
                    )
                    curr_o = self._last_opt
                    iters = 400
                else:
                    curr_p = new_initial
                    curr_o = optimizer.init(curr_p)
                    iters = self.n_iter

                params, self._last_opt, _, _ = _train_hmm_loop(
                    jnp.array(X_pad),
                    jnp.array(M_pad),
                    curr_p,
                    curr_o,
                    iters,
                    self.tol,
                    optimizer,
                    sticky_weight,
                )
                self._last_params = params

                returns_win = rs[win_start:win_end]
                obs_tr = jnp.asarray(X_train_raw)
                mask_tr = jnp.ones((L,))
                # Cache semantic mapping
                self._last_semantic_mapping = self._map_semantic_states(
                    params, returns_win, obs_tr, mask_tr, idx_macro_ret, idx_vol
                )

                # Full refresh of inference buffer after re-fitting
                X_inf_raw_full = _fast_robust_transform(X_raw[inf_start:t], transformers)
                L_inf = len(X_inf_raw_full)
                X_inf_pad.fill(0.0)
                X_inf_pad[:L_inf] = X_inf_raw_full.astype(np.float32)
                M_inf_pad.fill(0.0)
                M_inf_pad[:L_inf] = 1.0

                # [Institutional Quant] Perform full forward to establish new carry
                _, self._last_log_alpha = _jax_posterior_full_with_carry(
                    jnp.array(X_inf_pad), jnp.array(M_inf_pad), params
                )

            else:
                # [Institutional Quant] Incremental Feature Scaling
                # Only transform the NEW bars and append to buffer
                new_bars_raw = X_raw[t - self.predict_step : t]
                new_bars_scaled = _fast_robust_transform(
                    new_bars_raw, transformers
                ).astype(np.float32)
                
                # Update X_inf_pad for consistency, but we won't use the whole thing for forward pass
                if L_inf < MAX_HMM_WINDOW:
                    space_left = MAX_HMM_WINDOW - L_inf
                    to_add = min(len(new_bars_scaled), space_left)
                    X_inf_pad[L_inf : L_inf + to_add] = new_bars_scaled[:to_add]
                    L_inf += to_add
                    M_inf_pad[:L_inf] = 1.0
                else:
                    X_inf_pad = np.roll(X_inf_pad, -self.predict_step, axis=0)
                    X_inf_pad[-self.predict_step:] = new_bars_scaled

            try:
                # [Institutional Quant] High-Efficiency Forward Pass
                if self._last_log_alpha is not None:
                    # Incremental mode: Only process the latest bars
                    new_bars_scaled = X_inf_pad[L_inf - self.predict_step : L_inf]
                    new_mask = M_inf_pad[L_inf - self.predict_step : L_inf]
                    
                    post_incremental, self._last_log_alpha = _jax_posterior_incremental(
                        jnp.array(new_bars_scaled),
                        jnp.array(new_mask),
                        self._last_log_alpha,
                        params,
                    )
                    # We only need the latest bars' posterior
                    post_np = np.asarray(post_incremental)
                else:
                    # Fallback or initialization: Full forward
                    post_pad, self._last_log_alpha = _jax_posterior_full_with_carry(
                        jnp.array(X_inf_pad), jnp.array(M_inf_pad), params
                    )
                    post_np = np.asarray(post_pad)[:L_inf]
                    post_np = post_np[-self.predict_step:] # Align with write_start
                
                # [v8.7.1] Use cached mapping when params are stable
                mapping = (
                    self._last_semantic_mapping
                    or self._map_semantic_states_locs_fallback(params, idx_macro_ret, idx_vol)
                )
                
                # Quick Win: argmax from posterior instead of Viterbi
                raw_hard = np.argmax(post_np, axis=1)
                
                mapped_post = np.zeros_like(post_np)
                mapped_post[:, 0] = post_np[:, mapping["bull_calm"]]
                mapped_post[:, 1] = post_np[:, mapping["bull_vol_up"]]
                mapped_post[:, 2] = post_np[:, mapping["bear"]]
                mapped_post[:, 3] = post_np[:, mapping["chop"]]
                mapped_post[:, 4] = post_np[:, mapping["crisis"]]
                
                inv_mapping = {v: k for k, v in mapping.items()}
                semantic_to_idx = {
                    "bull_calm": 0,
                    "bull_vol_up": 1,
                    "bear": 2,
                    "chop": 3,
                    "crisis": 4,
                }
                mapped_hard = np.array([semantic_to_idx[inv_mapping[int(s)]] for s in raw_hard])
                
                combined_probs = mapped_post
                viterbi_write = mapped_hard
                
                viterbi_states_buf[write_start:t] = viterbi_write
                p_clip = np.clip(combined_probs, 1e-12, 1.0)
                entropy = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(
                    float(len(_SEMANTIC_ORDER))
                )
                results[write_start:t, : len(_SEMANTIC_ORDER)] = combined_probs
                results[write_start:t, len(_SEMANTIC_ORDER)] = entropy
                
            except Exception as e: 
                _logger.error("HMM Inference failed: %s", e)

        probs_df = pd.DataFrame(
            results, index=features_df.index, columns=out_cols
        ).ffill().bfill().fillna(0.0)

        # Posterior Smoothing (Phase 7: Macro Stability)
        smooth_method = OPT_FUTURES_CONFIG.get("FUTURES_HMM_SMOOTHING_METHOD", "EMA")
        smooth_span = int(OPT_FUTURES_CONFIG.get("FUTURES_HMM_SMOOTHING_SPAN", 8))
        if smooth_span > 1:
            sem_cols = list(_SEMANTIC_ORDER)
            if smooth_method == "EMA":
                probs_df[sem_cols] = probs_df[sem_cols].ewm(span=smooth_span).mean()
            elif smooth_method == "KAMA":
                for c in sem_cols:
                    probs_df[c] = talib.KAMA(probs_df[c].values, timeperiod=smooth_span)
            elif smooth_method == "DEMA":
                for c in sem_cols:
                    probs_df[c] = talib.DEMA(probs_df[c].values, timeperiod=smooth_span)
            elif smooth_method == "HMA":
                for c in sem_cols:
                    probs_df[c] = talib.WMA(
                    2 * talib.WMA(probs_df[c].values, smooth_span // 2)
                    - talib.WMA(probs_df[c].values, smooth_span),
                    int(np.sqrt(smooth_span)),
                )
            
            probs_df[sem_cols] = probs_df[sem_cols].ffill().fillna(1.0 / len(sem_cols))
            
            # Re-normalize to ensure sum to 1.0
            p_sum = probs_df[sem_cols].sum(axis=1)
            probs_df[sem_cols] = probs_df[sem_cols].div(p_sum, axis=0).fillna(1.0 / len(sem_cols))
            
            # Re-derive viterbi_states_buf from smoothed probabilities
            viterbi_states_buf = np.argmax(probs_df[sem_cols].to_numpy(), axis=1).astype(np.int32)

        probs_df["hmm_entropy"] = _normalized_entropy_k(
            probs_df[list(_SEMANTIC_ORDER)].values, len(_SEMANTIC_ORDER)
        )
        probs_df["hmm_prob_bull_trend"] = (
            probs_df["hmm_prob_bull_calm"] + probs_df["hmm_prob_bull_vol_up"]
        )

        dur_cfg = OPT_FUTURES_CONFIG.get("FUTURES_HMM_OUTPUT_STICKY_MIN_DURATION", [36, 12, 2, 12, 1])
        dur_arr = np.array(dur_cfg, dtype=np.int32)
        sticky_hard_states = _numba_sticky_labels(viterbi_states_buf.astype(np.int32), dur_arr)
        probs_df["hmm_hard_state"] = sticky_hard_states.astype(np.float64)
        probs_df["hmm_current_duration"] = _numba_current_duration(sticky_hard_states)

        if params is not None and transformers is not None:
            X_scaled_full = _fast_robust_transform(X_raw, transformers)
            Z_full = jnp.asarray(X_scaled_full[:, list(_TVTP_FEATURE_INDICES)])
            log_tm_full = _compute_tvtp_matrices(Z_full, params["tvtp_W"], params["tvtp_b"])
            P_full = np.asarray(jnp.exp(log_tm_full))
            n_row = len(features_df)
            sh = sticky_hard_states.astype(np.int64)
            diag_clipped = np.clip(np.diagonal(P_full, axis1=1, axis2=2), 0.0, 0.95)
            probs_df["hmm_expected_duration"] = (
                1.0 / (1.0 - diag_clipped[np.arange(n_row), sh])
            ).astype(np.float64)
        else:
            probs_df["hmm_expected_duration"] = 0.0

        if is_dual_tf:
            # [v8.8.0] Reindex 4h results back to original index and forward-fill
            probs_df = probs_df.reindex(orig_features_df.index).ffill().bfill()

        out = probs_df.reset_index().rename(columns={probs_df.index.name or "index": "datetime"})
        return out

    def _zeros_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        u = 1.0 / float(len(_SEMANTIC_ORDER))
        cols = [
            *_SEMANTIC_ORDER,
            "hmm_entropy",
            "hmm_expected_duration",
            "hmm_current_duration",
            "hmm_hard_state",
        ]
        out = pd.DataFrame(np.zeros((len(df), len(cols))), index=df.index, columns=cols)
        for c in _SEMANTIC_ORDER:
            out[c] = u
        out["hmm_prob_bull_trend"] = out["hmm_prob_bull_calm"] + out["hmm_prob_bull_vol_up"]
        return out.reset_index().rename(columns={out.index.name or "index": "datetime"})
