"""JAX-based Student-t HMM regime probabilities with SGD optimization and sticky priors."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
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
import talib
from numba import njit
from sklearn.preprocessing import RobustScaler

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_CACHE_DIR
from src.core.utils.cache_manager import CacheManager
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


def _calibrate_crisis_logit_offset(
    sem_probs: np.ndarray,
    target_mean_crisis: float = 0.07,
    delta_hi_max: float = 80.0,
    bisect_rounds: int = 48,
    crisis_col: int | None = None,
) -> np.ndarray:
    """Cap series-mean P(CRISIS) without global column scaling: CRISIS gets exp(-delta) vs others (logit shift).

    Keeps relative weights among non-crisis columns unchanged per row; avoids mean-scaling overcorrection that
    flips idxmax vs BEAR on strong-crisis bars.
    """
    p = np.asarray(sem_probs, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    cci = int(crisis_col) if crisis_col is not None else (p.shape[1] - 1)
    mean_c = float(np.mean(p[:, cci]))
    if mean_c <= target_mean_crisis + 1e-14:
        return p.copy()

    def apply_delta(delta: float) -> np.ndarray:
        w = np.ones_like(p)
        w[:, cci] = np.exp(-delta)
        q = p * w
        q /= q.sum(axis=1, keepdims=True)
        return q

    def crisis_mean(delta: float) -> float:
        q = apply_delta(delta)
        return float(np.mean(q[:, cci]))

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

# TVTP: macro_trend_168h(0), cs_dispersion(4), oi_delta(5), funding_mom(6), liq_proxy(7), lsr_delta(8).
# macro_ret_* observation dimensions (9–12) excluded from transition logits.
_TVTP_FEATURE_INDICES: tuple[int, ...] = (0, 4, 5, 6, 7, 8)
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
    sticky_weight = float(OPT_FUTURES_CONFIG.get("FUTURES_HMM_STICKY_PENALTY_WEIGHT", 100.0))
    sticky_penalty = -sticky_weight * jnp.sum(jnp.log(jnp.diag(avg_trans_mat) + 1e-6))

    # Soft Gravity NLL Penalty (v4.0.0 Pragmatic)
    # idx_ret = 9 (macro_ret_1h). Force CRISIS (4) negative, BULL_CALM (0) positive.
    gravity_penalty = 200.0 * (jnp.maximum(0.0, params["locs"][4, 9]) + jnp.maximum(0.0, -params["locs"][0, 9]))

    return -ll + sticky_penalty + gravity_penalty


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


def _empirical_mu_sigma_sharpe(hard: np.ndarray, rets: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _assign_semantic_five_state(emp_mu: np.ndarray, emp_sig: np.ndarray, sharpe: np.ndarray) -> dict[str, int]:
    """Map 5 latent states (v5 Sharpe-priority + μ crisis rule).

    - CRISIS: among states with μ<0 pick most negative μ; if none, global argmin(μ).
    - BULL_CALM: highest Sharpe among non-crisis states.
    - Remaining three: highest σ → BULL_VOL_UP; of the two low-σ, lower μ → BEAR, other → CHOP.
    """
    if len(emp_mu) != 5:
        raise ValueError("five-state semantic map requires K=5")

    sharpe_arr = np.asarray(sharpe, dtype=np.float64)
    emp_mu_arr = np.asarray(emp_mu, dtype=np.float64)

    neg_mask = emp_mu_arr < 0
    if np.any(neg_mask):
        crisis_idx = int(np.argmin(np.where(neg_mask, emp_mu_arr, np.inf)))
    else:
        crisis_idx = int(np.argmin(emp_mu_arr))

    other_idx = np.delete(np.arange(5), crisis_idx)
    bull_calm_idx = int(other_idx[np.argmax(sharpe_arr[other_idx])])

    remaining = np.delete(other_idx, np.where(other_idx == bull_calm_idx)[0])
    sig_order = remaining[np.argsort(emp_sig[remaining])]
    bull_vol_idx = int(sig_order[-1])
    low_vol = sig_order[:-1]
    mu_order = low_vol[np.argsort(emp_mu_arr[low_vol])]
    bear_idx = int(mu_order[0])
    chop_idx = int(mu_order[1])

    return {
        "bull_calm": bull_calm_idx,
        "bull_vol_up": bull_vol_idx,
        "bear": bear_idx,
        "chop": chop_idx,
        "crisis": crisis_idx,
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
    """5-state TVTP-HMM (Phase 5): CALM_BULL / VOL_UP / BEAR / CHOP / CRISIS + logit crisis cap + sticky durations."""

    n_states: int = 5
    max_states: int = 5
    n_iter: int = 2000
    predict_step: int = 24
    fit_step: int = 168
    _warmed_up: bool = False
    _last_params: dict[str, Any] | None = None
    _last_opt: Any | None = None
    _last_semantic_mapping: dict[str, int] | None = field(default=None, init=False, repr=False)

    def _map_semantic_states_locs_fallback(self, params: dict[str, Any], idx_macro_ret: int, idx_vol: int) -> dict[str, int]:
        """Fallback when empirical window is too short: v4.0.0 Pragmatic 2D logic.
        
        Uses locs[:, idx_vol] as sigma proxy and locs[:, idx_macro_ret] as mu proxy.
        """
        locs = np.asarray(params["locs"])
        ret_means = locs[:, idx_macro_ret]
        vol_locs = locs[:, idx_vol]
        
        # 1. Sort by vol locs (Sigma 2:3 split)
        sig_order = np.argsort(vol_locs)
        low_vol_group = sig_order[:3]   # Bottom 3
        high_vol_group = sig_order[3:]  # Top 2

        # 2. High Vol Group (2 states)
        hv_mu_order = high_vol_group[np.argsort(ret_means[high_vol_group])]
        crisis_idx = int(hv_mu_order[0])
        bull_vol_idx = int(hv_mu_order[1])

        # 3. Low Vol Group (3 states)
        lv_mu_order = low_vol_group[np.argsort(ret_means[low_vol_group])]
        bear_idx = int(lv_mu_order[0])
        chop_idx = int(lv_mu_order[1])
        bull_calm_idx = int(lv_mu_order[2])

        return {
            "bull_calm": bull_calm_idx,
            "bull_vol_up": bull_vol_idx,
            "bear": bear_idx,
            "chop": chop_idx,
            "crisis": crisis_idx,
        }

    def _map_semantic_states(
        self,
        params: dict[str, Any],
        returns_seg: np.ndarray,
        observations: jnp.ndarray,
        mask: jnp.ndarray,
        idx_macro_ret: int,
        idx_vol: int,
    ) -> dict[str, int]:
        """Viterbi + empirical μ/σ on returns; 5-way semantic assignment (Phase 5)."""
        T = int(observations.shape[0])
        rs = np.asarray(returns_seg, dtype=np.float64).reshape(-1)
        if T < 32 or rs.shape[0] != T:
            return self._map_semantic_states_locs_fallback(params, idx_macro_ret, idx_vol)
        try:
            v_one = _viterbi_hard_onehot(observations, mask, params)
            hard = np.argmax(np.asarray(v_one), axis=1).astype(np.int32)
        except Exception:
            return self._map_semantic_states_locs_fallback(params, idx_macro_ret, idx_vol)
        emp_mu, emp_sig, sharpe = _empirical_mu_sigma_sharpe(hard, rs, self.n_states)
        return _assign_semantic_five_state(emp_mu, emp_sig, sharpe)

    def _warmup_jit(self, n_feats: int) -> None:
        """Warm up JAX HMM JIT cache."""
        if self._warmed_up:
            return
        k = self.n_states
        _logger.info("Warming up JAX HMM JIT cache (K=%d states, TVTP, Unsupervised, FEATS=%d, v4.0.0 Pragmatic)...", k, n_feats)
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
        _ = _forward_pass(dummy_obs, dummy_mask, jax.nn.log_softmax(params["initial"]), log_trans_mats, params["locs"], jnp.exp(params["log_scales"]), jnp.exp(params["log_dfs"]) + 2.0)
        _ = _compute_nll(params, dummy_obs, dummy_mask)
        _, _, _, _ = _train_hmm_loop(
            dummy_obs, dummy_mask, params, opt_state, 10, 1e-4, optimizer
        )
        _ = _jax_posterior_and_viterbi(dummy_obs, dummy_mask, params)
        _ = _viterbi_hard_onehot(dummy_obs, dummy_mask, params)
        self._warmed_up = True

    def fit_predict_systemic(self, features_df: pd.DataFrame, returns_ser: pd.Series, is_end_idx: int, symbol: str = "Market", tf: str = "1h") -> pd.DataFrame:
        """Expanding-window unsupervised 5-state TVTP-HMM with empirical post-mapping."""
        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        for req_col in ["macro_trend_168h", "macro_vol_24h", "macro_downside_vol_24h"]:
            assert req_col in feat_cols, f"Required HMM feature missing: {req_col}"

        cm = CacheManager(FUTURES_CACHE_DIR, max_files=15, max_size_mb=1000.0)
        deps = {
            "symbol": symbol,
            "tf": tf,
            "data_len": len(features_df),
            "ver": "v55_smoothing_connected",
            "feat_cols": sorted(feat_cols),
            "smooth_method": OPT_FUTURES_CONFIG.get("FUTURES_HMM_SMOOTHING_METHOD", "EMA"),
            "smooth_span": int(OPT_FUTURES_CONFIG.get("FUTURES_HMM_SMOOTHING_SPAN", 8)),
            "sticky_durations": str(OPT_FUTURES_CONFIG.get("FUTURES_HMM_OUTPUT_STICKY_MIN_DURATION", [])),
            "sticky_weight": float(OPT_FUTURES_CONFIG.get("FUTURES_HMM_STICKY_PENALTY_WEIGHT", 100.0)),
        }
        tag = cm.generate_hash(deps, source_files=[Path(__file__).resolve()])
        cache_path = cm.get_cache_path(f"HMM_v53_unsupervised_{symbol}_{tf}_len{len(features_df)}", ".parquet", tag)
        if cache_path.exists():
            try: return pd.read_parquet(cache_path)
            except Exception: pass

        X_frame = features_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(X_frame)
        if n < 200: return self._zeros_semantic(features_df)
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

        out_cols = [*_SEMANTIC_ORDER, "hmm_entropy", "hmm_expected_duration", "hmm_current_duration", "hmm_hard_state"]
        results, params, transformers = np.full((n, len(out_cols)), 0.0, dtype=np.float64), None, None
        optimizer = optax.adamw(learning_rate=0.02, weight_decay=1e-4)
        viterbi_states_buf = np.zeros(n, dtype=np.int32)

        # Dynamic Index Mapping for Initialization
        idx_trend = feat_cols.index("macro_trend_168h")
        idx_vol = feat_cols.index("macro_vol_24h")
        idx_macro_ret = feat_cols.index("macro_ret_1h")
        k_states = self.n_states

        for t in range(500, n, self.predict_step):
            if t % self.fit_step == 0 or params is None:
                win_start, win_end = max(0, t - MAX_HMM_WINDOW), max(1, t - 1)
                X_win = X_raw[win_start:win_end]
                L = len(X_win)
                X_train_raw, transformers = _quantile_scaling(X_win)
                X_pad, M_pad = np.zeros((MAX_HMM_WINDOW, num_feats)), np.zeros(MAX_HMM_WINDOW)
                X_pad[:L], M_pad[:L] = X_train_raw, 1.0

                locs = np.zeros((k_states, num_feats), dtype=np.float32)
                locs[0, idx_trend], locs[0, idx_vol] = 2.0, -1.5  # calm bull
                locs[1, idx_trend], locs[1, idx_vol] = 1.5, 2.0  # vol-up bull
                locs[2, idx_trend], locs[2, idx_vol] = -1.0, 1.0  # bear
                locs[3, idx_trend], locs[3, idx_vol] = 0.0, -2.0  # chop
                locs[4, idx_trend], locs[4, idx_vol] = -4.0, 4.0  # crisis

                W_init = np.zeros((k_states, k_states, _N_TVTP_FEATS), dtype=np.float32)
                _tvtp_b_init = np.eye(k_states, dtype=np.float32) * 5.0  # Strong self-persistence

                new_initial: dict[str, Any] = {
                    "initial": jnp.zeros(k_states),
                    "tvtp_W": jnp.array(W_init),
                    "tvtp_b": jnp.array(_tvtp_b_init),
                    "locs": jnp.array(locs),
                    "log_scales": jnp.zeros((k_states, num_feats)),
                    "log_dfs": jnp.array([1.5] * (k_states - 1) + [1.0]) if k_states > 1 else jnp.array([1.0]),
                }

                if self._last_params is not None:
                    curr_p = jax.tree_util.tree_map(
                        lambda old, fresh: 0.8 * old + 0.2 * fresh,
                        self._last_params,
                        new_initial,
                    )
                    iters = 400
                else:
                    curr_p = new_initial
                    iters = self.n_iter

                curr_o = optimizer.init(curr_p)

                params, self._last_opt, _, _ = _train_hmm_loop(
                    jnp.array(X_pad),
                    jnp.array(M_pad),
                    curr_p,
                    curr_o,
                    iters,
                    1e-4,
                    optimizer,
                )
                self._last_params = params

                returns_win = rs[win_start:win_end]
                obs_tr = jnp.asarray(X_train_raw)
                mask_tr = jnp.ones((L,))
                self._last_semantic_mapping = self._map_semantic_states(
                    params, returns_win, obs_tr, mask_tr, idx_macro_ret, idx_vol
                )

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
                
                mapping = self._last_semantic_mapping
                if mapping is None:
                    mapping = self._map_semantic_states_locs_fallback(params, idx_macro_ret, idx_vol)
                
                post_np = np.asarray(post_pad)[:L_inf]
                vit_np = np.asarray(vit_pad)[:L_inf]
                
                mapped_post = np.zeros_like(post_np)
                mapped_post[:, 0] = post_np[:, mapping["bull_calm"]]
                mapped_post[:, 1] = post_np[:, mapping["bull_vol_up"]]
                mapped_post[:, 2] = post_np[:, mapping["bear"]]
                mapped_post[:, 3] = post_np[:, mapping["chop"]]
                mapped_post[:, 4] = post_np[:, mapping["crisis"]]
                
                raw_hard = np.argmax(vit_np, axis=1)
                inv_mapping = {v: k for k, v in mapping.items()}
                semantic_to_idx = {
                    "bull_calm": 0,
                    "bull_vol_up": 1,
                    "bear": 2,
                    "chop": 3,
                    "crisis": 4,
                }
                mapped_hard = np.array([semantic_to_idx[inv_mapping[int(s)]] for s in raw_hard])
                
                write_offset = write_start - inf_start
                combined_probs = mapped_post[write_offset:]
                viterbi_write = mapped_hard[write_offset:]
                
                viterbi_states_buf[write_start:t] = viterbi_write
                p_clip = np.clip(combined_probs, 1e-12, 1.0)
                entropy = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(float(len(_SEMANTIC_ORDER)))
                results[write_start:t, : len(_SEMANTIC_ORDER)], results[write_start:t, len(_SEMANTIC_ORDER)] = (
                    combined_probs,
                    entropy,
                )
                
            except Exception as e: 
                _logger.error("HMM Inference failed: %s", e)

        probs_df = pd.DataFrame(results, index=features_df.index, columns=out_cols).ffill().bfill().fillna(0.0)

        # [REFACTORED] Let HMM output its natural probability estimates
        # sem_cal = _calibrate_crisis_logit_offset(
        #     probs_df[list(_SEMANTIC_ORDER)].to_numpy(dtype=np.float64),
        #     target_mean_crisis=0.07,
        #     crisis_col=len(_SEMANTIC_ORDER) - 1,
        # )
        # probs_df[list(_SEMANTIC_ORDER)] = sem_cal

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
                    probs_df[c] = talib.WMA(2 * talib.WMA(probs_df[c].values, smooth_span // 2) - talib.WMA(probs_df[c].values, smooth_span), int(np.sqrt(smooth_span)))
            
            probs_df[sem_cols] = probs_df[sem_cols].ffill().fillna(1.0 / len(sem_cols))
            
            # Re-normalize to ensure sum to 1.0
            p_sum = probs_df[sem_cols].sum(axis=1)
            probs_df[sem_cols] = probs_df[sem_cols].div(p_sum, axis=0).fillna(1.0 / len(sem_cols))
            
            # Re-derive viterbi_states_buf from smoothed probabilities to enforce stability in hard states
            viterbi_states_buf = np.argmax(probs_df[sem_cols].to_numpy(), axis=1).astype(np.int32)

        probs_df["hmm_entropy"] = _normalized_entropy_k(probs_df[list(_SEMANTIC_ORDER)].values, len(_SEMANTIC_ORDER))
        probs_df["hmm_prob_bull_trend"] = (
            probs_df["hmm_prob_bull_calm"] + probs_df["hmm_prob_bull_vol_up"]
        )

        dur_cfg = OPT_FUTURES_CONFIG.get("FUTURES_HMM_OUTPUT_STICKY_MIN_DURATION", [36, 12, 2, 12, 1])
        dur_arr = np.array(dur_cfg, dtype=np.int32)
        sticky_hard_states = _numba_sticky_labels(viterbi_states_buf.astype(np.int32), dur_arr)
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
        for c in _SEMANTIC_ORDER:
            out[c] = u
        out["hmm_prob_bull_trend"] = out["hmm_prob_bull_calm"] + out["hmm_prob_bull_vol_up"]
        return out.reset_index().rename(columns={out.index.name or "index": "datetime"})
