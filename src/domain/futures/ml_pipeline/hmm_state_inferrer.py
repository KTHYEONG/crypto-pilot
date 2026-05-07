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


@jax.jit
def _multivariate_student_t_log_pdf(
    x: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray, df: jnp.ndarray
) -> jnp.ndarray:
    """Diagonal Multivariate Student-t log-pdf (sum of univariate)."""
    return jnp.sum(_student_t_log_pdf(x, loc, scale, df))

# --- JAX HMM Core Functions ---

MAX_HMM_WINDOW = 1500

# Indices of features used as exogenous drivers for TVTP (subset of SYSTEMIC_HMM_FEATURE_COLUMNS)
# Uses: cs_dispersion(4), oi_delta(5), funding_mom(6), liq_proxy(7), lsr_delta(8)
_TVTP_FEATURE_INDICES: tuple[int, ...] = (4, 5, 6, 7, 8)
_N_TVTP_FEATS: int = len(_TVTP_FEATURE_INDICES)

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
    guidance_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Negative Log-Likelihood with TVTP, Sticky Prior, and Guidance."""
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
    T = jnp.sum(mask) + 1e-9

    # LL-adaptive scale: penalties proportional to |ll|/T so they don't dominate
    ll_scale = jnp.abs(ll) / T

    avg_trans_mat = jnp.exp(jax.scipy.special.logsumexp(log_trans_mats, axis=0) - jnp.log(log_trans_mats.shape[0]))
    sticky_penalty = -30.0 * jnp.sum(jnp.log(jnp.diag(avg_trans_mat) + 1e-6))

    locs = params["locs"]
    # Semantic Penalties: Bull(0) > Chop(2) > Bear(1) > Crisis(3)
    # Use squared penalty for stronger enforcement when violated
    p_02 = jnp.square(jnp.maximum(0.0, locs[2, 0] - locs[0, 0] + 1.5))
    p_21 = jnp.square(jnp.maximum(0.0, locs[1, 0] - locs[2, 0] + 1.5))
    p_13 = jnp.square(jnp.maximum(0.0, locs[3, 0] - locs[1, 0] + 1.5))
    p_crisis_val = jnp.square(jnp.maximum(0.0, locs[3, 0] + 3.0)) # Force deeper crisis MU (Phase 4.1)

    # Volatility Penalties: Crisis > Bear > Others
    p_crisis_vol = jnp.square(jnp.maximum(0.0, jnp.max(locs[:3, 2]) - locs[3, 2] + 2.0))

    # Adaptive: 10000x ll_scale — extremely strong directional ordering enforcement
    semantic_penalty = 10000.0 * ll_scale * (p_02 + p_21 + p_13 + p_crisis_val + p_crisis_vol)

    posteriors = jax.nn.softmax(log_alphas, axis=1)
    # Adaptive: 10000x ll_scale for guidance (Phase 4 Final Push - Double strength)
    guidance_loss = 10000.0 * ll_scale * jnp.mean(mask * (posteriors[:, 3] - guidance_mask)**2)

    # State Frequency Penalty: CRISIS (state 3) should be < 20%
    crisis_freq = jnp.mean(posteriors[:, 3] * mask) / (jnp.mean(mask) + 1e-9)
    # Adaptive: 10x ll_scale
    freq_penalty = 10.0 * ll_scale * jnp.maximum(0.0, crisis_freq - 0.20)

    l2_penalty = 0.01 * jnp.sum(params["tvtp_W"]**2)
    return -ll + sticky_penalty + semantic_penalty + guidance_loss + freq_penalty + l2_penalty


@partial(jax.jit, static_argnames=("n_iter", "optimizer"))
def _train_hmm_loop(
    X: jnp.ndarray,
    mask: jnp.ndarray,
    guidance_mask: jnp.ndarray,
    params: dict[str, Any],
    opt_state: Any,
    n_iter: int,
    tol: float,
    optimizer: optax.GradientTransformation,
) -> tuple[dict[str, Any], Any, jnp.ndarray, bool]:
    """Compiled HMM training loop with early stopping and guidance."""
    def cond_fn(state):
        i, _, _, loss, prev_loss, converged = state
        return (i < n_iter) & (~converged)

    def body_fun(state):
        i, p, opt_s, loss, _, _ = state
        current_loss, grads = jax.value_and_grad(_compute_nll)(p, X, mask, guidance_mask)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        diff = jnp.abs(current_loss - loss)
        converged = (i > 5) & (diff < tol)
        return i + 1, new_p, new_opt_s, current_loss, loss, converged

    initial_loss = _compute_nll(params, X, mask, guidance_mask)
    init_val = (0, params, opt_state, initial_loss, initial_loss + 1e6, False)
    final_val = jax.lax.while_loop(cond_fn, body_fun, init_val)
    return final_val[1], final_val[2], final_val[3], final_val[5]


@jax.jit
def _jax_inference(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> jnp.ndarray:
    """Consolidated JAX inference call with TVTP using Viterbi decoding."""
    tvtp_z = observations[:, jnp.array(_TVTP_FEATURE_INDICES)]
    log_trans_mats = _compute_tvtp_matrices(tvtp_z, params["tvtp_W"], params["tvtp_b"])
    return _viterbi_decode(
        observations,
        mask,
        jax.nn.log_softmax(params["initial"]),
        log_trans_mats,
        params["locs"],
        jnp.exp(params["log_scales"]),
        jnp.exp(params["log_dfs"]) + 2.0,
    )


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
    """Perform mixed scaling: Log+RobustScaler for vol features, Rank-Gauss for others."""
    feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
    vol_indices = [i for i, c in enumerate(feat_cols) if "vol" in c or "cs_dispersion" in c]
    other_indices = [i for i in range(X.shape[1]) if i not in vol_indices]
    X_clean = X.copy()
    transformers: dict = {}
    if vol_indices:
        X_vol = np.log1p(np.maximum(X_clean[:, vol_indices], 0.0))
        rs = RobustScaler()
        X_clean[:, vol_indices] = rs.fit_transform(X_vol)
        transformers["vol_indices"] = vol_indices
        transformers["vol_rs"] = rs
    if other_indices:
        X_other = X_clean[:, other_indices].copy()
        for i_rel in range(X_other.shape[1]):
            col_data = X_other[:, i_rel]
            mu, std = np.mean(col_data), np.std(col_data) + 1e-12
            X_other[:, i_rel] = np.clip(col_data, mu - 3.0 * std, mu + 3.0 * std)
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(len(X_other), 1000), random_state=42)
        X_clean[:, other_indices] = qt.fit_transform(X_other)
        transformers["other_indices"] = other_indices
        transformers["other_qt"] = qt
    return np.clip(X_clean, -5.0, 5.0), transformers


def _quantile_transform(X: np.ndarray, transformers: dict) -> np.ndarray:
    """Apply pre-fitted mixed transformations."""
    X_out = X.copy()
    vol_indices = transformers.get("vol_indices", [])
    if vol_indices:
        rs = transformers["vol_rs"]
        X_vol = np.log1p(np.maximum(X_out[:, vol_indices], 0.0))
        X_out[:, vol_indices] = rs.transform(X_vol)
    other_indices = transformers.get("other_indices", [])
    if other_indices:
        qt = transformers["other_qt"]
        X_out[:, other_indices] = qt.transform(X_out[:, other_indices])
    return cast(np.ndarray, np.clip(X_out, -5.0, 5.0))


def _apply_posterior_smoothing(probs: np.ndarray | pd.DataFrame, span: int = 6) -> np.ndarray | pd.DataFrame:
    """Apply smoothing to probabilities using Numba."""
    if span <= 1: return probs
    is_df = isinstance(probs, pd.DataFrame)
    data = probs.to_numpy() if is_df else probs
    smoothed = _numba_ema_2d(data.astype(np.float64), span)
    smoothed = np.clip(smoothed, 0.0, 1.0)
    row_sums = np.sum(smoothed, axis=1).reshape(-1, 1)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    out = smoothed / row_sums
    return pd.DataFrame(out, index=probs.index, columns=probs.columns) if is_df else out


def _apply_sticky_posterior(regime_labels: np.ndarray, posteriors: np.ndarray, smoothing_window: int = 6, min_duration: int = 12) -> np.ndarray:
    """Posterior smoothing + min-duration constraint using Numba."""
    if posteriors.shape[0] == 0: return regime_labels.copy()
    smoothed = _numba_sliding_mean_2d(posteriors.astype(np.float64), max(1, smoothing_window))
    labels_smooth = np.argmax(smoothed, axis=1).astype(np.int32)
    return _numba_sticky_labels(labels_smooth, min_duration)


@dataclass
class HMMStateInferrer:
    """Infers market regimes using Guided 4-State TVTP-HMM (Phase 1)."""

    n_states: int = 4
    max_states: int = 4
    n_iter: int = 2000
    predict_step: int = 24
    fit_step: int = 168
    _warmed_up: bool = False
    _last_params: dict[str, Any] | None = None
    _last_opt: Any | None = None

    def _generate_stress_mask(self, features_df: pd.DataFrame, returns_ser: pd.Series) -> np.ndarray:
        """Generate guidance_mask using expanding-window return tails (Worst 15%, walk-forward clean)."""
        expanding_thresh = returns_ser.expanding(min_periods=200).quantile(0.15).bfill()
        stress_cond = (returns_ser < expanding_thresh)
        stress_mask_raw = stress_cond.astype(np.float32).to_numpy()
        return _numba_sticky_labels(stress_mask_raw.astype(np.int32), min_duration=4).astype(np.float32)

    def _warmup_jit(self, n_feats: int) -> None:
        """Warm up JAX HMM JIT cache."""
        if self._warmed_up: return
        _logger.info("Warming up JAX HMM JIT cache (4-states, TVTP, Guided, FEATS=%d)...", n_feats)
        dummy_obs, dummy_mask, dummy_gm = jnp.zeros((MAX_HMM_WINDOW, n_feats)), jnp.zeros(MAX_HMM_WINDOW), jnp.zeros(MAX_HMM_WINDOW)
        params = {"initial": jnp.zeros(4), "tvtp_W": jnp.zeros((4, 4, _N_TVTP_FEATS)), "tvtp_b": jnp.eye(4) * 1.0, "locs": jnp.zeros((4, n_feats)), "log_scales": jnp.zeros((4, n_feats)), "log_dfs": jnp.array([1.0, 1.0, 1.0, 0.5])}
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(params)
        dummy_tvtp_z = dummy_obs[:, jnp.array(_TVTP_FEATURE_INDICES)]
        log_trans_mats = _compute_tvtp_matrices(dummy_tvtp_z, params["tvtp_W"], params["tvtp_b"])
        _ = _forward_pass(dummy_obs, dummy_mask, jax.nn.log_softmax(params["initial"]), log_trans_mats, params["locs"], jnp.exp(params["log_scales"]), jnp.exp(params["log_dfs"]) + 2.0)
        _ = _compute_nll(params, dummy_obs, dummy_mask, dummy_gm)
        _, _, _, _ = _train_hmm_loop(dummy_obs, dummy_mask, dummy_gm, params, opt_state, 1, 1e-4, optimizer)
        _ = _jax_inference(dummy_obs, dummy_mask, params)
        self._warmed_up = True

    def fit_predict_systemic(self, features_df: pd.DataFrame, returns_ser: pd.Series, is_end_idx: int, symbol: str = "Market", tf: str = "1h") -> pd.DataFrame:
        """Expanding-window Guided 4-State TVTP-HMM."""
        cm = CacheManager(FUTURES_CACHE_DIR, max_files=15, max_size_mb=1000.0)
        deps = {"symbol": symbol, "tf": tf, "data_len": len(features_df), "ver": "v23_viterbi_hard_p4", "feat_cols": sorted(list(SYSTEMIC_HMM_FEATURE_COLUMNS))}
        tag = cm.generate_hash(deps, source_files=[Path(__file__).resolve()])
        cache_path = cm.get_cache_path(f"HMM_v23_viterbi_{symbol}_{tf}_len{len(features_df)}", ".parquet", tag)
        if cache_path.exists():
            try: return pd.read_parquet(cache_path)
            except Exception: pass
        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        X_frame = features_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(X_frame)
        if n < 200: return self._zeros_semantic(features_df)
        num_feats = len(feat_cols)
        self._warmup_jit(num_feats)
        X_raw, gm_raw = X_frame.to_numpy(dtype=np.float64), self._generate_stress_mask(features_df, returns_ser)
        out_cols = [*_SEMANTIC_ORDER, "hmm_entropy", "hmm_expected_duration", "hmm_current_duration"]
        results, params, transformers, optimizer = np.full((n, len(out_cols)), 0.0, dtype=np.float64), None, None, optax.adam(learning_rate=0.02)
        for t in range(500, n, self.predict_step):
            if t % self.fit_step == 0 or params is None:
                win_start, win_end = max(0, t - MAX_HMM_WINDOW), max(1, t - 1)
                X_win, gm_win = X_raw[win_start:win_end], gm_raw[win_start:win_end]
                L = len(X_win)
                X_train_raw, transformers = _quantile_scaling(X_win)
                X_pad, M_pad, GM_pad = np.zeros((MAX_HMM_WINDOW, num_feats)), np.zeros(MAX_HMM_WINDOW), np.zeros(MAX_HMM_WINDOW)
                X_pad[:L], M_pad[:L], GM_pad[:L] = X_train_raw, 1.0, gm_win
                if self._last_params:
                    curr_p = self._last_params
                    curr_o = optimizer.init(curr_p)
                    iters = 100
                else:
                    locs = np.zeros((4, num_feats), dtype=np.float32)
                    # BULL (0), BEAR (1), CHOP (2), CRISIS (3)
                    # macro_trend_168h (0), macro_vol_24h (2), macro_downside_vol_24h (3)
                    locs[0, 0], locs[0, 2], locs[0, 3] = 2.5, -1.0, -1.0
                    locs[1, 0], locs[1, 2], locs[1, 3] = -1.5, 1.0, 1.0
                    locs[2, 0], locs[2, 2], locs[2, 3] = 0.0, -1.0, -1.0
                    locs[3, 0], locs[3, 2], locs[3, 3] = -3.5, 3.5, 3.5
                    
                    W_init = np.zeros((4, 4, _N_TVTP_FEATS), dtype=np.float32)
                    # TVTP indices: cs_dispersion(0→4), oi_delta(1→5), funding_mom(2→6), liq_proxy(3→7), lsr_delta(4→8)
                    # Crisis entry driven by oi_delta(local idx 1) and liq_proxy(local idx 3)
                    W_init[:, 3, 1], W_init[:, 3, 3] = 3.0, -2.5
                    curr_p = {"initial": jnp.zeros(4), "tvtp_W": jnp.array(W_init), "tvtp_b": jnp.eye(4) * 7.0, "locs": jnp.array(locs), "log_scales": jnp.zeros((4, num_feats)), "log_dfs": jnp.array([1.0, 1.0, 1.0, 0.5])}
                    curr_o, iters = optimizer.init(curr_p), self.n_iter
                params, self._last_opt, _, _ = _train_hmm_loop(jnp.array(X_pad), jnp.array(M_pad), jnp.array(GM_pad), curr_p, curr_o, iters, 1e-4, optimizer)
                if self._last_params is not None:
                    params = jax.tree_util.tree_map(
                        lambda new, old: 0.7 * new + 0.3 * old, params, self._last_params
                    )
                self._last_params = params
            try:
                inf_start = max(0, t - MAX_HMM_WINDOW)
                write_start = max(0, t - self.predict_step)
                X_inf_raw = _quantile_transform(X_raw[inf_start:t], transformers)
                L_inf = len(X_inf_raw)
                X_inf_pad, M_inf_pad = np.zeros((MAX_HMM_WINDOW, num_feats)), np.zeros(MAX_HMM_WINDOW)
                X_inf_pad[:L_inf], M_inf_pad[:L_inf] = X_inf_raw, 1.0
                all_probs = np.asarray(_jax_inference(jnp.array(X_inf_pad), jnp.array(M_inf_pad), params))[:L_inf]
                write_offset = write_start - inf_start
                combined_probs = all_probs[write_offset:]
                p_clip = np.clip(combined_probs, 1e-12, 1.0)
                entropy = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(4)
                trans_mats = np.asarray(jnp.exp(_compute_tvtp_matrices(jnp.array(X_inf_raw[write_offset:, :][:, list(_TVTP_FEATURE_INDICES)]), params["tvtp_W"], params["tvtp_b"])))
                L_write = len(combined_probs)
                expected_dur = np.array([1.0 / (1.0 - np.diag(trans_mats[i]).clip(0.0, 0.95))[np.argmax(combined_probs[i])] for i in range(L_write)])
                results[write_start:t, :4], results[write_start:t, 4], results[write_start:t, 5] = combined_probs, entropy, expected_dur
            except Exception as e: _logger.error("HMM Inference failed: %s", e)
        probs_df = pd.DataFrame(results, index=features_df.index, columns=out_cols).ffill().bfill().fillna(0.0)
        
        # Viterbi decoding already returns one-hot; apply min-duration (sticky) constraint
        sem_probs_np = probs_df[_SEMANTIC_ORDER].to_numpy()
        hard_states = np.argmax(sem_probs_np, axis=1).astype(np.int32)
        sticky_hard_states = _numba_sticky_labels(hard_states, min_duration=12)
        
        # Convert back to one-hot for output consistency
        viterbi_onehot = np.zeros_like(sem_probs_np)
        viterbi_onehot[np.arange(len(sticky_hard_states)), sticky_hard_states] = 1.0
        probs_df[_SEMANTIC_ORDER] = viterbi_onehot
        
        probs_df["hmm_current_duration"] = _numba_current_duration(sticky_hard_states)
        out = probs_df.reset_index().rename(columns={probs_df.index.name or "index": "datetime"})
        try: out.to_parquet(cache_path)
        except Exception: pass
        return out

    def _zeros_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        u = 1.0 / float(len(_SEMANTIC_ORDER))
        cols = [*_SEMANTIC_ORDER, "hmm_entropy", "hmm_expected_duration", "hmm_current_duration"]
        out = pd.DataFrame(np.zeros((len(df), len(cols))), index=df.index, columns=cols)
        for c in _SEMANTIC_ORDER: out[c] = u
        return out.reset_index().rename(columns={out.index.name or "index": "datetime"})
