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

MAX_HMM_WINDOW = 2500

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
def _forward_pass(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    log_init_probs: jnp.ndarray,
    log_trans_mat: jnp.ndarray,
    locs: jnp.ndarray,
    scales: jnp.ndarray,
    dfs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run Forward algorithm in log-space with static shape masking."""
    num_states = log_init_probs.shape[0]

    def scan_fn(
        log_alpha_prev: jnp.ndarray, val: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, m = val
        # Log emissions for each state: (K,)
        log_emissions = _batched_student_t_log_pdf(x, locs, scales, dfs)
        # log_alpha_prev: (K,), log_trans_mat: (K, K)
        combined = log_alpha_prev[:, None] + log_trans_mat
        log_alpha_next_calc = jax.scipy.special.logsumexp(combined, axis=0) + log_emissions
        
        # Log-likelihood contribution of this step
        ll_t = jax.scipy.special.logsumexp(log_alpha_next_calc)
        
        # Normalize alpha to prevent drift
        log_alpha_next_normalized = log_alpha_next_calc - ll_t
        
        # If mask is 0, carry over previous alpha and add 0 to LL
        log_alpha_next = jnp.where(m, log_alpha_next_normalized, log_alpha_prev)
        incremental_ll = jnp.where(m, ll_t, 0.0)
        
        return log_alpha_next, (log_alpha_next, incremental_ll)

    # Initial step
    initial_emissions = _batched_student_t_log_pdf(observations[0], locs, scales, dfs)
    log_alpha_0_calc = log_init_probs + initial_emissions
    ll_0 = jax.scipy.special.logsumexp(log_alpha_0_calc)
    log_alpha_0_normalized = log_alpha_0_calc - ll_0
    
    log_alpha_0 = jnp.where(mask[0], log_alpha_0_normalized, log_init_probs)
    incremental_ll_0 = jnp.where(mask[0], ll_0, 0.0)

    _, (log_alphas, incremental_lls) = jax.lax.scan(scan_fn, log_alpha_0, (observations[1:], mask[1:]))
    log_alphas = jnp.concatenate([log_alpha_0[None, :], log_alphas], axis=0)
    incremental_lls = jnp.concatenate([incremental_ll_0[None], incremental_lls], axis=0)

    return log_alphas, incremental_lls


@jax.jit
def _compute_nll(
    params: dict[str, Any], 
    observations: jnp.ndarray, 
    mask: jnp.ndarray,
    sticky_alpha: float = 0.95
) -> jnp.ndarray:
    """Compute Negative Log-Likelihood with Sticky Prior penalty and Masking."""
    _, incremental_lls = _forward_pass(
        observations,
        mask,
        jax.nn.log_softmax(params["initial"]),
        jax.nn.log_softmax(params["transition"], axis=1),
        params["locs"],
        jnp.exp(params["log_scales"]),
        jnp.exp(params["log_dfs"]) + 2.0,
    )
    ll = jnp.sum(incremental_lls)

    # Sticky Prior: Encourage high diagonal in transition matrix
    trans_mat = jax.nn.softmax(params["transition"], axis=1)
    sticky_penalty = -10.0 * jnp.sum(jnp.log(jnp.diag(trans_mat) + 1e-6))

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
    """Compiled HMM training loop with early stopping."""
    
    def cond_fn(state):
        i, _, _, loss, prev_loss, converged = state
        return (i < n_iter) & (~converged)

    def body_fun(state):
        i, p, opt_s, loss, _, _ = state
        current_loss, grads = jax.value_and_grad(_compute_nll)(p, X, mask)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        
        # Convergence check: relative change in loss
        diff = jnp.abs(current_loss - loss)
        converged = (i > 5) & (diff < tol)
        
        return i + 1, new_p, new_opt_s, current_loss, loss, converged

    # Initial state for while_loop
    # i, params, opt_state, loss, prev_loss, converged
    initial_loss = _compute_nll(params, X, mask)
    init_val = (0, params, opt_state, initial_loss, initial_loss + 1e6, False)
    
    final_val = jax.lax.while_loop(cond_fn, body_fun, init_val)
    
    return final_val[1], final_val[2], final_val[3], final_val[5]


@jax.jit
def _jax_inference(
    observations: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> jnp.ndarray:
    """Consolidated JAX inference call."""
    log_alphas, _ = _forward_pass(
        observations,
        mask,
        jax.nn.log_softmax(params["initial"]),
        jax.nn.log_softmax(params["transition"], axis=1),
        params["locs"],
        jnp.exp(params["log_scales"]),
        jnp.exp(params["log_dfs"]) + 2.0,
    )
    return jax.nn.softmax(log_alphas, axis=1)


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
    """Fast sliding mean for 2D arrays (Zero-Loop Policy compliant)."""
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
    """Min-duration constraint to reduce regime flip noise (Numba optimized)."""
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
    """Calculate run length of states (Numba optimized)."""
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


def _quantile_scaling(X: np.ndarray) -> tuple[np.ndarray, dict]:  # type: ignore[type-arg]
    """Perform mixed scaling: Log+RobustScaler for vol features, Rank-Gauss for others."""
    feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
    vol_indices = [
        i for i, c in enumerate(feat_cols)
        if "vol" in c or "cs_dispersion" in c
    ]
    other_indices = [i for i in range(X.shape[1]) if i not in vol_indices]

    X_clean = X.copy()
    transformers: dict = {}  # type: ignore[type-arg]

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
            mu = np.mean(col_data)
            std = np.std(col_data) + 1e-12
            X_other[:, i_rel] = np.clip(col_data, mu - 3.0 * std, mu + 3.0 * std)
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(len(X_other), 1000),
            random_state=42,
        )
        X_clean[:, other_indices] = qt.fit_transform(X_other)
        transformers["other_indices"] = other_indices
        transformers["other_qt"] = qt

    X_clean = np.clip(X_clean, -5.0, 5.0)
    return X_clean, transformers


def _quantile_transform(X: np.ndarray, transformers: dict) -> np.ndarray:  # type: ignore[type-arg]
    """Apply pre-fitted mixed transformations."""
    X_out = X.copy()
    vol_indices: list[int] = transformers.get("vol_indices", [])
    if vol_indices:
        rs: RobustScaler = transformers["vol_rs"]
        X_vol = np.log1p(np.maximum(X_out[:, vol_indices], 0.0))
        X_out[:, vol_indices] = rs.transform(X_vol)
    other_indices: list[int] = transformers.get("other_indices", [])
    if other_indices:
        qt: QuantileTransformer = transformers["other_qt"]
        X_out[:, other_indices] = qt.transform(X_out[:, other_indices])
    return cast(np.ndarray, np.clip(X_out, -5.0, 5.0))


def _apply_posterior_smoothing(
    probs: np.ndarray | pd.DataFrame, span: int = 6
) -> np.ndarray | pd.DataFrame:
    """Apply smoothing to probabilities using Numba (Zero-Loop)."""
    if span <= 1:
        return probs

    is_df = isinstance(probs, pd.DataFrame)
    data = probs.to_numpy() if is_df else probs

    # 1. EMA via Numba
    smoothed = _numba_ema_2d(data.astype(np.float64), span)

    # 2. Vectorized Clip and Normalize
    smoothed = np.clip(smoothed, 0.0, 1.0)
    row_sums = np.sum(smoothed, axis=1).reshape(-1, 1)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    out = smoothed / row_sums

    if is_df:
        return pd.DataFrame(out, index=probs.index, columns=probs.columns)
    return out


def _apply_sticky_posterior(
    regime_labels: np.ndarray,
    posteriors: np.ndarray,
    smoothing_window: int = 6,
    min_duration: int = 12,
) -> np.ndarray:
    """Posterior smoothing + min-duration constraint using Numba."""
    t_len = posteriors.shape[0]
    if t_len == 0:
        return regime_labels.copy()

    # 1. Sliding Mean (Numba)
    smoothed = _numba_sliding_mean_2d(posteriors.astype(np.float64), max(1, smoothing_window))

    # 2. Argmax to get smooth labels
    labels_smooth = np.argmax(smoothed, axis=1).astype(np.int32)

    # 3. Sticky Logic (Numba)
    return _numba_sticky_labels(labels_smooth, min_duration)


LABEL_VERSION = "v18_hier_stress_v1"


@dataclass
class HMMStateInferrer:
    """Infers market regimes using Stress-Isolating Hierarchical HMM (Phase 3.2)."""

    n_states: int = 3  # Normal states: Bull, Bear, Chop
    max_states: int = 3
    n_iter: int = 500      # Default JAX SGD iterations
    predict_step: int = 24
    fit_step: int = 252
    _warmed_up: bool = False
    _last_params: dict[str, Any] | None = None
    _last_opt: Any | None = None

    def _generate_stress_mask(self, features_df: pd.DataFrame) -> np.ndarray:
        """Generate sticky stress_mask (1: Crisis, 0: Normal) using Vol_Z and Trend_Scaled."""
        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        trend_col = feat_cols[0]  # macro_trend_168h
        vol_col = feat_cols[1]    # macro_vol_24h
        
        trend_series = features_df[trend_col]
        vol_series = features_df[vol_col]
        
        # Vol_Z: rolling 24h Z-score of macro_vol_24h
        vol_mean = vol_series.rolling(window=24, min_periods=1).mean()
        vol_std = vol_series.rolling(window=24, min_periods=1).std()
        vol_z = (vol_series - vol_mean) / (vol_std + 1e-9)
        
        # Stress Mask: (Vol_Z > 1.5) & (Trend_Scaled < -0.5)
        stress_mask_raw = ((vol_z > 1.5) & (trend_series < -0.5)).astype(np.int32).to_numpy()
        
        # Apply sticky labels with min_duration=4
        return _numba_sticky_labels(stress_mask_raw, min_duration=4)

    def _warmup_jit(self, n_feats: int) -> None:
        """Run a dummy 1-iteration optimization to warm up JIT cache for 3-state HMM."""
        if self._warmed_up:
            return
        _logger.info("Warming up JAX HMM JIT cache (3-states, FEATS=%d)...", n_feats)
        dummy_obs = jnp.zeros((MAX_HMM_WINDOW, n_feats))
        dummy_mask = jnp.zeros(MAX_HMM_WINDOW)
        
        # 3-state HMM params
        params = {
            "initial": jnp.zeros(3),
            "transition": jnp.eye(3),
            "locs": jnp.zeros((3, n_feats)),
            "log_scales": jnp.zeros((3, n_feats)),
            "log_dfs": jnp.ones(3) * 3.0,
        }
        
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(params)
        
        # Test forward pass and NLL
        _ = _forward_pass(
            dummy_obs, dummy_mask, 
            jax.nn.log_softmax(params["initial"]), 
            jax.nn.log_softmax(params["transition"], axis=1),
            params["locs"], jnp.exp(params["log_scales"]), jnp.exp(params["log_dfs"]) + 2.0
        )
        _ = _compute_nll(params, dummy_obs, dummy_mask)
        
        # Test training loop
        _, _, _, _ = _train_hmm_loop(
            dummy_obs, dummy_mask, params, opt_state, 1, 1e-4, optimizer
        )
        
        # Test inference
        _ = _jax_inference(dummy_obs, dummy_mask, params)
        
        self._warmed_up = True

    def fit_predict_systemic(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
        symbol: str = "Market",
        tf: str = "1h",
    ) -> pd.DataFrame:
        """Expanding-window Stress-Isolating Hierarchical HMM (Phase 3.2)."""
        cm = CacheManager(FUTURES_CACHE_DIR, max_files=15, max_size_mb=1000.0)
        deps = {
            "symbol": symbol,
            "tf": tf,
            "data_len": len(features_df),
            "ver": "v18_hier_stress_v1",
            "feat_cols": sorted(list(SYSTEMIC_HMM_FEATURE_COLUMNS)),
        }
        src_files = [Path(__file__).resolve()]
        tag = cm.generate_hash(deps, source_files=src_files)
        prefix = f"HMM_v18_hier_stress_{symbol}_{tf}_len{len(features_df)}"
        cache_path = cm.get_cache_path(prefix, ".parquet", tag)

        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                _logger.info("[%s] Stress-Isolating HMM loaded from cache.", symbol)
                return cached_df
            except Exception as e:
                _logger.warning("Failed to load HMM cache for %s: %s", symbol, e)

        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        X_frame = features_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(X_frame)
        if n < 200:
            return self._zeros_semantic(features_df)

        num_feats = len(feat_cols)
        self._warmup_jit(num_feats)

        X_raw = X_frame.to_numpy(dtype=np.float64)
        stress_mask = self._generate_stress_mask(features_df)

        out_cols = [
            *_SEMANTIC_ORDER,
            "hmm_entropy",
            "hmm_expected_duration",
            "hmm_current_duration",
        ]
        results: np.ndarray = np.full((n, len(out_cols)), 0.0, dtype=np.float64)

        min_train = 500
        params: dict[str, Any] | None = None
        transformers: dict[str, Any] | None = None

        optimizer = optax.adam(learning_rate=0.01)

        for t in range(self.predict_step, n, self.predict_step):
            if t < min_train:
                continue

            is_fit_cycle = (t % self.fit_step == 0) or (params is None)

            if is_fit_cycle:
                win_end = max(1, t - 1)
                win_start = max(0, t - MAX_HMM_WINDOW)
                X_win = X_raw[win_start : win_end]
                sm_win = stress_mask[win_start : win_end]
                
                L = len(X_win)
                X_train_raw, new_transformers = _quantile_scaling(X_win)
                transformers = new_transformers
                
                # Static Shape Padding
                X_padded = np.zeros((MAX_HMM_WINDOW, num_feats), dtype=np.float64)
                X_padded[:L] = X_train_raw
                X_mask = np.zeros(MAX_HMM_WINDOW, dtype=np.float64)
                X_mask[:L] = 1.0
                
                sm_padded = np.zeros(MAX_HMM_WINDOW, dtype=np.float64)
                sm_padded[:L] = sm_win
                
                X_train_jax = jnp.array(X_padded)
                X_mask_jax = jnp.array(X_mask)
                sm_jax = jnp.array(sm_padded)

                # --- Train 3-State HMM on Normal Data (stress_mask == 0) ---
                mask_normal = X_mask_jax * (1.0 - sm_jax)
                
                if self._last_params is not None:
                    curr_p, curr_o, iters = self._last_params, self._last_opt, 50
                else:
                    # Init 3-state HMM: Bull, Bear, Chop
                    locs = np.zeros((3, num_feats), dtype=np.float32)
                    locs[0, 0], locs[0, 1] = 1.0, -1.0   # BULL_TREND: +Trend, -Vol
                    locs[1, 0], locs[1, 1] = -1.0, 1.0  # BEAR_TREND: -Trend, +Vol
                    locs[2, 0], locs[2, 1] = 0.0, -1.0  # CHOP: 0 Trend, -Vol
                    curr_p = {
                        "initial": jnp.zeros(3),
                        "transition": jnp.eye(3) * 5.0,
                        "locs": jnp.array(locs),
                        "log_scales": jnp.zeros((3, num_feats)),
                        "log_dfs": jnp.ones(3) * 3.0,
                    }
                    curr_o = optimizer.init(curr_p)
                    iters = self.n_iter
                
                params, self._last_opt, _, _ = _train_hmm_loop(
                    X_train_jax, mask_normal, curr_p, curr_o, iters, 1e-4, optimizer
                )
                self._last_params = params

            if params is None or transformers is None:
                continue

            try:
                apply_start = max(0, t - self.predict_step)
                x_seq = X_raw[apply_start : t]
                sm_seq = stress_mask[apply_start : t]
                x_seq_qt = _quantile_transform(x_seq, transformers)
                L_inf = len(x_seq_qt)
                
                X_inf_padded = np.zeros((MAX_HMM_WINDOW, num_feats), dtype=np.float64)
                X_inf_padded[:L_inf] = x_seq_qt
                X_inf_mask = np.zeros(MAX_HMM_WINDOW, dtype=np.float64)
                X_inf_mask[:L_inf] = 1.0
                X_inf_jax = jnp.array(X_inf_padded)
                M_inf_jax = jnp.array(X_inf_mask)

                # Inference on Normal HMM
                p_hmm = np.asarray(_jax_inference(X_inf_jax, M_inf_jax, params))[:L_inf]
                
                # Combine into 4 states: [bull_trend, bear_trend, chop, crisis]
                combined_probs = np.zeros((L_inf, 4), dtype=np.float64)
                
                normal_idx = (sm_seq == 0)
                stress_idx = (sm_seq == 1)
                
                combined_probs[normal_idx, :3] = p_hmm[normal_idx]
                combined_probs[stress_idx, 3] = 1.0 # Static Crisis

                # Entropy
                p_clip = np.clip(combined_probs, 1e-12, 1.0)
                entropy = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(4)

                # Expected duration
                trans = np.asarray(jax.nn.softmax(params["transition"], axis=1))
                dur = 1.0 / (1.0 - np.diag(trans).clip(0.0, 0.999))
                
                expected_dur = np.zeros(L_inf)
                if np.any(normal_idx):
                    expected_dur[normal_idx] = dur[np.argmax(p_hmm[normal_idx], axis=1)]
                expected_dur[stress_idx] = 4.0 # Min duration for crisis

                results[apply_start:t, :4] = combined_probs
                results[apply_start:t, 4] = entropy
                results[apply_start:t, 5] = expected_dur
            except Exception as e:
                _logger.error("Stress-Isolating HMM Inference failed at t=%d: %s", t, e)

        probs_df = pd.DataFrame(results, index=features_df.index, columns=out_cols)
        probs_df = probs_df.ffill().bfill().fillna(0.0)
        
        sem_cols = _SEMANTIC_ORDER
        probs_df[sem_cols] = _apply_posterior_smoothing(
            probs_df[sem_cols], span=int(OPT_FUTURES_CONFIG.get("FUTURES_HMM_SMOOTHING_SPAN", 12))
        )

        sem_probs_np = probs_df[sem_cols].to_numpy(dtype=np.float64)
        raw_hard_states = np.argmax(sem_probs_np, axis=1).astype(np.int32)
        sticky_hard_states = _apply_sticky_posterior(
            raw_hard_states,
            sem_probs_np,
            smoothing_window=int(OPT_FUTURES_CONFIG.get("FUTURES_HMM_SMOOTHING_SPAN", 12)),
            min_duration=6,
        )
        
        sticky_onehot = np.zeros_like(sem_probs_np)
        sticky_onehot[np.arange(len(sticky_hard_states)), sticky_hard_states] = 1.0
        blended = 0.8 * sticky_onehot + 0.2 * sem_probs_np
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        probs_df[sem_cols] = blended / row_sums

        hard_states_final = np.argmax(probs_df[sem_cols].to_numpy(), axis=1).astype(np.int32)
        probs_df["hmm_current_duration"] = _numba_current_duration(hard_states_final)
        
        out = probs_df.reset_index()
        if "datetime" not in out.columns:
            out = out.rename(columns={out.columns[0]: "datetime"})
        
        try:
            out.to_parquet(cache_path)
        except Exception:
            pass
        return out

    def _zeros_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        u = 1.0 / float(len(_SEMANTIC_ORDER))
        cols = [*_SEMANTIC_ORDER, "hmm_entropy", "hmm_expected_duration", "hmm_current_duration"]
        out = pd.DataFrame(np.zeros((len(df), len(cols))), index=df.index, columns=cols)
        for c in _SEMANTIC_ORDER:
            out[c] = u
        return out.reset_index().rename(columns={out.index.name or "index": "datetime"})
