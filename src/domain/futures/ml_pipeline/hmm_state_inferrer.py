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
        log_emissions = jax.vmap(
            lambda k: _multivariate_student_t_log_pdf(x, locs[k], scales[k], dfs[k])
        )(jnp.arange(num_states))
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
    initial_emissions = jax.vmap(
        lambda k: _multivariate_student_t_log_pdf(observations[0], locs[k], scales[k], dfs[k])
    )(jnp.arange(num_states))
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

# --- Utility Functions ---

def _assign_state_semantic_labels_v5(
    means: np.ndarray,
    state_returns: np.ndarray | None = None,
    state_vols: np.ndarray | None = None,
) -> dict[int, str]:
    """Map HMM state index to semantic label via G_LOG based Logic (v15.6 SOTA)."""
    k = means.shape[0]
    trend_idx = 0         # macro_trend_168h
    vol_idx = 1           # macro_vol_24h
    downside_vol_idx = 4  # macro_downside_vol_24h

    state_to_label: dict[int, str] = {}
    remaining = list(range(k))

    vol_means = means[:, vol_idx]
    down_vol_means = (
        means[:, downside_vol_idx] if means.shape[1] > downside_vol_idx else vol_means
    )
    composite_risk = (vol_means + down_vol_means) / 2.0

    crisis_idx = int(remaining[int(np.argmax(composite_risk[remaining]))])
    state_to_label[crisis_idx] = "crisis"
    remaining.remove(crisis_idx)

    if state_returns is not None and state_vols is not None:
        mu = np.asarray(state_returns, dtype=np.float64)
        sigma = np.asarray(state_vols, dtype=np.float64)
        proxy_g_log = mu - 0.5 * (sigma**2)
    else:
        proxy_g_log = means[:, trend_idx] - 0.5 * (means[:, vol_idx] ** 2)

    bull_idx_rel = int(np.argmax(proxy_g_log[remaining]))
    bull_idx = remaining[bull_idx_rel]
    state_to_label[bull_idx] = "bull_trend"
    remaining.remove(bull_idx)

    if remaining:
        bear_idx_rel = int(np.argmin(proxy_g_log[remaining]))
        bear_idx = remaining[bear_idx_rel]
        state_to_label[bear_idx] = "bear_trend"
        remaining.remove(bear_idx)

    for idx in remaining:
        state_to_label[idx] = "chop"

    return state_to_label


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
    df: pd.DataFrame, method: str = "EMA", span: int = 6
) -> pd.DataFrame:
    """Apply smoothing to probabilities to reduce whipsaws."""
    if span <= 1:
        return df
    out_df = df.ewm(span=span, adjust=False).mean()
    out_df = out_df.clip(0.0, 1.0)
    row_sums = out_df.sum(axis=1)
    out_df = out_df.div(row_sums, axis=0).fillna(1.0 / len(df.columns))
    return out_df


def _apply_sticky_posterior(
    regime_labels: np.ndarray,
    posteriors: np.ndarray,
    smoothing_window: int = 6,
    min_duration: int = 12,
) -> np.ndarray:
    """Posterior smoothing + min-duration constraint to reduce regime flip noise."""
    t_len = posteriors.shape[0]
    if t_len == 0:
        return regime_labels.copy()
    smoothed = np.zeros_like(posteriors)
    w = max(1, smoothing_window)
    for i in range(t_len):
        start = max(0, i - w + 1)
        smoothed[i] = posteriors[start : i + 1].mean(axis=0)
    labels_smooth: np.ndarray = np.argmax(smoothed, axis=1).astype(np.int32)
    if min_duration <= 1 or t_len < 2:
        return labels_smooth
    result = labels_smooth.copy()
    i = 0
    while i < t_len:
        current_state = result[i]
        j = i + 1
        while j < t_len and result[j] == current_state:
            j += 1
        run_len = j - i
        if run_len < min_duration and i > 0:
            prev_state = result[i - 1]
            result[i:j] = prev_state
            i = max(0, i - 1)
            while i > 0 and result[i - 1] == prev_state:
                i -= 1
        else:
            i = j
    return result


LABEL_VERSION = "v18_jax_student_t"


@dataclass
class HMMStateInferrer:
    """Infers market regimes using JAX-based Student-t HMM (v18 Optimized)."""

    n_states: int = 4
    max_states: int = 4
    n_iter: int = 500      # Default JAX SGD iterations
    predict_step: int = 24
    fit_step: int = 252
    _warmed_up: bool = False
    _last_params: dict[str, Any] | None = None
    _last_opt_state: Any | None = None

    def _warmup_jit(self, n_feats: int) -> None:
        """Run a dummy 1-iteration optimization to warm up JIT cache."""
        if self._warmed_up:
            return
        _logger.info("Warming up JAX HMM JIT cache (MAX_WINDOW=%d, FEATS=%d)...", MAX_HMM_WINDOW, n_feats)
        dummy_obs = jnp.zeros((MAX_HMM_WINDOW, n_feats))
        dummy_mask = jnp.zeros(MAX_HMM_WINDOW)
        
        # Simple dummy params
        params = {
            "initial": jnp.zeros(self.n_states),
            "transition": jnp.eye(self.n_states),
            "locs": jnp.zeros((self.n_states, n_feats)),
            "log_scales": jnp.zeros((self.n_states, n_feats)),
            "log_dfs": jnp.ones(self.n_states) * 3.0,
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
        
        self._warmed_up = True

    def fit_predict_systemic(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
        symbol: str = "Market",
        tf: str = "1h",
    ) -> pd.DataFrame:
        """Expanding-window systemic HMM with Optimized JAX logic."""
        cm = CacheManager(FUTURES_CACHE_DIR, max_files=15, max_size_mb=1000.0)
        deps = {
            "symbol": symbol,
            "tf": tf,
            "data_len": len(features_df),
            "n_states": self.n_states,
            "ver": "v18_jax_student_t",
            "feat_cols": sorted(list(SYSTEMIC_HMM_FEATURE_COLUMNS)),
        }
        src_files = [Path(__file__).resolve()]
        tag = cm.generate_hash(deps, source_files=src_files)
        prefix = f"HMM_v18_sys_{symbol}_{tf}_len{len(features_df)}"
        cache_path = cm.get_cache_path(prefix, ".parquet", tag)

        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                _logger.info("[%s] JAX Student-t HMM loaded from cache.", symbol)
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
        ret_raw = returns_ser.reindex(features_df.index).fillna(0.0).to_numpy(dtype=np.float64)

        out_cols = [
            *_SEMANTIC_ORDER,
            "hmm_entropy",
            "hmm_expected_duration",
            "hmm_current_duration",
        ]
        results: np.ndarray = np.full((n, len(out_cols)), np.nan, dtype=np.float64)

        min_train = 500
        params: dict[str, Any] | None = None
        hmm_transformers: dict[str, Any] | None = None
        map_matrix: np.ndarray | None = None

        optimizer = optax.adam(learning_rate=0.01)

        for t in range(self.predict_step, n, self.predict_step):
            if t < min_train:
                continue

            is_fit_cycle = (t % self.fit_step == 0) or (params is None)

            if is_fit_cycle:
                win_end = max(1, t - 1)
                win_start = max(0, t - MAX_HMM_WINDOW)
                X_win = X_raw[win_start : win_end]
                ret_win = ret_raw[win_start : win_end]
                
                L = len(X_win)
                X_train_raw, new_transformers = _quantile_scaling(X_win)
                
                # Static Shape Padding
                X_padded = np.zeros((MAX_HMM_WINDOW, num_feats), dtype=np.float64)
                X_padded[:L] = X_train_raw
                X_mask = np.zeros(MAX_HMM_WINDOW, dtype=np.float64)
                X_mask[:L] = 1.0
                
                X_train_jax = jnp.array(X_padded)
                X_mask_jax = jnp.array(X_mask)

                # Warm-start logic
                if self._last_params is not None:
                    curr_params = self._last_params
                    curr_opt_state = self._last_opt_state
                    iterations = 50  # Reduce iterations for warm-start
                else:
                    # Heuristic init
                    rng = jax.random.PRNGKey(42)
                    k1, _, _, _, _ = jax.random.split(rng, 5)
                    curr_params = {
                        "initial": jnp.zeros(self.n_states),
                        "transition": jnp.eye(self.n_states) * 5.0,  # Strong diagonal init
                        "locs": jax.random.normal(k1, (self.n_states, num_feats)),
                        "log_scales": jnp.zeros((self.n_states, num_feats)),
                        "log_dfs": jnp.ones(self.n_states) * 3.0,  # DoF ~ exp(3)+2 init
                    }
                    curr_opt_state = optimizer.init(curr_params)
                    iterations = self.n_iter

                # Run compiled training loop
                new_params, new_opt_state, last_loss, converged = _train_hmm_loop(
                    X_train_jax, X_mask_jax, curr_params, curr_opt_state, 
                    iterations, 1e-4, optimizer
                )

                self._last_params = new_params
                self._last_opt_state = new_opt_state
                params = new_params
                hmm_transformers = new_transformers

                _logger.debug("HMM Fit at t=%d: Loss=%.4f, Converged=%s", t, last_loss, converged)

                # Post-fit Labeling
                log_alphas, _ = _forward_pass(
                    X_train_jax,
                    X_mask_jax,
                    jax.nn.log_softmax(params["initial"]),
                    jax.nn.log_softmax(params["transition"], axis=1),
                    params["locs"],
                    jnp.exp(params["log_scales"]),
                    jnp.exp(params["log_dfs"]) + 2.0,
                )
                # Only use valid steps for labeling
                posteriors_full = jax.nn.softmax(log_alphas, axis=1)
                posteriors = np.asarray(posteriors_full[:L])
                hard_states = np.argmax(posteriors, axis=1)

                st_returns = np.zeros(self.n_states)
                st_vols = np.zeros(self.n_states)
                for s in range(self.n_states):
                    mask_s = hard_states == s
                    if np.any(mask_s):
                        st_returns[s] = float(np.mean(ret_win[mask_s]))
                        if X_train_raw.shape[1] > 2:
                            st_vols[s] = float(np.mean(X_train_raw[mask_s, 2]))

                state_to_label = _assign_state_semantic_labels_v5(
                    np.asarray(params["locs"]), st_returns, st_vols
                )
                map_matrix = np.zeros((self.n_states, len(_SEMANTIC_ORDER)), dtype=np.float64)
                for si in range(self.n_states):
                    lab = state_to_label.get(si, "chop")
                    prob_col = f"hmm_prob_{lab}"
                    if prob_col in _SEMANTIC_ORDER:
                        map_matrix[si, _SEMANTIC_ORDER.index(prob_col)] = 1.0

            if params is None or hmm_transformers is None or map_matrix is None:
                continue

            try:
                apply_start = max(0, t - self.predict_step)
                x_seq_qt = _quantile_transform(
                    X_raw[apply_start : t], hmm_transformers
                )
                L_inf = len(x_seq_qt)
                
                # Padding for Inference
                X_inf_padded = np.zeros((MAX_HMM_WINDOW, num_feats), dtype=np.float64)
                X_inf_padded[:L_inf] = x_seq_qt
                X_inf_mask = np.zeros(MAX_HMM_WINDOW, dtype=np.float64)
                X_inf_mask[:L_inf] = 1.0
                
                x_seq_jax = jnp.array(X_inf_padded)
                m_inf_jax = jnp.array(X_inf_mask)

                log_alphas, _ = _forward_pass(
                    x_seq_jax,
                    m_inf_jax,
                    jax.nn.log_softmax(params["initial"]),
                    jax.nn.log_softmax(params["transition"], axis=1),
                    params["locs"],
                    jnp.exp(params["log_scales"]),
                    jnp.exp(params["log_dfs"]) + 2.0,
                )
                p_seq_full = np.asarray(jax.nn.softmax(log_alphas, axis=1))
                p_seq = p_seq_full[:L_inf]
                
                semantic_probs = p_seq @ map_matrix

                p_clip = np.clip(p_seq, 1e-12, 1.0)
                entropy = -np.sum(p_clip * np.log(p_clip), axis=1) / jnp.log(max(2, self.n_states))

                trans_mat = np.asarray(jax.nn.softmax(params["transition"], axis=1))
                durations = 1.0 / (1.0 - np.diag(trans_mat).clip(0.0, 0.999))
                current_durations = durations[np.argmax(p_seq, axis=1)]

                results[apply_start:t, : len(_SEMANTIC_ORDER)] = semantic_probs
                results[apply_start:t, len(_SEMANTIC_ORDER)] = entropy
                results[apply_start:t, len(_SEMANTIC_ORDER) + 1] = current_durations
            except Exception as e:
                _logger.error("HMM Inference failed at t=%d: %s", t, e)

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

        hard_states = np.argmax(probs_df[sem_cols].to_numpy(), axis=1)
        dur_arr = np.zeros(len(hard_states), dtype=np.float64)
        if len(hard_states) > 0:
            c = 1.0
            dur_arr[0] = c
            for i in range(1, len(hard_states)):
                if hard_states[i] == hard_states[i-1]:
                    c += 1.0
                else:
                    c = 1.0
                dur_arr[i] = c
        probs_df["hmm_current_duration"] = dur_arr
        
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
