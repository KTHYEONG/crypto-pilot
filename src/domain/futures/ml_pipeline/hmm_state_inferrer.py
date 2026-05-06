"""JAX-based Skewed Student-t HMM regime probabilities with SGD optimization and sticky priors."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

# Control JAX memory preallocation and log backend info
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
_logger = logging.getLogger(__name__)

try:
    _logger.info("JAX Backend: %s", jax.default_backend())
    _logger.info("JAX Devices: %s", jax.devices())
except Exception as e:
    _logger.warning("JAX initialization info failed: %s", e)

import optax
import pandas as pd
from jax.scipy.special import betainc, gammaln
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

# --- JAX Skewed Student-t Distribution Implementation ---

@jax.jit
def _student_t_log_pdf(
    x: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray, df: jnp.ndarray
) -> jnp.ndarray:
    """Compute Student-t log-pdf."""
    y = (x - loc) / scale
    log_c = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * jnp.log(jnp.pi * df)
        - jnp.log(scale)
    )
    return log_c - (df + 1.0) / 2.0 * jnp.log1p(y**2 / df)


@jax.jit
def _student_t_cdf(x: jnp.ndarray, df: jnp.ndarray) -> jnp.ndarray:
    """Compute Student-t CDF using regularized incomplete beta function (betainc).

    Note: stop_gradient is used on df because jax.scipy.special.betainc
    does not support gradients w.r.t. a and b.
    """
    df_stop = jax.lax.stop_gradient(df)
    fac = df_stop / (df_stop + x**2)
    bi = betainc(df_stop / 2.0, 0.5, fac)
    return jnp.where(x > 0, 1.0 - 0.5 * bi, 0.5 * bi)

@jax.jit
def _skew_student_t_log_pdf(
    x: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray, skew: jnp.ndarray, df: jnp.ndarray
) -> jnp.ndarray:
    """Azzalini Skew-t log-pdf."""
    # x: (D,), loc: (D,), scale: (D,), skew: (D,), df: (1,)
    x_norm = (x - loc) / scale
    log_t = _student_t_log_pdf(x, loc, scale, df)
    
    # Skewness adjustment: 2 * t(x) * T(alpha * x_norm * sqrt((df+1)/(df + x_norm^2)))
    adj = skew * x_norm * jnp.sqrt((df + 1.0) / (df + x_norm**2))
    log_T = jnp.log(jnp.clip(_student_t_cdf(adj, df + 1.0), 1e-12, 1.0))
    
    return jnp.log(2.0) + log_t + log_T

@jax.jit
def _multivariate_skew_t_log_pdf(
    x: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray, skew: jnp.ndarray, df: jnp.ndarray
) -> jnp.ndarray:
    """Diagonal Multivariate Skew-t log-pdf (sum of univariate)."""
    # x: (D,), loc: (D,), scale: (D,), skew: (D,), df: scalar
    return jnp.sum(_skew_student_t_log_pdf(x, loc, scale, skew, df))

# --- JAX HMM Core Functions ---

@jax.jit
def _forward_pass(
    observations: jnp.ndarray,
    log_init_probs: jnp.ndarray,
    log_trans_mat: jnp.ndarray,
    locs: jnp.ndarray,
    scales: jnp.ndarray,
    skews: jnp.ndarray,
    dfs: jnp.ndarray,
) -> jnp.ndarray:
    """Run Forward algorithm in log-space."""
    num_states = log_init_probs.shape[0]

    def scan_fn(
        log_alpha_prev: jnp.ndarray, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Log emissions for each state: (K,)
        log_emissions = jax.vmap(
            lambda k: _multivariate_skew_t_log_pdf(x, locs[k], scales[k], skews[k], dfs[k])
        )(jnp.arange(num_states))
        # log_alpha_prev: (K,), log_trans_mat: (K, K)
        combined = log_alpha_prev[:, None] + log_trans_mat
        log_alpha_next = jax.scipy.special.logsumexp(combined, axis=0) + log_emissions
        return log_alpha_next, log_alpha_next

    # Initial step
    initial_emissions = jax.vmap(
        lambda k: _multivariate_skew_t_log_pdf(observations[0], locs[k], scales[k], skews[k], dfs[k])
    )(jnp.arange(num_states))
    log_alpha_0 = log_init_probs + initial_emissions

    _, log_alphas = jax.lax.scan(scan_fn, log_alpha_0, observations[1:])
    log_alphas = jnp.concatenate([log_alpha_0[None, :], log_alphas], axis=0)

    return log_alphas


@jax.jit
def _compute_nll(
    params: dict[str, Any], observations: jnp.ndarray, sticky_alpha: float = 0.95
) -> jnp.ndarray:
    """Compute Negative Log-Likelihood with Sticky Prior penalty."""
    log_alphas = _forward_pass(
        observations,
        jax.nn.log_softmax(params["initial"]),
        jax.nn.log_softmax(params["transition"], axis=1),
        params["locs"],
        jnp.exp(params["log_scales"]),
        params["skews"],
        jnp.exp(params["log_dfs"]),
    )
    ll = jax.scipy.special.logsumexp(log_alphas[-1])

    # Sticky Prior: Encourage high diagonal in transition matrix
    trans_mat = jax.nn.softmax(params["transition"], axis=1)
    sticky_penalty = -10.0 * jnp.sum(jnp.log(jnp.diag(trans_mat) + 1e-6))

    return -ll + sticky_penalty

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


LABEL_VERSION = "v16_jax_skew_t"


@dataclass
class HMMStateInferrer:
    """Infers market regimes using JAX-based Skewed Student-t HMM (v16 SOTA)."""

    n_states: int = 4
    max_states: int = 4
    n_iter: int = 500      # JAX SGD iterations
    predict_step: int = 24
    fit_step: int = 252

    def fit_predict_systemic(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
        symbol: str = "Market",
        tf: str = "1h",
    ) -> pd.DataFrame:
        """Expanding-window systemic HMM with JAX Skew-t logic."""
        k_cfg = int(OPT_FUTURES_CONFIG.get("FUTURES_HMM_K_STATES", self.n_states))
        self.n_states = max(4, k_cfg)
        
        cm = CacheManager(FUTURES_CACHE_DIR, max_files=15, max_size_mb=1000.0)
        deps = {
            "symbol": symbol,
            "tf": tf,
            "data_len": len(features_df),
            "n_states": self.n_states,
            "ver": LABEL_VERSION,
            "feat_cols": sorted(list(SYSTEMIC_HMM_FEATURE_COLUMNS)),
        }
        src_files = [Path(__file__).resolve()]
        tag = cm.generate_hash(deps, source_files=src_files)
        prefix = f"HMM_v16_sys_{symbol}_{tf}_len{len(features_df)}"
        cache_path = cm.get_cache_path(prefix, ".parquet", tag)

        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                _logger.info("[%s] JAX Skew-t HMM loaded from cache.", symbol)
                return cached_df
            except Exception as e:
                _logger.warning("Failed to load HMM cache for %s: %s", symbol, e)

        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        X_frame = features_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(X_frame)
        if n < 200:
            return self._zeros_semantic(features_df)

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
        max_window = 5040
        params: dict[str, Any] | None = None
        hmm_transformers: dict[str, Any] | None = None
        map_matrix: np.ndarray | None = None

        for t in range(self.predict_step, n, self.predict_step):
            if t < min_train:
                continue

            is_fit_cycle = (t % self.fit_step == 0) or (params is None)

            if is_fit_cycle:
                win_end = max(1, t - 1)
                X_win = X_raw[max(0, t - max_window) : win_end]
                ret_win = ret_raw[max(0, t - max_window) : win_end]

                X_train_raw, new_transformers = _quantile_scaling(X_win)
                X_train_jax = jnp.array(X_train_raw)

                # Initialize Params
                num_feats = X_train_jax.shape[1]
                rng = jax.random.PRNGKey(42)
                k1, k2, _, _, _ = jax.random.split(rng, 5)

                # Heuristic init
                initial_params = {
                    "initial": jnp.zeros(self.n_states),
                    "transition": jnp.eye(self.n_states) * 5.0,  # Strong diagonal init
                    "locs": jax.random.normal(k1, (self.n_states, num_feats)),
                    "log_scales": jnp.zeros((self.n_states, num_feats)),
                    "skews": jax.random.normal(k2, (self.n_states, num_feats)) * 0.1,
                    "log_dfs": jnp.ones(self.n_states) * 3.0,  # DoF ~ 20 init
                }

                # Optax SGD optimization
                optimizer = optax.adam(learning_rate=0.01)
                opt_state = optimizer.init(initial_params)

                @jax.jit
                def step(p: Any, opt_s: Any, x: jnp.ndarray) -> tuple[Any, Any]:
                    _, grads = jax.value_and_grad(_compute_nll)(p, x)
                    updates, new_opt_s = optimizer.update(grads, opt_s, p)
                    new_p = optax.apply_updates(p, updates)
                    return new_p, new_opt_s

                curr_params = initial_params
                for _ in range(self.n_iter):
                    curr_params, opt_state = step(curr_params, opt_state, X_train_jax)

                params = curr_params
                hmm_transformers = new_transformers

                # Post-fit Labeling
                log_alphas = _forward_pass(
                    X_train_jax,
                    jax.nn.log_softmax(params["initial"]),
                    jax.nn.log_softmax(params["transition"], axis=1),
                    params["locs"],
                    jnp.exp(params["log_scales"]),
                    params["skews"],
                    jnp.exp(params["log_dfs"]),
                )
                posteriors = jax.nn.softmax(log_alphas, axis=1)
                hard_states = np.argmax(np.asarray(posteriors), axis=1)

                st_returns = np.zeros(self.n_states)
                st_vols = np.zeros(self.n_states)
                for s in range(self.n_states):
                    mask = hard_states == s
                    if np.any(mask):
                        st_returns[s] = float(np.mean(ret_win[mask]))
                        if X_train_raw.shape[1] > 2:
                            st_vols[s] = float(np.mean(X_train_raw[mask, 2]))

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
                x_seq_qt = _quantile_transform(
                    X_raw[max(0, t - self.predict_step) : t], hmm_transformers
                )
                x_seq_jax = jnp.array(x_seq_qt)

                log_alphas = _forward_pass(
                    x_seq_jax,
                    jax.nn.log_softmax(params["initial"]),
                    jax.nn.log_softmax(params["transition"], axis=1),
                    params["locs"],
                    jnp.exp(params["log_scales"]),
                    params["skews"],
                    jnp.exp(params["log_dfs"]),
                )
                p_seq = np.asarray(jax.nn.softmax(log_alphas, axis=1))
                semantic_probs = p_seq @ map_matrix

                p_clip = np.clip(p_seq, 1e-12, 1.0)
                entropy = -np.sum(p_clip * np.log(p_clip), axis=1) / jnp.log(max(2, self.n_states))

                trans_mat = np.asarray(jax.nn.softmax(params["transition"], axis=1))
                durations = 1.0 / (1.0 - np.diag(trans_mat).clip(0.0, 0.999))
                current_durations = durations[np.argmax(p_seq, axis=1)]

                apply_start = max(0, t - self.predict_step)
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
