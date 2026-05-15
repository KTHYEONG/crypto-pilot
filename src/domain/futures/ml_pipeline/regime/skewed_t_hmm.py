"""Skewed-t emission HMM backend with Parallel Associative Scan (v11.0).

This backend implements Skewed-t emissions for better capture of asymmetric 
market regimes (e.g., fast crashes in BEAR state) and utilizes JAX's 
associative_scan for parallelized HMM filtering on GPUs.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

_logger = logging.getLogger(__name__)

_N_STATES = 4
_N_FEATURES = 6
_MAX_LEN = 3000
_EPS = 1e-6


@jax.jit
def _t_cdf_approx(x: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    """Robust approximation of Student-t CDF using Normal CDF (ndtr).
    
    Formula: T(x; nu) \approx Phi(x * (1 - 1/(4*nu)) / sqrt(1 + x^2/(2*nu)))
    """
    # nu should be > 2.0
    nu_safe = jnp.maximum(nu, 2.1)
    factor = (1.0 - 1.0 / (4.0 * nu_safe)) / jnp.sqrt(1.0 + jnp.square(x) / (2.0 * nu_safe))
    return jax.scipy.special.ndtr(x * factor)


@jax.jit
def _skewed_t_log_pdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    log_sig: jnp.ndarray,
    nu_raw: jnp.ndarray,
    lambda_raw: jnp.ndarray,
) -> jnp.ndarray:
    """Diagonal multivariate Skewed-t log-pdf (Azzalini formulation).

    Args:
        x: (F,) observation.
        mu: (K, F) state means.
        log_sig: (K, F) state log-scales.
        nu_raw: (K,) unconstrained dof parameters.
        lambda_raw: (K, F) skewness parameters.

    Returns:
        (K,) per-state log-probabilities.
    """
    # nu: (K, 1), nu_p1: (K, 1)
    nu = jnp.clip(jax.nn.softplus(nu_raw) + 2.1, 2.1, 100.0)[:, None]
    nu_p1 = nu + 1.0
    sig = jnp.exp(jnp.clip(log_sig, -6.0, 3.0))
    alpha = lambda_raw  # (K, F)
    
    z = (x - mu) / jnp.maximum(sig, _EPS)
    z2 = jnp.square(z)

    # Standard Student-t log-pdf component
    log_norm = (
        jax.scipy.special.gammaln(nu_p1 * 0.5)
        - jax.scipy.special.gammaln(nu * 0.5)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(jnp.maximum(sig, _EPS))
    )
    log_kernel = -0.5 * nu_p1 * jnp.log1p(z2 / jnp.maximum(nu, _EPS))
    log_t_pdf = log_norm + log_kernel

    # Skewness component: log(2 * T(alpha * z * sqrt((nu+1)/(nu+z^2)); nu+1))
    skew_arg = alpha * z * jnp.sqrt(nu_p1 / (nu + z2))
    skew_log_cdf = jnp.log(jnp.maximum(_t_cdf_approx(skew_arg, nu_p1), _EPS))
    
    # log(2) + log_t_pdf + skew_log_cdf
    logp_elements = jnp.log(2.0) + log_t_pdf + skew_log_cdf
    logp = jnp.sum(logp_elements, axis=1)
    
    return jnp.nan_to_num(logp, nan=-1e6, posinf=-1e6, neginf=-1e6)


@jax.jit
def _log_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Log-space matrix multiplication: C = A @ B."""
    # A: (..., K, K), B: (..., K, K)
    # result[..., i, j] = logsumexp(A[..., i, k] + B[..., k, j])
    return jax.scipy.special.logsumexp(A[..., :, :, None] + B[..., None, :, :], axis=-2)


@jax.jit
def _hmm_forward(
    obs: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Parallel forward filtering pass using associative_scan."""
    mu = params["mu"]
    log_sig = params["log_sig"]
    nu_raw = params["nu_raw"]
    lambda_raw = params["lambda_raw"]
    log_init = jax.nn.log_softmax(params["log_init"])
    log_trans = jax.nn.log_softmax(params["log_trans"], axis=1)

    # 1. Precompute all emission log-probs
    # log_emits: (T, K)
    log_emits = jax.vmap(_skewed_t_log_pdf, in_axes=(0, None, None, None, None))(
        obs, mu, log_sig, nu_raw, lambda_raw
    )
    
    # 2. Define transition matrices M_t(i, j) = log_trans(i, j) + log_emit(t, j)
    # log_Ms: (T, K, K)
    # We want alpha_t^T = alpha_{t-1}^T @ M_t
    # So M_t[i, j] is the log-prob of transitioning from i to j and emitting obs[t] at j.
    log_Ms = log_trans[None, :, :] + log_emits[:, None, :]
    
    # Apply mask: if mask is 0, M_t should be Identity in log-space (0 on diag, -inf off)
    # Actually, if mask is 0, we just want to keep alpha_prev.
    # In matrix terms, alpha_t = alpha_prev @ I.
    eye_log = jnp.eye(_N_STATES)
    eye_log = jnp.where(eye_log > 0.5, 0.0, -1e6)
    mask_v = mask[:, None, None]
    log_Ms = jnp.where(mask_v > 0.5, log_Ms, eye_log)

    # 3. Parallel scan to get prefix products of transition matrices
    # prefix_Ms[t] = M_0 @ M_1 @ ... @ M_t
    prefix_Ms = jax.lax.associative_scan(_log_matmul, log_Ms)
    
    # 4. Compute log_alphas: alpha_t = log_init @ prefix_Ms[t]
    # alpha_0 is special in standard HMM, but here M_0 already includes log_emit_0.
    # So alpha_t = log_init @ M_0 @ ... @ M_t
    log_alphas_unnorm = jax.vmap(lambda M: jax.scipy.special.logsumexp(log_init[:, None] + M, axis=0))(prefix_Ms)
    
    # 5. Compute Log-Likelihoods
    # Total LL at t is logsumexp(alpha_t)
    lls_total = jax.scipy.special.logsumexp(log_alphas_unnorm, axis=1)
    
    # Normalize log_alphas
    log_alphas = log_alphas_unnorm - lls_total[:, None]
    
    # Incremental LLs: ll_t = lls_total[t] - lls_total[t-1]
    lls = jnp.concatenate([lls_total[:1], jnp.diff(lls_total)], axis=0)
    lls = jnp.where(mask, lls, 0.0)
    
    return log_alphas, lls


def _hmm_nll(
    params: dict[str, Any],
    obs: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    _, lls = _hmm_forward(obs, mask, params)
    nll = -jnp.sum(lls)

    # Transition priors (Stickiness)
    trans_probs = jax.nn.softmax(params["log_trans"], axis=1)
    diag = jnp.diag(trans_probs)
    
    vol_z = obs[:, 1]
    avg_vol = jnp.sum(vol_z * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    
    sigmoid_vol = jax.nn.sigmoid(2.0 * (avg_vol - 1.0))
    sticky_base = 2200.0  # Slightly stronger for Skewed-t
    sticky_weight = sticky_base * (1.0 - 0.5 * sigmoid_vol)
    sticky_prior = -sticky_weight * jnp.sum(jnp.log(jnp.maximum(diag, _EPS)))

    # Semantic priors for state separation
    mu = params["mu"]
    n_obs = jnp.maximum(jnp.sum(mask), 1.0)
    semantic_prior = n_obs * (
        1000.0 * jnp.square(jnp.maximum(0.0, 0.6 - mu[0, 0]))    # BULL f1 > 0.6
        + 1000.0 * jnp.square(jnp.maximum(0.0, mu[1, 0] + 0.6))  # BEAR f1 < -0.6
        + 500.0 * jnp.square(jnp.maximum(0.0, 1.0 - mu[2, 1]))   # CHOP_HIGH f2 > 1.0
        + 500.0 * jnp.square(jnp.maximum(0.0, mu[3, 1] + 0.6))   # CHOP_LOW f2 < -0.6
    )

    # DOF (nu) Prior
    nu = jax.nn.softplus(params["nu_raw"]) + 2.1
    nu_targets = jnp.array([40.0, 3.0, 10.0, 15.0]) # BULL: Gaussian-like, BEAR: Heavy tail
    nu_prior = 15.0 * jnp.sum(jnp.square(jnp.log(nu) - jnp.log(nu_targets)))

    # Skewness (lambda) Prior
    # BULL: Positive skew on trend (f1), BEAR: Negative skew on trend (f1)
    lambdas = params["lambda_raw"] # (K, F)
    # Target: BULL trend skew > 0, BEAR trend skew < -1.5 (Crash risk)
    skew_prior = n_obs * (
        0.5 * jnp.square(jnp.maximum(0.0, 0.5 - lambdas[0, 0]))   # BULL f1 skew
        + 1.5 * jnp.square(jnp.maximum(0.0, lambdas[1, 0] + 1.5)) # BEAR f1 skew << 0
    )
    # L2 regularization on other skewness params
    skew_l2 = 0.1 * jnp.sum(jnp.square(lambdas))

    loss = nll + sticky_prior + semantic_prior + nu_prior + skew_prior + skew_l2
    return jnp.nan_to_num(loss, nan=1e9, posinf=1e9, neginf=1e9)


@partial(jax.jit, static_argnames=("n_iter", "optimizer"))
def _train_hmm(
    obs: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
    opt_state: Any,
    n_iter: int,
    tol: float,
    optimizer: optax.GradientTransformation,
) -> tuple[dict[str, Any], Any]:
    def cond(state):
        i, _, _, loss, prev = state
        return (i < n_iter) & (jnp.abs(loss - prev) > tol)

    def body(state):
        i, p, opt_s, loss, _prev = state
        new_loss, grads = jax.value_and_grad(_hmm_nll)(p, obs, mask)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
        )
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return i + 1, new_p, new_opt_s, new_loss, loss

    init_loss = _hmm_nll(params, obs, mask)
    state = (0, params, opt_state, init_loss, init_loss + 1e6)
    final = jax.lax.while_loop(cond, body, state)
    return final[1], final[2]


class SkewedTMultivariateHMM:
    """4-state HMM with diagonal Skewed-t emissions and Parallel Scan.

    Features (6):
        f1: trend_z, f2: vol_z, f3: downside_vol_z, f4: cs_dispersion_z,
        f5: oi_delta_z, f6: funding_mom_z
    """

    def __init__(self, n_iter: int = 1000, tol: float = 1e-4):
        self.n_iter = n_iter
        self.tol = tol
        self._params = None
        self._warmed = False
        
        # Check for GPU
        devices = jax.devices()
        _logger.info("JAX devices: %s", devices)
        self._device = devices[0]

    def _init_params(self, obs: np.ndarray) -> dict[str, Any]:
        f = [obs[:, i] for i in range(obs.shape[1])]
        
        def get_mu(p_list):
            return [np.percentile(f[i], p_list[i]) for i in range(len(f))]

        # State 0: BULL (High f1, Low f2, Low f3, High f5, High f6)
        mu0 = get_mu([75, 25, 20, 50, 70, 70])
        # State 1: BEAR (Low f1, Mid f2, High f3, Low f5, Low f6)
        mu1 = get_mu([25, 50, 80, 50, 30, 30])
        # State 2: CHOP_HIGH (Mid f1, High f2, High f4)
        mu2 = get_mu([50, 75, 50, 80, 50, 50])
        # State 3: CHOP_LOW (Mid f1, Low f2, Low f4)
        mu3 = get_mu([50, 10, 25, 20, 50, 50])
        
        return {
            "log_init": jnp.zeros(_N_STATES),
            "log_trans": jnp.eye(_N_STATES) * 4.5,
            "mu": jnp.array([mu0, mu1, mu2, mu3], dtype=jnp.float32),
            "log_sig": jnp.zeros((_N_STATES, _N_FEATURES), dtype=jnp.float32) - 0.4,
            "nu_raw": jnp.ones((_N_STATES,), dtype=jnp.float32) * 2.0,
            "lambda_raw": jnp.zeros((_N_STATES, _N_FEATURES), dtype=jnp.float32),
        }

    def _warmup(self) -> None:
        if self._warmed:
            return
        _logger.debug("Skewed-t HMM warmup on %s...", self._device)
        dummy_obs = jnp.zeros((64, _N_FEATURES), dtype=jnp.float32)
        dummy_mask = jnp.ones((64,), dtype=jnp.float32)
        p = self._init_params(np.random.normal(size=(64, _N_FEATURES)).astype(np.float32))
        opt = optax.adamw(0.01)
        os = opt.init(p)
        _hmm_forward(dummy_obs, dummy_mask, p)
        _train_hmm(dummy_obs, dummy_mask, p, os, 2, 1e-4, opt)
        self._warmed = True

    def _prep_obs(self, obs_df: pd.DataFrame) -> np.ndarray:
        obs_arr = obs_df.fillna(0.0).to_numpy(dtype=np.float64)
        obs_arr = np.clip(obs_arr, -8.0, 8.0)
        obs_arr = np.nan_to_num(obs_arr, nan=0.0, posinf=8.0, neginf=-8.0)
        return obs_arr.astype(np.float32)

    def _to_prob_df(self, probs: np.ndarray, obs_df: pd.DataFrame) -> pd.DataFrame:
        safe = np.nan_to_num(probs, nan=1.0 / _N_STATES, posinf=1.0 / _N_STATES, neginf=0.0)
        safe = np.clip(safe, _EPS, 1.0)
        safe = safe / np.maximum(safe.sum(axis=1, keepdims=True), _EPS)
        return pd.DataFrame(
            safe,
            index=obs_df.index,
            columns=["bull_trend", "bear_trend", "chop_high", "chop_low"],
        )

    def fit(self, obs_df: pd.DataFrame) -> SkewedTMultivariateHMM:
        obs_arr = self._prep_obs(obs_df)
        n = len(obs_arr)
        if n <= 0: return self

        self._warmup()
        opt = optax.adamw(learning_rate=0.02, weight_decay=1e-4)

        for t in range(min(n, 500), n + 168, 168):
            t_end = min(t, n)
            win_start = max(0, t_end - _MAX_LEN)
            obs_win = obs_arr[win_start:t_end]
            if len(obs_win) <= 0: continue
            
            p = self._params if self._params is not None else self._init_params(obs_win)
            m = np.ones((len(obs_win),), dtype=np.float32)
            os = opt.init(p)
            self._params, _ = _train_hmm(
                jnp.array(obs_win), jnp.array(m), p, os, self.n_iter, self.tol, opt
            )
        return self

    def filter(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        obs_arr = self._prep_obs(obs_df)
        n = len(obs_arr)
        if n <= 0:
            return self._to_prob_df(np.empty((0, _N_STATES), dtype=np.float64), obs_df)

        self._warmup()
        if self._params is None:
            self.fit(obs_df)

        m = np.ones((n,), dtype=np.float32)
        log_alphas, _ = _hmm_forward(jnp.array(obs_arr), jnp.array(m), self._params)
        probs = np.asarray(jax.nn.softmax(log_alphas, axis=1))
        return self._to_prob_df(probs, obs_df)

    def fit_filter_train_oos(self, obs_df: pd.DataFrame, is_end_idx: int) -> pd.DataFrame:
        n = len(obs_df)
        if n <= 0:
            return self._to_prob_df(np.empty((0, _N_STATES), dtype=np.float64), obs_df)
        cut = int(np.clip(is_end_idx, 0, n - 1))
        self._params = None
        self.fit(obs_df.iloc[: cut + 1])
        return self.filter(obs_df)

    def fit_predict(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        n = len(obs_df)
        if n <= 0:
            return self._to_prob_df(np.empty((0, _N_STATES), dtype=np.float64), obs_df)
        return self.fit_filter_train_oos(obs_df, n - 1)
