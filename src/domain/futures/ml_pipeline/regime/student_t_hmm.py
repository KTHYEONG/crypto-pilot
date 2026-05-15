"""Student-t emission HMM backend for regime inference (v10.1, 6-feature).

This backend mirrors the public API of the existing HMM backend while using
diagonal Student-t emissions for heavier-tail robustness.

Features:
    f1: Trend           (Rolling Z-score of EMA(12)/EMA(144)-1)
    f2: Volatility      (Rolling Z-score of Log(ATR(14)/Close))
    f3: Downside Vol    (Robust Z-score of macro_downside_vol_24h)
    f4: Funding Mom     (Robust Z-score of macro_funding_mom_24h)
    f5: OI Delta        (Robust Z-score of macro_oi_delta_24h)
    f6: CS Dispersion   (Robust Z-score of macro_cs_dispersion_24h)
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any

# Force CPU backend to avoid CUDA plugin probing/segfault in this runtime.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
def _student_t_log_pdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    log_sig: jnp.ndarray,
    nu_raw: jnp.ndarray,
) -> jnp.ndarray:
    """Diagonal multivariate Student-t log-pdf.

    Args:
        x: (F,) observation.
        mu: (K, F) state means.
        log_sig: (K, F) state log-scales.
        nu_raw: (K,) unconstrained dof parameters.

    Returns:
        (K,) per-state log-probabilities.

    """
    # Keep dof away from invalid region and cap extremely large values.
    nu = jnp.clip(jax.nn.softplus(nu_raw) + 2.1, 2.1, 100.0)[:, None]  # (K, 1)
    sig = jnp.exp(jnp.clip(log_sig, -6.0, 3.0))
    z2 = jnp.square((x - mu) / jnp.maximum(sig, _EPS))

    log_norm = (
        jax.scipy.special.gammaln((nu + 1.0) * 0.5)
        - jax.scipy.special.gammaln(nu * 0.5)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(jnp.maximum(sig, _EPS))
    )
    log_kernel = -0.5 * (nu + 1.0) * jnp.log1p(z2 / jnp.maximum(nu, _EPS))
    logp = jnp.sum(log_norm + log_kernel, axis=1)
    return jnp.nan_to_num(logp, nan=-1e6, posinf=-1e6, neginf=-1e6)


@jax.jit
def _hmm_forward(
    obs: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Log-space forward filtering pass."""
    log_trans = jax.nn.log_softmax(params["log_trans"], axis=1)
    mu = params["mu"]
    log_sig = params["log_sig"]
    nu_raw = params["nu_raw"]
    log_init = jax.nn.log_softmax(params["log_init"])

    def step_fn(log_alpha_prev, inputs):
        x_t, m_t = inputs
        log_emit = _student_t_log_pdf(x_t, mu, log_sig, nu_raw)
        log_alpha_pred = jax.scipy.special.logsumexp(
            log_alpha_prev[:, None] + log_trans, axis=0
        )
        log_alpha_unnorm = log_alpha_pred + log_emit
        ll_t = jax.scipy.special.logsumexp(log_alpha_unnorm)
        log_alpha_norm = log_alpha_unnorm - ll_t
        log_alpha_out = jnp.where(m_t, log_alpha_norm, log_alpha_prev)
        ll_out = jnp.where(m_t, ll_t, 0.0)
        return log_alpha_out, (log_alpha_out, ll_out)

    log_emit_0 = _student_t_log_pdf(obs[0], mu, log_sig, nu_raw)
    log_alpha_0_unnorm = log_init + log_emit_0
    ll_0 = jax.scipy.special.logsumexp(log_alpha_0_unnorm)
    log_alpha_0 = log_alpha_0_unnorm - ll_0
    log_alpha_0 = jnp.where(mask[0], log_alpha_0, log_init)
    ll_0_out = jnp.where(mask[0], ll_0, 0.0)

    _, (log_alphas_tail, lls_tail) = jax.lax.scan(
        step_fn, log_alpha_0, (obs[1:], mask[1:])
    )
    log_alphas = jnp.concatenate([log_alpha_0[None], log_alphas_tail], axis=0)
    lls = jnp.concatenate([ll_0_out[None], lls_tail], axis=0)
    return log_alphas, lls


def _hmm_nll(
    params: dict[str, Any],
    obs: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    _, lls = _hmm_forward(obs, mask, params)
    nll = -jnp.sum(lls)

    # Phase 3: TVTP (Time-Varying Transition Probabilities) Sigmoid Smoothing
    trans_probs = jax.nn.softmax(params["log_trans"], axis=1)
    diag = jnp.diag(trans_probs)
    
    vol_z = obs[:, 1]
    avg_vol = jnp.sum(vol_z * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    
    # Nonlinear sigmoid weighting: smoother transition than jnp.where
    sigmoid_vol = jax.nn.sigmoid(2.0 * (avg_vol - 1.0))
    sticky_base = 2000.0
    sticky_weight = sticky_base * (1.0 - 0.5 * sigmoid_vol)
    sticky_prior = -sticky_weight * jnp.sum(jnp.log(jnp.maximum(diag, _EPS)))

    mu = params["mu"]
    n_obs = jnp.maximum(jnp.sum(mask), 1.0)
    semantic_prior = n_obs * (
        900.0 * jnp.square(jnp.maximum(0.0, 0.5 - mu[0, 0]))     # BULL f1 > 0.5
        + 900.0 * jnp.square(jnp.maximum(0.0, mu[1, 0] + 0.5))   # BEAR f1 < -0.5
        + 450.0 * jnp.square(jnp.maximum(0.0, 0.8 - mu[2, 1]))   # CHOP_HIGH f2 > 0.8
        + 450.0 * jnp.square(jnp.maximum(0.0, mu[3, 1] + 0.5))   # CHOP_LOW f2 < -0.5
    )

    # Phase 1: Asymmetric nu Prior
    nu = jax.nn.softplus(params["nu_raw"]) + 2.1
    # BULL: 30.0 (Normal), BEAR: 3.5 (Heavy Tail), CHOP: 8.0~12.0
    nu_targets = jnp.array([30.0, 3.5, 8.0, 12.0])
    nu_prior = 10.0 * jnp.sum(jnp.square(jnp.log(jnp.maximum(nu, _EPS)) - jnp.log(nu_targets)))
    loss = nll + sticky_prior + semantic_prior + nu_prior
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


class StudentTMultivariateHMM:
    """4-state HMM with diagonal Student-t emissions.

    Output columns are fixed to:
    `["bull_trend", "bear_trend", "chop_high", "chop_low"]`.
    """

    def __init__(self, n_iter: int = 800, tol: float = 1e-4):
        self.n_iter = n_iter
        self.tol = tol
        self._params = None
        self._warmed = False

    def _init_params(self, obs: np.ndarray) -> dict[str, Any]:
        """Quantile-based initialization for 6-feature state separation.

        6-feature layout:
            f1: trend_z, f2: vol_z, f3: downside_vol_z, f4: cs_dispersion_z,
            f5: oi_delta_z, f6: funding_mom_z
        """
        f = [obs[:, i] for i in range(obs.shape[1])]
        
        # Helper to get percentiles for 6 features
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
            "log_trans": jnp.eye(_N_STATES) * 4.0,
            "mu": jnp.array([mu0, mu1, mu2, mu3], dtype=jnp.float32),
            "log_sig": jnp.zeros((_N_STATES, _N_FEATURES), dtype=jnp.float32) - 0.4,
            "nu_raw": jnp.ones((_N_STATES,), dtype=jnp.float32) * 1.5,
        }

    def _warmup(self) -> None:
        if self._warmed:
            return
        _logger.debug("Student-t HMM warmup...")
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

    def fit(self, obs_df: pd.DataFrame) -> StudentTMultivariateHMM:
        """Fit parameters on the provided observation frame."""
        obs_arr = self._prep_obs(obs_df)
        n = len(obs_arr)
        if n <= 0:
            return self

        self._warmup()
        opt = optax.adamw(learning_rate=0.02, weight_decay=1e-4)

        for t in range(min(n, 500), n + 168, 168):
            t_end = min(t, n)
            win_start = max(0, t_end - _MAX_LEN)
            obs_win = obs_arr[win_start:t_end]
            if len(obs_win) <= 0:
                continue
            if self._params is None:
                p = self._init_params(obs_win)
            else:
                p = self._params
            m = np.ones((len(obs_win),), dtype=np.float32)
            os = opt.init(p)
            self._params, _ = _train_hmm(
                jnp.array(obs_win), jnp.array(m), p, os, self.n_iter, self.tol, opt
            )
        return self

    def filter(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """Run causal forward filtering with fixed model parameters."""
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
        """Fit on `[0..is_end_idx]` and filter the full frame with frozen params."""
        n = len(obs_df)
        if n <= 0:
            return self._to_prob_df(np.empty((0, _N_STATES), dtype=np.float64), obs_df)
        cut = int(np.clip(is_end_idx, 0, n - 1))
        self._params = None
        self.fit(obs_df.iloc[: cut + 1])
        return self.filter(obs_df)

    def fit_predict(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible wrapper: fit full sample, then filter full sample."""
        n = len(obs_df)
        if n <= 0:
            return self._to_prob_df(np.empty((0, _N_STATES), dtype=np.float64), obs_df)
        return self.fit_filter_train_oos(obs_df, n - 1)
