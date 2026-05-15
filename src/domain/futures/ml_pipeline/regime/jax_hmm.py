"""Track A: 4-State JAX Multivariate HMM (v10.1).

States:
    0: BULL_TREND    (High f1, Low f2)
    1: BEAR_TREND    (Low f1, Low f2, High f3, Low f4)
    2: CHOP_HIGH_VOL (Mid f1, High f2)
    3: CHOP_LOW_VOL  (Mid f1, Low f2)

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
_N_FEATURES = 4
_MAX_LEN = 3000

@jax.jit
def _gauss_log_pdf(x: jnp.ndarray, mu: jnp.ndarray, log_sig: jnp.ndarray) -> jnp.ndarray:
    """Multivariate Gaussian log-pdf with diagonal covariance.

    Args:
        x: (F,) observation.
        mu: (K, F) means.
        log_sig: (K, F) log-sigmas.

    Returns:
        (K,) log-probabilities.

    """
    sig = jnp.exp(log_sig)
    # (K, F)
    log_probs = -0.5 * jnp.log(2.0 * jnp.pi) - log_sig - 0.5 * ((x - mu) / sig) ** 2
    return jnp.sum(log_probs, axis=1)

@jax.jit
def _hmm_forward(
    obs: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Log-space HMM forward pass."""
    K = _N_STATES
    log_trans = jax.nn.log_softmax(params["log_trans"], axis=1)
    mu = params["mu"]
    log_sig = params["log_sig"]
    log_init = jax.nn.log_softmax(params["log_init"])

    def step_fn(log_alpha_prev, inputs):
        x_t, m_t = inputs
        log_emit = _gauss_log_pdf(x_t, mu, log_sig)
        log_alpha_pred = jax.scipy.special.logsumexp(
            log_alpha_prev[:, None] + log_trans, axis=0
        )
        log_alpha_unnorm = log_alpha_pred + log_emit
        ll_t = jax.scipy.special.logsumexp(log_alpha_unnorm)
        log_alpha_norm = log_alpha_unnorm - ll_t
        
        log_alpha_out = jnp.where(m_t, log_alpha_norm, log_alpha_prev)
        ll_out = jnp.where(m_t, ll_t, 0.0)
        return log_alpha_out, (log_alpha_out, ll_out)

    # t=0
    log_emit_0 = _gauss_log_pdf(obs[0], mu, log_sig)
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
    """NLL with priors for JAX HMM."""
    _, lls = _hmm_forward(obs, mask, params)
    nll = -jnp.sum(lls)

    # 1. Transition Matrix Priors (Stickiness ~0.95-0.98)
    # Step 6: TVTP (Time-Varying Transition Probabilities)
    # 평시(Low f2_vol_z)에는 sticky_weight 강화, 고변동성 시기에는 완화
    trans_probs = jax.nn.softmax(params["log_trans"], axis=1)
    diag = jnp.diag(trans_probs)
    
    vol_z = obs[:, 1]
    avg_vol = jnp.sum(vol_z * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    sticky_base = 1500.0
    sticky_weight = jnp.where(avg_vol < 0.0, sticky_base * 1.5, sticky_base * 0.7)
    sticky_prior = -sticky_weight * jnp.sum(jnp.log(jnp.maximum(diag, 1e-6)))

    # 2. Semantic State Separation Priors
    # BULL (0): f1 >> 0
    # BEAR (1): f1 << 0
    # CHOP_HIGH (2): f2 >> 0
    # CHOP_LOW (3): f2 << 0
    mu = params["mu"]
    n_obs = jnp.maximum(jnp.sum(mask), 1.0)

    semantic_prior = n_obs * (
        1000.0 * jnp.square(jnp.maximum(0.0, 0.5 - mu[0, 0]))    # BULL f1 > 0.5
        + 1000.0 * jnp.square(jnp.maximum(0.0, mu[1, 0] + 0.5))  # BEAR f1 < -0.5
        + 500.0  * jnp.square(jnp.maximum(0.0, 0.8 - mu[2, 1]))  # CHOP_HIGH f2 > 0.8
        + 500.0  * jnp.square(jnp.maximum(0.0, mu[3, 1] + 0.5))  # CHOP_LOW f2 < -0.5
    )

    return nll + sticky_prior + semantic_prior

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
        i, _, _, loss, prev, _ = state
        return (i < n_iter) & (jnp.abs(loss - prev) > tol)

    def body(state):
        i, p, opt_s, loss, _prev, _c = state
        new_loss, grads = jax.value_and_grad(_hmm_nll)(p, obs, mask)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return i + 1, new_p, new_opt_s, new_loss, loss, False

    init_loss = _hmm_nll(params, obs, mask)
    state = (0, params, opt_state, init_loss, init_loss + 1e6, False)
    final = jax.lax.while_loop(cond, body, state)
    return final[1], final[2]

class JAXMultivariateHMM:
    def __init__(self, n_iter: int = 1000, tol: float = 1e-4):
        self.n_iter = n_iter
        self.tol = tol
        self._params = None
        self._warmed = False

    def _init_params(self, obs: np.ndarray) -> dict[str, Any]:
        """Quantile-based initialization for 4-feature state separation.

        4-feature layout:
            f1: trend_z, f2: vol_z, f3: downside_vol_z, f4: cs_dispersion_z
        """
        f1, f2, f3, f4 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]

        # State 0: BULL (High f1, Low f2, Low f3)
        mu0 = [np.percentile(f1, 75), np.percentile(f2, 25), np.percentile(f3, 20), np.percentile(f4, 50)]
        # State 1: BEAR (Low f1, Mid f2, High f3)
        mu1 = [np.percentile(f1, 25), np.percentile(f2, 50), np.percentile(f3, 80), np.percentile(f4, 50)]
        # State 2: CHOP_HIGH (Mid f1, High f2, High f4)
        mu2 = [np.percentile(f1, 50), np.percentile(f2, 75), np.percentile(f3, 50), np.percentile(f4, 80)]
        # State 3: CHOP_LOW (Mid f1, Low f2, Low f4)
        mu3 = [np.percentile(f1, 50), np.percentile(f2, 10), np.percentile(f3, 25), np.percentile(f4, 20)]

        return {
            "log_init": jnp.zeros(_N_STATES),
            "log_trans": jnp.eye(_N_STATES) * 4.0,  # Strong self-transition
            "mu": jnp.array([mu0, mu1, mu2, mu3], dtype=jnp.float32),
            "log_sig": jnp.zeros((_N_STATES, _N_FEATURES), dtype=jnp.float32) - 0.5,
        }

    def _warmup(self):
        if self._warmed: return
        _logger.debug("JAX HMM Warmup...")
        dummy_obs = jnp.zeros((_MAX_LEN, _N_FEATURES))
        dummy_mask = jnp.zeros(_MAX_LEN)
        p = self._init_params(np.random.normal(size=(_MAX_LEN, _N_FEATURES)))
        opt = optax.adamw(0.01)
        os = opt.init(p)
        _hmm_forward(dummy_obs, dummy_mask, p)
        _train_hmm(dummy_obs, dummy_mask, p, os, 3, 1e-4, opt)
        self._warmed = True

    def _prep_obs(self, obs_df: pd.DataFrame) -> np.ndarray:
        obs_arr = obs_df.fillna(0.0).to_numpy(dtype=np.float64)
        obs_arr = np.clip(obs_arr, -5.0, 5.0).astype(np.float32)
        return obs_arr

    def _to_prob_df(self, probs: np.ndarray, obs_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            probs,
            index=obs_df.index,
            columns=["bull_trend", "bear_trend", "chop_high", "chop_low"],
        )

    def fit(self, obs_df: pd.DataFrame) -> JAXMultivariateHMM:
        """Train HMM parameters on the provided observations only."""
        obs_arr = self._prep_obs(obs_df)
        n = len(obs_arr)
        if n <= 0:
            return self

        self._warmup()
        opt = optax.adamw(learning_rate=0.03, weight_decay=1e-4)

        # Expanding window training (causal by construction).
        step = 168
        for t in range(min(n, 500), n + step, step):
            t_end = min(t, n)
            win_start = max(0, t_end - _MAX_LEN)
            L = t_end - win_start
            if L <= 0:
                continue

            obs_win = obs_arr[win_start:t_end]
            obs_pad = np.zeros((_MAX_LEN, _N_FEATURES), dtype=np.float32)
            m_pad = np.zeros(_MAX_LEN, dtype=np.float32)
            obs_pad[:L] = obs_win
            m_pad[:L] = 1.0

            if self._params is None:
                p = self._init_params(obs_win)
            else:
                p = self._params

            os = opt.init(p)
            self._params, _ = _train_hmm(
                jnp.array(obs_pad), jnp.array(m_pad), p, os, self.n_iter, self.tol, opt
            )
        return self

    def filter(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """Run forward filtering with fixed params.

        Causal guarantee: posterior at t depends only on observations up to t.
        """
        obs_arr = self._prep_obs(obs_df)
        n = len(obs_arr)
        if n <= 0:
            return self._to_prob_df(np.empty((0, _N_STATES), dtype=np.float64), obs_df)

        self._warmup()
        if self._params is None:
            # Backward-compatible behavior: if unfitted, train on provided sample first.
            self.fit(obs_df)

        probs_all = np.full((n, _N_STATES), 1.0 / _N_STATES, dtype=np.float64)
        # Causal rolling filtering: each t uses a trailing window ending at t.
        # This avoids any future-to-past overwrite.
        step = 168
        for t_end in range(step, n + step, step):
            end = min(t_end, n)
            win_start = max(0, end - _MAX_LEN)
            obs_win = obs_arr[win_start:end]
            L = len(obs_win)
            if L <= 0:
                continue
            obs_p = np.zeros((_MAX_LEN, _N_FEATURES), dtype=np.float32)
            m_p = np.zeros(_MAX_LEN, dtype=np.float32)
            obs_p[:L] = obs_win
            m_p[:L] = 1.0
            log_alphas, _ = _hmm_forward(jnp.array(obs_p), jnp.array(m_p), self._params)
            post = np.asarray(jax.nn.softmax(log_alphas[:L], axis=1))
            probs_all[win_start:end] = post

        return self._to_prob_df(probs_all, obs_df)

    def fit_filter_train_oos(self, obs_df: pd.DataFrame, is_end_idx: int) -> pd.DataFrame:
        """Fit on train slice and filter full slice with frozen params.

        Train slice: [0, is_end_idx] inclusive.
        Causal guarantee: no future bar is used to re-estimate posteriors on earlier bars.
        """
        n = len(obs_df)
        if n <= 0:
            return self._to_prob_df(np.empty((0, _N_STATES), dtype=np.float64), obs_df)

        cut = int(np.clip(is_end_idx, 0, n - 1))
        self._params = None
        self.fit(obs_df.iloc[: cut + 1])
        return self.filter(obs_df)

    def fit_predict(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible wrapper.

        Semantics: fit on full sample, then causal filtering on full sample.
        """
        n = len(obs_df)
        if n <= 0:
            return self._to_prob_df(np.empty((0, _N_STATES), dtype=np.float64), obs_df)
        return self.fit_filter_train_oos(obs_df, n - 1)
