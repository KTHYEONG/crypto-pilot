"""Layer 2: 3-state Direction HMM on vol-normalised returns.

Architecture:
    Emission: Gaussian on y_t = r_6h / sigma_6h
    sigma_6h  = posterior-weighted √(Σ_k p_k * σ²_k) from Layer 1
    Transition: state-conditioned (3 separate transition rows)

States:
    0: BULL  (μ > 0)
    1: RANGE (μ ≈ 0)
    2: BEAR  (μ < 0)
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from config.opt_config import OPT_FUTURES_CONFIG

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
jax.config.update("jax_platform_name", "cpu")

_logger = logging.getLogger(__name__)

_N_DIR_STATES: int = 3   # BULL=0, RANGE=1, BEAR=2
_DIR_MAX_LEN: int = 1000


@jax.jit
def _gauss_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, log_sig: jnp.ndarray) -> jnp.ndarray:
    """Diagonal Gaussian log-pdf per state.

    Args:
        y:       Scalar observation (vol-normalised return).
        mu:      (K,) state means.
        log_sig: (K,) log(standard deviation) for numerical stability.

    Returns:
        (K,) per-state log-pdf values.

    """
    sig = jnp.exp(log_sig)
    return -0.5 * jnp.log(2.0 * jnp.pi) - log_sig - 0.5 * ((y - mu) / sig) ** 2


@jax.jit
def _dir_forward(
    obs: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run HMM forward pass (log-space) for Direction model.

    Args:
        obs:    (T,) vol-normalised returns (padded to _DIR_MAX_LEN).
        mask:   (T,) 1.0 for valid, 0.0 for padding.
        params: Direction HMM parameter dict.

    Returns:
        log_alphas:    (T, K) filtered log-posteriors.
        incremental_ll:(T,)  per-step LL increments.

    Time complexity: O(T * K²).

    """
    log_trans = jax.nn.log_softmax(params["log_trans"], axis=1)  # (K, K)
    mu = params["mu"]          # (K,)
    log_sig = params["log_sig"]  # (K,)
    log_init = jax.nn.log_softmax(params["log_init"])  # (K,)

    def step_fn(
        log_alpha_prev: jnp.ndarray,
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        y_t, m_t = inputs
        log_emit = _gauss_log_pdf(y_t, mu, log_sig)
        log_alpha_pred = jax.scipy.special.logsumexp(
            log_alpha_prev[:, None] + log_trans, axis=0
        )
        log_alpha_unnorm = log_alpha_pred + log_emit
        ll_t = jax.scipy.special.logsumexp(log_alpha_unnorm)
        log_alpha_norm = log_alpha_unnorm - ll_t
        log_alpha_out = jnp.where(m_t, log_alpha_norm, log_alpha_prev)
        ll_out = jnp.where(m_t, ll_t, 0.0)
        return log_alpha_out, (log_alpha_out, ll_out)

    # t = 0
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


def _dir_nll(
    params: dict[str, Any],
    obs: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """NLL + soft semantic priors for Direction HMM.

    Time complexity: O(T * K²).
    """
    _, lls = _dir_forward(obs, mask, params)
    nll = -jnp.sum(lls)

    mu = params["mu"]
    n_obs = jnp.maximum(jnp.sum(mask), 1.0)

    # BULL μ > 0, BEAR μ < 0, RANGE μ ≈ 0
    mu_prior = n_obs * (
        2000.0 * jnp.square(jnp.maximum(0.0, -mu[0]))   # BULL: μ > 0
        + 2000.0 * jnp.square(jnp.maximum(0.0, mu[2]))  # BEAR: μ < 0
        + 500.0 * jnp.square(mu[1])                     # RANGE: μ ≈ 0
    )

    # State separation: BULL vs BEAR
    sep_prior = n_obs * 1000.0 * jnp.exp(-jnp.abs(mu[0] - mu[2]) * 100.0)

    # Stickiness: diagonal > 0.6
    avg_diag = jnp.diag(jax.nn.softmax(params["log_trans"], axis=1))
    sticky_prior = -500.0 * jnp.sum(jnp.log(jnp.maximum(avg_diag, 1e-6)))

    return nll + mu_prior + sep_prior + sticky_prior


@partial(jax.jit, static_argnames=("n_iter", "optimizer"))
def _train_dir(
    obs: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
    opt_state: Any,
    n_iter: int,
    tol: float,
    optimizer: optax.GradientTransformation,
) -> tuple[dict[str, Any], Any]:
    """JIT-compiled Direction HMM training loop."""
    def cond(state: tuple[Any, ...]) -> jnp.ndarray:
        i, _, _, loss, prev, _ = state
        return (i < n_iter) & (jnp.abs(loss - prev) > tol)

    def body(state: tuple[Any, ...]) -> tuple[Any, ...]:
        i, p, opt_s, loss, _prev, _c = state
        new_loss, grads = jax.value_and_grad(_dir_nll)(p, obs, mask)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        conv = (i > 5) & (jnp.abs(new_loss - loss) < tol)
        return i + 1, new_p, new_opt_s, new_loss, loss, conv

    init_loss = _dir_nll(params, obs, mask)
    state = (0, params, opt_state, init_loss, init_loss + 1e6, False)
    final = jax.lax.while_loop(cond, body, state)
    return final[1], final[2]


def _warmup_dir() -> None:
    """Pre-compile Direction HMM JIT functions."""
    _logger.debug("⚙️  JAX JIT warmup (Dir)")
    dummy_obs = jnp.zeros(_DIR_MAX_LEN)
    dummy_mask = jnp.zeros(_DIR_MAX_LEN)
    p = {
        "log_init": jnp.zeros(_N_DIR_STATES),
        "log_trans": jnp.eye(_N_DIR_STATES) * 5.0,
        "mu": jnp.array([0.3, 0.0, -0.3]),
        "log_sig": jnp.zeros(_N_DIR_STATES),
    }
    opt = optax.adamw(0.01)
    os_ = opt.init(p)
    _dir_forward(dummy_obs, dummy_mask, p)
    _train_dir(dummy_obs, dummy_mask, p, os_, 3, 1e-4, opt)
    _logger.debug("✅ JAX JIT ready (Dir)")


class DirRegimeModel:
    """Layer 2: 3-state Direction HMM on vol-normalised returns.

    Training uses expanding windows at 6h granularity.
    Output probability columns: ["dir_bull", "dir_range", "dir_bear"].
    """

    def __init__(self, n_iter: int = 800, tol: float = 1e-4) -> None:
        """Initialise DirRegimeModel with training hyper-parameters."""
        self.n_iter = n_iter
        self.tol = tol
        self._params: dict[str, Any] | None = None
        self._warmed: bool = False
        self._jax_enabled: bool = bool(
            OPT_FUTURES_CONFIG.get("FUTURES_HMM_JAX_BACKEND_ENABLED", False)
        )

    def _init_params(self, obs_data: np.ndarray) -> dict[str, Any]:
        """Percentile-anchored Direction HMM parameter initialisation."""
        std = float(np.std(obs_data)) if len(obs_data) > 1 else 1.0
        std = max(std, 1e-5)
        return {
            "log_init": jnp.zeros(_N_DIR_STATES),
            "log_trans": jnp.array(
                [
                    [5.0, -1.0, -3.0],  # BULL: persistent
                    [-1.0, 5.0, -1.0],  # RANGE: persistent
                    [-3.0, -1.0, 5.0],  # BEAR: persistent
                ],
                dtype=jnp.float32,
            ),
            "mu": jnp.array([0.5 * std, 0.0, -0.5 * std], dtype=jnp.float32),
            "log_sig": jnp.array(
                [np.log(std), np.log(std * 0.6), np.log(std)], dtype=jnp.float32
            ),
        }

    def fit_predict(
        self,
        norm_returns: pd.Series,
        vol_probs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit Direction HMM and produce per-bar direction posteriors.

        Args:
            norm_returns: Vol-normalised returns (index = 6h DatetimeIndex).
            vol_probs:    Layer 1 output aligned to norm_returns.index
                          (columns: vol_low, vol_mid, vol_high).

        Returns:
            DataFrame(index=norm_returns.index, columns=["dir_bull","dir_range","dir_bear"])

        Time complexity: O(n_windows * n_iter * T * K²).

        """
        if not self._jax_enabled:
            return self._fit_predict_fallback(norm_returns, vol_probs)

        if not self._warmed:
            _warmup_dir()
            self._warmed = True

        obs_arr = norm_returns.fillna(0.0).to_numpy(dtype=np.float64)
        n = len(obs_arr)

        # Clip extreme values for numerical stability
        obs_arr = np.clip(obs_arr, -10.0, 10.0).astype(np.float32)

        probs_all = np.full((n, _N_DIR_STATES), 1.0 / _N_DIR_STATES, dtype=np.float64)
        opt = optax.adamw(learning_rate=0.02, weight_decay=1e-4)

        first_fit = True
        for t in range(200, n, 84):
            win_start = max(0, t - _DIR_MAX_LEN)
            L = t - win_start
            obs_win = obs_arr[win_start:t]

            obs_pad = np.zeros(_DIR_MAX_LEN, dtype=np.float32)
            m_pad = np.zeros(_DIR_MAX_LEN, dtype=np.float32)
            obs_pad[:L] = obs_win
            m_pad[:L] = 1.0

            if first_fit or self._params is None:
                p = self._init_params(obs_win)
                opt_state = opt.init(p)
                iters = self.n_iter
                first_fit = False
            else:
                p = self._params
                opt_state = opt.init(p)
                iters = 300

            self._params, _ = _train_dir(
                jnp.array(obs_pad),
                jnp.array(m_pad),
                p,
                opt_state,
                iters,
                self.tol,
                opt,
            )

            # Inference on [win_start, inf_end) — cap so that length <= _DIR_MAX_LEN
            inf_end = min(t + 84, n, win_start + _DIR_MAX_LEN)
            obs_inf = obs_arr[win_start:inf_end]
            L_inf = len(obs_inf)  # guaranteed <= _DIR_MAX_LEN

            obs_p = np.zeros(_DIR_MAX_LEN, dtype=np.float32)
            m_p = np.zeros(_DIR_MAX_LEN, dtype=np.float32)
            obs_p[:L_inf] = obs_inf
            m_p[:L_inf] = 1.0

            log_alphas, _ = _dir_forward(
                jnp.array(obs_p), jnp.array(m_p), self._params
            )
            post = np.asarray(jax.nn.softmax(log_alphas[:L_inf], axis=1))
            probs_all[win_start:inf_end] = post

        return pd.DataFrame(
            probs_all,
            index=norm_returns.index,
            columns=["dir_bull", "dir_range", "dir_bear"],
        )

    @staticmethod
    def _fit_predict_fallback(
        norm_returns: pd.Series,
        vol_probs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stable NumPy fallback to avoid JAX segfaults on long runs."""
        idx = norm_returns.index
        x = norm_returns.reindex(idx).fillna(0.0).to_numpy(dtype=np.float64)
        x = np.clip(x, -10.0, 10.0)

        vol_hi = (
            vol_probs["vol_high"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
            if "vol_high" in vol_probs.columns
            else np.zeros_like(x)
        )
        vol_mid = (
            vol_probs["vol_mid"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
            if "vol_mid" in vol_probs.columns
            else np.zeros_like(x)
        )

        # Smoothened directional signal + volatility-aware range boost
        x_s = pd.Series(x, index=idx).ewm(span=8, adjust=False).mean().to_numpy(dtype=np.float64)
        range_penalty = np.abs(x_s)
        range_boost = 0.6 * vol_mid + 0.35 * vol_hi

        logit_bull = 1.15 * x_s - 0.45 * vol_hi
        logit_bear = -1.15 * x_s + 0.15 * vol_hi
        logit_range = 0.90 - 1.25 * range_penalty + range_boost

        logits = np.column_stack([logit_bull, logit_range, logit_bear])
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)

        return pd.DataFrame(
            probs,
            index=idx,
            columns=["dir_bull", "dir_range", "dir_bear"],
        )
