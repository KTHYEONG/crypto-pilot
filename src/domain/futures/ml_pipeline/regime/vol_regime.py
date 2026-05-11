"""Layer 1: MS-GARCH with Skew-t innovations -- 3-state Volatility Regime Model.

Mathematical Model:
    r_t        = mu_{s_t} + eps_t
    sig2_{t,k} = omega_k + alpha_k * eps2_{t-1} + beta_k * sig2_{t-1,k}  (GARCH per state)
    z_t        = eps_t / sig_{t,s_t}                                      (standardized)
    z_t        ~ SkewT(nu_{s_t}, lam_{s_t})                               (split-t)
    P(s_t|s_{t-1}, Z_t) = TVTP(W, Z_t)                                   (time-varying)

States:
    0: LOW_VOL  (calm bull, low persistence variance)
    1: MID_VOL  (transitional, moderate vol)
    2: HIGH_VOL (crash/crisis, left-skewed, high vol)
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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
jax.config.update("jax_platform_name", "cpu")

_logger = logging.getLogger(__name__)

_N_VOL_STATES: int = 3

# SYSTEMIC_HMM_FEATURE_COLUMNS indices:
# 0=macro_trend_168h, 1=macro_trend_24h, 2=macro_vol_24h, 3=macro_downside_vol_24h,
# 4=macro_cs_dispersion_24h, 5=macro_oi_delta_24h, 6=macro_funding_mom_24h,
# 7=macro_liq_proxy_24h, 8=macro_lsr_delta_24h, 9=macro_ret_1h, 10=macro_breadth_168h
_TVTP_FEAT_IDX: tuple[int, ...] = (4, 5, 6)  # cs_dispersion, oi_delta, funding_mom
_N_TVTP: int = len(_TVTP_FEAT_IDX)

MAX_LEN: int = 1500


@jax.jit
def _skew_t_log_pdf(z: jnp.ndarray, log_nu: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Split-t log-pdf for scalar (or element-wise broadcastable) inputs.

    Args:
        z:      Standardized residual.
        log_nu: log(nu - 2) so that nu = exp(log_nu) + 2 > 2 (finite variance).
        lam:    Skewness parameter; lam < 0 = left-heavy tail.

    Returns:
        Scalar log-pdf value.

    Time complexity: O(1) per element.

    """
    nu = jnp.exp(log_nu) + 2.0
    z_adj = z * jnp.exp(-lam * jnp.sign(z))
    log_t = (
        jax.scipy.special.gammaln((nu + 1.0) / 2.0)
        - jax.scipy.special.gammaln(nu / 2.0)
        - 0.5 * jnp.log(jnp.pi * nu)
        - (nu + 1.0) / 2.0 * jnp.log1p(z_adj ** 2 / nu)
    )
    log_jacobian = -lam * jnp.sign(z)
    log_norm = -jnp.log(2.0 * jnp.cosh(lam / 2.0 + 1e-8))
    return log_t + log_jacobian + log_norm


@jax.jit
def _ms_garch_forward(
    returns: jnp.ndarray,
    mask: jnp.ndarray,
    tvtp_z: jnp.ndarray,
    params: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Hamilton (1994) MS-GARCH filter in log-space.

    Args:
        returns:  (MAX_LEN,) 12h compound returns (padded).
        mask:     (MAX_LEN,) 1.0 for valid bars, 0.0 for padding.
        tvtp_z:   (MAX_LEN, _N_TVTP) scaled macro TVTP features.
        params:   Parameter dict (see VolRegimeModel._init_params).

    Returns:
        log_alphas:     (MAX_LEN, K) filtered log-posteriors (log-space).
        incremental_ll: (MAX_LEN,)  per-step log-likelihood increments.

    Time complexity: O(T * K²).

    """
    K = _N_VOL_STATES
    mu = params["mu"]                                                          # (K,)
    omega = jnp.exp(params["log_omega"]) + 1e-8                               # (K,)
    alpha_g = jax.nn.sigmoid(params["logit_alpha"])                           # (K,) in (0,1)
    beta_g = jax.nn.sigmoid(params["logit_beta"]) * (1.0 - alpha_g) * 0.99   # (K,) a+b < 1
    log_nu = params["log_nu"]                                                  # (K,)
    lam = params["lam"]                                                        # (K,)
    tvtp_W = params["tvtp_W"]                                                  # (K, K, _N_TVTP)
    tvtp_b = params["tvtp_b"]                                                  # (K, K)

    def step_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        log_alpha_prev, sigma2_prev, eps_prev = carry
        r_t, m_t, z_t = inputs

        # GARCH variance update with previous expected residual
        eps2_prev = eps_prev ** 2
        sigma2_t = omega + alpha_g * eps2_prev + beta_g * sigma2_prev

        # Time-varying transition logits: logit_{k→j} = Σ_d W[k,j,d]*z_t[d] + b[k,j]
        logits_kj = jnp.einsum("kjd,d->kj", tvtp_W, z_t) + tvtp_b  # (K, K)
        log_trans = jax.nn.log_softmax(logits_kj, axis=1)            # row=from, col=to

        # Prediction step: marginalise previous state
        log_alpha_pred = jax.scipy.special.logsumexp(
            log_alpha_prev[:, None] + log_trans, axis=0
        )  # (K,)

        # Emission: skew-t on standardized residual
        sigma_t = jnp.sqrt(jnp.maximum(sigma2_t, 1e-12))
        z_stand = (r_t - mu) / sigma_t  # (K,)
        log_emit = jax.vmap(lambda zi, lni, li: _skew_t_log_pdf(zi, lni, li))(
            z_stand, log_nu, lam
        )  # (K,)

        # Update step
        log_alpha_unnorm = log_alpha_pred + log_emit
        ll_t = jax.scipy.special.logsumexp(log_alpha_unnorm)
        log_alpha_norm = log_alpha_unnorm - ll_t

        # Masking: skip update on padding
        log_alpha_out = jnp.where(m_t, log_alpha_norm, log_alpha_prev)
        ll_out = jnp.where(m_t, ll_t, 0.0)

        # Expected residual for next GARCH step (posterior-weighted)
        post = jax.nn.softmax(log_alpha_out)
        eps_new = jnp.sum(post * (r_t - mu))

        new_carry = (log_alpha_out, sigma2_t, eps_new)
        return new_carry, (log_alpha_out, ll_out)

    # ── Initialise t=0 ──────────────────────────────────────────────────────
    init_sigma2 = jnp.var(returns) * jnp.ones(K) + 1e-8
    log_init = jax.nn.log_softmax(params["log_init"])

    sigma2_0 = omega + alpha_g * 0.0 + beta_g * init_sigma2
    sigma_0 = jnp.sqrt(jnp.maximum(sigma2_0, 1e-12))
    z0_stand = (returns[0] - mu) / sigma_0
    log_emit_0 = jax.vmap(lambda zi, lni, li: _skew_t_log_pdf(zi, lni, li))(
        z0_stand, log_nu, lam
    )
    log_alpha_0_unnorm = log_init + log_emit_0
    ll_0 = jax.scipy.special.logsumexp(log_alpha_0_unnorm)
    log_alpha_0 = log_alpha_0_unnorm - ll_0
    log_alpha_0 = jnp.where(mask[0], log_alpha_0, log_init)
    ll_0_out = jnp.where(mask[0], ll_0, 0.0)
    eps_0 = jnp.sum(jax.nn.softmax(log_alpha_0) * (returns[0] - mu))

    carry_init = (log_alpha_0, sigma2_0, eps_0)
    _, (log_alphas_tail, lls_tail) = jax.lax.scan(
        step_fn, carry_init, (returns[1:], mask[1:], tvtp_z[1:])
    )

    log_alphas = jnp.concatenate([log_alpha_0[None], log_alphas_tail], axis=0)
    lls = jnp.concatenate([ll_0_out[None], lls_tail], axis=0)
    return log_alphas, lls


def _ms_garch_nll(
    params: dict[str, Any],
    returns: jnp.ndarray,
    mask: jnp.ndarray,
    tvtp_z: jnp.ndarray,
) -> jnp.ndarray:
    """Negative log-likelihood for MS-GARCH with soft stationarity & skew priors.

    Time complexity: O(T * K²) forward pass + O(K) prior terms.
    """
    _, lls = _ms_garch_forward(returns, mask, tvtp_z, params)
    nll = -jnp.sum(lls)

    # Soft priors on semantic state identity
    mu = params["mu"]
    lam = params["lam"]

    # HIGH_VOL (2): mu < -0.1%, lam < -0.2 (left-skewed)
    prior = (
        5000.0 * jnp.square(jnp.maximum(0.0, mu[2] + 0.001))
        + 1000.0 * jnp.square(jnp.maximum(0.0, lam[2] + 0.2))
        + 1000.0 * jnp.square(jnp.maximum(0.0, -lam[0]))
    )

    # GARCH stationarity: alpha + beta < 0.98
    alpha_g = jax.nn.sigmoid(params["logit_alpha"])
    beta_g = jax.nn.sigmoid(params["logit_beta"]) * (1.0 - alpha_g) * 0.99
    stat_penalty = 1000.0 * jnp.sum(
        jnp.square(jnp.maximum(0.0, alpha_g + beta_g - 0.98))
    )

    return nll + prior + stat_penalty


@partial(jax.jit, static_argnames=("n_iter", "optimizer"))
def _train_vol_regime(
    returns: jnp.ndarray,
    mask: jnp.ndarray,
    tvtp_z: jnp.ndarray,
    params: dict[str, Any],
    opt_state: Any,
    n_iter: int,
    tol: float,
    optimizer: optax.GradientTransformation,
) -> tuple[dict[str, Any], Any, jnp.ndarray]:
    """JIT-compiled training loop with early-stopping via while_loop.

    Time complexity: O(n_iter * T * K²).
    """
    def cond(state: tuple[Any, ...]) -> jnp.ndarray:
        i, _, _, loss, prev_loss, _ = state
        return (i < n_iter) & (jnp.abs(loss - prev_loss) > tol)

    def body(state: tuple[Any, ...]) -> tuple[Any, ...]:
        i, p, opt_s, loss, _prev, _conv = state
        new_loss, grads = jax.value_and_grad(_ms_garch_nll)(p, returns, mask, tvtp_z)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        conv = (i > 5) & (jnp.abs(new_loss - loss) < tol)
        return i + 1, new_p, new_opt_s, new_loss, loss, conv

    init_loss = _ms_garch_nll(params, returns, mask, tvtp_z)
    state = (0, params, opt_state, init_loss, init_loss + 1e6, False)
    final = jax.lax.while_loop(cond, body, state)
    return final[1], final[2], final[3]


class VolRegimeModel:
    """Layer 1: 3-state MS-GARCH with Skew-t innovations.

    Expanding-window training at 12h granularity with warm-start.
    Output probability columns: ["vol_low", "vol_mid", "vol_high"].
    """

    def __init__(self, n_iter: int = 1500, tol: float = 1e-4) -> None:
        """Initialise VolRegimeModel with training hyper-parameters."""
        self.n_iter = n_iter
        self.tol = tol
        self._params: dict[str, Any] | None = None
        self._warmed: bool = False

    def _init_params(self, ret_data: np.ndarray) -> dict[str, Any]:
        """Percentile-anchored parameter initialisation."""
        K = _N_VOL_STATES
        vol = float(np.std(ret_data)) if len(ret_data) > 1 else 0.01
        vol = max(vol, 1e-5)
        return {
            "log_init": jnp.zeros(K),
            "mu": jnp.array([0.0002, 0.0, -0.0005], dtype=jnp.float32),
            "log_omega": jnp.log(
                jnp.array([vol * 0.1, vol * 0.3, vol * 0.8], dtype=jnp.float32)
                + 1e-8
            ),
            "logit_alpha": jnp.array([-2.0, -1.0, 0.0], dtype=jnp.float32),
            "logit_beta": jnp.array([2.0, 1.5, 1.0], dtype=jnp.float32),
            "log_nu": jnp.array([2.0, 1.5, 1.0], dtype=jnp.float32),
            "lam": jnp.array([0.1, -0.1, -0.5], dtype=jnp.float32),
            "tvtp_W": jnp.zeros((K, K, _N_TVTP), dtype=jnp.float32),
            "tvtp_b": jnp.eye(K, dtype=jnp.float32) * 8.0,
        }

    def _warmup(self) -> None:
        """Pre-compile JAX functions with dummy tensors."""
        if self._warmed:
            return
        _logger.debug("⚙️  JAX JIT warmup (Vol)")
        dummy_r = jnp.zeros(MAX_LEN)
        dummy_m = jnp.zeros(MAX_LEN)
        dummy_z = jnp.zeros((MAX_LEN, _N_TVTP))
        p = self._init_params(np.zeros(100))
        opt = optax.adamw(0.01)
        os_init = opt.init(p)
        _ms_garch_forward(dummy_r, dummy_m, dummy_z, p)
        _train_vol_regime(dummy_r, dummy_m, dummy_z, p, os_init, 3, 1e-4, opt)
        self._warmed = True
        _logger.debug("✅ JAX JIT ready (Vol)")

    def fit_predict(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
    ) -> pd.DataFrame:
        """Fit MS-GARCH and produce per-bar vol-state posteriors.

        Args:
            features_df: DataFrame indexed by DatetimeIndex with SYSTEMIC_HMM_FEATURE_COLUMNS.
                         Should already be resampled to 12h by the caller.
            returns_ser: Compound returns aligned to features_df.index.
            is_end_idx:  Unused; kept for interface consistency.

        Returns:
            DataFrame(index=features_df.index, columns=["vol_low","vol_mid","vol_high"])

        Time complexity: O(n_windows * n_iter * T * K²).

        """
        from src.domain.futures.ml_pipeline.features.engineering import (
            SYSTEMIC_HMM_FEATURE_COLUMNS,
        )

        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        ret_arr = (
            returns_ser.reindex(features_df.index).fillna(0.0).to_numpy(dtype=np.float64)
        )
        tvtp_cols = [feat_cols[i] for i in _TVTP_FEAT_IDX]
        tvtp_arr = (
            features_df[tvtp_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

        # IQR robust scaling of TVTP features
        med = np.nanmedian(tvtp_arr, axis=0)
        iqr = np.nanpercentile(tvtp_arr, 75, axis=0) - np.nanpercentile(
            tvtp_arr, 25, axis=0
        )
        iqr = np.where(np.abs(iqr) < 1e-9, 1.0, iqr)
        tvtp_scaled = np.clip((tvtp_arr - med) / iqr, -5.0, 5.0)

        n = len(ret_arr)
        self._warmup()

        probs_all = np.full((n, _N_VOL_STATES), 1.0 / _N_VOL_STATES, dtype=np.float64)
        opt = optax.adamw(learning_rate=0.02, weight_decay=1e-4)

        first_fit = True
        for t in range(400, n, 168):
            win_start = max(0, t - MAX_LEN)
            L = t - win_start
            r_win = ret_arr[win_start:t].astype(np.float32)
            z_win = tvtp_scaled[win_start:t].astype(np.float32)

            # Padded training tensors (static shape for JIT)
            r_pad = np.zeros(MAX_LEN, dtype=np.float32)
            m_pad = np.zeros(MAX_LEN, dtype=np.float32)
            z_pad = np.zeros((MAX_LEN, _N_TVTP), dtype=np.float32)
            r_pad[:L] = r_win
            m_pad[:L] = 1.0
            z_pad[:L] = z_win

            if first_fit or self._params is None:
                p = self._init_params(r_win)
                opt_state = opt.init(p)
                iters = self.n_iter
                first_fit = False
            else:
                p = self._params
                opt_state = opt.init(p)
                iters = 500

            self._params, _, _ = _train_vol_regime(
                jnp.array(r_pad),
                jnp.array(m_pad),
                jnp.array(z_pad),
                p,
                opt_state,
                iters,
                self.tol,
                opt,
            )

            # Inference on [win_start, inf_end) — cap so that length <= MAX_LEN
            inf_end = min(t + 168, n, win_start + MAX_LEN)
            r_inf = ret_arr[win_start:inf_end].astype(np.float32)
            z_inf = tvtp_scaled[win_start:inf_end].astype(np.float32)
            L_inf = len(r_inf)  # guaranteed <= MAX_LEN

            r_p = np.zeros(MAX_LEN, dtype=np.float32)
            m_p = np.zeros(MAX_LEN, dtype=np.float32)
            z_p = np.zeros((MAX_LEN, _N_TVTP), dtype=np.float32)
            r_p[:L_inf] = r_inf
            m_p[:L_inf] = 1.0
            z_p[:L_inf] = z_inf

            log_alphas, _ = _ms_garch_forward(
                jnp.array(r_p), jnp.array(m_p), jnp.array(z_p), self._params
            )
            post = np.asarray(jax.nn.softmax(log_alphas[:L_inf], axis=1))
            probs_all[win_start:inf_end] = post

        return pd.DataFrame(
            probs_all,
            index=features_df.index,
            columns=["vol_low", "vol_mid", "vol_high"],
        )
