"""Skewed-t emission HMM backend with Parallel Associative Scan (v11.5 - GPU Native).

This backend implements Skewed-t emissions for better capture of asymmetric 
market regimes and utilizes JAX's lax.scan for extreme GPU-native 
Walk-Forward optimization on RTX 40-series Tensor Cores.
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

# 1. TF32 활성화 (RTX 4070 Ti Tensor Core 가속)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

_logger = logging.getLogger(__name__)

_N_STATES = 4
_N_FEATURES = 6
_MAX_LEN = 3000
_EPS = 1e-6


@jax.jit
def _t_cdf_approx(x: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    """Robust approximation of Student-t CDF using Normal CDF (ndtr)."""
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
    """Diagonal multivariate Skewed-t log-pdf."""
    nu = jnp.clip(jax.nn.softplus(nu_raw) + 2.1, 2.1, 100.0)[:, None]
    nu_p1 = nu + 1.0
    sig = jnp.exp(jnp.clip(log_sig, -6.0, 3.0))
    alpha = lambda_raw
    
    z = (x - mu) / jnp.maximum(sig, _EPS)
    z2 = jnp.square(z)

    log_norm = (
        jax.scipy.special.gammaln(nu_p1 * 0.5)
        - jax.scipy.special.gammaln(nu * 0.5)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(jnp.maximum(sig, _EPS))
    )
    log_kernel = -0.5 * nu_p1 * jnp.log1p(z2 / jnp.maximum(nu, _EPS))
    log_t_pdf = log_norm + log_kernel

    skew_arg = alpha * z * jnp.sqrt(nu_p1 / (nu + z2))
    skew_log_cdf = jnp.log(jnp.maximum(_t_cdf_approx(skew_arg, nu_p1), _EPS))
    
    logp = jnp.sum(jnp.log(2.0) + log_t_pdf + skew_log_cdf, axis=1)
    return jnp.nan_to_num(logp, nan=-1e6, posinf=-1e6, neginf=-1e6)


@jax.jit
def _log_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Log-space matrix multiplication."""
    return jax.scipy.special.logsumexp(A[..., :, :, None] + B[..., None, :, :], axis=-2)


@jax.jit
def _hmm_forward(
    obs: jnp.ndarray,
    mask: jnp.ndarray,
    params: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Parallel forward filtering pass."""
    mu = params["mu"]
    log_sig = params["log_sig"]
    nu_raw = params["nu_raw"]
    lambda_raw = params["lambda_raw"]
    log_init = jax.nn.log_softmax(params["log_init"])
    log_trans = jax.nn.log_softmax(params["log_trans"], axis=1)

    log_emits = jax.vmap(_skewed_t_log_pdf, in_axes=(0, None, None, None, None))(
        obs, mu, log_sig, nu_raw, lambda_raw
    )
    
    log_Ms = log_trans[None, :, :] + log_emits[:, None, :]
    
    eye_log = jnp.where(jnp.eye(_N_STATES) > 0.5, 0.0, -1e6)
    log_Ms = jnp.where(mask[:, None, None] > 0.5, log_Ms, eye_log)

    prefix_Ms = jax.lax.associative_scan(_log_matmul, log_Ms)
    
    log_alphas_unnorm = jax.vmap(lambda M: jax.scipy.special.logsumexp(log_init[:, None] + M, axis=0))(prefix_Ms)
    lls_total = jax.scipy.special.logsumexp(log_alphas_unnorm, axis=1)
    log_alphas = log_alphas_unnorm - lls_total[:, None]
    
    lls = jnp.concatenate([lls_total[:1], jnp.diff(lls_total)], axis=0)
    lls = jnp.where(mask, lls, 0.0)
    
    return log_alphas, lls


def _compute_invariants(obs: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_feats = obs.shape[1]
    raw_tail_flag = (obs[:, 0] < -1.5) | (obs[:, 2] > 1.5 if n_feats > 2 else obs[:, 0] < -1.5)
    tail_flag = raw_tail_flag * mask
    outcome_w = jnp.where(tail_flag > 0.5, 2.5, 1.0)
    n_obs = jnp.maximum(jnp.sum(mask), 1.0)
    vol_z = obs[:, 1]
    avg_vol = jnp.sum(vol_z * mask) / n_obs
    sticky_weight = 3200.0 * (1.0 - 0.5 * jax.nn.sigmoid(2.0 * (avg_vol - 1.0)))
    return tail_flag, outcome_w, sticky_weight, n_obs


def _hmm_nll(
    params: dict[str, Any],
    obs: jnp.ndarray,
    mask: jnp.ndarray,
    tail_flag: Any = None,
    outcome_w: Any = None,
    sticky_weight: Any = None,
    n_obs: Any = None,
) -> jnp.ndarray:
    if tail_flag is None:
        tail_flag, outcome_w, sticky_weight, n_obs = _compute_invariants(obs, mask)

    log_alphas, lls = _hmm_forward(obs, mask, params)
    nll = -jnp.sum(lls * outcome_w)

    probs = jax.nn.softmax(log_alphas, axis=1)
    avg_occupancy = jnp.sum(probs * mask[:, None], axis=0) / n_obs
    occ_penalty = 5000.0 * jnp.sum(jnp.square(jnp.maximum(0.0, 0.05 - avg_occupancy)))
    tail_force_penalty = -2.0 * jnp.sum(tail_flag * log_alphas[:, 1])

    trans_probs = jax.nn.softmax(params["log_trans"], axis=1)
    diag = jnp.diag(trans_probs)
    
    sticky_prior = -sticky_weight * jnp.sum(jnp.log(jnp.maximum(diag, _EPS)))

    mu = params["mu"]
    semantic_prior = n_obs * (
        1000.0 * jnp.square(jnp.maximum(0.0, 0.6 - mu[0, 0]))
        + 1000.0 * jnp.square(jnp.maximum(0.0, mu[1, 0] + 0.6))
        + 500.0 * jnp.square(jnp.maximum(0.0, 1.0 - mu[2, 1]))
        + 500.0 * jnp.square(jnp.maximum(0.0, mu[3, 1] + 0.6))
        + 800.0 * (jnp.square(mu[2, 0]) + jnp.square(mu[3, 0]))
    )

    nu = jax.nn.softplus(params["nu_raw"]) + 2.1
    nu_targets = jnp.array([40.0, 3.0, 10.0, 15.0])
    nu_prior = 15.0 * jnp.sum(jnp.square(jnp.log(nu) - jnp.log(nu_targets)))

    lambdas = params["lambda_raw"]
    skew_prior = n_obs * (
        0.8 * jnp.square(jnp.maximum(0.0, 0.5 - lambdas[0, 0]))
        + 2.5 * jnp.square(jnp.maximum(0.0, lambdas[1, 0] + 1.5))
    )
    skew_l2 = 0.1 * jnp.sum(jnp.square(lambdas))

    loss = nll + occ_penalty + tail_force_penalty + sticky_prior + semantic_prior + nu_prior + skew_prior + skew_l2
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
    
    tail_flag, outcome_w, sticky_weight, n_obs = _compute_invariants(obs, mask)

    def cond(state):
        i, _, _, loss, prev = state
        return (i < n_iter) & (jnp.abs(loss - prev) > tol * jnp.maximum(jnp.abs(prev), 1.0))

    def body(state):
        i, p, opt_s, loss, _prev = state
        new_loss, grads = jax.value_and_grad(_hmm_nll)(p, obs, mask, tail_flag, outcome_w, sticky_weight, n_obs)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
        )
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return i + 1, new_p, new_opt_s, new_loss, loss

    init_loss = _hmm_nll(params, obs, mask, tail_flag, outcome_w, sticky_weight, n_obs)
    state = (0, params, opt_state, init_loss, init_loss + 1e6)
    final = jax.lax.while_loop(cond, body, state)
    return final[1], final[2]


@partial(jax.jit, static_argnames=("n_iter", "optimizer"))
def _scan_step(
    carry: tuple[dict[str, Any], Any],
    t_end: jnp.ndarray,
    obs_full: jnp.ndarray,
    n_iter: int,
    tol: float,
    optimizer: optax.GradientTransformation,
) -> tuple[tuple[dict[str, Any], Any], None]:
    """GPU-Native Walk-Forward step using lax.scan."""
    params, opt_state = carry
    
    win_start = jnp.maximum(0, t_end - _MAX_LEN)
    cur_len = t_end - win_start
    
    # 윈도우 추출 (GPU 내재화)
    obs_win = jax.lax.dynamic_slice(obs_full, (win_start, 0), (_MAX_LEN, _N_FEATURES))
    mask = jnp.where(jnp.arange(_MAX_LEN) < cur_len, 1.0, 0.0)
    
    # 모델 학습
    new_params, new_opt_state = _train_hmm(
        obs_win, mask, params, opt_state, n_iter, tol, optimizer
    )
    return (new_params, new_opt_state), None


class SkewedTMultivariateHMM:
    """4-state HMM with GPU-Native Scan optimization (v11.5)."""

    def __init__(self, n_iter: int = 1000, tol: float = 1e-4):
        self.n_iter = n_iter
        self.tol = tol
        self._params = None
        self._warmed = False
        
        devices = jax.devices()
        _logger.info("JAX devices: %s", devices)
        self._device = devices[0]

    def _init_params(self, obs: np.ndarray, init_type: str = "standard") -> dict[str, Any]:
        f = [obs[:, i] for i in range(obs.shape[1])]
        def get_mu(p_list):
            return [np.percentile(f[i], p_list[i]) for i in range(len(f))]

        if init_type == "tail_sensitive":
            mu = [get_mu([65, 30, 30, 50, 60, 60]), get_mu([10, 60, 90, 50, 20, 20]),
                  get_mu([50, 80, 50, 85, 50, 50]), get_mu([50, 20, 30, 15, 50, 50])]
        elif init_type == "trend_focused":
            mu = [get_mu([90, 40, 30, 50, 80, 80]), get_mu([10, 40, 70, 50, 20, 20]),
                  get_mu([50, 70, 50, 70, 50, 50]), get_mu([50, 30, 50, 30, 50, 50])]
        else:
            mu = [get_mu([75, 25, 20, 50, 70, 70]), get_mu([25, 50, 80, 50, 30, 30]),
                  get_mu([50, 75, 50, 80, 50, 50]), get_mu([50, 10, 25, 20, 50, 50])]
        
        return {
            "log_init": jnp.zeros(_N_STATES),
            "log_trans": jnp.eye(_N_STATES) * 4.5,
            "mu": jnp.array(mu, dtype=jnp.float32),
            "log_sig": jnp.zeros((_N_STATES, _N_FEATURES), dtype=jnp.float32) - 0.4,
            "nu_raw": jnp.ones((_N_STATES,), dtype=jnp.float32) * 2.0,
            "lambda_raw": jnp.zeros((_N_STATES, _N_FEATURES), dtype=jnp.float32),
        }

    def _warmup(self) -> None:
        if self._warmed: return
        _logger.debug("Warmup on %s...", self._device)
        dummy_obs = jnp.zeros((_MAX_LEN, _N_FEATURES), dtype=jnp.float32)
        dummy_mask = jnp.ones((_MAX_LEN,), dtype=jnp.float32)
        p = self._init_params(np.random.normal(size=(_MAX_LEN, _N_FEATURES)).astype(np.float32))
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
        safe = np.clip(np.nan_to_num(probs, nan=1.0/_N_STATES), _EPS, 1.0)
        safe = safe / np.maximum(safe.sum(axis=1, keepdims=True), _EPS)
        return pd.DataFrame(safe, index=obs_df.index, columns=["bull_trend", "bear_trend", "chop_high", "chop_low"])

    def fit(self, obs_df: pd.DataFrame) -> SkewedTMultivariateHMM:
        obs_arr = self._prep_obs(obs_df)
        n = len(obs_arr)
        if n <= 0: return self

        self._warmup()
        opt = optax.adamw(learning_rate=0.02, weight_decay=1e-4)
        
        # 2. Host-to-Device 통신 최소화: 전체 데이터를 GPU로 업로드
        # Pad with zeros at the end to allow dynamic_slice to always take _MAX_LEN
        obs_full = jnp.zeros((n + _MAX_LEN, _N_FEATURES), dtype=jnp.float32)
        obs_full = obs_full.at[:n, :].set(jnp.array(obs_arr))
        
        t_starts = np.arange(min(n, 500), n + 168, 168)
        if len(t_starts) == 0: return self
        
        # First step: Multi-start or Initial Training
        t_first = t_starts[0]
        win_start = max(0, t_first - _MAX_LEN)
        obs_win_first = obs_arr[win_start:t_first]
        cur_len_first = len(obs_win_first)
        
        obs_pad = np.zeros((_MAX_LEN, _N_FEATURES), dtype=np.float32)
        m_pad = np.zeros(_MAX_LEN, dtype=np.float32)
        obs_pad[:cur_len_first] = obs_win_first
        m_pad[:cur_len_first] = 1.0
        
        obs_j = jnp.array(obs_pad)
        m_j = jnp.array(m_pad)
        
        if self._params is None:
            best_loss, best_p = 1e18, None
            for itype in ["standard", "tail_sensitive", "trend_focused"]:
                p_c = self._init_params(obs_win_first, init_type=itype)
                os = opt.init(p_c)
                p_f, _ = _train_hmm(obs_j, m_j, p_c, os, self.n_iter, self.tol, opt)
                loss = float(_hmm_nll(p_f, obs_j, m_j))
                if loss < best_loss:
                    best_loss, best_p = loss, p_f
            self._params = best_p
        else:
            os = opt.init(self._params)
            self._params, _ = _train_hmm(obs_j, m_j, self._params, os, self.n_iter, self.tol, opt)
            
        # 3. Walk-Forward Loop의 GPU 내재화: jax.lax.scan 사용
        if len(t_starts) > 1:
            t_remaining = jnp.array(t_starts[1:])
            init_os = opt.init(self._params)
            
            (final_params, _), _ = jax.lax.scan(
                partial(_scan_step, obs_full=obs_full, n_iter=self.n_iter, tol=self.tol, optimizer=opt),
                (self._params, init_os),
                t_remaining
            )
            self._params = final_params

        return self

    def filter(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        obs_arr = self._prep_obs(obs_df)
        n = len(obs_arr)
        if n <= 0: return self._to_prob_df(np.empty((0, _N_STATES)), obs_df)

        self._warmup()
        if self._params is None: self.fit(obs_df)

        m = jnp.ones((n,), dtype=jnp.float32)
        log_alphas, _ = _hmm_forward(jnp.array(obs_arr), m, self._params)
        probs = np.asarray(jax.nn.softmax(log_alphas, axis=1))
        return self._to_prob_df(probs, obs_df)

    def fit_filter_train_oos(self, obs_df: pd.DataFrame, is_end_idx: int) -> pd.DataFrame:
        n = len(obs_df)
        if n <= 0: return self._to_prob_df(np.empty((0, _N_STATES)), obs_df)
        cut = int(np.clip(is_end_idx, 0, n - 1))
        self._params = None
        self.fit(obs_df.iloc[: cut + 1])
        return self.filter(obs_df)

    def fit_predict(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        n = len(obs_df)
        if n <= 0: return self._to_prob_df(np.empty((0, _N_STATES)), obs_df)
        return self.fit_filter_train_oos(obs_df, n - 1)
