from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.ml_pipeline.regime.jax_hmm import (
    JAXMultivariateHMM,
    _hmm_forward,
    _train_hmm,
)


def _synthetic_obs(n: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    f1 = 0.5 * np.sin(t / 13.0) + 0.2 * np.sign(np.sin(t / 41.0)) + 0.10 * rng.normal(size=n)
    f2 = 0.6 * np.cos(t / 19.0) + 0.08 * rng.normal(size=n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"f1": f1, "f2": f2}, index=idx)


def _to_obs_arr(df: pd.DataFrame) -> np.ndarray:
    arr = df.fillna(0.0).to_numpy(dtype=np.float64)
    return np.clip(arr, -5.0, 5.0).astype(np.float32)


def _fit_params(train_df: pd.DataFrame, n_iter: int = 25, tol: float = 1e-6):
    model = JAXMultivariateHMM(n_iter=n_iter, tol=tol)
    obs = _to_obs_arr(train_df)
    params = model._init_params(obs)
    optimizer = optax.adamw(learning_rate=0.03, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    mask = np.ones(len(obs), dtype=np.float32)
    fitted, _ = _train_hmm(
        jnp.array(obs),
        jnp.array(mask),
        params,
        opt_state,
        model.n_iter,
        model.tol,
        optimizer,
    )
    return fitted


def _filter_probs(df: pd.DataFrame, params) -> np.ndarray:
    obs = _to_obs_arr(df)
    mask = np.ones(len(obs), dtype=np.float32)
    log_alphas, _ = _hmm_forward(jnp.array(obs), jnp.array(mask), params)
    return np.asarray(jax.nn.softmax(log_alphas, axis=1))


def test_hmm_filtered_prefix_truncation_invariant_under_fixed_params() -> None:
    train_n = 140
    prefix_n = 210
    full_n = 260

    df = _synthetic_obs(full_n)
    params = _fit_params(df.iloc[:train_n], n_iter=20, tol=1e-6)

    probs_prefix = _filter_probs(df.iloc[:prefix_n], params)
    probs_full = _filter_probs(df, params)

    np.testing.assert_allclose(probs_prefix, probs_full[:prefix_n], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(probs_full.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
