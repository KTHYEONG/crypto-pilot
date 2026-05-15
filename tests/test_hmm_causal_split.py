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


def _synthetic_obs(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    f1 = 0.8 * np.sin(t / 17.0) + 0.15 * rng.normal(size=n)
    f2 = 0.7 * np.cos(t / 23.0) + 0.12 * rng.normal(size=n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"f1": f1, "f2": f2}, index=idx)


def _to_obs_arr(df: pd.DataFrame) -> np.ndarray:
    arr = df.fillna(0.0).to_numpy(dtype=np.float64)
    return np.clip(arr, -5.0, 5.0).astype(np.float32)


class _CausalSplitHarness:
    """Minimal fit(train)/filter(oos) harness using frozen HMM params."""

    def __init__(self, n_iter: int = 30, tol: float = 1e-6) -> None:
        self._model = JAXMultivariateHMM(n_iter=n_iter, tol=tol)
        self._params = None
        self._train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        obs = _to_obs_arr(train_df)
        params = self._model._init_params(obs)
        optimizer = optax.adamw(learning_rate=0.03, weight_decay=1e-4)
        opt_state = optimizer.init(params)
        mask = np.ones(len(obs), dtype=np.float32)
        self._params, _ = _train_hmm(
            jnp.array(obs),
            jnp.array(mask),
            params,
            opt_state,
            self._model.n_iter,
            self._model.tol,
            optimizer,
        )
        self._train_df = train_df.copy()

    def filter(self, oos_df: pd.DataFrame) -> np.ndarray:
        assert self._params is not None and self._train_df is not None
        full = pd.concat([self._train_df, oos_df], axis=0)
        obs = _to_obs_arr(full)
        mask = np.ones(len(obs), dtype=np.float32)
        log_alphas, _ = _hmm_forward(jnp.array(obs), jnp.array(mask), self._params)
        probs = np.asarray(jax.nn.softmax(log_alphas, axis=1))
        return probs[len(self._train_df) :]

    def filter_full(self, full_df: pd.DataFrame) -> np.ndarray:
        assert self._params is not None
        obs = _to_obs_arr(full_df)
        mask = np.ones(len(obs), dtype=np.float32)
        log_alphas, _ = _hmm_forward(jnp.array(obs), jnp.array(mask), self._params)
        return np.asarray(jax.nn.softmax(log_alphas, axis=1))


def test_hmm_fit_train_filter_oos_causal_split() -> None:
    split = 180
    full_df = _synthetic_obs(260)
    train_df = full_df.iloc[:split]
    oos_df = full_df.iloc[split:]

    harness = _CausalSplitHarness(n_iter=25, tol=1e-6)
    harness.fit(train_df)

    probs_oos = harness.filter(oos_df)
    probs_full = harness.filter_full(full_df)

    assert probs_oos.shape == (len(oos_df), 4)
    np.testing.assert_allclose(probs_oos.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        probs_oos,
        probs_full[split:],
        rtol=1e-6,
        atol=1e-6,
    )
