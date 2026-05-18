from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parents[6])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.ml_pipeline.regime.jax_hmm import (
    JAXMultivariateHMM,
)

def _synthetic_obs(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    f1 = 0.8 * np.sin(t / 17.0) + 0.15 * rng.normal(size=n)
    f2 = 0.7 * np.cos(t / 23.0) + 0.12 * rng.normal(size=n)
    f3 = 0.4 * np.sin(t / 13.0) + 0.10 * rng.normal(size=n)
    f4 = 0.35 * np.cos(t / 29.0) + 0.10 * rng.normal(size=n)
    f5 = 0.3 * np.sin(t / 11.0) + 0.10 * rng.normal(size=n)
    f6 = 0.3 * np.cos(t / 31.0) + 0.10 * rng.normal(size=n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6},
        index=idx,
    )

# --- From test_hmm_causal_split.py ---

class _CausalSplitHarness:
    """Minimal fit(train)/filter(oos) harness using frozen HMM params."""

    def __init__(self, n_iter: int = 30, tol: float = 1e-6) -> None:
        self._model = JAXMultivariateHMM(n_iter=n_iter, tol=tol)
        self._params = None
        self._train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self._model.fit(train_df)
        self._train_df = train_df.copy()

    def filter(self, oos_df: pd.DataFrame) -> np.ndarray:
        assert self._train_df is not None
        full = pd.concat([self._train_df, oos_df], axis=0)
        probs_df = self._model.filter(full)
        probs = probs_df.to_numpy(dtype=np.float64)
        return probs[len(self._train_df) :]

    def filter_full(self, full_df: pd.DataFrame) -> np.ndarray:
        probs_df = self._model.filter(full_df)
        return probs_df.to_numpy(dtype=np.float64)


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

# --- From test_hmm_truncation_invariance.py ---

def _fit_model(train_df: pd.DataFrame, n_iter: int = 25, tol: float = 1e-6) -> JAXMultivariateHMM:
    model = JAXMultivariateHMM(n_iter=n_iter, tol=tol)
    model.fit(train_df)
    return model


def test_hmm_filter_truncation_invariance() -> None:
    """Tests that filtering on [A, B] gives same result for B as filtering on [A, B, C]."""
    full_df = _synthetic_obs(300, seed=42)
    split_1 = 150
    split_2 = 220
    
    train_df = full_df.iloc[:split_1]
    model = _fit_model(train_df)
    
    # Filter on [train + OOS_part1]
    df_short = full_df.iloc[:split_2]
    probs_short = model.filter(df_short).to_numpy(dtype=np.float64)
    
    # Filter on [train + OOS_part1 + OOS_part2]
    probs_long = model.filter(full_df).to_numpy(dtype=np.float64)
    
    # Results for df_short should be identical
    np.testing.assert_allclose(
        probs_short,
        probs_long[:split_2],
        rtol=1e-6,
        atol=1e-6
    )
    
    # The sum should always be 1.0
    np.testing.assert_allclose(probs_long.sum(axis=1), 1.0, rtol=1e-6)
