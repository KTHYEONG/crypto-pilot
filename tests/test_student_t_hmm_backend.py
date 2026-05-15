from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _load_student_t_backend_class() -> type | None:
    try:
        mod = importlib.import_module("src.domain.futures.ml_pipeline.regime.student_t_hmm")
    except ModuleNotFoundError:
        return None

    for name in ("StudentTMultivariateHMM", "StudentTHMM", "StudentTRegimeHMM"):
        cls = getattr(mod, name, None)
        if isinstance(cls, type):
            return cls

    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type):
            lname = attr.lower()
            if "student" in lname and "hmm" in lname:
                return obj
    return None


def _synthetic_obs(n: int = 180, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    f1 = 0.9 * np.sin(t / 19.0) + 0.15 * rng.normal(size=n)
    f2 = 0.7 * np.cos(t / 27.0) + 0.12 * rng.normal(size=n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"f1": f1, "f2": f2}, index=idx)


def _prob_cols(df: pd.DataFrame) -> list[str]:
    preferred = [c for c in df.columns if any(k in c for k in ("bull", "bear", "chop", "state"))]
    if preferred:
        return preferred
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _assert_prob_frame(df: pd.DataFrame, n_expected: int) -> None:
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_expected
    cols = _prob_cols(df)
    assert cols, "No numeric probability-like columns found"
    probs = df[cols].to_numpy(dtype=np.float64)
    assert np.isfinite(probs).all()
    assert ((probs >= -1e-8) & (probs <= 1.0 + 1e-8)).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-5)


def test_student_t_backend_fit_filter_train_oos_api() -> None:
    backend_cls = _load_student_t_backend_class()
    if backend_cls is None:
        pytest.skip("Student-t HMM backend module/class not available in this branch")

    obs = _synthetic_obs()
    model = backend_cls(n_iter=20, tol=1e-4)

    fitted = model.fit(obs)
    assert fitted is model

    filt = model.filter(obs)
    _assert_prob_frame(filt, len(obs))

    split = 120
    split_probs = model.fit_filter_train_oos(obs, is_end_idx=split)
    _assert_prob_frame(split_probs, len(obs))

