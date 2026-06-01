from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.domain.futures.strategy.candidate_gate import fit_candidate_gate, predict_candidate_gate
from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(slots=True, frozen=True)
class _Dataset:
    X: np.ndarray
    y_gate: np.ndarray
    sample_weight: np.ndarray
    feature_names: tuple[str, ...]


def _make_dataset(seed: int, n: int) -> _Dataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0.0).astype(np.int8)
    w = np.ones(n, dtype=np.float32)
    return _Dataset(X=x, y_gate=y, sample_weight=w, feature_names=("f0", "f1", "f2", "f3"))


def test_fit_predict_candidate_gate_returns_calibrated_probability() -> None:
    train = _make_dataset(seed=11, n=160)
    valid = _make_dataset(seed=13, n=80)
    cfg = CandidateStrategyConfig(seed=17)

    model = fit_candidate_gate(train=train, valid=valid, cfg=cfg)
    probs = predict_candidate_gate(model=model, dataset=valid)

    assert probs.shape == (80,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert model.calibrator is not None


def test_fit_candidate_gate_is_deterministic_given_seed() -> None:
    train = _make_dataset(seed=21, n=140)
    valid = _make_dataset(seed=22, n=70)
    cfg = CandidateStrategyConfig(seed=5)

    model_a = fit_candidate_gate(train=train, valid=valid, cfg=cfg)
    model_b = fit_candidate_gate(train=train, valid=valid, cfg=cfg)

    probs_a = predict_candidate_gate(model=model_a, dataset=valid)
    probs_b = predict_candidate_gate(model=model_b, dataset=valid)

    assert np.allclose(probs_a, probs_b)
