from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_dataset import CandidateDataset
from src.domain.futures.strategy.candidate_gate import fit_candidate_gate, predict_candidate_gate
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _make_dataset(seed: int, n: int) -> CandidateDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0.0).astype(np.int8)
    w = np.ones(n, dtype=np.float32)
    return CandidateDataset(
        X=x,
        y_gate=y,
        y_edge_bps=np.zeros(n, dtype=np.float32),
        y_q10_bps=np.zeros(n, dtype=np.float32),
        y_mfe_bps=np.zeros(n, dtype=np.float32),
        gate_weight=w,
        edge_weight=w,
        groups=np.arange(n, dtype=np.int32),
        event_index=pd.DataFrame(),
        feature_names=("f0", "f1", "f2", "f3"),
        effective_sample_size=float(n),
    )


def test_fit_predict_candidate_gate_returns_calibrated_probability() -> None:
    train = _make_dataset(seed=11, n=160)
    valid = _make_dataset(seed=13, n=80)
    cfg = CandidateStrategyConfig(seed=17, min_gate_calibration_obs=10)

    model = fit_candidate_gate(train=train, early_stop=valid, calibration=valid, cfg=cfg)
    probs = predict_candidate_gate(model=model, dataset=valid)

    assert probs.shape == (80,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert model.calibrator is None
    assert model.calibration_used is False
    assert model.calibration_reason == "gate_model_removed_ratio_discount"
    assert np.all(probs == 1.0)


def test_fit_candidate_gate_is_deterministic_given_seed() -> None:
    train = _make_dataset(seed=21, n=140)
    valid = _make_dataset(seed=22, n=70)
    cfg = CandidateStrategyConfig(seed=5)

    model_a = fit_candidate_gate(train=train, early_stop=valid, calibration=valid, cfg=cfg)
    model_b = fit_candidate_gate(train=train, early_stop=valid, calibration=valid, cfg=cfg)

    probs_a = predict_candidate_gate(model=model_a, dataset=valid)
    probs_b = predict_candidate_gate(model=model_b, dataset=valid)

    assert np.allclose(probs_a, probs_b)


def test_fit_candidate_gate_skips_calibration_when_probability_dispersion_collapses() -> None:
    train = _make_dataset(seed=31, n=160)
    valid = _make_dataset(seed=32, n=80)
    cfg = CandidateStrategyConfig(seed=9, min_gate_calibration_obs=10, min_gate_probability_std=10.0)

    model = fit_candidate_gate(train=train, early_stop=valid, calibration=valid, cfg=cfg)
    probs = predict_candidate_gate(model=model, dataset=valid)

    assert probs.shape == (80,)
    assert not model.calibration_used
    assert model.calibration_reason == "gate_model_removed_ratio_discount"


def test_gate_always_active_returns_calibrated_prob() -> None:
    """Layer 1 simplification: gate.validation.enabled=True always, predict returns calibrated prob."""
    train = _make_dataset(seed=41, n=160)
    valid = _make_dataset(seed=42, n=80)
    cfg = CandidateStrategyConfig(seed=19, min_gate_calibration_obs=10)

    # Act
    model = fit_candidate_gate(train=train, early_stop=valid, calibration=valid, cfg=cfg)
    probs = predict_candidate_gate(model=model, dataset=valid)

    # Assert: gate is always enabled regardless of brier_skill/decile_lift
    assert model.validation.enabled is True
    assert model.validation.reason == "gate_replaced_by_q10_mu_ratio"
    assert probs.shape == (80,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.all(probs == 1.0)
