from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_dataset import CandidateDataset
from src.domain.futures.strategy.candidate_edge import fit_candidate_edge_models, predict_candidate_edges
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _make_dataset(seed: int, n: int) -> CandidateDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    edge = (15.0 * x[:, 0] - 5.0 * x[:, 1]).astype(np.float32)
    q10 = (edge - 8.0).astype(np.float32)
    mfe = (edge + 8.0).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    return CandidateDataset(
        X=x,
        y_gate=np.ones(n, dtype=np.int8),
        y_edge_bps=edge,
        y_q10_bps=q10,
        y_mfe_bps=mfe,
        sample_weight=w,
        groups=np.arange(n, dtype=np.int32),
        event_index=pd.DataFrame(),
        feature_names=("f0", "f1", "f2", "f3"),
    )


def test_predict_candidate_edges_exposes_all_required_outputs() -> None:
    train = _make_dataset(seed=100, n=180)
    valid = _make_dataset(seed=101, n=60)
    cfg = CandidateStrategyConfig(seed=3)

    models = fit_candidate_edge_models(train=train, valid=valid, cfg=cfg)
    p_pass = np.full(valid.X.shape[0], 0.6, dtype=np.float64)
    out = predict_candidate_edges(models=models, dataset=valid, p_pass=p_pass, cfg=cfg)

    assert out.mu_gross_bps.shape == (60,)
    assert out.mu_net_decision_bps.shape == (60,)
    assert out.q10_net_bps.shape == (60,)
    assert out.q90_net_bps.shape == (60,)
    assert out.utility_score.shape == (60,)


def test_predict_candidate_edges_applies_cost_and_utility_formula() -> None:
    train = _make_dataset(seed=200, n=180)
    valid = _make_dataset(seed=201, n=30)
    cfg = CandidateStrategyConfig(
        seed=4,
        expected_cost_bps=2.0,
        downside_penalty=0.5,
        turnover_penalty=1.0,
        concentration_penalty=0.0,
    )

    models = fit_candidate_edge_models(train=train, valid=valid, cfg=cfg)
    p_pass = np.full(valid.X.shape[0], 0.5, dtype=np.float64)
    out = predict_candidate_edges(models=models, dataset=valid, p_pass=p_pass, cfg=cfg)

    expected_utility = 0.5 * out.mu_net_decision_bps - 0.5 * np.abs(np.minimum(out.q10_net_bps, 0.0)) - 1.0

    assert np.allclose(out.mu_net_decision_bps, out.mu_gross_bps)
    assert np.allclose(out.utility_score, expected_utility)
