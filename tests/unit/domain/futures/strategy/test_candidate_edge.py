from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_dataset import CandidateDataset
from src.domain.futures.strategy.candidate_edge import fit_candidate_edge_models, predict_candidate_edges
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _make_dataset(seed: int, n: int, *, family: str = "trend_ma", variant: str = "ema_12_72") -> CandidateDataset:
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
        event_index=pd.DataFrame(
            {
                "family": [family] * n,
                "variant": [variant] * n,
            }
        ),
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
    assert "utility_min" in out.selection_thresholds


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


def test_edge_prior_residual_preserves_positive_variant_prior() -> None:
    train = _make_dataset(seed=300, n=120, family="funding_carry", variant="funding_24")
    train = CandidateDataset(
        X=train.X,
        y_gate=train.y_gate,
        y_edge_bps=np.full(train.X.shape[0], 25.0, dtype=np.float32),
        y_q10_bps=np.full(train.X.shape[0], -5.0, dtype=np.float32),
        y_mfe_bps=np.full(train.X.shape[0], 40.0, dtype=np.float32),
        sample_weight=train.sample_weight,
        groups=train.groups,
        event_index=train.event_index,
        feature_names=train.feature_names,
    )
    valid = _make_dataset(seed=301, n=30, family="funding_carry", variant="funding_24")
    cfg = CandidateStrategyConfig(seed=11, edge_prior_min_obs=10, edge_prior_shrinkage_obs=50)

    models = fit_candidate_edge_models(train=train, valid=valid, cfg=cfg)
    out = predict_candidate_edges(
        models=models,
        dataset=valid,
        p_pass=np.full(valid.X.shape[0], 0.6, dtype=np.float64),
        cfg=cfg,
    )

    assert models.target_mode == "prior_residual"
    assert models.variant_prior_bps["funding_carry:funding_24"] > 0.0
    assert float(np.max(out.mu_net_decision_bps)) > 0.0


def test_predict_candidate_edges_flags_prediction_collapse() -> None:
    train = _make_dataset(seed=400, n=120)
    valid = _make_dataset(seed=401, n=40)
    cfg = CandidateStrategyConfig(
        seed=12,
        edge_prediction_min_std_bps=100.0,
        edge_prediction_min_positive_rate=0.99,
    )

    models = fit_candidate_edge_models(train=train, valid=valid, cfg=cfg)
    out = predict_candidate_edges(
        models=models,
        dataset=valid,
        p_pass=np.full(valid.X.shape[0], 0.4, dtype=np.float64),
        cfg=cfg,
    )

    assert out.selection_thresholds["prediction_collapse"] is True
    assert float(out.selection_thresholds["mu_std_bps"]) < 100.0
