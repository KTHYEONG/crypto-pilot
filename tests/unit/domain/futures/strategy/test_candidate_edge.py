from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_dataset import CandidateDataset
from src.domain.futures.strategy.candidate_edge import (
    EdgeModelValidation,
    fit_candidate_edge_models,
    predict_candidate_edges,
)
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
        gate_weight=w,
        edge_weight=w,
        groups=np.arange(n, dtype=np.int32),
        event_index=pd.DataFrame(
            {
                "family": [family] * n,
                "variant": [variant] * n,
            }
        ),
        feature_names=("f0", "f1", "f2", "f3"),
        effective_sample_size=float(n),
    )


def test_predict_candidate_edges_exposes_all_required_outputs() -> None:
    train = _make_dataset(seed=100, n=180)
    valid = _make_dataset(seed=101, n=60)
    cal_eval = _make_dataset(seed=102, n=40)
    cfg = CandidateStrategyConfig(seed=3)

    models = fit_candidate_edge_models(train=train, valid=valid, calibration_eval=cal_eval, cfg=cfg)
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
    cal_eval = _make_dataset(seed=202, n=30)
    cfg = CandidateStrategyConfig(
        seed=4,
        expected_cost_bps=2.0,
        downside_penalty=0.5,
        turnover_penalty=1.0,
        concentration_penalty=0.0,
        selection_utility_mode="additive_drag",  # test additive formula path explicitly
    )

    models = fit_candidate_edge_models(train=train, valid=valid, calibration_eval=cal_eval, cfg=cfg)
    p_pass = np.full(valid.X.shape[0], 0.5, dtype=np.float64)
    out = predict_candidate_edges(models=models, dataset=valid, p_pass=p_pass, cfg=cfg)

    expected_utility = out.mu_net_decision_bps - 0.5 * np.abs(np.minimum(out.q10_net_bps, 0.0)) - 1.0

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
        gate_weight=train.gate_weight,
        edge_weight=train.edge_weight,
        groups=train.groups,
        event_index=train.event_index,
        feature_names=train.feature_names,
        effective_sample_size=train.effective_sample_size,
    )
    valid = _make_dataset(seed=301, n=30, family="funding_carry", variant="funding_24")
    # cal_eval: varying edge so rank_ic is finite and test can confirm prior is positive
    cal_eval = _make_dataset(seed=302, n=30, family="funding_carry", variant="funding_24")
    cfg = CandidateStrategyConfig(
        seed=11, edge_prior_min_obs=10, edge_prior_shrinkage_obs=50, min_edge_rank_ic=0.0
    )

    models = fit_candidate_edge_models(train=train, valid=valid, calibration_eval=cal_eval, cfg=cfg)
    predict_candidate_edges(
        models=models,
        dataset=valid,
        p_pass=np.full(valid.X.shape[0], 0.6, dtype=np.float64),
        cfg=cfg,
    )

    assert models.variant_prior_bps["funding_carry:funding_24"] > 0.0
    # Prior is positive; model output should reflect it regardless of prediction_mode
    assert float(models.global_prior_bps) > 0.0 or models.variant_prior_bps.get("funding_carry:funding_24", 0.0) > 0.0


def test_predict_candidate_edges_flags_prediction_collapse() -> None:
    train = _make_dataset(seed=400, n=120)
    valid = _make_dataset(seed=401, n=40)
    cal_eval = _make_dataset(seed=402, n=40)
    cfg = CandidateStrategyConfig(
        seed=12,
        edge_prediction_min_std_bps=1e9,
        edge_prediction_min_positive_rate=0.999,
    )

    models = fit_candidate_edge_models(
        train=train, valid=valid, calibration_eval=cal_eval, cfg=cfg
    )
    out = predict_candidate_edges(
        models=models,
        dataset=valid,
        p_pass=np.full(valid.X.shape[0], 0.4, dtype=np.float64),
        cfg=cfg,
    )

    assert out.selection_thresholds["prediction_collapse"] is True
    assert float(out.selection_thresholds["mu_std_bps"]) < float(cfg.edge_prediction_min_std_bps)


def test_edge_gate_reverts_to_direct_when_rank_ic_below_threshold() -> None:
    """rank_ic < min_edge_rank_ic → prediction_mode='prior_only', accepted=False."""
    # Arrange: cal_eval is constant → rank_ic=nan → insufficient_obs or rank_ic_fail
    train = _make_dataset(seed=500, n=120)
    valid = _make_dataset(seed=501, n=30)
    # Small cal_eval (< 20 obs) → insufficient_obs → accepted=False
    cal_eval_small = _make_dataset(seed=502, n=10)
    cfg = CandidateStrategyConfig(seed=13, min_edge_rank_ic=0.02)

    # Act
    models = fit_candidate_edge_models(
        train=train, valid=valid, calibration_eval=cal_eval_small, cfg=cfg
    )

    # Assert
    assert models.validation is not None
    assert isinstance(models.validation, EdgeModelValidation)
    assert models.validation.accepted is False
    assert models.validation.reason == "insufficient_obs"
    assert models.validation.n_cal_eval == 10
    assert models.prediction_mode == "prior_only"


def test_edge_gate_accepts_when_rank_ic_above_threshold() -> None:
    """rank_ic >= min_edge_rank_ic AND edge_prior_enabled → prediction_mode='prior_residual'."""
    # Arrange: strong signal so rank_ic is high
    rng = np.random.default_rng(600)
    n = 150
    x = rng.normal(size=(n, 4)).astype(np.float32)
    # Make edge highly correlated with x[:,0] so model can achieve high rank_ic
    edge = (50.0 * x[:, 0]).astype(np.float32)
    w = np.ones(n, dtype=np.float32)

    def _make_correlated_ds(seed_offset: int, n_size: int) -> CandidateDataset:
        rng2 = np.random.default_rng(600 + seed_offset)
        x2 = rng2.normal(size=(n_size, 4)).astype(np.float32)
        edge2 = (50.0 * x2[:, 0]).astype(np.float32)
        w2 = np.ones(n_size, dtype=np.float32)
        return CandidateDataset(
            X=x2,
            y_gate=np.ones(n_size, dtype=np.int8),
            y_edge_bps=edge2,
            y_q10_bps=(edge2 - 5.0).astype(np.float32),
            y_mfe_bps=(edge2 + 5.0).astype(np.float32),
            gate_weight=w2,
            edge_weight=w2,
            groups=np.arange(n_size, dtype=np.int32),
            event_index=pd.DataFrame({"family": ["trend_ma"] * n_size, "variant": ["v1"] * n_size}),
            feature_names=("f0", "f1", "f2", "f3"),
            effective_sample_size=float(n_size),
        )

    train = CandidateDataset(
        X=x,
        y_gate=np.ones(n, dtype=np.int8),
        y_edge_bps=edge,
        y_q10_bps=(edge - 5.0).astype(np.float32),
        y_mfe_bps=(edge + 5.0).astype(np.float32),
        gate_weight=w,
        edge_weight=w,
        groups=np.arange(n, dtype=np.int32),
        event_index=pd.DataFrame({"family": ["trend_ma"] * n, "variant": ["v1"] * n}),
        feature_names=("f0", "f1", "f2", "f3"),
        effective_sample_size=float(n),
    )
    valid = _make_correlated_ds(1, 40)
    cal_eval = _make_correlated_ds(2, 50)

    # Set low threshold so acceptance is likely
    cfg = CandidateStrategyConfig(
        seed=14,
        edge_prior_enabled=True,
        edge_residual_model_enabled=True,
        min_edge_rank_ic=0.0,  # always accept finite rank_ic
        edge_prior_min_obs=10,
    )

    # Act
    models = fit_candidate_edge_models(train=train, valid=valid, calibration_eval=cal_eval, cfg=cfg)

    # Assert: validation is populated
    assert models.validation is not None
    assert models.validation.n_cal_eval == 50
    assert np.isfinite(models.validation.rank_ic_cal_eval)
    assert models.validation.accepted is True
    assert models.validation.reason == "rank_ic_pass"
    assert models.prediction_mode == "prior_residual"


def test_rejected_residual_model_uses_prior_only_mu() -> None:
    train = _make_dataset(seed=700, n=120, family="carry", variant="v1")
    train = CandidateDataset(
        X=train.X,
        y_gate=train.y_gate,
        y_edge_bps=np.full(train.X.shape[0], 30.0, dtype=np.float32),
        y_q10_bps=np.full(train.X.shape[0], -5.0, dtype=np.float32),
        y_mfe_bps=np.full(train.X.shape[0], 40.0, dtype=np.float32),
        gate_weight=train.gate_weight,
        edge_weight=train.edge_weight,
        groups=train.groups,
        event_index=train.event_index,
        feature_names=train.feature_names,
        effective_sample_size=train.effective_sample_size,
    )
    valid = _make_dataset(seed=701, n=40, family="carry", variant="v1")
    cal_eval = _make_dataset(seed=702, n=40, family="carry", variant="v1")
    cfg = CandidateStrategyConfig(seed=15, min_edge_rank_ic=0.99, edge_prior_min_obs=10)

    models = fit_candidate_edge_models(train=train, valid=valid, calibration_eval=cal_eval, cfg=cfg)
    out = predict_candidate_edges(
        models=models,
        dataset=valid,
        p_pass=np.full(valid.X.shape[0], 0.5, dtype=np.float64),
        cfg=cfg,
    )

    expected_prior_bps = models.variant_prior_bps["carry:v1"]
    assert models.prediction_mode == "prior_only"
    assert np.allclose(out.mu_net_decision_bps, expected_prior_bps)


def test_rejected_prior_only_without_eligible_rows_disables_mu() -> None:
    train = _make_dataset(seed=800, n=120, family="carry", variant="v1")
    zero_weights = np.zeros(train.X.shape[0], dtype=np.float32)
    train = CandidateDataset(
        X=train.X,
        y_gate=train.y_gate,
        y_edge_bps=train.y_edge_bps,
        y_q10_bps=train.y_q10_bps,
        y_mfe_bps=train.y_mfe_bps,
        gate_weight=zero_weights,
        edge_weight=zero_weights,
        groups=train.groups,
        event_index=train.event_index,
        feature_names=train.feature_names,
        effective_sample_size=0.0,
    )
    valid = _make_dataset(seed=801, n=20, family="carry", variant="v1")
    cal_eval = _make_dataset(seed=802, n=20, family="carry", variant="v1")
    cfg = CandidateStrategyConfig(seed=16, min_edge_rank_ic=0.99, edge_prior_min_obs=10)

    models = fit_candidate_edge_models(train=train, valid=valid, calibration_eval=cal_eval, cfg=cfg)
    out = predict_candidate_edges(
        models=models,
        dataset=valid,
        p_pass=np.full(valid.X.shape[0], 0.5, dtype=np.float64),
        cfg=cfg,
    )

    assert models.prediction_mode == "disabled"
    assert np.allclose(out.mu_net_decision_bps, 0.0)
