from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.calibrator import (
    compute_conservative_ev,
    fit_quantile_calibrators,
    predict_conservative_ev,
    predict_ev_quantiles,
)
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LongMatrixDataset


def _dataset(rows: int, groups: int, features: int = 5) -> LongMatrixDataset:
    rng = np.random.default_rng(7)
    group_size = rows // groups
    x = rng.normal(size=(rows, features)).astype(np.float32)
    y_rank = np.tile(np.arange(group_size, dtype=np.int32), groups)
    y_ev = rng.normal(scale=1e-3, size=rows).astype(np.float32)
    group = np.full((groups,), group_size, dtype=np.int32)
    sample_weight = np.ones((rows,), dtype=np.float32)
    index_map = np.column_stack(
        [np.arange(rows, dtype=np.int64), np.zeros((rows,), dtype=np.int64)]
    )
    return LongMatrixDataset(
        X=x,
        y_rank=y_rank,
        y_ev=y_ev,
        group=group,
        sample_weight=sample_weight,
        index_map=index_map,
        feature_names=tuple(f"f{i}" for i in range(features)),
    )


def test_fit_quantile_calibrators_and_predict_ev_clip() -> None:
    train = _dataset(rows=180, groups=20)
    valid = _dataset(rows=45, groups=5)
    test = _dataset(rows=36, groups=4)
    cfg = StrategyMLConfig(calibrator_n_estimators=20, early_stopping_rounds=10, n_jobs=1)

    rank_train = np.linspace(-1.0, 1.0, train.X.shape[0], dtype=np.float32)
    rank_valid = np.linspace(-0.8, 0.8, valid.X.shape[0], dtype=np.float32)
    rank_test = np.linspace(-0.5, 0.5, test.X.shape[0], dtype=np.float32)

    models = fit_quantile_calibrators(
        train=train,
        valid=valid,
        rank_score_train=rank_train,
        rank_score_valid=rank_valid,
        cfg=cfg,
    )
    ev = predict_conservative_ev(models=models, dataset=test, rank_score=rank_test, cfg=cfg)

    clip = cfg.alpha_clip_bps / 10000.0
    assert ev.shape == (test.X.shape[0],)
    assert ev.dtype == np.float32
    assert np.all(np.isfinite(ev))
    assert np.max(np.abs(ev)) <= clip + 1e-12


def test_predict_conservative_ev_empty_returns_empty() -> None:
    train = _dataset(rows=180, groups=20)
    valid = _dataset(rows=45, groups=5)
    cfg = StrategyMLConfig(calibrator_n_estimators=10, early_stopping_rounds=5, n_jobs=1)

    rank_train = np.zeros((train.X.shape[0],), dtype=np.float32)
    rank_valid = np.zeros((valid.X.shape[0],), dtype=np.float32)
    models = fit_quantile_calibrators(
        train=train,
        valid=valid,
        rank_score_train=rank_train,
        rank_score_valid=rank_valid,
        cfg=cfg,
    )

    empty = LongMatrixDataset(
        X=np.zeros((0, train.X.shape[1]), dtype=np.float32),
        y_rank=np.zeros((0,), dtype=np.int32),
        y_ev=np.zeros((0,), dtype=np.float32),
        group=np.zeros((0,), dtype=np.int32),
        sample_weight=np.zeros((0,), dtype=np.float32),
        index_map=np.zeros((0, 2), dtype=np.int64),
        feature_names=train.feature_names,
    )
    pred = predict_conservative_ev(
        models=models,
        dataset=empty,
        rank_score=np.zeros((0,), dtype=np.float32),
        cfg=cfg,
    )

    assert pred.shape == (0,)


def test_predict_ev_quantiles_and_compute_conservative_ev() -> None:
    train = _dataset(rows=180, groups=20)
    valid = _dataset(rows=45, groups=5)
    test = _dataset(rows=36, groups=4)
    cfg = StrategyMLConfig(calibrator_n_estimators=10, early_stopping_rounds=5, n_jobs=1)
    models = fit_quantile_calibrators(train=train, valid=valid, cfg=cfg)
    rank_test = np.linspace(-0.5, 0.5, test.X.shape[0], dtype=np.float32)

    quantiles = predict_ev_quantiles(models=models, dataset=test, rank_score=rank_test)
    ev = compute_conservative_ev(quantiles.q10, quantiles.q50, quantiles.q90, cfg)

    assert quantiles.q10.shape == (test.X.shape[0],)
    assert quantiles.q50.shape == (test.X.shape[0],)
    assert quantiles.q90.shape == (test.X.shape[0],)
    assert ev.shape == (test.X.shape[0],)
