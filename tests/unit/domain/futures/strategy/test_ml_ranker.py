from __future__ import annotations

import typing

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LongMatrixDataset
from src.domain.futures.strategy.ranker import fit_ranker, predict_rank_score


def _dataset(rows: int, groups: int, features: int = 6) -> LongMatrixDataset:
    rng = np.random.default_rng(123)
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


def test_fit_ranker_and_predict_rank_score() -> None:
    train = _dataset(rows=160, groups=20)
    valid = _dataset(rows=40, groups=5)
    cfg = StrategyMLConfig(ranker_n_estimators=20, early_stopping_rounds=10, n_jobs=1)

    fit_result = fit_ranker(train=train, valid=valid, cfg=cfg)
    score = predict_rank_score(fit_result.model, valid)

    assert score.shape == (valid.X.shape[0],)
    assert score.dtype == np.float32
    assert np.all(np.isfinite(score))


def test_fit_ranker_uses_lambdarank_hyperparams(monkeypatch: MonkeyPatch) -> None:
    train = _dataset(rows=40, groups=5)
    valid = _dataset(rows=16, groups=2)
    captured_ctor: dict[str, typing.Any] = {}
    captured_fit: dict[str, typing.Any] = {}

    class _FakeRanker:
        def __init__(self, **kwargs: object) -> None:
            captured_ctor.update(kwargs)

        def fit(self, *args: object, **kwargs: object) -> _FakeRanker:
            captured_fit["args"] = args
            captured_fit["kwargs"] = kwargs
            return self

        def predict(self, x: typing.Sized) -> np.ndarray:
            return np.linspace(0.0, 1.0, len(x), dtype=np.float64)

    monkeypatch.setattr("src.domain.futures.strategy.ranker.lgb.LGBMRanker", _FakeRanker)
    cfg = StrategyMLConfig(
        model_family="lgbm_lambdarank",
        ranking_mode="group_ndcg",
        ranker_learning_rate=0.017,
        ranker_feature_fraction=0.66,
        ranker_bagging_fraction=0.61,
        ranker_bagging_freq=3,
        ranker_lambda_l2=2.5,
        ranker_reg_alpha=0.9,
    )
    fit_result = fit_ranker(train=train, valid=valid, cfg=cfg)
    score = predict_rank_score(fit_result.model, valid)

    assert captured_ctor["objective"] == "lambdarank"
    assert captured_ctor["metric"] == "ndcg"
    assert captured_ctor["learning_rate"] == 0.017
    assert captured_ctor["feature_fraction"] == 0.66
    assert captured_ctor["bagging_fraction"] == 0.61
    assert captured_ctor["bagging_freq"] == 3
    assert captured_ctor["lambda_l2"] == 2.5
    assert captured_ctor["reg_alpha"] == 0.9
    assert np.array_equal(captured_fit["kwargs"]["group"], train.group)
    assert np.array_equal(captured_fit["kwargs"]["eval_group"][0], valid.group)
    assert np.array_equal(captured_fit["kwargs"]["eval_set"][0][1], valid.y_rank)
    assert score.shape == (valid.X.shape[0],)
    assert score.dtype == np.float32


def test_predict_rank_score_empty_dataset_returns_empty() -> None:
    train = _dataset(rows=160, groups=20)
    valid = _dataset(rows=40, groups=5)
    empty = LongMatrixDataset(
        X=np.zeros((0, train.X.shape[1]), dtype=np.float32),
        y_rank=np.zeros((0,), dtype=np.int32),
        y_ev=np.zeros((0,), dtype=np.float32),
        group=np.zeros((0,), dtype=np.int32),
        sample_weight=np.zeros((0,), dtype=np.float32),
        index_map=np.zeros((0, 2), dtype=np.int64),
        feature_names=train.feature_names,
    )
    cfg = StrategyMLConfig(ranker_n_estimators=5, early_stopping_rounds=3, n_jobs=1)

    fit_result = fit_ranker(train=train, valid=valid, cfg=cfg)
    score = predict_rank_score(fit_result.model, empty)
    assert score.shape == (0,)
