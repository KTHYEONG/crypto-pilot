from __future__ import annotations

import numpy as np

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

    fit = fit_ranker(train=train, valid=valid, cfg=cfg)
    pred = predict_rank_score(fit.model, valid)

    assert pred.shape == (valid.X.shape[0],)
    assert pred.dtype == np.float32
    assert np.all(np.isfinite(pred))


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

    fit = fit_ranker(train=train, valid=valid, cfg=cfg)
    pred = predict_rank_score(fit.model, empty)

    assert pred.shape == (0,)
