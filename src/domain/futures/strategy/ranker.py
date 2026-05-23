from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LongMatrixDataset


@dataclass(slots=True, frozen=True)
class RankerFitResult:
    """Ranker fit output."""

    model: lgb.LGBMRanker


def fit_ranker(
    train: LongMatrixDataset,
    valid: LongMatrixDataset,
    cfg: StrategyMLConfig,
) -> RankerFitResult:
    """Train LambdaMART ranker."""
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=cfg.ranker_n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction,
        bagging_freq=1,
        lambda_l2=cfg.lambda_l2,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
    )
    if train.X.shape[0] == 0:
        raise RuntimeError("ranker train dataset is empty")
    if valid.X.shape[0] > 0 and valid.group.shape[0] > 0:
        model.fit(
            train.X,
            train.y_rank,
            group=train.group,
            sample_weight=train.sample_weight,
            eval_set=[(valid.X, valid.y_rank)],
            eval_group=[valid.group],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
    else:
        model.fit(
            train.X,
            train.y_rank,
            group=train.group,
            sample_weight=train.sample_weight,
        )
    return RankerFitResult(model=model)


def predict_rank_score(
    model: lgb.LGBMRanker, dataset: LongMatrixDataset
) -> NDArray[np.float32]:
    """Predict ranker score."""
    if dataset.X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    pred = cast(NDArray[np.float64], model.predict(dataset.X))
    out: NDArray[np.float32] = np.asarray(pred, dtype=np.float32).reshape(-1).copy()
    return out
