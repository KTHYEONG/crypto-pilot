from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LongMatrixDataset


@dataclass(slots=True, frozen=True)
class RankerFitResult:
    """Ranker fit output."""

    model: lgb.LGBMRanker
    models: list[lgb.LGBMRanker] | None = None


def _as_feature_frame(x: np.ndarray, feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Build deterministic feature frame for sklearn feature-name consistency."""
    if x.shape[1] != len(feature_names):
        raise ValueError("feature column count mismatch")
    return pd.DataFrame(x, columns=list(feature_names), copy=False)


def fit_ranker(
    train: LongMatrixDataset,
    valid: LongMatrixDataset,
    cfg: StrategyMLConfig,
) -> RankerFitResult:
    """Train signed LambdaRank models on relevance labels."""
    if train.X.shape[0] == 0:
        raise RuntimeError("ranker train dataset is empty")

    x_train = _as_feature_frame(train.X, train.feature_names)
    train_target: NDArray[np.int32] = train.y_rank
    valid_target: NDArray[np.int32] = valid.y_rank
    models: list[lgb.LGBMRanker] = []

    for seed in cfg.ensemble_seeds:
        model = _build_ranker_model(cfg, seed=seed)
        if valid.X.shape[0] > 0:
            x_valid = _as_feature_frame(valid.X, valid.feature_names)
            model.fit(
                x_train,
                train_target,
                group=train.group,
                sample_weight=train.sample_weight,
                eval_set=[(x_valid, valid_target)],
                eval_group=[valid.group],
                eval_at=(3, 5),
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )
        else:
            model.fit(
                x_train,
                train_target,
                group=train.group,
                sample_weight=train.sample_weight,
            )
        models.append(model)

    return RankerFitResult(
        model=models[0],
        models=models,
    )


def _build_ranker_model(
    cfg: StrategyMLConfig, seed: int | None = None
) -> lgb.LGBMRanker:
    """Build signed LambdaRank model."""
    run_seed = seed if seed is not None else cfg.seed
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=cfg.ranker_n_estimators,
        learning_rate=cfg.ranker_learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.ranker_feature_fraction,
        bagging_fraction=cfg.ranker_bagging_fraction,
        bagging_freq=cfg.ranker_bagging_freq,
        lambda_l2=cfg.ranker_lambda_l2,
        reg_alpha=cfg.ranker_reg_alpha,
        random_state=run_seed,
        n_jobs=cfg.n_jobs,
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
    )


def predict_rank_score(
    model: lgb.LGBMRanker | list[lgb.LGBMRanker],
    dataset: LongMatrixDataset,
) -> NDArray[np.float32]:
    """Predict CS-demeaned expected return score (with support for ensemble list)."""
    if dataset.X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x = _as_feature_frame(dataset.X, dataset.feature_names)
    if isinstance(model, list):
        if not model:
            raise ValueError("empty model list for ranker ensemble prediction")
        estimates = [cast(NDArray[np.float64], m.predict(x)) for m in model]
        estimate = np.mean(estimates, axis=0)
    else:
        estimate = cast(NDArray[np.float64], model.predict(x))
    score: NDArray[np.float32] = np.asarray(estimate, dtype=np.float32).reshape(-1).copy()
    return score
