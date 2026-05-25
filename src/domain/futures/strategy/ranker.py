from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LongMatrixDataset

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RankerFitResult:
    """Regressor fit output (CS-demeaned GBT regression)."""

    model: lgb.LGBMRegressor | lgb.LGBMRanker


def _as_feature_frame(x: np.ndarray, feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Build deterministic feature frame for sklearn feature-name consistency."""
    if x.shape[1] != len(feature_names):
        raise ValueError("feature column count mismatch")
    return pd.DataFrame(x, columns=list(feature_names), copy=False)


def _cs_demean(y_ev: NDArray[np.float32], group: NDArray[np.int32]) -> NDArray[np.float32]:
    """Subtract cross-sectional mean within each group (timestep)."""
    y_out = y_ev.copy()
    offset = 0
    for g in group:
        g_int = int(g)
        sl = slice(offset, offset + g_int)
        y_out[sl] = y_ev[sl] - np.mean(y_ev[sl], dtype=np.float32)
        offset += g_int
    return y_out


_MIN_GROUPS_FOR_NDCG: int = 5  # minimum distinct timestep groups for stable lambdarank


def fit_ranker(
    train: LongMatrixDataset,
    valid: LongMatrixDataset,
    cfg: StrategyMLConfig,
) -> RankerFitResult:
    """Train CS-demeaned GBT regressor (replaces LambdaMART).

    Cross-sectional demeaning removes market-beta from the target so the model
    learns relative expected-return signals rather than directional market bets.
    Works correctly with any group size N >= min_group_size (no NDCG constraint).
    """
    if train.X.shape[0] == 0:
        raise RuntimeError("ranker train dataset is empty")

    # Auto-fallback: group_ndcg requires enough distinct timestep groups for stable NDCG.
    # When group cardinality is insufficient, silently downgrade to pointwise regression.
    n_groups = len(train.group)
    is_lambdarank = cfg.ranking_mode == "group_ndcg"
    if is_lambdarank and n_groups < _MIN_GROUPS_FOR_NDCG:
        _logger.warning(
            "[RANKER] group_ndcg fallback to pointwise: n_groups=%d < min=%d",
            n_groups,
            _MIN_GROUPS_FOR_NDCG,
        )
        is_lambdarank = False

    model = _build_ranker_model(cfg, force_pointwise=not is_lambdarank)
    x_train = _as_feature_frame(train.X, train.feature_names)
    train_target: NDArray[np.float32] | NDArray[np.int32]
    valid_target: NDArray[np.float32] | NDArray[np.int32]
    if is_lambdarank:
        train_target = train.y_rank
        valid_target = valid.y_rank
    else:
        train_target = _cs_demean(train.y_ev, train.group)
        valid_target = _cs_demean(valid.y_ev, valid.group) if valid.X.shape[0] > 0 else valid.y_ev

    if valid.X.shape[0] > 0:
        x_valid = _as_feature_frame(valid.X, valid.feature_names)
        if is_lambdarank:
            ranker_model = cast(lgb.LGBMRanker, model)
            ranker_model.fit(
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
            regressor_model = cast(lgb.LGBMRegressor, model)
            regressor_model.fit(
                x_train,
                train_target,
                sample_weight=train.sample_weight,
                eval_set=[(x_valid, valid_target)],
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )
    else:
        if is_lambdarank:
            ranker_model = cast(lgb.LGBMRanker, model)
            ranker_model.fit(
                x_train,
                train_target,
                group=train.group,
                sample_weight=train.sample_weight,
            )
        else:
            regressor_model = cast(lgb.LGBMRegressor, model)
            regressor_model.fit(x_train, train_target, sample_weight=train.sample_weight)
    return RankerFitResult(model=model)


def _build_ranker_model(
    cfg: StrategyMLConfig, force_pointwise: bool = False
) -> lgb.LGBMRegressor | lgb.LGBMRanker:
    """Build ranker model by configured model family."""
    if not force_pointwise and (
        cfg.ranking_mode == "group_ndcg" or cfg.model_family == "lgbm_lambdarank"
    ):
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
            eval_at=(3, 5),
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            verbose=-1,
        )
    # force_pointwise=True with lgbm_lambdarank family → fall back to regression
    if cfg.model_family in {"lgbm_regression", "lgbm_lambdarank"}:
        return lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
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
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            verbose=-1,
        )
    if cfg.model_family == "lgbm_huber":
        return lgb.LGBMRegressor(
            objective="huber",
            metric="huber",
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
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            verbose=-1,
        )
    raise ValueError(f"unsupported model_family: {cfg.model_family}")


def predict_rank_score(
    model: lgb.LGBMRegressor | lgb.LGBMRanker, dataset: LongMatrixDataset
) -> NDArray[np.float32]:
    """Predict CS-demeaned expected return score."""
    if dataset.X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x = _as_feature_frame(dataset.X, dataset.feature_names)
    estimate = cast(NDArray[np.float64], model.predict(x))
    score: NDArray[np.float32] = np.asarray(estimate, dtype=np.float32).reshape(-1).copy()
    return score
