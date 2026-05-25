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
    """Regressor fit output (CS-demeaned GBT regression)."""

    model: lgb.LGBMRegressor


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

    y_train_cs = _cs_demean(train.y_ev, train.group)
    y_valid_cs = _cs_demean(valid.y_ev, valid.group) if valid.X.shape[0] > 0 else valid.y_ev

    model = _build_ranker_model(cfg)
    x_train = _as_feature_frame(train.X, train.feature_names)
    if valid.X.shape[0] > 0:
        x_valid = _as_feature_frame(valid.X, valid.feature_names)
        model.fit(
            x_train,
            y_train_cs,
            sample_weight=train.sample_weight,
            eval_set=[(x_valid, y_valid_cs)],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
    else:
        model.fit(x_train, y_train_cs, sample_weight=train.sample_weight)
    return RankerFitResult(model=model)


def _build_ranker_model(cfg: StrategyMLConfig) -> lgb.LGBMRegressor:
    """Build ranker model by configured model family."""
    if cfg.model_family == "lgbm_regression":
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
    model: lgb.LGBMRegressor, dataset: LongMatrixDataset
) -> NDArray[np.float32]:
    """Predict CS-demeaned expected return score."""
    if dataset.X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x = _as_feature_frame(dataset.X, dataset.feature_names)
    pred = cast(NDArray[np.float64], model.predict(x))
    out: NDArray[np.float32] = np.asarray(pred, dtype=np.float32).reshape(-1).copy()
    return out
