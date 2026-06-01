from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LongMatrixDataset


@dataclass(slots=True, frozen=True)
class RankerFitResult:
    """Ranker fit output supporting hybrid blending."""

    model: lgb.LGBMRanker | lgb.LGBMRegressor
    models: list[lgb.LGBMRanker] | list[lgb.LGBMRegressor] | None = None
    regressor_model: lgb.LGBMRegressor | None = None
    regressor_models: list[lgb.LGBMRegressor] | None = None
    hybrid_blending_enabled: bool = False
    hybrid_rank_weight: float = 0.6


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
    """Train signed LambdaRank and optionally Huber Regressor models on relevance/returns."""
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

    regressor_model: lgb.LGBMRegressor | None = None
    regressor_models: list[lgb.LGBMRegressor] | None = None

    if cfg.hybrid_blending_enabled:
        regressor_models = []
        train_ev: NDArray[np.float32] = train.y_ev
        valid_ev: NDArray[np.float32] = valid.y_ev
        for seed in cfg.ensemble_seeds:
            reg_model = _build_regressor_model(cfg, seed=seed)
            if valid.X.shape[0] > 0:
                x_valid = _as_feature_frame(valid.X, valid.feature_names)
                reg_model.fit(
                    x_train,
                    train_ev,
                    sample_weight=train.sample_weight,
                    eval_set=[(x_valid, valid_ev)],
                    callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                )
            else:
                reg_model.fit(
                    x_train,
                    train_ev,
                    sample_weight=train.sample_weight,
                )
            regressor_models.append(reg_model)
        regressor_model = regressor_models[0]

    return RankerFitResult(
        model=models[0],
        models=models,
        regressor_model=regressor_model,
        regressor_models=regressor_models,
        hybrid_blending_enabled=cfg.hybrid_blending_enabled,
        hybrid_rank_weight=cfg.hybrid_rank_weight,
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


def _build_regressor_model(
    cfg: StrategyMLConfig, seed: int | None = None
) -> lgb.LGBMRegressor:
    """Build signed Regressor model with Huber loss."""
    run_seed = seed if seed is not None else cfg.seed
    return lgb.LGBMRegressor(
        objective="huber",
        boosting_type="gbdt",
        n_estimators=cfg.ranker_n_estimators,
        learning_rate=cfg.learning_rate,
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
    model: Any,
    dataset: LongMatrixDataset,
) -> NDArray[np.float32]:
    """Predict CS-demeaned expected return score (with support for hybrid ensembling)."""
    if dataset.X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    # Hybrid blending flow
    if hasattr(model, "hybrid_blending_enabled") and getattr(model, "hybrid_blending_enabled", False):
        ranker_models = model.models if model.models is not None else [model.model]
        regressor_models = model.regressor_models if model.regressor_models is not None else (
            [model.regressor_model] if model.regressor_model is not None else []
        )

        x = _as_feature_frame(dataset.X, dataset.feature_names)
        ranker_estimates = [cast(NDArray[np.float64], m.predict(x)) for m in ranker_models]
        ranker_score = np.mean(ranker_estimates, axis=0)

        if regressor_models:
            regressor_estimates = [cast(NDArray[np.float64], m.predict(x)) for m in regressor_models]
            regressor_score = np.mean(regressor_estimates, axis=0)

            # Robust Standardization before blending
            r_mean = float(np.mean(ranker_score))
            r_std = float(np.std(ranker_score))
            ranker_norm = (ranker_score - r_mean) / max(r_std, 1e-9)

            reg_mean = float(np.mean(regressor_score))
            reg_std = float(np.std(regressor_score))
            reg_norm = (regressor_score - reg_mean) / max(reg_std, 1e-9)

            w = float(model.hybrid_rank_weight)
            estimate = w * ranker_norm + (1.0 - w) * reg_norm
        else:
            estimate = ranker_score
    else:
        # Standard flow
        models_list = model
        if hasattr(model, "models") and getattr(model, "models", None) is not None:
            models_list = model.models
        elif hasattr(model, "model"):
            models_list = model.model

        x = _as_feature_frame(dataset.X, dataset.feature_names)
        if isinstance(models_list, list):
            if not models_list:
                raise ValueError("empty model list for ranker ensemble prediction")
            estimates = [cast(NDArray[np.float64], m.predict(x)) for m in models_list]
            estimate = np.mean(estimates, axis=0)
        else:
            estimate = cast(NDArray[np.float64], models_list.predict(x))

    score: NDArray[np.float32] = np.asarray(estimate, dtype=np.float32).reshape(-1).copy()
    return score
