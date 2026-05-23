from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LongMatrixDataset


@dataclass(slots=True, frozen=True)
class CalibratorFitResult:
    """Quantile calibrator models."""

    q10: lgb.LGBMRegressor
    q50: lgb.LGBMRegressor
    q90: lgb.LGBMRegressor


@dataclass(slots=True, frozen=True)
class EVQuantiles:
    """Quantile EV predictions."""

    q10: NDArray[np.float32]
    q50: NDArray[np.float32]
    q90: NDArray[np.float32]


def _fit_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    cfg: StrategyMLConfig,
    alpha: float,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=cfg.calibrator_n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=min(cfg.max_depth, 5),
        min_data_in_leaf=max(cfg.min_data_in_leaf, 100),
        feature_fraction=min(cfg.feature_fraction + 0.05, 0.95),
        bagging_fraction=min(cfg.bagging_fraction + 0.05, 0.95),
        bagging_freq=1,
        lambda_l2=max(cfg.lambda_l2, 5.0),
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
    )
    if valid_x.shape[0] > 0:
        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
    else:
        model.fit(train_x, train_y)
    return model


def fit_quantile_calibrators(
    train: LongMatrixDataset,
    valid: LongMatrixDataset,
    cfg: StrategyMLConfig,
    rank_score_train: np.ndarray | None = None,
    rank_score_valid: np.ndarray | None = None,
) -> CalibratorFitResult:
    """Train q10/q50/q90 models."""
    if train.X.shape[0] == 0:
        raise RuntimeError("calibrator train dataset is empty")
    if rank_score_train is None:
        rank_score_train = np.zeros((train.X.shape[0],), dtype=np.float32)
    if rank_score_valid is None:
        rank_score_valid = np.zeros((valid.X.shape[0],), dtype=np.float32)
    if rank_score_train.shape[0] != train.X.shape[0]:
        raise ValueError("rank_score_train length mismatch")
    if rank_score_valid.shape[0] != valid.X.shape[0]:
        raise ValueError("rank_score_valid length mismatch")
    x_train = np.column_stack([train.X, rank_score_train])
    x_valid = (
        np.column_stack([valid.X, rank_score_valid])
        if valid.X.shape[0] > 0
        else np.zeros((0, x_train.shape[1]), dtype=np.float32)
    )
    q10 = _fit_one(x_train, train.y_ev, x_valid, valid.y_ev, cfg, 0.10)
    q50 = _fit_one(x_train, train.y_ev, x_valid, valid.y_ev, cfg, 0.50)
    q90 = _fit_one(x_train, train.y_ev, x_valid, valid.y_ev, cfg, 0.90)
    return CalibratorFitResult(q10=q10, q50=q50, q90=q90)


def predict_ev_quantiles(
    models: CalibratorFitResult,
    dataset: LongMatrixDataset,
    rank_score: np.ndarray,
) -> EVQuantiles:
    """Predict q10/q50/q90 EV in return units."""
    if dataset.X.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return EVQuantiles(q10=empty, q50=empty, q90=empty)
    if rank_score.shape[0] != dataset.X.shape[0]:
        raise ValueError("rank_score length mismatch")
    x = np.column_stack([dataset.X, rank_score])
    q10 = np.asarray(cast(NDArray[np.float64], models.q10.predict(x)), dtype=np.float32)
    q50 = np.asarray(cast(NDArray[np.float64], models.q50.predict(x)), dtype=np.float32)
    q90 = np.asarray(cast(NDArray[np.float64], models.q90.predict(x)), dtype=np.float32)
    return EVQuantiles(
        q10=q10.reshape(-1).copy(),
        q50=q50.reshape(-1).copy(),
        q90=q90.reshape(-1).copy(),
    )


def compute_conservative_ev(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    cfg: StrategyMLConfig,
) -> NDArray[np.float32]:
    """Compute conservative EV from predicted quantiles."""
    if q10.shape != q50.shape or q50.shape != q90.shape:
        raise ValueError("quantile vectors must have identical shapes")
    if q10.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    q10f = np.asarray(q10, dtype=np.float32)
    q50f = np.asarray(q50, dtype=np.float32)
    q90f = np.asarray(q90, dtype=np.float32)
    downside = np.maximum(q50f - q10f, 0.0)
    upside = np.maximum(q90f - q50f, 0.0)
    asym = upside / np.maximum(downside, np.float32(1e-12))
    ev = q50f - np.float32(cfg.lambda_tail) * downside
    ev = ev * np.clip(asym, 0.25, 2.0)
    clip = np.float32(cfg.alpha_clip_bps / 10000.0)
    return np.asarray(np.clip(ev, -clip, clip), dtype=np.float32).reshape(-1).copy()


def predict_conservative_ev(
    models: CalibratorFitResult,
    dataset: LongMatrixDataset,
    rank_score: np.ndarray,
    cfg: StrategyMLConfig,
) -> NDArray[np.float32]:
    """Predict conservative EV in return units."""
    quantiles = predict_ev_quantiles(models=models, dataset=dataset, rank_score=rank_score)
    return compute_conservative_ev(
        q10=quantiles.q10,
        q50=quantiles.q50,
        q90=quantiles.q90,
        cfg=cfg,
    )
