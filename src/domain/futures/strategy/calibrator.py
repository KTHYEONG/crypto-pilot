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


def _as_feature_frame(x: np.ndarray, feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Build deterministic feature frame for sklearn feature-name consistency."""
    if x.shape[1] != len(feature_names):
        raise ValueError("feature column count mismatch")
    return pd.DataFrame(x, columns=list(feature_names), copy=False)


def _fit_one(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    valid_y: np.ndarray,
    cfg: StrategyMLConfig,
    alpha: float,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=cfg.calibrator_n_estimators,
        learning_rate=cfg.calibrator_learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=min(cfg.max_depth, cfg.calibrator_max_depth_cap),
        min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.calibrator_feature_fraction,
        bagging_fraction=cfg.calibrator_bagging_fraction,
        bagging_freq=cfg.calibrator_bagging_freq,
        lambda_l2=cfg.calibrator_lambda_l2,
        reg_alpha=cfg.calibrator_reg_alpha,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbose=-1,
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
    """Train q10/q50/q90 models on absolute executable EV target."""
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
    # Keep raw y_ev for calibrator so EV magnitude remains executable against cost wall.
    y_train = train.y_ev
    y_valid = valid.y_ev
    x_train_np = np.column_stack([train.X, rank_score_train])
    x_train_names = (*train.feature_names, "rank_score")
    x_train = _as_feature_frame(x_train_np, x_train_names)
    x_valid = (
        _as_feature_frame(
            np.column_stack([valid.X, rank_score_valid]),
            (*valid.feature_names, "rank_score"),
        )
        if valid.X.shape[0] > 0
        else _as_feature_frame(
            np.zeros((0, x_train_np.shape[1]), dtype=np.float32),
            x_train_names,
        )
    )
    q10 = _fit_one(x_train, y_train, x_valid, y_valid, cfg, 0.10)
    q50 = _fit_one(x_train, y_train, x_valid, y_valid, cfg, 0.50)
    q90 = _fit_one(x_train, y_train, x_valid, y_valid, cfg, 0.90)
    return CalibratorFitResult(q10=q10, q50=q50, q90=q90)


def predict_ev_quantiles(
    models: CalibratorFitResult,
    dataset: LongMatrixDataset,
    rank_score: np.ndarray,
) -> EVQuantiles:
    """Predict q10/q50/q90 EV in return units."""
    if dataset.X.shape[0] == 0:
        empty: NDArray[np.float32] = np.zeros((0,), dtype=np.float32)
        return EVQuantiles(q10=empty, q50=empty, q90=empty)
    if rank_score.shape[0] != dataset.X.shape[0]:
        raise ValueError("rank_score length mismatch")
    x = _as_feature_frame(
        np.column_stack([dataset.X, rank_score]),
        (*dataset.feature_names, "rank_score"),
    )
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
    """Compute conservative EV from predicted quantiles.

    Uses sign-symmetric adjustment so both long (q50>0) and short (q50<0)
    positions are penalized by their own tail risk, avoiding the negative bias
    of always subtracting downside from a CS-demeaned (mean≈0) q50.

    Long (q50 >= 0): ev = q50 - lambda * downside_risk
    Short (q50 < 0): ev = q50 + lambda * upside_risk  (reduces magnitude)
    """
    if q10.shape != q50.shape or q50.shape != q90.shape:
        raise ValueError("quantile vectors must have identical shapes")
    if q10.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    q10f = np.asarray(q10, dtype=np.float32)
    q50f = np.asarray(q50, dtype=np.float32)
    q90f = np.asarray(q90, dtype=np.float32)
    uncertainty = np.maximum(q90f - q10f, np.float32(1e-8))
    if cfg.ev_mode == "prob_x_magnitude":
        magnitude = np.maximum(np.abs(q50f), np.float32(0.0))
        prob_up = np.clip((q50f - q10f) / uncertainty, np.float32(0.0), np.float32(1.0))
        directional_prob = (np.float32(2.0) * prob_up) - np.float32(1.0)
        ev = directional_prob * magnitude
    else:
        downside = np.maximum(q50f - q10f, np.float32(0.0))
        upside = np.maximum(q90f - q50f, np.float32(0.0))
        med_unc = np.median(uncertainty) if uncertainty.size > 0 else np.float32(1e-4)
        lam = np.float32(cfg.lambda_tail)
        lam_dynamic = np.clip(
            lam * (uncertainty / np.maximum(med_unc, np.float32(1e-8))),
            np.float32(0.0),
            lam * np.float32(2.0),
        )
        is_long = q50f >= np.float32(0.0)
        penalty_ratio = np.where(is_long, downside / uncertainty, upside / uncertainty)
        penalty_term = np.clip(lam_dynamic * penalty_ratio, np.float32(0.0), np.float32(0.99))
        ev = q50f * (np.float32(1.0) - penalty_term)
    ev = np.asarray(ev, dtype=np.float32)
    clip = np.float32(cfg.alpha_clip_bps / 10000.0)
    clipped = np.clip(ev, -clip, clip)
    return np.asarray(clipped, dtype=np.float32).reshape(-1).copy()


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
