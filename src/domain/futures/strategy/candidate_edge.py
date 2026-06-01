from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Any

import numpy as np
from lightgbm import LGBMRegressor
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput
from src.domain.futures.strategy.candidate_dataset import CandidateDataset
from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(slots=True, frozen=True)
class CandidateEdgeModels:
    center_model: LGBMRegressor
    q10_model: LGBMRegressor
    q90_model: LGBMRegressor
    feature_names: tuple[str, ...]


def _seed_from_cfg(cfg: Any) -> int:
    return int(getattr(cfg, "seed", getattr(getattr(cfg, "ml", cfg), "seed", 42)))


def _read_float(cfg: Any, key: str, default: float) -> float:
    return float(getattr(cfg, key, getattr(getattr(cfg, "ml", cfg), key, default)))


def fit_candidate_edge_models(
    *,
    train: CandidateDataset,
    valid: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> CandidateEdgeModels:
    """Fit robust expected edge and quantile downside models."""
    seed = _seed_from_cfg(cfg)

    center = LGBMRegressor(
        objective="huber",
        random_state=seed,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=1,
        verbose=-1,
    )
    q10 = LGBMRegressor(
        objective="quantile",
        alpha=0.10,
        random_state=seed,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=1,
        verbose=-1,
    )
    q90 = LGBMRegressor(
        objective="quantile",
        alpha=0.90,
        random_state=seed,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=1,
        verbose=-1,
    )

    # Use valid split if provided for early stopping
    eval_set_center: Any = [(valid.X, valid.y_edge_bps)] if valid.X.shape[0] > 0 else None
    eval_set_q10: Any = [(valid.X, valid.y_q10_bps)] if valid.X.shape[0] > 0 else None
    eval_set_q90: Any = [(valid.X, valid.y_edge_bps)] if valid.X.shape[0] > 0 else None

    if eval_set_center is not None and len(np.unique(valid.y_edge_bps)) > 1:
        center.fit(
            train.X,
            train.y_edge_bps,
            sample_weight=train.sample_weight,
            eval_set=eval_set_center,
            callbacks=[],
        )
    else:
        center.fit(train.X, train.y_edge_bps, sample_weight=train.sample_weight)

    if eval_set_q10 is not None and len(np.unique(valid.y_q10_bps)) > 1:
        q10.fit(
            train.X,
            train.y_q10_bps,
            sample_weight=train.sample_weight,
            eval_set=eval_set_q10,
            callbacks=[],
        )
    else:
        q10.fit(train.X, train.y_q10_bps, sample_weight=train.sample_weight)

    # Fit Q90 on y_edge_bps (representing the target upside distribution)
    if eval_set_q90 is not None and len(np.unique(valid.y_edge_bps)) > 1:
        q90.fit(
            train.X,
            train.y_edge_bps,
            sample_weight=train.sample_weight,
            eval_set=eval_set_q90,
            callbacks=[],
        )
    else:
        q90.fit(train.X, train.y_edge_bps, sample_weight=train.sample_weight)

    return CandidateEdgeModels(
        center_model=center,
        q10_model=q10,
        q90_model=q90,
        feature_names=tuple(train.feature_names),
    )


def predict_candidate_edges(
    *,
    models: CandidateEdgeModels,
    dataset: CandidateDataset,
    p_pass: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
) -> CandidateModelOutput:
    """Return expected edge, downside quantiles, and utility scores."""
    mu_gross_bps = cast(NDArray[np.float64], models.center_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )
    q10_gross_bps = cast(NDArray[np.float64], models.q10_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )
    q90_gross_bps = cast(NDArray[np.float64], models.q90_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )

    expected_cost_bps = _read_float(cfg, "expected_cost_bps", 0.0)
    downside_penalty = _read_float(cfg, "downside_penalty", 0.0)
    turnover_penalty = _read_float(cfg, "turnover_penalty", 0.0)
    concentration_penalty = _read_float(cfg, "concentration_penalty", 0.0)

    # Extract turnover_proxy from features or fallback to a constant 1.0 (so turnover_term = turnover_penalty)
    turnover_proxy = np.ones_like(mu_gross_bps)
    if "turnover_proxy" in dataset.feature_names:
        t_idx = dataset.feature_names.index("turnover_proxy")
        turnover_proxy = dataset.X[:, t_idx].astype(np.float64, copy=False)

    mu_net_decision_bps = mu_gross_bps - expected_cost_bps
    q10_net_bps = q10_gross_bps - expected_cost_bps
    q90_net_bps = q90_gross_bps - expected_cost_bps

    downside_term = downside_penalty * np.abs(np.minimum(q10_net_bps, 0.0))
    turnover_term = turnover_penalty * turnover_proxy
    utility_score = p_pass * mu_net_decision_bps - downside_term - turnover_term - concentration_penalty

    return CandidateModelOutput(
        events=None,
        p_pass=p_pass.astype(np.float64, copy=False),
        mu_gross_bps=mu_gross_bps,
        mu_net_decision_bps=mu_net_decision_bps,
        q10_net_bps=q10_net_bps,
        q90_net_bps=q90_net_bps,
        utility_score=utility_score,
    )

