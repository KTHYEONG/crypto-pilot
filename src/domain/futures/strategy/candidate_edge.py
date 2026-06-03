from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
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


def _diagnostic_top_k(cfg: Any) -> int:
    return max(1, int(getattr(cfg, "diagnostic_top_k", 10)))


def _edge_shortfall_pass_rate(
    values: NDArray[np.float32] | NDArray[np.float64],
    threshold: float,
) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float((finite >= threshold).mean())


def _shortfall_threshold(cfg: Any) -> float:
    return -float(getattr(cfg, "max_expected_shortfall_bps", 80.0))


def _log_edge_target_distribution(*, split: str, dataset: CandidateDataset, cfg: CandidateStrategyConfig) -> None:
    """Log the target distribution for a dataset split."""
    if dataset.X.shape[0] == 0:
        return

    y_edge = np.asarray(dataset.y_edge_bps, dtype=np.float64)
    y_q10 = np.asarray(dataset.y_q10_bps, dtype=np.float64)
    finite_edge = y_edge[np.isfinite(y_edge)]
    finite_q10 = y_q10[np.isfinite(y_q10)]
    _logger = logging.getLogger(__name__)
    _logger.debug(
        (
            "[DIAG][EDGE_TARGET] split=%s n=%d mean=%.1f median=%.1f "
            "p10=%.1f p90=%.1f pct_pos=%.3f q10_mean=%.1f q10_pass=%.3f"
        ),
        split,
        int(dataset.X.shape[0]),
        float(np.mean(finite_edge)) if finite_edge.size > 0 else float("nan"),
        float(np.median(finite_edge)) if finite_edge.size > 0 else float("nan"),
        float(np.percentile(finite_edge, 10)) if finite_edge.size > 0 else float("nan"),
        float(np.percentile(finite_edge, 90)) if finite_edge.size > 0 else float("nan"),
        float((finite_edge > 0.0).mean()) if finite_edge.size > 0 else 0.0,
        float(np.mean(finite_q10)) if finite_q10.size > 0 else float("nan"),
        _edge_shortfall_pass_rate(y_q10, _shortfall_threshold(cfg)),
    )


def _log_edge_target_variants(*, split: str, dataset: CandidateDataset, cfg: CandidateStrategyConfig) -> None:
    """Log target summaries grouped by candidate variant."""
    event_index = getattr(dataset, "event_index", pd.DataFrame())
    if dataset.X.shape[0] == 0 or event_index.empty:
        return

    frame = event_index.copy()
    frame["_edge"] = np.asarray(dataset.y_edge_bps, dtype=np.float64)
    frame["_q10"] = np.asarray(dataset.y_q10_bps, dtype=np.float64)
    grouped = frame.groupby(["family", "variant"], sort=False, dropna=False)
    rows: list[tuple[str, float, float, float]] = []
    for (family, variant), group in grouped:
        edge = pd.to_numeric(group["_edge"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        q10 = pd.to_numeric(group["_q10"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite_edge = edge[np.isfinite(edge)]
        rows.append(
            (
                f"{family}:{variant}",
                float(np.mean(finite_edge)) if finite_edge.size > 0 else float("nan"),
                float((finite_edge > 0.0).mean()) if finite_edge.size > 0 else 0.0,
                _edge_shortfall_pass_rate(q10, _shortfall_threshold(cfg)),
            )
        )

    top = sorted(rows, key=lambda item: item[1], reverse=True)[: _diagnostic_top_k(cfg)]
    for key, mean_edge, pct_pos, q10_pass in top:
        logging.getLogger(__name__).debug(
            "[DIAG][EDGE_TARGET_VARIANT] split=%s key=%s mean=%.1f pct_pos=%.3f q10_pass=%.3f",
            split,
            key,
            mean_edge,
            pct_pos,
            q10_pass,
        )


def _log_edge_prediction_variants(
    *,
    dataset: CandidateDataset,
    mu_net_decision_bps: NDArray[np.float64],
    q10_net_bps: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
) -> None:
    """Log prediction summaries grouped by candidate variant."""
    event_index = getattr(dataset, "event_index", pd.DataFrame())
    if dataset.X.shape[0] == 0 or event_index.empty:
        return

    frame = event_index.copy()
    frame["_mu_net"] = np.asarray(mu_net_decision_bps, dtype=np.float64)
    frame["_q10_net"] = np.asarray(q10_net_bps, dtype=np.float64)
    grouped = frame.groupby(["family", "variant"], sort=False, dropna=False)
    rows: list[tuple[str, float, float, float, float, int]] = []
    for (family, variant), group in grouped:
        mu = pd.to_numeric(group["_mu_net"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        q10 = pd.to_numeric(group["_q10_net"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite_mu = mu[np.isfinite(mu)]
        finite_q10 = q10[np.isfinite(q10)]
        rows.append(
            (
                f"{family}:{variant}",
                float(np.mean(finite_mu)) if finite_mu.size > 0 else float("nan"),
                float(np.max(finite_mu)) if finite_mu.size > 0 else float("nan"),
                float((finite_mu >= 1.0).mean()) if finite_mu.size > 0 else 0.0,
                float((finite_q10 >= _shortfall_threshold(cfg)).mean()) if finite_q10.size > 0 else 0.0,
                int(group.shape[0]),
            )
        )

    top = sorted(rows, key=lambda item: item[1], reverse=True)[: _diagnostic_top_k(cfg)]
    for key, mean_mu, max_mu, pct_mu_pass, pct_q10_pass, n in top:
        logging.getLogger(__name__).debug(
            "[DIAG][EDGE_VARIANT_TOP] key=%s n=%d mean_mu=%.1f max_mu=%.1f pct_mu_pass=%.3f pct_q10_pass=%.3f",
            key,
            n,
            mean_mu,
            max_mu,
            pct_mu_pass,
            pct_q10_pass,
        )


def _selection_thresholds(
    *,
    utility_score: NDArray[np.float64],
    p_pass: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
) -> dict[str, float]:
    finite_utility = utility_score[np.isfinite(utility_score)]
    utility_min = float("-inf")
    if finite_utility.size > 0:
        quantile = max(0.0, min(1.0, 1.0 - float(cfg.selection_top_quantile)))
        utility_min = float(np.quantile(finite_utility, quantile))
    return {
        "utility_min": utility_min,
        "p_pass_min": float(np.nanmin(p_pass)) if p_pass.size > 0 else 0.0,
        "edge_min": max(0.0, float(cfg.min_expected_net_bps)),
        "q10_catastrophic_min": -float(cfg.catastrophic_shortfall_bps),
    }


def fit_candidate_edge_models(
    *,
    train: CandidateDataset,
    valid: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> CandidateEdgeModels:
    """Fit robust expected edge and quantile downside models."""
    seed = _seed_from_cfg(cfg)
    _log_edge_target_distribution(split="train", dataset=train, cfg=cfg)
    _log_edge_target_distribution(split="valid", dataset=valid, cfg=cfg)
    _log_edge_target_variants(split="train", dataset=train, cfg=cfg)
    _log_edge_target_variants(split="valid", dataset=valid, cfg=cfg)

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
    eval_set_q90: Any = [(valid.X, valid.y_mfe_bps)] if valid.X.shape[0] > 0 else None

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

    # Fit Q90 on y_mfe_bps (upside realisation — maximum favourable excursion)
    if eval_set_q90 is not None and len(np.unique(valid.y_mfe_bps)) > 1:
        q90.fit(
            train.X,
            train.y_mfe_bps,
            sample_weight=train.sample_weight,
            eval_set=eval_set_q90,
            callbacks=[],
        )
    else:
        q90.fit(train.X, train.y_mfe_bps, sample_weight=train.sample_weight)

    return CandidateEdgeModels(
        center_model=center,
        q10_model=q10,
        q90_model=q90,
        feature_names=tuple(train.feature_names),
    )


def predict_candidate_edges(
    *,
    models: CandidateEdgeModels | None,
    dataset: CandidateDataset,
    p_pass: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
) -> CandidateModelOutput:
    """Return expected edge, downside quantiles, and utility scores."""
    if models is None or dataset.X.shape[0] == 0:
        zeros = np.zeros(dataset.X.shape[0], dtype=np.float64)
        return CandidateModelOutput(
            events=None,
            p_pass=p_pass.astype(np.float64, copy=False),
            mu_gross_bps=zeros,
            mu_net_decision_bps=zeros,
            q10_net_bps=zeros,
            q90_net_bps=zeros,
            utility_score=zeros,
            selection_thresholds={
                "utility_min": 0.0,
                "p_pass_min": 0.0,
                "edge_min": 0.0,
                "q10_catastrophic_min": 0.0,
            },
        )

    mu_model_bps = cast(NDArray[np.float64], models.center_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )
    q10_model_bps = cast(NDArray[np.float64], models.q10_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )
    q90_model_bps = cast(NDArray[np.float64], models.q90_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )

    expected_cost_bps = float(getattr(cfg, "expected_cost_bps", 24.0))
    downside_penalty = float(getattr(cfg, "downside_penalty", 1.0))
    turnover_penalty = float(getattr(cfg, "turnover_penalty", 0.5))
    concentration_penalty = float(getattr(cfg, "concentration_penalty", 0.0))

    # Extract turnover_proxy from features or fallback to a constant 1.0 (so turnover_term = turnover_penalty)
    turnover_proxy = np.ones_like(mu_model_bps)
    if "turnover_proxy" in dataset.feature_names:
        t_idx = dataset.feature_names.index("turnover_proxy")
        turnover_proxy = dataset.X[:, t_idx].astype(np.float64, copy=False)

    # The fitted targets already use net-of-cost/hurdle labels.
    mu_net_decision_bps = mu_model_bps
    q10_net_bps = q10_model_bps
    q90_net_bps = q90_model_bps

    downside_term = downside_penalty * np.abs(np.minimum(q10_net_bps, 0.0))
    turnover_term = turnover_penalty * turnover_proxy
    utility_score = p_pass * mu_net_decision_bps - downside_term - turnover_term - concentration_penalty

    _logger = logging.getLogger(__name__)
    _logger.debug(
        "[DIAG][EDGE] n=%d target_scale=net cost_bps=%.1f "
        "mu_model mean=%.1f max=%.1f pct_ge25=%.3f | "
        "mu_decision mean=%.1f max=%.1f pct_ge1=%.3f | "
        "q10_net mean=%.1f min=%.1f | utility mean=%.3f max=%.3f",
        len(mu_model_bps),
        expected_cost_bps,
        float(mu_model_bps.mean()),
        float(mu_model_bps.max()),
        float((mu_model_bps >= 25.0).mean()),
        float(mu_net_decision_bps.mean()),
        float(mu_net_decision_bps.max()),
        float((mu_net_decision_bps >= 1.0).mean()),
        float(q10_net_bps.mean()),
        float(q10_net_bps.min()),
        float(utility_score.mean()),
        float(utility_score.max()),
    )
    _log_edge_prediction_variants(
        dataset=dataset,
        mu_net_decision_bps=mu_net_decision_bps,
        q10_net_bps=q10_net_bps,
        cfg=cfg,
    )
    return CandidateModelOutput(
        events=None,
        p_pass=p_pass.astype(np.float64, copy=False),
        # Legacy field name: values are now on the fitted target scale (net of cost/hurdle).
        mu_gross_bps=mu_model_bps,
        mu_net_decision_bps=mu_net_decision_bps,
        q10_net_bps=q10_net_bps,
        q90_net_bps=q90_net_bps,
        utility_score=utility_score,
        selection_thresholds=_selection_thresholds(
            utility_score=utility_score,
            p_pass=p_pass.astype(np.float64, copy=False),
            cfg=cfg,
        ),
    )
