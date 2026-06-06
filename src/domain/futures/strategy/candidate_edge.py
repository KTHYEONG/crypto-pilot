from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

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
    variant_prior_bps: dict[str, float]
    variant_prior_obs: dict[str, int]
    global_prior_bps: float
    target_mode: Literal["direct", "prior_residual"]
    prediction_diagnostics: dict[str, float | int | str]


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


def _variant_keys(dataset: CandidateDataset) -> list[str]:
    """Return stable per-row variant keys or global fallbacks."""
    event_index = getattr(dataset, "event_index", pd.DataFrame())
    if (
        not event_index.empty
        and {"family", "variant"}.issubset(event_index.columns)
        and len(event_index) == dataset.X.shape[0]
    ):
        keys = event_index["family"].astype(str).str.cat(event_index["variant"].astype(str), sep=":")
        return list(keys.to_numpy(dtype=object))
    return ["__global__"] * dataset.X.shape[0]


def _weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Return a finite weighted mean with uniform fallback."""
    finite_mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(finite_mask.any()):
        finite_values = values[np.isfinite(values)]
        return float(np.mean(finite_values)) if finite_values.size > 0 else 0.0
    return float(np.average(values[finite_mask], weights=weights[finite_mask]))


def _build_variant_priors(
    *,
    dataset: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> tuple[dict[str, float], dict[str, int], float, NDArray[np.float64]]:
    """Return shrunk per-variant priors and per-row prior values."""
    y_edge = np.asarray(dataset.y_edge_bps, dtype=np.float64)
    weights = np.asarray(dataset.edge_weight, dtype=np.float64)
    keys = _variant_keys(dataset)
    global_prior = _weighted_mean(y_edge, weights)
    variant_prior_bps: dict[str, float] = {}
    variant_prior_obs: dict[str, int] = {}
    row_priors = np.full(dataset.X.shape[0], global_prior, dtype=np.float64)
    shrinkage_obs = float(cfg.edge_prior_shrinkage_obs)
    min_obs = int(cfg.edge_prior_min_obs)

    key_to_indices: dict[str, list[int]] = {}
    for idx, key in enumerate(keys):
        key_to_indices.setdefault(key, []).append(idx)

    for key, indices in key_to_indices.items():
        indexer = np.asarray(indices, dtype=np.int32)
        variant_values = y_edge[indexer]
        variant_weights = weights[indexer]
        obs = int(indexer.shape[0])
        variant_prior_obs[key] = obs
        if obs < min_obs:
            row_priors[indexer] = global_prior
            continue
        variant_mean = _weighted_mean(variant_values, variant_weights)
        shrink = obs / (obs + shrinkage_obs)
        prior = float(shrink * variant_mean + (1.0 - shrink) * global_prior)
        max_dev = float(getattr(cfg, "edge_prior_max_deviation_bps", float("inf")))
        if np.isfinite(max_dev) and max_dev > 0.0:
            prior = float(np.clip(prior, global_prior - max_dev, global_prior + max_dev))
        variant_prior_bps[key] = prior
        row_priors[indexer] = prior

    return variant_prior_bps, variant_prior_obs, global_prior, row_priors


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
    mu_net_decision_bps: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
) -> dict[str, float | bool]:
    utility_min = float(getattr(cfg, "selection_min_expected_utility_bps", 0.0))
    finite_mu = mu_net_decision_bps[np.isfinite(mu_net_decision_bps)]
    breakeven_floor_bps = float(cfg.min_net_floor_cost_fraction) * float(cfg.cost_floor_bps)
    mu_std_bps = float(np.std(finite_mu)) if finite_mu.size > 0 else 0.0
    mu_positive_rate = float((finite_mu > 0.0).mean()) if finite_mu.size > 0 else 0.0
    mu_floor_pass_rate = float((finite_mu >= breakeven_floor_bps).mean()) if finite_mu.size > 0 else 0.0
    prediction_collapse = bool(
        finite_mu.size > 0
        and mu_std_bps < float(cfg.edge_prediction_min_std_bps)
        and mu_positive_rate < float(cfg.edge_prediction_min_positive_rate)
    )
    return {
        "utility_min": utility_min,
        "p_pass_min": float(np.nanmin(p_pass)) if p_pass.size > 0 else 0.0,
        "edge_min": max(0.0, float(cfg.min_expected_net_bps)),
        "q10_catastrophic_min": -float(cfg.catastrophic_shortfall_bps),
        "mu_std_bps": mu_std_bps,
        "mu_positive_rate": mu_positive_rate,
        "mu_floor_pass_rate": mu_floor_pass_rate,
        "prediction_collapse": prediction_collapse,
    }


def _rank_ic(pred: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    pred_s = pd.Series(pred, dtype="float64")
    target_s = pd.Series(target, dtype="float64")
    corr = pred_s.corr(target_s, method="spearman")
    return float(corr) if corr is not None and np.isfinite(corr) else float("nan")


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
    variant_prior_bps, variant_prior_obs, global_prior_bps, train_row_priors = _build_variant_priors(
        dataset=train,
        cfg=cfg,
    )
    use_prior_residual = bool(cfg.edge_prior_enabled and cfg.edge_residual_model_enabled)
    target_mode: Literal["direct", "prior_residual"] = "prior_residual" if use_prior_residual else "direct"
    center_train_target = (
        train.y_edge_bps.astype(np.float64, copy=False) - train_row_priors
        if use_prior_residual
        else train.y_edge_bps.astype(np.float64, copy=False)
    )
    center_valid_target = valid.y_edge_bps.astype(np.float64, copy=False)
    if use_prior_residual and valid.X.shape[0] > 0:
        valid_keys = _variant_keys(valid)
        valid_priors = np.asarray(
            [variant_prior_bps.get(key, global_prior_bps) for key in valid_keys],
            dtype=np.float64,
        )
        center_valid_target = center_valid_target - valid_priors

    max_depth = int(getattr(cfg, "edge_lgbm_max_depth", 4))
    reg_lambda = float(getattr(cfg, "edge_lgbm_reg_lambda", 20.0))
    num_leaves = 2**max_depth - 1 if max_depth > 0 else 31

    center = LGBMRegressor(
        objective="huber",
        random_state=seed,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=2.0,
        reg_lambda=reg_lambda,
        n_jobs=1,
        verbose=-1,
    )
    q10 = LGBMRegressor(
        objective="quantile",
        alpha=0.10,
        random_state=seed,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=2.0,
        reg_lambda=reg_lambda,
        n_jobs=1,
        verbose=-1,
    )
    q90 = LGBMRegressor(
        objective="quantile",
        alpha=0.90,
        random_state=seed,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=2.0,
        reg_lambda=reg_lambda,
        n_jobs=1,
        verbose=-1,
    )

    # Use valid split if provided for early stopping
    eval_set_center: Any = [(valid.X, center_valid_target)] if valid.X.shape[0] > 0 else None
    eval_set_q10: Any = [(valid.X, valid.y_q10_bps)] if valid.X.shape[0] > 0 else None
    eval_set_q90: Any = [(valid.X, valid.y_mfe_bps)] if valid.X.shape[0] > 0 else None

    from lightgbm import early_stopping as lgbm_early_stopping
    from lightgbm import log_evaluation as lgbm_log_evaluation

    _es_cb: list[Any] = [lgbm_early_stopping(stopping_rounds=30, verbose=False), lgbm_log_evaluation(period=-1)]

    if eval_set_center is not None and len(np.unique(valid.y_edge_bps)) > 1:
        center.fit(
            train.X,
            center_train_target,
            sample_weight=train.edge_weight,
            eval_set=eval_set_center,
            callbacks=_es_cb,
        )
    else:
        center.fit(train.X, center_train_target, sample_weight=train.edge_weight)

    if eval_set_q10 is not None and len(np.unique(valid.y_q10_bps)) > 1:
        q10.fit(
            train.X,
            train.y_q10_bps,
            sample_weight=train.edge_weight,
            eval_set=eval_set_q10,
            callbacks=_es_cb,
        )
    else:
        q10.fit(train.X, train.y_q10_bps, sample_weight=train.edge_weight)

    # Fit Q90 on y_mfe_bps (upside realisation — maximum favourable excursion)
    if eval_set_q90 is not None and len(np.unique(valid.y_mfe_bps)) > 1:
        q90.fit(
            train.X,
            train.y_mfe_bps,
            sample_weight=train.edge_weight,
            eval_set=eval_set_q90,
            callbacks=_es_cb,
        )
    else:
        q90.fit(train.X, train.y_mfe_bps, sample_weight=train.edge_weight)

    return CandidateEdgeModels(
        center_model=center,
        q10_model=q10,
        q90_model=q90,
        feature_names=tuple(train.feature_names),
        variant_prior_bps=variant_prior_bps,
        variant_prior_obs=variant_prior_obs,
        global_prior_bps=global_prior_bps,
        target_mode=target_mode,
        prediction_diagnostics={
            "target_mode": target_mode,
            "global_prior_bps": global_prior_bps,
            "variant_prior_count": len(variant_prior_bps),
        },
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
                "mu_std_bps": 0.0,
                "mu_positive_rate": 0.0,
                "mu_floor_pass_rate": 0.0,
                "prediction_collapse": False,
            },
        )

    center_pred_bps = cast(NDArray[np.float64], models.center_model.predict(dataset.X)).astype(
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
    turnover_proxy = np.ones_like(center_pred_bps)
    if "turnover_proxy" in dataset.feature_names:
        t_idx = dataset.feature_names.index("turnover_proxy")
        turnover_proxy = dataset.X[:, t_idx].astype(np.float64, copy=False)

    prior_bps = np.zeros(dataset.X.shape[0], dtype=np.float64)
    if models.target_mode == "prior_residual":
        keys = _variant_keys(dataset)
        prior_bps = np.asarray(
            [models.variant_prior_bps.get(key, models.global_prior_bps) for key in keys],
            dtype=np.float64,
        )
    mu_net_decision_bps = center_pred_bps + prior_bps
    q10_net_bps = q10_model_bps
    q90_net_bps = q90_model_bps

    prob_adjusted_mu_bps = p_pass * mu_net_decision_bps
    downside_term = downside_penalty * np.abs(np.minimum(q10_net_bps, 0.0))
    turnover_term = turnover_penalty * turnover_proxy
    expected_utility_bps = (
        prob_adjusted_mu_bps
        + (1.0 - p_pass) * np.minimum(q10_net_bps, 0.0)
        - turnover_term
        - concentration_penalty
    )
    if getattr(cfg, "selection_utility_mode", "additive_drag") == "expected_edge_direct":
        # Direct mode: mu_net is the unconditional E[return]; p_pass/q10 are sizing-only.
        utility_score = mu_net_decision_bps
    elif bool(getattr(cfg, "selection_use_expected_utility", True)):
        utility_score = expected_utility_bps
    else:
        utility_score = prob_adjusted_mu_bps - downside_term - turnover_term - concentration_penalty

    breakeven_floor_bps = float(getattr(cfg, "min_net_floor_cost_fraction", 0.5)) * float(
        getattr(cfg, "cost_floor_bps", expected_cost_bps)
    )
    catastrophic_shortfall_bps = float(getattr(cfg, "catastrophic_shortfall_bps", 300.0))
    max_shortfall_bps = float(getattr(cfg, "max_expected_shortfall_bps", 80.0))
    finite_mu = mu_net_decision_bps[np.isfinite(mu_net_decision_bps)]
    finite_q10 = q10_net_bps[np.isfinite(q10_net_bps)]
    finite_utility = utility_score[np.isfinite(utility_score)]
    finite_expected_utility = expected_utility_bps[np.isfinite(expected_utility_bps)]

    _logger = logging.getLogger(__name__)
    prediction_collapse = bool(
        finite_mu.size > 0
        and float(np.std(finite_mu)) < float(cfg.edge_prediction_min_std_bps)
        and float((finite_mu > 0.0).mean()) < float(cfg.edge_prediction_min_positive_rate)
    )
    _logger.info(
        (
            "[DIAG][EDGE] n=%d target_scale=net mode=%s cost_bps=%.1f floor_bps=%.1f "
            "mu_mean=%.1f mu_p50=%.1f mu_p90=%.1f mu_max=%.1f "
            "q10_mean=%.1f q10_p10=%.1f q10_p50=%.1f q10_min=%.1f "
            "exp_util_mean=%.3f exp_util_p50=%.3f exp_util_p90=%.3f "
            "utility_mean=%.3f utility_p50=%.3f utility_p90=%.3f utility_max=%.3f "
            "pct_mu_ge1=%.3f pct_mu_ge_floor=%.3f pct_q10_ge_cat=%.3f pct_q10_ge_max=%.3f"
        ),
        len(center_pred_bps),
        models.target_mode,
        expected_cost_bps,
        breakeven_floor_bps,
        float(np.mean(finite_mu)) if finite_mu.size > 0 else float("nan"),
        float(np.median(finite_mu)) if finite_mu.size > 0 else float("nan"),
        float(np.percentile(finite_mu, 90)) if finite_mu.size > 0 else float("nan"),
        float(np.max(finite_mu)) if finite_mu.size > 0 else float("nan"),
        float(np.mean(finite_q10)) if finite_q10.size > 0 else float("nan"),
        float(np.percentile(finite_q10, 10)) if finite_q10.size > 0 else float("nan"),
        float(np.median(finite_q10)) if finite_q10.size > 0 else float("nan"),
        float(np.min(finite_q10)) if finite_q10.size > 0 else float("nan"),
        float(np.mean(finite_expected_utility)) if finite_expected_utility.size > 0 else float("nan"),
        float(np.median(finite_expected_utility)) if finite_expected_utility.size > 0 else float("nan"),
        float(np.percentile(finite_expected_utility, 90)) if finite_expected_utility.size > 0 else float("nan"),
        float(np.mean(finite_utility)) if finite_utility.size > 0 else float("nan"),
        float(np.median(finite_utility)) if finite_utility.size > 0 else float("nan"),
        float(np.percentile(finite_utility, 90)) if finite_utility.size > 0 else float("nan"),
        float(np.max(finite_utility)) if finite_utility.size > 0 else float("nan"),
        float((finite_mu >= 1.0).mean()) if finite_mu.size > 0 else 0.0,
        float((finite_mu >= breakeven_floor_bps).mean()) if finite_mu.size > 0 else 0.0,
        float((finite_q10 >= -catastrophic_shortfall_bps).mean()) if finite_q10.size > 0 else 0.0,
        float((finite_q10 >= -max_shortfall_bps).mean()) if finite_q10.size > 0 else 0.0,
    )
    if prediction_collapse:
        _logger.warning(
            "[DIAG][EDGE_COLLAPSE] std=%.3f positive_rate=%.3f threshold_std=%.3f threshold_pos=%.3f",
            float(np.std(finite_mu)) if finite_mu.size > 0 else 0.0,
            float((finite_mu > 0.0).mean()) if finite_mu.size > 0 else 0.0,
            float(cfg.edge_prediction_min_std_bps),
            float(cfg.edge_prediction_min_positive_rate),
        )
    _log_edge_prediction_variants(
        dataset=dataset,
        mu_net_decision_bps=mu_net_decision_bps,
        q10_net_bps=q10_net_bps,
        cfg=cfg,
    )
    if dataset.X.shape[0] > 0 and dataset.y_edge_bps.shape[0] == dataset.X.shape[0]:
        diag_frame = pd.DataFrame(
            {
                "mu": mu_net_decision_bps,
                "p_pass": p_pass,
                "expected_utility": expected_utility_bps,
                "y_edge": np.asarray(dataset.y_edge_bps, dtype=np.float64),
            }
        )
        for col in ("mu", "p_pass", "expected_utility"):
            valid = diag_frame[[col, "y_edge"]].replace([np.inf, -np.inf], np.nan).dropna()
            if valid.empty:
                continue
            valid["decile"] = pd.qcut(valid[col], q=10, labels=False, duplicates="drop")
            grouped = valid.groupby("decile", sort=True)
            for decile, group in grouped:
                _logger.debug(
                    "[DIAG][EDGE_DECILE] col=%s decile=%s n=%d realized_mean=%.3f hit_rate=%.3f",
                    col,
                    int(decile),
                    int(group.shape[0]),
                    float(group["y_edge"].mean()),
                    float((group["y_edge"] > 0.0).mean()),
                )
        _logger.debug(
            "[DIAG][EDGE_RANK] mu_ic=%.4f expected_utility_ic=%.4f",
            _rank_ic(mu_net_decision_bps, np.asarray(dataset.y_edge_bps, dtype=np.float64)),
            _rank_ic(expected_utility_bps, np.asarray(dataset.y_edge_bps, dtype=np.float64)),
        )
    return CandidateModelOutput(
        events=None,
        p_pass=p_pass.astype(np.float64, copy=False),
        # Legacy field name: values are now on the fitted target scale (net of cost/hurdle).
        mu_gross_bps=mu_net_decision_bps,
        mu_net_decision_bps=mu_net_decision_bps,
        q10_net_bps=q10_net_bps,
        q90_net_bps=q90_net_bps,
        utility_score=utility_score,
        selection_thresholds=_selection_thresholds(
            utility_score=utility_score,
            p_pass=p_pass.astype(np.float64, copy=False),
            mu_net_decision_bps=mu_net_decision_bps,
            cfg=cfg,
        ),
    )
