from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
    EdgePredictionMode,
    EdgeSource,
)
from src.domain.futures.strategy.candidate_dataset import CandidateDataset
from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(slots=True, frozen=True)
class EdgeModelValidation:
    """Rank IC validation result for edge model acceptance gate."""

    rank_ic_cal_eval: float
    n_cal_eval: int
    accepted: bool
    reason: str
    overlay_lift_bps: float = 0.0
    overlay_lift_tstat: float = 0.0
    n_eff: float = 0.0


@dataclass(slots=True, frozen=True)
class CandidateEdgeModels:
    center_model: LGBMRegressor
    q10_model: LGBMRegressor
    q90_model: LGBMRegressor
    feature_names: tuple[str, ...]
    variant_prior_r: dict[str, float]
    global_prior_r: float
    variant_prior_bps: dict[str, float]
    variant_prior_obs: dict[str, int]
    global_prior_bps: float
    prediction_mode: EdgePredictionMode
    prediction_diagnostics: dict[str, float | int | str]
    validation: EdgeModelValidation | None = None


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
    """Return a finite weighted mean using positive-weight rows only."""
    finite_mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(finite_mask.any()):
        return 0.0
    return float(np.average(values[finite_mask], weights=weights[finite_mask]))


def _weighted_tstat(values: NDArray[np.float64], weights: NDArray[np.float64]) -> tuple[float, float]:
    finite_mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(finite_mask.any()):
        return 0.0, 0.0
    x = values[finite_mask]
    w = weights[finite_mask]
    w_sum = float(np.sum(w))
    w_sq_sum = float(np.sum(np.square(w)))
    if w_sum <= 0.0 or w_sq_sum <= 0.0:
        return 0.0, 0.0
    n_eff = (w_sum * w_sum) / w_sq_sum
    if n_eff <= 1.0:
        return 0.0, n_eff
    mean = float(np.sum(w * x) / w_sum)
    centered = x - mean
    var = float(np.sum(w * centered * centered) / w_sum)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0, n_eff
    se = np.sqrt(var / n_eff)
    if not np.isfinite(se) or se <= 0.0:
        return 0.0, n_eff
    return mean / se, n_eff


def _rank_ic_tstat(pred: NDArray[np.float64], target: NDArray[np.float64]) -> tuple[float, float]:
    mask = np.isfinite(pred) & np.isfinite(target)
    n_eff = float(np.sum(mask))
    if n_eff <= 1.0:
        return 0.0, n_eff
    rank_ic = _rank_ic(pred[mask], target[mask])
    if not np.isfinite(rank_ic):
        return 0.0, n_eff
    return float(rank_ic * np.sqrt(n_eff)), n_eff


def _compute_kelly_fraction(
    mu: NDArray[np.float64],
    q10: NDArray[np.float64],
    q90: NDArray[np.float64],
    *,
    kelly_fraction: float,
    max_symbol_weight: float,
) -> NDArray[np.float64]:
    sigma = np.maximum((q90 - q10) / 2.563, 1e-6)
    denom = np.square(sigma) + np.square(mu) + 1e-9
    raw = kelly_fraction * np.maximum(mu, 0.0) / denom
    return np.clip(raw, 0.0, max_symbol_weight)


def _downside_upside_ratio_p_pass(
    mu_return_r: NDArray[np.float64],
    q10_return_r: NDArray[np.float64],
) -> NDArray[np.float64]:
    positive_mu = np.maximum(mu_return_r, 1e-9)
    ratio = q10_return_r / positive_mu
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    return np.clip(ratio, 0.0, 1.0)


def _build_variant_priors(
    *,
    dataset: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> tuple[dict[str, float], dict[str, int], float, NDArray[np.float64], bool]:
    """Return shrunk per-variant priors and per-row prior values."""
    y_edge = np.asarray(
        dataset.y_return_r if dataset.y_return_r is not None else dataset.y_edge_bps,
        dtype=np.float64,
    )
    weights = np.asarray(dataset.edge_weight, dtype=np.float64)
    keys = _variant_keys(dataset)
    eligible_global = np.isfinite(y_edge) & np.isfinite(weights) & (weights > 0.0)
    global_prior = _weighted_mean(y_edge, weights)
    variant_prior_bps: dict[str, float] = {}
    variant_prior_obs: dict[str, int] = {}
    row_priors = np.full(dataset.X.shape[0], global_prior, dtype=np.float64)
    shrinkage_obs = float(cfg.edge_prior_shrinkage_obs)
    min_obs = int(cfg.edge_prior_min_obs)

    key_to_indices: dict[str, list[int]] = {}
    if len(keys) > 0:
        keys_ser = pd.Series(keys)
        key_to_indices = {str(k): list(v) for k, v in keys_ser.groupby(keys_ser, sort=False).groups.items()}

    variant_means: list[float] = []
    variant_se: list[float] = []
    eligible_variants: list[tuple[str, NDArray[np.int32], float, int]] = []
    for key, indices in key_to_indices.items():
        indexer = np.asarray(indices, dtype=np.int32)
        variant_values = y_edge[indexer]
        variant_weights = weights[indexer]
        eligible_variant = np.isfinite(variant_values) & np.isfinite(variant_weights) & (variant_weights > 0.0)
        obs = int(eligible_variant.sum())
        variant_prior_obs[key] = obs
        if obs < min_obs:
            row_priors[indexer] = global_prior
            continue
        variant_mean = _weighted_mean(variant_values, variant_weights)
        eligible_values = variant_values[eligible_variant]
        finite_std = float(np.std(eligible_values, ddof=1)) if obs > 1 else 0.0
        variant_means.append(variant_mean)
        variant_se.append(finite_std / max(np.sqrt(obs), 1.0))
        eligible_variants.append((key, indexer, variant_mean, obs))
        shrink = obs / (obs + shrinkage_obs)
        prior = float(shrink * variant_mean + (1.0 - shrink) * global_prior)
        reference_risk_unit_bps = float(
            np.median(dataset.risk_unit_bps.astype(np.float64, copy=False)[dataset.risk_unit_bps > 0.0])
        ) if dataset.risk_unit_bps is not None and np.any(dataset.risk_unit_bps > 0.0) else float(
            getattr(cfg, "min_risk_unit_bps", 25.0)
        )
        max_dev_bps = float(getattr(cfg, "edge_prior_max_deviation_bps", float("inf")))
        max_dev = (
            max_dev_bps / max(reference_risk_unit_bps, 1e-6)
            if np.isfinite(max_dev_bps) and max_dev_bps > 0.0
            else float("inf")
        )
        if np.isfinite(max_dev) and max_dev > 0.0:
            prior = float(np.clip(prior, global_prior - max_dev, global_prior + max_dev))
        variant_prior_bps[key] = prior
        row_priors[indexer] = prior

    if bool(getattr(cfg, "use_empirical_bayes_shrinkage", True)) and len(eligible_variants) >= 2:
        means = np.asarray(variant_means, dtype=np.float64)
        se = np.asarray(variant_se, dtype=np.float64)
        sigma2_within = float(np.mean(np.square(se))) if se.size > 0 else 0.0
        sigma2_between = max(float(np.var(means, ddof=0)) - sigma2_within, 0.0)
        shrinkage_lambda = sigma2_within / max(sigma2_within + sigma2_between, 1e-12)
        for key, indexer, variant_mean, _obs in eligible_variants:
            prior = float(shrinkage_lambda * global_prior + (1.0 - shrinkage_lambda) * variant_mean)
            variant_prior_bps[key] = prior
            row_priors[indexer] = prior

    return variant_prior_bps, variant_prior_obs, global_prior, row_priors, bool(eligible_global.any())


def _log_edge_target_distribution(*, split: str, dataset: CandidateDataset, cfg: CandidateStrategyConfig) -> None:
    """Log the target distribution for a dataset split."""
    if dataset.X.shape[0] == 0:
        return

    y_edge = np.asarray(
        dataset.y_return_r if dataset.y_return_r is not None else dataset.y_edge_bps,
        dtype=np.float64,
    )
    y_q10 = np.asarray(
        dataset.y_return_r if dataset.y_return_r is not None else dataset.y_q10_bps,
        dtype=np.float64,
    )
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
    frame["_edge"] = np.asarray(
        dataset.y_return_r if dataset.y_return_r is not None else dataset.y_edge_bps,
        dtype=np.float64,
    )
    frame["_q10"] = np.asarray(
        dataset.y_return_r if dataset.y_return_r is not None else dataset.y_q10_bps,
        dtype=np.float64,
    )
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
    calibration_eval: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> CandidateEdgeModels:
    """Fit robust expected edge and quantile downside models."""
    seed = _seed_from_cfg(cfg)
    _log_edge_target_distribution(split="train", dataset=train, cfg=cfg)
    _log_edge_target_distribution(split="valid", dataset=valid, cfg=cfg)
    _log_edge_target_variants(split="train", dataset=train, cfg=cfg)
    _log_edge_target_variants(split="valid", dataset=valid, cfg=cfg)
    (
        variant_prior_bps,
        variant_prior_obs,
        global_prior_bps,
        train_row_priors,
        has_eligible_prior_rows,
    ) = _build_variant_priors(dataset=train, cfg=cfg)
    use_prior_residual = bool(cfg.edge_prior_enabled and cfg.edge_residual_model_enabled)
    train_return_r = np.asarray(
        train.y_return_r if train.y_return_r is not None else train.y_edge_bps,
        dtype=np.float64,
    )
    valid_return_r = np.asarray(
        valid.y_return_r if valid.y_return_r is not None else valid.y_edge_bps,
        dtype=np.float64,
    )
    center_train_target = train_return_r - train_row_priors if use_prior_residual else train_return_r
    center_valid_target = valid_return_r
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
    eval_set_q10: Any = [(valid.X, valid_return_r)] if valid.X.shape[0] > 0 else None
    eval_set_q90: Any = [(valid.X, valid_return_r)] if valid.X.shape[0] > 0 else None

    from lightgbm import early_stopping as lgbm_early_stopping
    from lightgbm import log_evaluation as lgbm_log_evaluation

    _es_cb: list[Any] = [lgbm_early_stopping(stopping_rounds=30, verbose=False), lgbm_log_evaluation(period=-1)]

    if eval_set_center is not None and len(np.unique(valid_return_r)) > 1:
        center.fit(
            train.X,
            center_train_target,
            sample_weight=train.edge_weight,
            eval_set=eval_set_center,
            callbacks=_es_cb,
        )
    else:
        center.fit(train.X, center_train_target, sample_weight=train.edge_weight)

    if eval_set_q10 is not None and len(np.unique(valid_return_r)) > 1:
        q10.fit(
            train.X,
            train_return_r,
            sample_weight=train.edge_weight,
            eval_set=eval_set_q10,
            callbacks=_es_cb,
        )
    else:
        q10.fit(train.X, train_return_r, sample_weight=train.edge_weight)

    # Fit Q90 on y_mfe_bps (upside realisation — maximum favourable excursion)
    if eval_set_q90 is not None and len(np.unique(valid_return_r)) > 1:
        q90.fit(
            train.X,
            train_return_r,
            sample_weight=train.edge_weight,
            eval_set=eval_set_q90,
            callbacks=_es_cb,
        )
    else:
        q90.fit(train.X, train_return_r, sample_weight=train.edge_weight)

    median_risk_unit_bps = float(
        np.median(train.risk_unit_bps.astype(np.float64, copy=False))
        if train.risk_unit_bps is not None and train.risk_unit_bps.shape[0] > 0
        else getattr(cfg, "min_risk_unit_bps", 25.0)
    )

    # --- Layer 2: Rank IC gate on calibration_eval set ---
    _logger = logging.getLogger(__name__)
    _breakeven_floor = float(cfg.min_net_floor_cost_fraction) * float(cfg.cost_floor_bps)
    _logger.debug(
        "[DIAG][PRIOR] global_prior_bps=%.2f breakeven_floor=%.2f n_variants=%d",
        global_prior_bps,
        _breakeven_floor,
        len(variant_prior_bps),
    )
    for _k, _p in sorted(variant_prior_bps.items(), key=lambda x: -x[1])[:8]:
        _logger.debug(
            "[DIAG][VARIANT_PRIOR] variant=%-45s prior=%.2f bps  obs=%d",
            _k,
            _p,
            variant_prior_obs.get(_k, 0),
        )
    accepted = False
    rank_ic_val = float("nan")
    overlay_lift_bps = 0.0
    overlay_lift_tstat = 0.0
    n_eff = 0.0
    reason = "insufficient_obs"
    n_cal_eval = int(calibration_eval.X.shape[0]) if calibration_eval.X is not None else 0
    gate_mode = str(getattr(cfg, "edge_gate_mode", "rank_ic"))
    if gate_mode == "overlay_lift":
        cal_events = getattr(calibration_eval, "event_index", pd.DataFrame())
        if cal_events.empty or "overlay_mult" not in cal_events.columns:
            reason = "missing_overlay_context"
        else:
            realized_bps = np.asarray(
                calibration_eval.y_return_bps
                if calibration_eval.y_return_bps is not None
                else calibration_eval.y_edge_bps,
                dtype=np.float64,
            )
            overlay_mult = pd.to_numeric(cal_events["overlay_mult"], errors="coerce").to_numpy(
                dtype=np.float64,
                copy=False,
            )
            cal_weights = np.asarray(calibration_eval.edge_weight, dtype=np.float64)
            lift_diff_bps = realized_bps * (overlay_mult - 1.0)
            overlay_lift_bps = _weighted_mean(lift_diff_bps, cal_weights)
            overlay_lift_tstat, n_eff = _weighted_tstat(lift_diff_bps, cal_weights)
            min_tstat = float(getattr(cfg, "edge_gate_min_lift_tstat", 1.0))
            min_n_eff = float(getattr(cfg, "edge_gate_min_n_eff", 60))
            if n_eff < min_n_eff:
                reason = "insufficient_n_eff"
            elif overlay_lift_bps <= 0.0:
                reason = "overlay_lift_non_positive"
            elif overlay_lift_tstat < min_tstat:
                reason = "overlay_lift_below_threshold"
            else:
                accepted = True
                reason = "overlay_lift_pass"
    elif n_cal_eval >= 20:
        cal_keys = _variant_keys(calibration_eval)
        if use_prior_residual:
            cal_prior_r = np.asarray(
                [variant_prior_bps.get(key, global_prior_bps) / max(median_risk_unit_bps, 1e-6)
                 for key in cal_keys],
                dtype=np.float64,
            )
        else:
            cal_prior_r = np.zeros(n_cal_eval, dtype=np.float64)
        pred_cal = cast(
            NDArray[np.float64],
            center.predict(calibration_eval.X),
        ).astype(np.float64, copy=False) + cal_prior_r
        realized_cal = np.asarray(
            calibration_eval.y_return_r
            if calibration_eval.y_return_r is not None
            else calibration_eval.y_edge_bps,
            dtype=np.float64,
        )
        rank_ic_val = _rank_ic(pred_cal, realized_cal)
        rank_ic_tstat, rank_ic_n_eff = _rank_ic_tstat(pred_cal, realized_cal)
        min_ic_tstat = float(getattr(cfg, "min_ic_tstat", 1.5))
        n_eff = rank_ic_n_eff
        if np.isfinite(rank_ic_val) and rank_ic_tstat >= min_ic_tstat:
            accepted = True
            reason = "rank_ic_pass"
        elif np.isfinite(rank_ic_val):
            reason = "rank_ic_fail"
    else:
        reason = "insufficient_obs"

    if accepted:
        final_prediction_mode: EdgePredictionMode = "prior_residual" if use_prior_residual else "direct"
    elif cfg.edge_prior_enabled and has_eligible_prior_rows:
        final_prediction_mode = "prior_only"
    else:
        final_prediction_mode = "disabled"

    if gate_mode == "overlay_lift":
        _logger.debug(
            "[EDGE_GATE] mode=overlay_lift lift=%.1f t=%.2f n_eff=%.1f decision=%s reason=%s inference_mode=%s",
            overlay_lift_bps,
            overlay_lift_tstat,
            n_eff,
            "accepted" if accepted else "rejected",
            reason,
            final_prediction_mode,
        )
    else:
        _logger.debug(
            (
                "[EDGE_GATE] mode=rank_ic rank_ic=%.4f t=%.2f threshold_t=%.2f "
                "decision=%s n=%d reason=%s inference_mode=%s"
            ),
            rank_ic_val,
            rank_ic_tstat if "rank_ic_tstat" in locals() else 0.0,
            float(getattr(cfg, "min_ic_tstat", 1.5)),
            "accepted" if accepted else "rejected",
            n_cal_eval,
            reason,
            final_prediction_mode,
        )

    edge_validation = EdgeModelValidation(
        rank_ic_cal_eval=rank_ic_val,
        n_cal_eval=n_cal_eval,
        accepted=accepted,
        reason=reason,
        overlay_lift_bps=overlay_lift_bps,
        overlay_lift_tstat=overlay_lift_tstat,
        n_eff=n_eff,
    )

    return CandidateEdgeModels(
        center_model=center,
        q10_model=q10,
        q90_model=q90,
        feature_names=tuple(train.feature_names),
        variant_prior_r=variant_prior_bps,
        global_prior_r=global_prior_bps,
        variant_prior_bps={key: value * median_risk_unit_bps for key, value in variant_prior_bps.items()},
        variant_prior_obs=variant_prior_obs,
        global_prior_bps=global_prior_bps * median_risk_unit_bps,
        prediction_mode=final_prediction_mode,
        prediction_diagnostics={
            "prediction_mode": final_prediction_mode,
            "global_prior_bps": global_prior_bps * median_risk_unit_bps,
            "global_prior_r": global_prior_bps,
            "variant_prior_count": len(variant_prior_bps),
            "rank_ic_cal_eval": rank_ic_val,
            "rank_ic_tstat": rank_ic_tstat if "rank_ic_tstat" in locals() else 0.0,
            "overlay_lift_bps": overlay_lift_bps,
            "overlay_lift_tstat": overlay_lift_tstat,
            "edge_gate_n_eff": n_eff,
            "edge_gate_accepted": accepted,
            "has_eligible_prior_rows": int(has_eligible_prior_rows),
        },
        validation=edge_validation,
    )


def predict_candidate_edges(
    *,
    models: CandidateEdgeModels | None,
    dataset: CandidateDataset,
    p_pass: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
    gate_enabled: bool = False,
    gate_threshold: float = 0.5,
    edge_source: EdgeSource = EdgeSource.PRIOR_ONLY,
) -> CandidateModelOutput:
    """Return expected edge, downside quantiles, and utility scores."""
    if models is None or dataset.X.shape[0] == 0:
        zeros = np.zeros(dataset.X.shape[0], dtype=np.float64)
        return CandidateModelOutput(
            events=dataset.event_index,
            p_pass=p_pass.astype(np.float64, copy=False),
            gate_enabled=gate_enabled,
            gate_threshold=gate_threshold,
            edge_source=edge_source,
            expected_return_r=zeros,
            expected_net_bps=zeros,
            q10_return_r=zeros,
            q10_net_bps=zeros,
            q90_return_r=zeros,
            q90_net_bps=zeros,
            selection_score=zeros,
            kelly_fraction=zeros,
            validation_diagnostics={
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

    center_pred_r = cast(NDArray[np.float64], models.center_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )
    q10_model_r = cast(NDArray[np.float64], models.q10_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )
    q90_model_r = cast(NDArray[np.float64], models.q90_model.predict(dataset.X)).astype(
        np.float64, copy=False
    )

    expected_cost_bps = float(getattr(cfg, "expected_cost_bps", 24.0))
    downside_penalty = float(getattr(cfg, "downside_penalty", 1.0))
    turnover_penalty = float(getattr(cfg, "turnover_penalty", 0.5))
    concentration_penalty = float(getattr(cfg, "concentration_penalty", 0.0))

    # Extract turnover_proxy from features or fallback to a constant 1.0 (so turnover_term = turnover_penalty)
    turnover_proxy = np.ones_like(center_pred_r)
    if "turnover_proxy" in dataset.feature_names:
        t_idx = dataset.feature_names.index("turnover_proxy")
        turnover_proxy = dataset.X[:, t_idx].astype(np.float64, copy=False)

    risk_unit_bps = (
        dataset.risk_unit_bps.astype(np.float64, copy=False)
        if dataset.risk_unit_bps is not None
        else np.full(dataset.X.shape[0], float(getattr(cfg, "min_risk_unit_bps", 25.0)), dtype=np.float64)
    )
    prior_r = np.zeros(dataset.X.shape[0], dtype=np.float64)
    if models.prediction_mode in {"prior_only", "prior_residual"}:
        keys = _variant_keys(dataset)
        prior_r = np.asarray(
            [models.variant_prior_r.get(key, models.global_prior_r) for key in keys],
            dtype=np.float64,
        )
    center_component_r = (
        center_pred_r if models.prediction_mode in {"direct", "prior_residual"} else np.zeros_like(center_pred_r)
    )
    prior_component_r = (
        prior_r if models.prediction_mode in {"prior_only", "prior_residual"} else np.zeros_like(prior_r)
    )
    mu_return_r = center_component_r + prior_component_r
    q10_return_r = q10_model_r
    q90_return_r = q90_model_r
    effective_p_pass = (
        p_pass if gate_enabled
        else _downside_upside_ratio_p_pass(mu_return_r, q10_return_r)
    )
    mu_net_decision_bps = mu_return_r * risk_unit_bps
    q10_net_bps = q10_return_r * risk_unit_bps
    q90_net_bps = q90_return_r * risk_unit_bps

    downside_term = downside_penalty * np.abs(np.minimum(q10_net_bps, 0.0))
    turnover_term = turnover_penalty * turnover_proxy
    expected_utility_bps = mu_net_decision_bps - downside_term - turnover_term - concentration_penalty
    if getattr(cfg, "selection_utility_mode", "additive_drag") == "expected_edge_direct":
        # Direct mode: mu_net is the unconditional E[return]; p_pass/q10 are sizing-only.
        utility_score = mu_net_decision_bps
    elif bool(getattr(cfg, "selection_use_expected_utility", True)):
        utility_score = expected_utility_bps
    else:
        utility_score = expected_utility_bps

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
    _logger.debug(
        (
            "[DIAG][EDGE] n=%d target_scale=net mode=%s cost_bps=%.1f floor_bps=%.1f "
            "mu_mean=%.1f mu_p50=%.1f mu_p90=%.1f mu_max=%.1f "
            "q10_mean=%.1f q10_p10=%.1f q10_p50=%.1f q10_min=%.1f "
            "exp_util_mean=%.3f exp_util_p50=%.3f exp_util_p90=%.3f "
            "utility_mean=%.3f utility_p50=%.3f utility_p90=%.3f utility_max=%.3f "
            "pct_mu_ge1=%.3f pct_mu_ge_floor=%.3f pct_q10_ge_cat=%.3f pct_q10_ge_max=%.3f"
        ),
        len(center_pred_r),
        models.prediction_mode,
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
    target_for_diag = np.asarray(
        dataset.y_return_bps if dataset.y_return_bps is not None else dataset.y_edge_bps,
        dtype=np.float64,
    )
    if dataset.X.shape[0] > 0 and target_for_diag.shape[0] == dataset.X.shape[0]:
        diag_frame = pd.DataFrame(
            {
                "mu": mu_net_decision_bps,
                "p_pass": effective_p_pass,
                "expected_utility": expected_utility_bps,
                "y_edge": target_for_diag,
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
            _rank_ic(mu_net_decision_bps, target_for_diag),
            _rank_ic(expected_utility_bps, target_for_diag),
        )

    kelly_fraction_arr = _compute_kelly_fraction(
        mu_return_r,
        q10_return_r,
        q90_return_r,
        kelly_fraction=float(cfg.kelly_fraction),
        max_symbol_weight=float(cfg.max_symbol_weight),
    )

    return CandidateModelOutput(
        events=dataset.event_index,
        p_pass=effective_p_pass.astype(np.float64, copy=False),
        gate_enabled=gate_enabled,
        gate_threshold=gate_threshold,
        edge_source=edge_source,
        expected_return_r=mu_return_r,
        expected_net_bps=mu_net_decision_bps,
        q10_return_r=q10_return_r,
        q10_net_bps=q10_net_bps,
        q90_return_r=q90_return_r,
        q90_net_bps=q90_net_bps,
        selection_score=utility_score,
        kelly_fraction=kelly_fraction_arr,
        validation_diagnostics=cast(
            dict[str, float | int | str | bool],
            {
                **_selection_thresholds(
                utility_score=utility_score,
                p_pass=effective_p_pass.astype(np.float64, copy=False),
                mu_net_decision_bps=mu_net_decision_bps,
                cfg=cfg,
                ),
                "prediction_mode": models.prediction_mode,
                "center_component_p90_bps": float(np.percentile(center_component_r * risk_unit_bps, 90))
                if center_component_r.size > 0
                else 0.0,
                "prior_component_p90_bps": float(np.percentile(prior_component_r * risk_unit_bps, 90))
                if prior_component_r.size > 0
                else 0.0,
            },
        ),
    )
