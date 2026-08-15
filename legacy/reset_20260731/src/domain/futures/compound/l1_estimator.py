from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L1EstimatorConfig
from src.domain.futures.compound.contracts import (
    AlphaForecastTape,
    AlphaLifecycle,
    AlphaLifecycleEvidence,
    CausalAlphaFold,
    MarketFeatureCube,
    RawAlphaTape,
)

_logger = logging.getLogger(__name__)


def build_causal_alpha_folds(
    *,
    n_bars: int,
    fit_start: int,
    holdout_start: int,
    n_folds: int,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[CausalAlphaFold, ...]:
    assert fit_start < holdout_start <= n_bars
    available_fit = holdout_start - fit_start
    fold_size = available_fit // n_folds
    assert fold_size > purge_bars + embargo_bars + 1

    folds: list[CausalAlphaFold] = []
    for i in range(n_folds):
        fit_end = fit_start + (i + 1) * fold_size
        oos_start = fit_end + purge_bars
        oos_end = min(oos_start + fold_size, holdout_start)
        if i == n_folds - 1:
            oos_end = holdout_start
        folds.append(
            CausalAlphaFold(
                fold_id=i,
                fit_start=fit_start,
                fit_end_exclusive=fit_end,
                oos_start=oos_start,
                oos_end_exclusive=oos_end,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )
        )
    return tuple(folds)


def _ridge_slope(
    x: NDArray[np.float32], y: NDArray[np.float32], alpha: float = 1.0
) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    n_eff = float(np.sum(mask))
    if n_eff < 3:
        return 0.0, 0.0, 0.0
    x_masked = x[mask].astype(np.float64)
    y_masked = y[mask].astype(np.float64)
    sx = np.sum(x_masked)
    sy = np.sum(y_masked)
    sxx = np.sum(x_masked * x_masked) + alpha
    sxy = np.sum(x_masked * y_masked)
    denom = n_eff * sxx - sx * sx
    if abs(denom) < 1e-15:
        return 0.0, n_eff, float(np.var(y_masked))
    beta = (n_eff * sxy - sx * sy) / denom
    residuals = y_masked - beta * x_masked
    residual_var = float(np.var(residuals)) if len(residuals) > 1 else 1e-8
    return float(beta), n_eff, max(residual_var, 1e-8)


def _compute_labels(
    cube: MarketFeatureCube, idx_slice: slice, horizon_bars: int,
) -> NDArray[np.float32]:
    open_p = cube.fields_2d["open"]
    start = idx_slice.start or 0
    stop = idx_slice.stop or open_p.shape[0]
    n = stop - start
    labels = np.full((n, open_p.shape[1]), np.nan, dtype=np.float32)
    for t in range(n):
        entry_t = start + t + 1
        exit_t = entry_t + horizon_bars
        if exit_t >= open_p.shape[0]:
            break
        entry_vals = open_p[entry_t]
        exit_vals = open_p[exit_t]
        mask = (entry_vals > 0) & (exit_vals > 0)
        if mask.any():
            labels[t, mask] = np.where(mask, np.log(exit_vals / entry_vals), np.nan).astype(np.float32)[mask]
    return labels


def _compute_lifecycle(
    estimated_3d: NDArray[np.bool_],
    config: L1EstimatorConfig,
) -> tuple[AlphaLifecycle, ...]:
    n_recipes = estimated_3d.shape[2]
    lifecycle: list[AlphaLifecycle] = []
    for k in range(n_recipes):
        n_eff = float(np.sum(estimated_3d[:, :, k]))
        if n_eff < config.active_effective_n:
            lifecycle.append(AlphaLifecycle.SHADOW)
        else:
            lifecycle.append(AlphaLifecycle.ACTIVE)
    return tuple(lifecycle)


def build_causal_alpha_forecasts(
    *,
    raw: RawAlphaTape,
    cube: MarketFeatureCube,
    folds: tuple[CausalAlphaFold, ...],
    holdout_start_idx: int,
    config: L1EstimatorConfig,
) -> AlphaForecastTape:
    n_bars = raw.timestamps_ns.size
    n_syms = len(raw.symbols)
    n_recipes = len(raw.recipe_ids)

    gross_mu = np.zeros((n_bars, n_syms, n_recipes), dtype=np.float32)
    mean_edge_var = np.full((n_bars, n_syms, n_recipes), 1e-4, dtype=np.float32)
    residual_var = np.full((n_bars, n_syms, n_recipes), 1e-4, dtype=np.float32)
    reliability = np.zeros((n_bars, n_syms, n_recipes), dtype=np.float32)
    estimated = np.zeros((n_bars, n_syms, n_recipes), dtype=np.bool_)
    valid = raw.valid_3d.copy()

    for k in range(n_recipes):
        horizon = int(raw.horizon_bars_1d[k])
        label_matrix = _compute_labels(cube, slice(0, n_bars), horizon)

        recipe_slopes: list[float] = []
        for fold in folds:
            fit_slice = slice(fold.fit_start, fold.fit_end_exclusive - fold.purge_bars)
            fit_len = (fold.fit_end_exclusive - fold.purge_bars) - fold.fit_start
            for n in range(n_syms):
                x_fit = raw.scores_3d[fit_slice, n, k]
                y_fit = label_matrix[fit_slice, n] if fit_len > 0 else np.array([], dtype=np.float32)
                beta, n_eff, _ = _ridge_slope(x_fit, y_fit)
                if n_eff > 0:
                    recipe_slopes.append(beta)

        beta_recipe = float(np.median(recipe_slopes)) if recipe_slopes else 0.0

        for fold in folds:
            if fold.oos_start >= fold.oos_end_exclusive:
                continue
            fit_slice = slice(fold.fit_start, fold.fit_end_exclusive - fold.purge_bars)
            oos_slice = slice(fold.oos_start, fold.oos_end_exclusive)
            for n in range(n_syms):
                x_fit = raw.scores_3d[fit_slice, n, k]
                y_fit = label_matrix[fit_slice, n]
                beta_n, n_eff, res_var = _ridge_slope(x_fit, y_fit)

                if n_eff < 1:
                    continue

                w = n_eff / (n_eff + config.prior_effective_n)
                beta_post = w * beta_n + (1 - w) * beta_recipe

                x_oos = raw.scores_3d[oos_slice, n, k]
                oos_valid_mask = raw.valid_3d[oos_slice, n, k] & np.isfinite(x_oos)

                if not np.any(oos_valid_mask):
                    continue

                mu = beta_post * x_oos

                if n_eff > 3:
                    var_x_fit = float(np.var(x_fit[np.isfinite(x_fit)]))
                    se_beta = np.sqrt(res_var / max(n_eff * var_x_fit, 1e-15))
                else:
                    se_beta = 1.0
                param_var_oos = float(se_beta * se_beta) * x_oos * x_oos

                h = float(horizon)
                mu_bar = mu / h
                mean_var_bar = param_var_oos / (h * h)

                gross_mu[oos_slice, n, k] = np.where(oos_valid_mask, mu_bar, 0.0)
                mean_edge_var[oos_slice, n, k] = np.where(oos_valid_mask, mean_var_bar, 1e-4)
                residual_var[oos_slice, n, k] = np.where(oos_valid_mask, res_var, 1e-4)
                reliability[oos_slice, n, k] = np.where(
                    oos_valid_mask,
                    np.float32(min(1.0, n_eff / max(config.active_effective_n, 1))),
                    0.0,
                )
                estimated[oos_slice, n, k] = oos_valid_mask

    for k in range(n_recipes):
        holdout_valid = valid[holdout_start_idx:, :, k].copy()
        valid[holdout_start_idx:, :, k] = holdout_valid & estimated[holdout_start_idx:, :, k]

    lifecycle = _compute_lifecycle(estimated, config)

    model_version = "compound-v1"
    data_hash = cube.data_manifest_hash
    fold_str = "-".join(str(f.fold_id) for f in folds)
    fold_hash = f"folds_{len(folds)}_{fold_str}"

    return AlphaForecastTape(
        timestamps_ns=raw.timestamps_ns,
        symbols=raw.symbols,
        recipe_ids=raw.recipe_ids,
        gross_mu_3d=gross_mu,
        mean_edge_var_3d=mean_edge_var,
        residual_var_3d=residual_var,
        reliability_3d=reliability,
        estimated_3d=estimated,
        valid_3d=valid,
        horizon_bars_1d=raw.horizon_bars_1d,
        lifecycle_by_recipe=lifecycle,
        model_version=model_version,
        data_manifest_hash=data_hash,
        fold_manifest_hash=fold_hash,
    )


def estimate_cross_fitted_alpha_tape(
    *,
    raw: RawAlphaTape,
    cube: MarketFeatureCube,
    folds: Sequence[CausalAlphaFold],
    config: L1EstimatorConfig,
) -> AlphaForecastTape:
    folds_tuple = tuple(folds)
    holdout_start_idx = max(f.oos_end_exclusive for f in folds_tuple)
    return build_causal_alpha_forecasts(
        raw=raw,
        cube=cube,
        folds=folds_tuple,
        holdout_start_idx=holdout_start_idx,
        config=config,
    )


def update_alpha_lifecycle(
    *,
    current: Sequence[AlphaLifecycle],
    evidence: Sequence[AlphaLifecycleEvidence],
    config: L1EstimatorConfig,
) -> tuple[AlphaLifecycle, ...]:
    updated: list[AlphaLifecycle] = []
    for i, state in enumerate(current):
        ev = evidence[i] if i < len(evidence) else AlphaLifecycleEvidence(
            recipe_id=f"recipe_{i}",
            effective_n=0.0,
            probability_net_positive=0.5,
            consecutive_negative_versions=0,
            data_valid=True,
        )
        if not ev.data_valid or (
            ev.effective_n >= config.retire_effective_n
            and ev.probability_net_positive <= config.retire_probability_max
            and ev.consecutive_negative_versions >= config.retire_consecutive_versions
        ):
            updated.append(AlphaLifecycle.RETIRED)
        elif ev.effective_n >= config.active_effective_n and ev.data_valid:
            updated.append(AlphaLifecycle.ACTIVE)
        elif state == AlphaLifecycle.ACTIVE and ev.effective_n < config.active_effective_n:
            updated.append(AlphaLifecycle.SHADOW)
        else:
            updated.append(state)
    return tuple(updated)
