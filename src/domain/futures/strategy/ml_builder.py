from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.core.settings import round_trip_cost_bps
from src.domain.futures.optimization.opt_config import (
    OPT_FUTURES_CONFIG,
    default_ev_hurdle_bps,
)
from src.domain.futures.strategy.alpha_evaluation import (
    derive_signed_rank_signal,
    diagnose_alpha_ic_decomposition,
    effective_breadth_corr,
)
from src.domain.futures.strategy.cache import build_manifest_hash
from src.domain.futures.strategy.common.alignment import align_data_maps
from src.domain.futures.strategy.common.normalization import (
    apply_missing_value_imputer,
    apply_robust_bounds,
    fit_missing_value_imputer,
    fit_robust_bounds,
)
from src.domain.futures.strategy.common.validation import (
    validate_feature_panel,
    validate_label_panel,
    validate_long_matrix,
)
from src.domain.futures.strategy.config import StrategyConfig, StrategyMLConfig
from src.domain.futures.strategy.contracts import (
    FeaturePanel,
    FoldSpec,
    LabelPanel,
    LongMatrixDataset,
)
from src.domain.futures.strategy.dataset import (
    build_long_matrix,
    make_walk_forward_folds,
)
from src.domain.futures.strategy.diagnostics import (
    alpha_gate_diagnostics,
    build_quality_report,
    feature_cs_ic_audit,
    ic_summary,
    ml_alpha_metrics,
    passes_ic_gate,
    passes_quality_gate,
    preservation_ratio,
    rolling_ic,
    side_alpha_tail_metrics,
)
from src.domain.futures.strategy.features import build_feature_panel
from src.domain.futures.strategy.integrity import (
    select_features,
    verify_data_integrity,
    verify_feature_integrity,
)
from src.domain.futures.strategy.labels import build_label_panel
from src.domain.futures.strategy.rank_selection import (
    RankSelectionPolicy,
    apply_rank_selection_policy,
    calibrate_rank_selection_policy,
    policy_to_dict,
)
from src.domain.futures.strategy.ranker import RankerFitResult, fit_ranker, predict_rank_score
from src.domain.futures.strategy.regime_gate import apply_regime_gate

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EVQuantiles:
    """Compatibility quantile container for downstream metadata contracts."""

    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray


def _build_side_matrix(**kwargs: Any) -> Any:
    """Build side-specific matrix with compatibility fallback for older signatures."""
    try:
        return build_long_matrix(**kwargs)
    except TypeError:
        kwargs.pop("relevance_override", None)
        kwargs.pop("ev_target_override", None)
        return build_long_matrix(**kwargs)


def _resolve_side_targets(
    labels: LabelPanel,
    ml_cfg: StrategyMLConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve side-specific relevance and magnitude targets."""
    long_rel = labels.relevance_long if labels.relevance_long is not None else labels.relevance
    short_rel = labels.relevance_short if labels.relevance_short is not None else labels.relevance
    long_mag = (
        labels.magnitude_target_long
        if labels.magnitude_target_long is not None
        else (
            labels.magnitude_target if labels.magnitude_target is not None else labels.exec_net_ret
        )
    )
    short_mag = (
        labels.magnitude_target_short
        if labels.magnitude_target_short is not None
        else np.maximum(
            -(
                labels.magnitude_target
                if labels.magnitude_target is not None
                else labels.exec_net_ret
            ),
            0.0,
        )
    )
    long_rank_target: np.ndarray | None = None
    short_rank_target: np.ndarray | None = None
    if (
        ml_cfg.rank_target_mode == "forward_gross_rank"
        and labels.forward_gross_rank_target is not None
        and labels.forward_gross_relevance is not None
    ):
        long_rank_target = labels.forward_gross_rank_target
        short_rank_target = -labels.forward_gross_rank_target
        long_rel = labels.forward_gross_relevance
        short_rel = 4 - labels.forward_gross_relevance
    elif ml_cfg.rank_target_mode == "cs_residual" and labels.rank_target is not None:
        long_rank_target = labels.rank_target
        short_rank_target = -labels.rank_target
        long_rel = labels.relevance
        short_rel = 4 - labels.relevance
    return long_rank_target, short_rank_target, long_rel, short_rel, long_mag, short_mag


def _rank_score(
    fit_result: RankerFitResult | None, dataset: LongMatrixDataset
) -> np.ndarray:
    """Return rank score or NaNs when ranker is disabled.

    Args:
        fit_result: Trained ranker result, or None when ranker_enabled=False.
        dataset: Dataset to score.

    Returns:
        Rank score array [N] as float32; NaNs when fit_result is None.

    Time complexity: O(N * T_trees) when fit_result is not None, O(N) otherwise.
    Space complexity: O(N).

    """
    if fit_result is None:
        return np.full(dataset.X.shape[0], np.nan, dtype=np.float32)
    models_to_predict: (
        lgb.LGBMRegressor
        | lgb.LGBMRanker
        | list[lgb.LGBMRegressor | lgb.LGBMRanker]
    ) = fit_result.model
    if getattr(fit_result, "models", None) is not None:
        models_list = fit_result.models
        if models_list is not None:
            models_to_predict = models_list
    return predict_rank_score(models_to_predict, dataset)


def _btc_close_from_data_maps(
    data_maps: dict[str, dict[str, Any]],
    tf: str,
    prefer_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
) -> pd.Series:
    """Extract BTC-proxy close series from data_maps for regime classification.

    Args:
        data_maps: symbol → timeframe → DataFrame with 'datetime' and 'close' columns.
        tf: Timeframe key to look up (e.g. "4h").
        prefer_symbols: Symbols tried in order; first hit wins.

    Returns:
        pd.Series indexed by datetime, or empty Series if no match found.

    Time Complexity: O(|prefer_symbols|).

    Space Complexity: O(T) for the selected close series.

    """
    for sym in prefer_symbols:
        df: pd.DataFrame | None = data_maps.get(sym, {}).get(tf)
        if (
            df is not None
            and not df.empty
            and "close" in df.columns
            and "datetime" in df.columns
        ):
            return df.set_index("datetime")["close"]
    return pd.Series(dtype=np.float64)


def _predict_quantiles_with_fallback(
    models: Any,
    dataset: LongMatrixDataset,
    rank_score: np.ndarray,
    ev_pred: np.ndarray,
) -> EVQuantiles:
    """Build degenerate quantiles from signed score-only prediction."""
    _ = (models, dataset, rank_score)
    fallback = np.asarray(ev_pred, dtype=np.float32).reshape(-1).copy()
    return EVQuantiles(q10=fallback, q50=fallback, q90=fallback)


@dataclass(frozen=True)
class _FoldPredictResult:
    """공통 fold 학습·예측 결과 컨테이너 (WF·AWF·virtual 공유)."""

    ev_test_long: np.ndarray
    ev_test_short: np.ndarray
    quant_test_long: EVQuantiles
    quant_test_short: EVQuantiles
    conf_test_long: np.ndarray
    conf_test_short: np.ndarray
    score_test_long: np.ndarray
    score_test_short: np.ndarray
    ev_valid_long: np.ndarray
    ev_valid_short: np.ndarray
    score_valid_long: np.ndarray
    score_valid_short: np.ndarray
def _emit_rank_sized_alpha(
    rank_score_long_2d: np.ndarray,
    rank_score_short_2d: np.ndarray,
    eligible_2d: np.ndarray,
    *,
    select_q: float,
    weight_k: float,
    clip_lim: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense rank score to cross-sectional continuous long/short weights.

    Replaces EV-magnitude hard clip. Broad quantile selection + tanh rank-weight:
    (1) emit breadth increases → N_eff increases → be_eff decreases,
    (2) dense ranking skill preserved (presv improves).
    Time: O(T·N log N). Space: O(T·N).
    """
    from src.domain.futures.forecast.compose import _rank_weight_1d

    t_n = rank_score_long_2d.shape[0]
    al = np.zeros_like(rank_score_long_2d, dtype=np.float64)
    as_ = np.zeros_like(rank_score_short_2d, dtype=np.float64)
    for t in range(t_n):
        elig = eligible_2d[t]
        n_elig = int(np.count_nonzero(elig))
        if n_elig < 2:
            continue
        wl = _rank_weight_1d(np.where(elig, rank_score_long_2d[t], np.nan), k=weight_k)
        ws = _rank_weight_1d(np.where(elig, -rank_score_short_2d[t], np.nan), k=weight_k)
        keep = max(1, int(np.ceil(n_elig * select_q)))
        lo_idx = np.argsort(wl)[::-1][:keep]
        so_idx = np.argsort(ws)[::-1][:keep]
        al[t, lo_idx] = np.clip(np.maximum(wl[lo_idx], 0.0), 0.0, clip_lim)
        as_[t, so_idx] = np.clip(np.maximum(ws[so_idx], 0.0), 0.0, clip_lim)
    return al.astype(np.float32, copy=False), as_.astype(np.float32, copy=False)


def _train_predict_single_fold(
    fold: FoldSpec,
    features: FeaturePanel,
    labels: LabelPanel,
    ml_cfg: StrategyMLConfig,
    long_rank_target: np.ndarray | None,
    short_rank_target: np.ndarray | None,
    long_rel: np.ndarray,
    short_rel: np.ndarray,
    long_mag: np.ndarray,
    short_mag: np.ndarray,
) -> dict[str, Any]:
    """Train and predict a single walk-forward or virtual fold in parallel-safe manner."""
    t_start = time.perf_counter()
    fold_id = fold.fold_id

    # 1. Feature normalization and imputation on fold slices
    t_slice = time.perf_counter()
    train_values = features.values[fold.train_start : fold.train_end].astype(
        np.float64, copy=False
    )
    bounds = fit_robust_bounds(train_values, clip_quantile=0.995)
    clipped_values = apply_robust_bounds(features.values.astype(np.float64, copy=False), bounds)
    imputer = fit_missing_value_imputer(train_values)
    normalized = apply_missing_value_imputer(clipped_values, imputer).astype(
        np.float32, copy=False
    )
    normalized_features = FeaturePanel(
        datetimes=features.datetimes,
        symbols=features.symbols,
        values=normalized,
        feature_names=features.feature_names,
        valid_mask=features.valid_mask,
        availability_masks=features.availability_masks,
        metadata={
            **features.metadata,
            "train_imputer_applied": True,
            "missing_imputer": "train_median",
        },
    )
    _logger.debug(
        "[perf-ml-fold] [Fold %d] data slicing and robust scaling took %.4fs",
        fold_id,
        time.perf_counter() - t_slice,
    )

    # 2. Build Side Matrices
    t_matrices = time.perf_counter()
    train_long = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.train_start,
        end=fold.train_end,
        fold=fold,
        split="train",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=long_rank_target,
        relevance_override=long_rel,
        ev_target_override=long_mag,
    )
    valid_long = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.valid_start,
        end=fold.valid_end,
        fold=fold,
        split="valid",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=long_rank_target,
        relevance_override=long_rel,
        ev_target_override=long_mag,
    )
    test_long = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.test_start,
        end=fold.test_end,
        fold=fold,
        split="test",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=long_rank_target,
        relevance_override=long_rel,
        ev_target_override=long_mag,
    )
    train_short = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.train_start,
        end=fold.train_end,
        fold=fold,
        split="train",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=short_rank_target,
        relevance_override=short_rel,
        ev_target_override=short_mag,
    )
    valid_short = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.valid_start,
        end=fold.valid_end,
        fold=fold,
        split="valid",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=short_rank_target,
        relevance_override=short_rel,
        ev_target_override=short_mag,
    )
    test_short = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.test_start,
        end=fold.test_end,
        fold=fold,
        split="test",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=short_rank_target,
        relevance_override=short_rel,
        ev_target_override=short_mag,
    )

    validate_long_matrix(train_long)
    validate_long_matrix(valid_long)
    validate_long_matrix(test_long)
    validate_long_matrix(train_short)
    validate_long_matrix(valid_short)
    validate_long_matrix(test_short)
    _logger.debug(
        "[perf-ml-fold] [Fold %d] side matrix build took %.4fs",
        fold_id,
        time.perf_counter() - t_matrices,
    )

    # 3. Model training & prediction
    t_fit_predict = time.perf_counter()
    _fold_result = _fit_predict_fold_dual_side(
        train_long=train_long,
        valid_long=valid_long,
        test_long=test_long,
        train_short=train_short,
        valid_short=valid_short,
        test_short=test_short,
        ml_cfg=ml_cfg,
    )
    _logger.debug(
        "[perf-ml-fold] [Fold %d] _fit_predict_fold_dual_side took %.4fs",
        fold_id,
        time.perf_counter() - t_fit_predict,
    )
    _logger.debug(
        "[perf-ml-fold] [Fold %d] total run took %.4fs",
        fold_id,
        time.perf_counter() - t_start,
    )

    return {
        "fold_id": fold.fold_id,
        "test_long_index_map": test_long.index_map,
        "test_short_index_map": test_short.index_map,
        "valid_long_index_map": valid_long.index_map,
        "valid_short_index_map": valid_short.index_map,
        "ev_test_long": _fold_result.ev_test_long,
        "ev_test_short": _fold_result.ev_test_short,
        "quant_test_long": _fold_result.quant_test_long,
        "quant_test_short": _fold_result.quant_test_short,
        "conf_test_long": _fold_result.conf_test_long,
        "conf_test_short": _fold_result.conf_test_short,
        "score_test_long": _fold_result.score_test_long,
        "score_test_short": _fold_result.score_test_short,
        "ev_valid_long": _fold_result.ev_valid_long,
        "ev_valid_short": _fold_result.ev_valid_short,
        "score_valid_long": _fold_result.score_valid_long,
        "score_valid_short": _fold_result.score_valid_short,
        "test_long_shape_0": test_long.X.shape[0],
        "train_long_shape_0": train_long.X.shape[0],
        "valid_long_shape_0": valid_long.X.shape[0],
    }


def _fit_predict_fold_dual_side(
    train_long: LongMatrixDataset,
    valid_long: LongMatrixDataset,
    test_long: LongMatrixDataset,
    train_short: LongMatrixDataset,
    valid_short: LongMatrixDataset,
    test_short: LongMatrixDataset,
    ml_cfg: StrategyMLConfig,
) -> _FoldPredictResult:
    """공통 fold 학습·예측 엔진 (WF·AWF·virtual 공유).

    Args:
        train_long: Long side training dataset.
        valid_long: Long side validation dataset.
        test_long: Long side test dataset.
        train_short: Short side training dataset.
        valid_short: Short side validation dataset.
        test_short: Short side test dataset.
        ml_cfg: ML strategy configuration.

    Returns:
        _FoldPredictResult with all EV/quantile/confidence/score arrays.

    Note:
        Time Complexity: O(N * T) where N=samples, T=trees.
        Space Complexity: O(N) per output array.

    """
    t_fit_ranker = time.perf_counter()
    ranker_long: RankerFitResult | None = (
        fit_ranker(train=train_long, valid=valid_long, cfg=ml_cfg)
        if ml_cfg.ranker_enabled
        else None
    )
    _logger.debug(
        "[perf-ml-fold] fit_ranker (signed) took %.4fs",
        time.perf_counter() - t_fit_ranker,
    )

    _ = _rank_score(ranker_long, train_long)
    score_valid_long = _rank_score(ranker_long, valid_long)
    score_test_long = _rank_score(ranker_long, test_long)
    score_valid_short = _rank_score(ranker_long, valid_short)
    score_test_short = _rank_score(ranker_long, test_short)

    t_predict = time.perf_counter()
    ev_valid_long = np.maximum(score_valid_long, 0.0).astype(np.float32, copy=False)
    ev_valid_short = np.maximum(-score_valid_short, 0.0).astype(np.float32, copy=False)
    ev_test_long = np.maximum(score_test_long, 0.0).astype(np.float32, copy=False)
    ev_test_short = np.maximum(-score_test_short, 0.0).astype(np.float32, copy=False)
    quant_test_long = _predict_quantiles_with_fallback(
        None, test_long, score_test_long, ev_test_long
    )
    quant_test_short = _predict_quantiles_with_fallback(
        None, test_short, score_test_short, ev_test_short
    )
    conf_test_long = np.ones_like(ev_test_long, dtype=np.float32)
    conf_test_short = np.ones_like(ev_test_short, dtype=np.float32)
    _logger.debug(
        "[perf-ml-fold] signed score split prediction took %.4fs",
        time.perf_counter() - t_predict,
    )
    return _FoldPredictResult(
        ev_test_long=ev_test_long,
        ev_test_short=ev_test_short,
        quant_test_long=quant_test_long,
        quant_test_short=quant_test_short,
        conf_test_long=conf_test_long,
        conf_test_short=conf_test_short,
        score_test_long=score_test_long,
        score_test_short=score_test_short,
        ev_valid_long=ev_valid_long,
        ev_valid_short=ev_valid_short,
        score_valid_long=score_valid_long,
        score_valid_short=score_valid_short,
    )


def _resolve_horizon_candidates(ml_cfg: StrategyMLConfig) -> tuple[int, ...]:
    """Resolve executable horizon candidates while preserving default behavior."""
    if not ml_cfg.horizon_experiment_enabled:
        return (int(ml_cfg.label_horizon_bars),)
    seen: set[int] = set()
    out: list[int] = []
    for h in ml_cfg.horizon_candidates:
        h_int = int(h)
        if h_int >= 1 and h_int not in seen:
            seen.add(h_int)
            out.append(h_int)
    return tuple(out) if out else (int(ml_cfg.label_horizon_bars),)


def _subset_feature_panel(panel: FeaturePanel, selected_names: tuple[str, ...]) -> FeaturePanel:
    """Return feature panel reduced to selected feature names."""
    if not selected_names:
        return panel
    idx_map = {name: i for i, name in enumerate(panel.feature_names)}
    indices = [idx_map[name] for name in selected_names if name in idx_map]
    if len(indices) == len(panel.feature_names):
        return panel
    values = panel.values[:, :, indices]
    availability_masks = {
        name: mask for name, mask in panel.availability_masks.items() if name in selected_names
    }
    metadata = dict(panel.metadata)
    metadata["selected_feature_names"] = list(selected_names)
    metadata["selected_feature_count"] = len(selected_names)
    return FeaturePanel(
        datetimes=panel.datetimes,
        symbols=panel.symbols,
        values=values,
        feature_names=selected_names,
        valid_mask=panel.valid_mask,
        availability_masks=availability_masks,
        metadata=metadata,
    )


def _integrity_ready(aligned: Any) -> bool:
    required = (
        "open_2d",
        "high_2d",
        "low_2d",
        "close_2d",
        "active_mask",
        "warm_mask",
        "kill_mask",
        "datetimes",
    )
    return all(hasattr(aligned, key) for key in required)


@dataclass(slots=True, frozen=True)
class AnchoredMLPrecomputedPanels:
    """Reusable causal ML panels for anchored refit legs."""

    features: FeaturePanel
    labels: LabelPanel


def precompute_anchored_ml_panels(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> AnchoredMLPrecomputedPanels:
    """Build feature/label panels once and reuse across anchored legs."""
    ml_cfg = cfg.ml
    aligned = align_data_maps(data_maps=data_maps, symbols=symbols, tf=tf)
    if len(aligned.symbols) < ml_cfg.min_group_size:
        raise ValueError(
            f"anchored strategy needs >= {ml_cfg.min_group_size} symbols, "
            f"got {len(aligned.symbols)}"
        )
    features = build_feature_panel(aligned, ml_cfg)
    labels = build_label_panel(aligned, ml_cfg)
    validate_feature_panel(features)
    validate_label_panel(labels, t=features.values.shape[0], n=features.values.shape[1])
    if _integrity_ready(aligned):
        oos_start_idx = max(int(features.values.shape[0] * 0.8), 1)
        _ic_target = (
            labels.forward_gross_ret.astype(np.float64)
            if labels.forward_gross_ret is not None
            else labels.signed_net_ret.astype(np.float64)
        )
        _raw_elig = (
            labels.raw_eligible_mask
            if labels.raw_eligible_mask is not None
            else labels.eligible_mask
        )
        data_integrity = verify_data_integrity(
            aligned,
            oos_start_idx=oos_start_idx,
            forward_gross_ret=_ic_target,
            eligible_mask=_raw_elig,
        )
        _avg_breadth = float(np.mean(np.sum(labels.eligible_mask, axis=1)))
        breakeven_ic = 24.0 / (500.0 * max(_avg_breadth, 1.0) ** 0.5)
        feature_integrity = verify_feature_integrity(
            features,
            train_slice=slice(0, oos_start_idx),
            oos_slice=slice(oos_start_idx, features.values.shape[0]),
            target_2d=_ic_target,
            breakeven_ic=float(breakeven_ic),
        )
        if ml_cfg.feature_selection_enabled:
            selected_names = select_features(
                feature_integrity,
                features.feature_names,
                ml_cfg.feature_integrity,
            )
            features = _subset_feature_panel(features, selected_names)
        features.metadata["integrity"] = {
            "data": asdict(data_integrity),
            "feature": asdict(feature_integrity),
        }
        if data_integrity.hard_fail and ml_cfg.integrity_gate_enabled:
            raise RuntimeError(f"data integrity hard-fail: {data_integrity.fail_reasons}")
    return AnchoredMLPrecomputedPanels(features=features, labels=labels)


def build_ml_strategy_alpha(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
    trading_symbols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build ML strategy alpha panel."""
    if cfg.ml.horizon_experiment_enabled:
        candidates = _resolve_horizon_candidates(cfg.ml)
        friction_bps = round_trip_cost_bps()
        hurdle_bps = float(default_ev_hurdle_bps(OPT_FUTURES_CONFIG))
        floor_bps = friction_bps + hurdle_bps
        _logger.info(
            "[ML-HARNESS] mode=horizon_experiment candidates=%s floor_bps=%.2f",
            list(candidates),
            floor_bps,
        )
        best_score = float("-inf")
        best_panel: pd.DataFrame | None = None
        best_horizon = int(cfg.ml.label_horizon_bars)
        records: list[dict[str, Any]] = []
        for horizon in candidates:
            horizon_cfg = replace(
                cfg.ml,
                horizon_experiment_enabled=False,
                label_horizon_bars=int(horizon),
                purge_bars=max(int(cfg.ml.purge_bars), int(horizon)),
                embargo_bars=max(int(cfg.ml.embargo_bars), int(horizon)),
            )
            candidate_cfg = replace(cfg, ml=horizon_cfg)
            panel = build_ml_strategy_alpha(
                data_maps=data_maps,
                symbols=symbols,
                tf=tf,
                cfg=candidate_cfg,
            )
            report = panel.attrs.get("quality_report", {})
            active_p95_bps = float(
                report.get("alpha_active_p95_bps", report.get("alpha_p95_bps", 0.0))
            )
            full_matrix_p95_bps = float(report.get("alpha_p95_bps", 0.0))
            tradable_density = min(
                float(report.get("alpha_long_tradable_nz", 0.0)),
                float(report.get("alpha_short_tradable_nz", 0.0)),
            )
            score_bps = (active_p95_bps - floor_bps) + 10.0 * tradable_density
            record = {
                "horizon": int(horizon),
                "active_p95_bps": active_p95_bps,
                "full_matrix_p95_bps": full_matrix_p95_bps,
                "tradable_density": tradable_density,
                "score_bps": score_bps,
                "clears_cost_wall": bool(active_p95_bps >= floor_bps),
            }
            records.append(record)
            _logger.info(
                (
                    "[ML-HORIZON] horizon=%d active_p95=%.2fbps full_p95=%.2fbps "
                    "floor=%.2fbps tradable=%.4f score=%.2fbps pass=%s"
                ),
                int(horizon),
                active_p95_bps,
                full_matrix_p95_bps,
                floor_bps,
                tradable_density,
                score_bps,
                str(active_p95_bps >= floor_bps),
            )
            if score_bps > best_score:
                best_score = score_bps
                best_panel = panel
                best_horizon = int(horizon)
        if best_panel is None:
            raise RuntimeError("horizon experiment failed to produce any candidate panel")
        best_panel.attrs["horizon_experiment"] = records
        best_panel.attrs["selected_horizon"] = best_horizon
        best_panel.attrs["selected_horizon_score_bps"] = best_score
        best_panel.attrs["selected_horizon_floor_bps"] = floor_bps
        best_panel.attrs["baseline_harness"] = {
            "version": "v1",
            "mode": "horizon_experiment",
            "selected_horizon": best_horizon,
            "selected_horizon_score_bps": best_score,
            "cost_floor_bps": floor_bps,
            "candidate_count": len(records),
        }
        _logger.info(
            "[ML-HARNESS] mode=horizon_experiment selected_horizon=%d selected_score=%.2fbps",
            best_horizon,
            best_score,
        )
        return best_panel

    # Timing Accumulators (Time Complexity: O(1) space overhead)
    t_align_feature_label = 0.0
    t_fold_split_preprocess = 0.0
    t_dataset_build = 0.0
    t_fit_predict_fold = 0.0
    t_grid_assembly = 0.0

    t_start_total = time.perf_counter()

    ml_cfg = cfg.ml
    t_align_start = time.perf_counter()
    aligned = align_data_maps(data_maps=data_maps, symbols=symbols, tf=tf)
    if len(aligned.symbols) < ml_cfg.min_group_size:
        raise ValueError(
            f"strategy needs >= {ml_cfg.min_group_size} symbols, got {len(aligned.symbols)}"
        )
    
    t_feat_build = time.perf_counter()
    features = build_feature_panel(aligned, ml_cfg)
    _logger.debug(
        "[perf-ml-prep] build_feature_panel took %.4fs",
        time.perf_counter() - t_feat_build,
    )

    t_lbl_build = time.perf_counter()
    labels = build_label_panel(aligned, ml_cfg)
    _logger.debug("[perf-ml-prep] build_label_panel took %.4fs", time.perf_counter() - t_lbl_build)

    t_val = time.perf_counter()
    validate_feature_panel(features)
    validate_label_panel(labels, t=features.values.shape[0], n=features.values.shape[1])
    _logger.debug("[perf-ml-prep] validate panels took %.4fs", time.perf_counter() - t_val)

    data_integrity: Any | None = None
    feature_integrity: Any | None = None
    if _integrity_ready(aligned):
        oos_start_idx = max(int(features.values.shape[0] * 0.8), 1)
        _ic_target_int = (
            labels.forward_gross_ret.astype(np.float64)
            if labels.forward_gross_ret is not None
            else labels.signed_net_ret.astype(np.float64)
        )
        _raw_elig_int = (
            labels.raw_eligible_mask
            if labels.raw_eligible_mask is not None
            else labels.eligible_mask
        )
        t_data_int = time.perf_counter()
        data_integrity = verify_data_integrity(
            aligned,
            oos_start_idx=oos_start_idx,
            forward_gross_ret=_ic_target_int,
            eligible_mask=_raw_elig_int,
        )
        _logger.debug(
            "[perf-ml-prep] verify_data_integrity took %.4fs",
            time.perf_counter() - t_data_int,
        )

        _cofinite = np.sum(
            np.isfinite(_ic_target_int[oos_start_idx:]) & labels.eligible_mask[oos_start_idx:],
            axis=1,
        )
        _cofinite_p50 = float(np.median(_cofinite)) if _cofinite.size else 1.0
        breakeven_ic = 24.0 / (500.0 * max(_cofinite_p50, 1.0) ** 0.5)
        
        t_feat_int = time.perf_counter()
        feature_integrity = verify_feature_integrity(
            features,
            train_slice=slice(0, oos_start_idx),
            oos_slice=slice(oos_start_idx, features.values.shape[0]),
            target_2d=_ic_target_int,
            breakeven_ic=float(breakeven_ic),
        )
        _logger.debug(
            "[perf-ml-prep] verify_feature_integrity took %.4fs",
            time.perf_counter() - t_feat_int,
        )

        if ml_cfg.feature_selection_enabled:
            _total_feat_before = len(features.feature_names)
            t_feat_select = time.perf_counter()
            selected_names = select_features(
                feature_integrity, features.feature_names, ml_cfg.feature_integrity
            )
            features = _subset_feature_panel(features, selected_names)
            _logger.debug(
                "[perf-ml-prep] select_features took %.4fs",
                time.perf_counter() - t_feat_select,
            )
        _logger.info(
            "🛡️ [DATA-INT] zero_price=%.6f ohlc_violation=%.6f bar_gap=%d nan_decomp=%s",
            data_integrity.zero_price_ratio,
            data_integrity.ohlc_violation_ratio,
            data_integrity.bar_gap_count,
            data_integrity.nan_decomposition,
        )
        _logger.info(
            "🧬 [FEAT-INT] constant=%d drifted=%d redundant=%d leakage=%d",
            len(feature_integrity.constant_features),
            len(feature_integrity.drifted_features),
            len(feature_integrity.redundant_pairs),
            len(feature_integrity.leakage_suspects),
        )
        if feature_integrity.constant_features:
            _logger.info("🧬 [FEAT-INT] constant=%s", feature_integrity.constant_features)
        if feature_integrity.redundant_pairs:
            _logger.info(
                "🧬 [FEAT-INT] redundant=%s",
                [(a, b, round(c, 3)) for a, b, c in feature_integrity.redundant_pairs],
            )
        if ml_cfg.feature_selection_enabled:
            _logger.info(
                "🧬 [FEAT-SELECT] kept=%d/%d names=%s",
                len(selected_names),
                _total_feat_before,
                list(selected_names),
            )
        if data_integrity.hard_fail:
            msg = f"data integrity hard-fail: {data_integrity.fail_reasons}"
            if ml_cfg.integrity_gate_enabled:
                raise RuntimeError(msg)
            _logger.warning("🛡️ [DATA-INT] %s", msg)
    t_align_feature_label += time.perf_counter() - t_align_start
    _logger.debug(
        "[perf-ml-prep] total features + alignment phase took %.4fs",
        t_align_feature_label,
    )

    # [Dynamic Train Window] In-Sample 또는 Leg Refit 등 데이터 기간이 부족할 경우
    # train_months를 유연하게 동적 조정합니다.
    idx = pd.to_datetime(features.datetimes)
    if idx.size > 0:
        total_months = round((idx[-1] - idx[0]).days / 30.4375)
        required_minimum = ml_cfg.valid_months + ml_cfg.test_months
        needed = ml_cfg.train_months + required_minimum
        if total_months < needed:
            # 1개월의 세이프티 마진을 차감하여 날짜 오프셋 경계 불일치를 원천 차단합니다.
            adjusted_train = max(12, total_months - required_minimum - 1)
            if adjusted_train != ml_cfg.train_months:
                _logger.info(
                    "🧠 ML_INIT: adj_train=%dm -> %dm (safety_margin applied)",
                    ml_cfg.train_months,
                    adjusted_train,
                )
                ml_cfg = replace(ml_cfg, train_months=adjusted_train)

    t_folds = time.perf_counter()
    folds = make_walk_forward_folds(features.datetimes, ml_cfg)
    _logger.debug(
        "[perf-ml-prep] make_walk_forward_folds took %.4fs",
        time.perf_counter() - t_folds,
    )
    if not folds:
        raise RuntimeError("no walk-forward folds can be built")
    _logger.info(
        "🧠 ML_INIT: feats=%d symbols=%d rows=%d train_w=%dm | 🧩 folds=%d",
        features.values.shape[2],
        features.values.shape[1],
        features.values.shape[0],
        ml_cfg.train_months,
        len(folds),
    )
    _logger.debug(
        ".. ML_LABEL: eligible=%.4f sample_weight_mean=%.4f",
        float(np.mean(labels.eligible_mask)),
        (
            float(np.mean(labels.sample_weight[labels.eligible_mask]))
            if np.any(labels.eligible_mask)
            else 0.0
        ),
    )

    ev_long_grid = np.zeros((features.values.shape[0], features.values.shape[1]), dtype=np.float32)
    ev_short_grid = np.zeros((features.values.shape[0], features.values.shape[1]), dtype=np.float32)
    q10_long_grid = np.zeros_like(ev_long_grid)
    q50_long_grid = np.zeros_like(ev_long_grid)
    q90_long_grid = np.zeros_like(ev_long_grid)
    q10_short_grid = np.zeros_like(ev_long_grid)
    q50_short_grid = np.zeros_like(ev_long_grid)
    q90_short_grid = np.zeros_like(ev_long_grid)
    confidence_long_grid = np.zeros_like(ev_long_grid)
    confidence_short_grid = np.zeros_like(ev_long_grid)
    score_grid = np.full_like(ev_long_grid, np.nan, dtype=np.float32)
    rank_score_long_grid = np.full_like(ev_long_grid, np.nan, dtype=np.float32)
    rank_score_short_grid = np.full_like(ev_long_grid, np.nan, dtype=np.float32)
    valid_ev_long_all: list[np.ndarray] = []
    valid_ev_short_all: list[np.ndarray] = []
    rank_policies_by_fold: list[dict[str, float | int | str]] = []
    fold_policy_masks: list[tuple[np.ndarray, np.ndarray, RankSelectionPolicy, int]] = []

    # 1. Check if there is an uncovered live/OOS window at the end of the timeline
    last_test_end = folds[-1].test_end
    total_bars = features.values.shape[0]
    all_folds = list(folds)
    v_fold = None
    if last_test_end < total_bars:
        _logger.debug(
            "[ML-OOS-FILL] Uncovered OOS/live window detected: [%d, %d)", last_test_end, total_bars
        )
        # Construct virtual fold Spec
        v_size = folds[-1].valid_end - folds[-1].valid_start
        v_train_end = last_test_end - v_size
        _v_purge = getattr(ml_cfg, "purge_bars", 0)
        _v_embargo = getattr(ml_cfg, "embargo_bars", 0)
        v_fold = FoldSpec(
            fold_id=len(folds),
            train_start=0,
            train_end=v_train_end,
            valid_start=min(v_train_end + _v_purge, max(v_train_end, last_test_end - 1)),
            valid_end=last_test_end,
            test_start=min(last_test_end + _v_embargo, max(last_test_end, total_bars - 1)),
            test_end=total_bars,
            purge_bars=_v_purge,
            embargo_bars=_v_embargo,
        )
        all_folds.append(v_fold)

    # 2. Dynamic CPU & Thread Allocation to maximize CPU and prevent WSL OOM
    import sys
    cpu_count = os.cpu_count() or 4
    is_pytest = "pytest" in sys.modules
    if ml_cfg.parallel_folds and len(all_folds) > 1 and not is_pytest:
        if ml_cfg.parallel_fold_workers <= 0:
            # Optimal fallback for WSL with 8 processes and 16GB RAM:
            # Set to 6 folds in parallel. Under loky, memory reuse prevents OOM.
            folds_jobs = min(6, max(1, cpu_count - 2))
        else:
            folds_jobs = ml_cfg.parallel_fold_workers
        
        # Enforce n_jobs = 1 to prevent CPU oversubscription thrashing.
        lgb_n_jobs = 1
        _logger.info(
            "🧠 ML-PARALLEL: Training %d folds in parallel. LightGBM n_jobs forced to %d.",
            folds_jobs,
            lgb_n_jobs,
        )
    else:
        folds_jobs = 1
        lgb_n_jobs = max(1, cpu_count - 2) if ml_cfg.n_jobs <= 0 else ml_cfg.n_jobs
        _logger.info(
            "🧠 ML-SEQUENTIAL: Training folds sequentially. LightGBM n_jobs=%d.",
            lgb_n_jobs,
        )

    # Apply resolved dynamic n_jobs
    ml_cfg = replace(ml_cfg, n_jobs=lgb_n_jobs)

    # 3. Resolve target arrays once to optimize memory allocation
    (
        long_rank_target,
        short_rank_target,
        long_rel,
        short_rel,
        long_mag,
        short_mag,
    ) = _resolve_side_targets(labels, ml_cfg)

    # 4. joblib Parallel execution with sequential fallback (pytest-friendly)
    t_parallel_start = time.perf_counter()
    if folds_jobs > 1:
        results = Parallel(n_jobs=folds_jobs, backend="loky")(
            delayed(_train_predict_single_fold)(
                fold=fold,
                features=features,
                labels=labels,
                ml_cfg=ml_cfg,
                long_rank_target=long_rank_target,
                short_rank_target=short_rank_target,
                long_rel=long_rel,
                short_rel=short_rel,
                long_mag=long_mag,
                short_mag=short_mag,
            )
            for fold in all_folds
        )
    else:
        results = [
            _train_predict_single_fold(
                fold=fold,
                features=features,
                labels=labels,
                ml_cfg=ml_cfg,
                long_rank_target=long_rank_target,
                short_rank_target=short_rank_target,
                long_rel=long_rel,
                short_rel=short_rel,
                long_mag=long_mag,
                short_mag=short_mag,
            )
            for fold in all_folds
        ]
    t_parallel_elapsed = time.perf_counter() - t_parallel_start
    _logger.info(
        "🧠 ML-PARALLEL: Completed all %d folds in %.2f ms",
        len(all_folds),
        t_parallel_elapsed * 1000.0,
    )

    # 5. Grid Assembly and Post-processing
    t_grid_start = time.perf_counter()
    for res in results:
        fold_id = res["fold_id"]
        is_virtual = (v_fold is not None and fold_id == v_fold.fold_id)

        ev_test_long = res["ev_test_long"]
        ev_test_short = res["ev_test_short"]
        quant_test_long = res["quant_test_long"]
        quant_test_short = res["quant_test_short"]
        conf_test_long = res["conf_test_long"]
        conf_test_short = res["conf_test_short"]
        score_test = res["score_test_long"]
        score_test_short = res["score_test_short"]

        if not is_virtual:
            valid_ev_long_all.append(res["ev_valid_long"])
            valid_ev_short_all.append(res["ev_valid_short"])
            valid_long_index_map = np.asarray(res["valid_long_index_map"])
            valid_short_index_map = np.asarray(res["valid_short_index_map"])
            score_valid_long = np.asarray(res["score_valid_long"], dtype=np.float64)
            score_valid_short = np.asarray(res["score_valid_short"], dtype=np.float64)
            
            _logger.debug(
                ".. ML_RANKER: fold=%d train_n=%d valid_n=%d test_n=%d",
                fold_id,
                int(res["train_long_shape_0"]),
                int(res["valid_long_shape_0"]),
                int(res["test_long_shape_0"]),
            )
            if ev_test_long.size > 0:
                _ev_neg_ratio = float(np.mean(ev_test_long < 0.0))
                _ev_p50 = float(np.percentile(ev_test_long, 50)) * 1e4
                _ev_p90 = float(np.percentile(ev_test_long, 90)) * 1e4
                _ev_p95 = float(np.percentile(ev_test_long, 95)) * 1e4
                _logger.info(
                    "🔬 [EV-PRECLIP] fold=%d neg=%.1f%% p50=%.1fbps p90=%.1fbps p95=%.1fbps n=%d",
                    fold_id,
                    100.0 * _ev_neg_ratio,
                    _ev_p50,
                    _ev_p90,
                    _ev_p95,
                    int(ev_test_long.size),
                )
        else:
            if ev_test_long.size > 0:
                _vev_neg_ratio = float(np.mean(ev_test_long < 0.0))
                _vev_p50 = float(np.percentile(ev_test_long, 50)) * 1e4
                _vev_p90 = float(np.percentile(ev_test_long, 90)) * 1e4
                _vev_p95 = float(np.percentile(ev_test_long, 95)) * 1e4
                _logger.info(
                    "🔬 [EV-PRECLIP] vrefit neg=%.1f%% p50=%.1fbps p90=%.1fbps p95=%.1fbps n=%d",
                    100.0 * _vev_neg_ratio,
                    _vev_p50,
                    _vev_p90,
                    _vev_p95,
                    int(ev_test_long.size),
                )

        # Write directly into grids via fancy indexing
        for row, (t_idx, s_idx) in enumerate(res["test_long_index_map"]):
            q10_long_grid[int(t_idx), int(s_idx)] = quant_test_long.q10[row]
            q50_long_grid[int(t_idx), int(s_idx)] = quant_test_long.q50[row]
            q90_long_grid[int(t_idx), int(s_idx)] = quant_test_long.q90[row]
            confidence_long_grid[int(t_idx), int(s_idx)] = conf_test_long[row]
            ev_long_grid[int(t_idx), int(s_idx)] = np.float32(max(ev_test_long[row], 0.0))
            score_grid[int(t_idx), int(s_idx)] = score_test[row]
            rank_score_long_grid[int(t_idx), int(s_idx)] = score_test[row]

        for row, (t_idx, s_idx) in enumerate(res["test_short_index_map"]):
            q10_short_grid[int(t_idx), int(s_idx)] = quant_test_short.q10[row]
            q50_short_grid[int(t_idx), int(s_idx)] = quant_test_short.q50[row]
            q90_short_grid[int(t_idx), int(s_idx)] = quant_test_short.q90[row]
            confidence_short_grid[int(t_idx), int(s_idx)] = conf_test_short[row]
            ev_short_grid[int(t_idx), int(s_idx)] = np.float32(max(ev_test_short[row], 0.0))
            rank_score_short_grid[int(t_idx), int(s_idx)] = score_test_short[row]
            if np.isnan(score_grid[int(t_idx), int(s_idx)]):
                score_grid[int(t_idx), int(s_idx)] = score_test_short[row]

        if not is_virtual and bool(getattr(ml_cfg, "rank_policy_enabled", True)):
            valid_score_long_grid = np.full_like(rank_score_long_grid, np.nan, dtype=np.float64)
            valid_score_short_grid = np.full_like(rank_score_short_grid, np.nan, dtype=np.float64)
            for row, (t_idx, s_idx) in enumerate(valid_long_index_map):
                valid_score_long_grid[int(t_idx), int(s_idx)] = score_valid_long[row]
            for row, (t_idx, s_idx) in enumerate(valid_short_index_map):
                valid_score_short_grid[int(t_idx), int(s_idx)] = score_valid_short[row]

            valid_signed = derive_signed_rank_signal(valid_score_long_grid, valid_score_short_grid)
            h_candidates = tuple(int(h) for h in ml_cfg.rank_policy_holding_candidates)
            hold = int(ml_cfg.label_horizon_bars)
            if hold not in h_candidates and len(h_candidates) > 0:
                hold = int(h_candidates[0])
            policy = calibrate_rank_selection_policy(
                signed_score_2d=valid_signed,
                realized_fwd_ret_2d=labels.exec_net_ret.astype(np.float64),
                eligible_2d=labels.eligible_mask.astype(bool),
                quantiles=tuple(float(q) for q in ml_cfg.rank_policy_quantiles),
                min_abs_z_grid=tuple(float(z) for z in ml_cfg.rank_policy_min_abs_z_grid),
                holding_bars=hold,
                cost_bps=24.0,
                min_obs=int(ml_cfg.rank_policy_min_validation_obs),
                weight_k=float(ml_cfg.alpha_emit_weight_k),
                weighting=ml_cfg.rank_policy_weighting,
            )
            fold_policy_masks.append(
                (
                    np.asarray(res["test_long_index_map"]),
                    np.asarray(res["test_short_index_map"]),
                    policy,
                    int(fold_id),
                )
            )
            pol_dict = policy_to_dict(policy)
            pol_dict["fold_id"] = int(fold_id)
            rank_policies_by_fold.append(pol_dict)
            _logger.info(
                "[RANK-POLICY] fold=%d polarity=%d q=%.2f floor=%.2f hold=%d val_lcb=%.2f val_ir=%.2f mono=%.2f",
                int(fold_id),
                int(policy.polarity),
                float(policy.quantile),
                float(policy.min_abs_z),
                int(policy.holding_bars),
                float(policy.validation_net_lcb_bps),
                float(policy.validation_ir_t),
                float(policy.validation_monotonicity),
            )

        if not is_virtual:
            _logger.debug(
                ".. ML_SCORE_SPLIT: fold=%d ev_mean=%.6e ev_p10=%.6e ev_p90=%.6e",
                fold_id,
                float(np.mean(ev_test_long, dtype=np.float32)) if ev_test_long.size > 0 else 0.0,
                float(np.percentile(ev_test_long, 10)) if ev_test_long.size > 0 else 0.0,
                float(np.percentile(ev_test_long, 90)) if ev_test_long.size > 0 else 0.0,
            )
            _logger.debug(
                ".. ML_OOS: fold=%d test_rows=%d alpha_nz=%.4f",
                fold_id,
                int(res["test_long_shape_0"]),
                float(np.count_nonzero(ev_test_long) / max(1, ev_test_long.size)),
            )
        else:
            nz_l = (
                float(np.count_nonzero(ev_test_long) / max(1, ev_test_long.size))
                if ev_test_long.size > 0
                else 0.0
            )
            nz_s = (
                float(np.count_nonzero(ev_test_short) / max(1, ev_test_short.size))
                if ev_test_short.size > 0
                else 0.0
            )
            _logger.info(
                "🧩 ML_OOS_FILL: virtual_refit complete (rows=%d L_nz=%.3f S=%.3f)",
                int(res["test_long_shape_0"]),
                nz_l,
                nz_s,
            )

    t_grid_assembly += time.perf_counter() - t_grid_start

    t_grid_start = time.perf_counter()
    clip_lim = float(ml_cfg.alpha_clip_bps / 10000.0)
    eligible_2d = labels.eligible_mask
    _emit_mode = str(getattr(ml_cfg, "alpha_emit_mode", "rank_sized"))
    if _emit_mode == "rank_sized":
        # clip_lim=1.0: tanh weights가 (-1,+1) 범위이므로 alpha_clip_bps(0.0075) 대신
        # 1.0을 사용해 continuous weight 보존. portfolio sizing은 backtest에서 별도 결정.
        if bool(getattr(ml_cfg, "rank_policy_enabled", True)) and rank_policies_by_fold:
            alpha_long_final = np.zeros_like(rank_score_long_grid, dtype=np.float32)
            alpha_short_final = np.zeros_like(rank_score_short_grid, dtype=np.float32)
            signed_all = derive_signed_rank_signal(
                rank_score_long_grid.astype(np.float64),
                rank_score_short_grid.astype(np.float64),
            )
            for test_l_idx, test_s_idx, policy, _fold_id in fold_policy_masks:
                lmask_2d = np.zeros_like(eligible_2d, dtype=bool)
                smask_2d = np.zeros_like(eligible_2d, dtype=bool)
                for t_idx, s_idx in test_l_idx:
                    lmask_2d[int(t_idx), int(s_idx)] = True
                for t_idx, s_idx in test_s_idx:
                    smask_2d[int(t_idx), int(s_idx)] = True
                apply_mask = lmask_2d | smask_2d
                pol_long, pol_short = apply_rank_selection_policy(
                    signed_score_2d=signed_all,
                    eligible_2d=eligible_2d & apply_mask,
                    policy=policy,
                )
                alpha_long_final = np.maximum(alpha_long_final, pol_long)
                alpha_short_final = np.maximum(alpha_short_final, pol_short)
            if v_fold is not None:
                aggregate_policy = sorted(
                    rank_policies_by_fold,
                    key=lambda p: float(p["validation_net_lcb_bps"]),
                    reverse=True,
                )[0]
                live_policy = RankSelectionPolicy(
                    polarity=int(aggregate_policy["polarity"]),  # type: ignore[arg-type]
                    quantile=float(aggregate_policy["quantile"]),
                    min_abs_z=float(aggregate_policy["min_abs_z"]),
                    weighting=str(aggregate_policy["weighting"]),  # type: ignore[arg-type]
                    weight_k=float(aggregate_policy["weight_k"]),
                    holding_bars=int(aggregate_policy["holding_bars"]),
                    validation_net_lcb_bps=float(aggregate_policy["validation_net_lcb_bps"]),
                    validation_gross_bps=float(aggregate_policy["validation_gross_bps"]),
                    validation_ir_t=float(aggregate_policy["validation_ir_t"]),
                    validation_monotonicity=float(aggregate_policy["validation_monotonicity"]),
                    n_obs=int(aggregate_policy["n_obs"]),
                )
                vmask = np.zeros_like(eligible_2d, dtype=bool)
                for t_idx, s_idx in np.asarray(results[-1]["test_long_index_map"]):
                    vmask[int(t_idx), int(s_idx)] = True
                for t_idx, s_idx in np.asarray(results[-1]["test_short_index_map"]):
                    vmask[int(t_idx), int(s_idx)] = True
                pol_long_v, pol_short_v = apply_rank_selection_policy(
                    signed_score_2d=signed_all,
                    eligible_2d=eligible_2d & vmask,
                    policy=live_policy,
                )
                alpha_long_final = np.maximum(alpha_long_final, pol_long_v)
                alpha_short_final = np.maximum(alpha_short_final, pol_short_v)
                panel_policy_summary = dict(aggregate_policy)
            else:
                panel_policy_summary = max(
                    rank_policies_by_fold,
                    key=lambda p: float(p["validation_net_lcb_bps"]),
                )
        else:
            alpha_long_final, alpha_short_final = _emit_rank_sized_alpha(
                rank_score_long_grid,
                rank_score_short_grid,
                eligible_2d,
                select_q=float(getattr(ml_cfg, "alpha_emit_select_q", 0.40)),
                weight_k=float(getattr(ml_cfg, "alpha_emit_weight_k", 3.0)),
                clip_lim=1.0,
            )
            panel_policy_summary = None
    else:
        alpha_long_final = np.where(
            eligible_2d,
            np.clip(np.maximum(ev_long_grid, 0.0), 0.0, clip_lim),
            0.0,
        ).astype(np.float32, copy=False)
        alpha_short_final = np.where(
            eligible_2d,
            np.clip(np.maximum(ev_short_grid, 0.0), 0.0, clip_lim),
            0.0,
        ).astype(np.float32, copy=False)
    alpha_ic_score = (alpha_long_final - alpha_short_final).astype(np.float64, copy=False)
    forecast_metadata = {
        "q10_long": q10_long_grid.reshape(-1),
        "q50_long": q50_long_grid.reshape(-1),
        "q90_long": q90_long_grid.reshape(-1),
        "q10_short": q10_short_grid.reshape(-1),
        "q50_short": q50_short_grid.reshape(-1),
        "q90_short": q90_short_grid.reshape(-1),
        "confidence_long": confidence_long_grid.reshape(-1),
        "confidence_short": confidence_short_grid.reshape(-1),
        "rank_score_long": rank_score_long_grid.reshape(-1),
        "rank_score_short": rank_score_short_grid.reshape(-1),
    }
    _idx = pd.MultiIndex.from_product(
        [features.datetimes, features.symbols], names=["datetime", "symbol"]
    )
    panel = pd.DataFrame(
        {
            "alpha_long": alpha_long_final.reshape(-1),
            "alpha_short": alpha_short_final.reshape(-1),
            "rank_score_long": rank_score_long_grid.reshape(-1),
            "rank_score_short": rank_score_short_grid.reshape(-1),
        },
        index=_idx,
    ).sort_index()
    if forecast_metadata is not None:
        from src.domain.futures.strategy.contracts import ALPHA_FORECAST_CONTRACT
        panel.attrs["forecast_contract_version"] = ALPHA_FORECAST_CONTRACT
        panel.attrs["alpha_forecast_metadata"] = dict(forecast_metadata)
    panel.attrs["rank_score_contract"] = {
        "version": 1,
        "mode": "signed_single_ranker",
        "long_higher_is_better": True,
        "short_lower_is_better": True,
        "signed_signal_formula": "derive_signed_rank_signal(rank_score_long, rank_score_short)",
    }
    if _emit_mode == "rank_sized" and rank_policies_by_fold:
        assert panel_policy_summary is not None
        panel.attrs["rank_selection_policy"] = dict(panel_policy_summary)
        panel.attrs["rank_selection_policy_by_fold"] = list(rank_policies_by_fold)
    panel.attrs["strategy_name"] = cfg.name
    panel.attrs["feature_names"] = list(features.feature_names)
    panel.attrs["fold_count"] = len(folds)
    panel.attrs["config_hash"] = build_manifest_hash(asdict(cfg.ml))
    panel.attrs["feature_config_hash"] = build_manifest_hash(
        {"feature_names": sorted(features.feature_names)}
    )
    panel.attrs["label_config_hash"] = build_manifest_hash(
        {
            "label_horizon_bars": ml_cfg.label_horizon_bars,
        }
    )
    panel.attrs["fold_spec_hash"] = build_manifest_hash(
        {
            "fold_count": len(folds),
            "folds": [
                {
                    "fold_id": getattr(f, "fold_id", i),
                    "train_start": getattr(f, "train_start", 0),
                    "train_end": getattr(f, "train_end", 0),
                }
                for i, f in enumerate(folds)
            ],
        }
    )
    panel.attrs["train_window_hash"] = build_manifest_hash(
        {
            "windows": [
                {
                    "train_start": getattr(f, "train_start", 0),
                    "train_end": getattr(f, "train_end", 0),
                }
                for f in folds
            ]
        }
    )
    from src.domain.futures.forecast.contracts import AlphaArtifactHash

    _aah = AlphaArtifactHash(
        alpha_config_hash=str(panel.attrs["config_hash"]),
        feature_config_hash=str(panel.attrs["feature_config_hash"]),
        label_config_hash=str(panel.attrs["label_config_hash"]),
        train_window_hash=str(panel.attrs["train_window_hash"]),
        fold_spec_hash=str(panel.attrs["fold_spec_hash"]),
        model_family="lightgbm_signed_lambdarank",
        selected_horizon=int(ml_cfg.label_horizon_bars),
    )
    panel.attrs["alpha_artifact_combined_hash"] = _aah.combined()
    panel.attrs["alpha_artifact_structural_hash"] = _aah.structural_hash()
    panel.attrs["selected_horizon"] = int(ml_cfg.label_horizon_bars)
    panel.attrs["baseline_harness"] = {
        "version": "v1",
        "mode": "single_horizon",
        "selected_horizon": int(ml_cfg.label_horizon_bars),
        "cost_floor_bps": float(
            round_trip_cost_bps()
            + float(default_ev_hurdle_bps(OPT_FUTURES_CONFIG))
        ),
        "candidate_count": 1,
    }
    panel.attrs["model_family"] = "lightgbm_signed_lambdarank"
    panel.attrs["ranking_mode"] = "group_ndcg_signed"
    panel.attrs["alpha_contract"] = "return_unit_grinold_rank"
    panel.attrs["feature_groups_enabled"] = list(features.availability_masks.keys())
    if data_integrity is not None and feature_integrity is not None:
        panel.attrs["integrity"] = {
            "data": asdict(data_integrity),
            "feature": asdict(feature_integrity),
        }
    quality_report: dict[str, Any] = build_quality_report(
        feature_values=features.values,
        feature_valid_mask=features.valid_mask,
        label_eligible_mask=labels.eligible_mask,
        score_2d=score_grid,
        signed_ret_2d=labels.signed_net_ret.astype(np.float64),
        relevance_2d=labels.relevance.astype(np.float64),
        alpha_long_2d=alpha_long_final,
        alpha_short_2d=alpha_short_final,
        ic_score_2d=alpha_ic_score,
    )
    # OOS rank-IC 진단
    _score_grid_f = score_grid.astype(np.float64)
    if ml_cfg.oos_ic_target_source == "forward_gross_ret" and labels.forward_gross_ret is not None:
        _ic_target = labels.forward_gross_ret.astype(np.float64)
        _target_source = "forward_gross_ret"
    else:
        _ic_target = labels.signed_net_ret.astype(np.float64)
        _target_source = "signed_net_ret"

    # OOS 시간 범위 감지: score_grid에 ≥1 finite가 있는 bar
    _score_finite_per_bar = np.sum(np.isfinite(_score_grid_f), axis=1)  # [T]
    _oos_time_mask: np.ndarray = _score_finite_per_bar >= 1  # OOS로 추정되는 bar들

    # Co-finite: score AND signed_net_ret 모두 finite인 (t,s) 쌍 수 / bar
    _cofinite_per_bar = np.sum(np.isfinite(_score_grid_f) & np.isfinite(_ic_target), axis=1)  # [T]
    _cofinite_oos = _cofinite_per_bar[_oos_time_mask]  # OOS 구간만

    # dense 비율 — 로깅 호환용으로만 유지 (분모에 비활성 심볼 포함하므로 진단 기준 아님)
    _target_oos_finite = float(
        np.mean(np.isfinite(_ic_target[_oos_time_mask]))
    ) if _oos_time_mask.any() else 0.0

    _cofinite_p50 = float(np.median(_cofinite_oos)) if _cofinite_oos.size > 0 else 0.0
    _bars_ge5 = int(np.sum(_cofinite_oos >= 5))
    _bars_ge5_ratio = float(_bars_ge5 / max(int(_cofinite_oos.size), 1))

    # eligibility-조건부 커버리지: raw eligible 셀(finite_long 적용 전) 중 target이 finite인 비율.
    # raw_eligible을 써야 tautology를 방지한다 — eligible_mask=eligible&finite_long을 쓰면
    # 분모가 이미 finite를 보장하므로 cov_elig ≡ 1.0이 되어 결손 탐지력이 없다.
    _raw_elig_labels = (
        labels.raw_eligible_mask
        if labels.raw_eligible_mask is not None
        else labels.eligible_mask
    )
    _elig = np.asarray(_raw_elig_labels, dtype=bool)
    _elig_oos = _elig[_oos_time_mask]
    _tgt_oos = _ic_target[_oos_time_mask]
    _elig_cnt = int(np.count_nonzero(_elig_oos))
    _cov_within_elig = (
        float(np.count_nonzero(np.isfinite(_tgt_oos) & _elig_oos) / _elig_cnt)
        if _elig_cnt > 0 else 0.0
    )

    # SCORE-IC 원인 진단 — dense 비율 대신 eligibility-조건부 커버리지 기준 사용
    if not _oos_time_mask.any():
        _oos_diag = "no_oos_predictions"
    elif _cofinite_p50 < 5 and _cov_within_elig < 0.9:
        _oos_diag = "both_cofinite_starvation_AND_target_dropout"
    elif _cofinite_p50 < 5:
        _oos_diag = "cofinite_starvation(score∩snr<5/bar)"
    elif _cov_within_elig < 0.9:
        _oos_diag = "target_dropout_within_eligible"
    else:
        _oos_diag = "sufficient_cofinite_check_ic"

    # OOS rank-IC 계산 (co-finite ≥ 5인 bar 한정)
    _oos_ge5_mask = (_oos_time_mask) & (_cofinite_per_bar >= 5)
    if _oos_ge5_mask.any():
        _oos_ic_series = rolling_ic(
            _score_grid_f[_oos_ge5_mask],
            _ic_target[_oos_ge5_mask],
            method="spearman",
        )
        _oos_ic_stats = ic_summary(_oos_ic_series)
    else:
        _oos_ic_stats = {"mean_ic": 0.0, "t_stat": 0.0, "hit_ratio": 0.0, "n_obs": 0.0}

    _score_breadth = float(np.mean(_score_finite_per_bar))
    _logger.info(
        "🔬 [SCORE-IC] dense_ranker ic=%.4f t=%.2f hit=%.3f breadth=%.1f"
        " (cf. emit_breadth≈1, target_breadth≥8)",
        float(_oos_ic_stats["mean_ic"]),
        float(_oos_ic_stats["t_stat"]),
        float(_oos_ic_stats["hit_ratio"]),
        _score_breadth,
    )
    _logger.info(
        "🔬 [OOS-RANKIC] ic=%.4f t=%.2f n_bars=%d cofinite_p50=%.1f"
        " bars_ge5_ratio=%.3f snr_oos_finite=%.3f cov_elig=%.3f",
        float(_oos_ic_stats["mean_ic"]),
        float(_oos_ic_stats["t_stat"]),
        int(_oos_ic_stats.get("n_obs", 0.0)),
        _cofinite_p50,
        _bars_ge5_ratio,
        _target_oos_finite,
        _cov_within_elig,
    )
    _logger.info(
        "🔬 [OOS-DIAG] cause=%s oos_bars=%d ge5_bars=%d",
        _oos_diag,
        int(_oos_time_mask.sum()),
        _bars_ge5,
    )
    panel.attrs["oos_forward_rank_ic"] = {
        "mean_ic": float(_oos_ic_stats.get("mean_ic", 0.0)),
        "t_stat": float(_oos_ic_stats.get("t_stat", 0.0)),
        "hit_ratio": float(_oos_ic_stats.get("hit_ratio", 0.0)),
        "n_obs": int(_oos_ic_stats.get("n_obs", 0.0)),
        "cofinite_p50": float(_cofinite_p50),
        "bars_ge5_ratio": float(_bars_ge5_ratio),
        "target_source": _target_source,
        "coverage_within_eligible": float(_cov_within_elig),
    }
    _is_rank_ic = float(quality_report.get("spearman_rank_ic", 0.0))
    _valid_rank_ic = float(quality_report.get("ranker_valid_ndcg_at_5", 0.0))
    _test_rank_ic = float(_oos_ic_stats.get("mean_ic", 0.0))
    _retention_ratio = float(
        _test_rank_ic / max(abs(_is_rank_ic), 1e-12) if abs(_is_rank_ic) > 0.0 else 0.0
    )
    # 게이트를 OOS-only IC 기준으로 교체: 전체 패널 IC(IS+OOS)는 vrefit 구간 노이즈에 취약.
    # is_rank_ic(alpha_long-short 전체 패널)는 diagnostic 용도로만 유지.
    quality_report["full_panel_alpha_ic"] = _is_rank_ic
    quality_report["spearman_rank_ic"] = _test_rank_ic
    quality_report["fold_oos_ic"] = _test_rank_ic
    # breakeven IC: cost24bps / (sigma500bps * sqrt(breadth)); breadth ≈ cofinite_p50
    _be_ic = 24.0 / (500.0 * max(_cofinite_p50, 1.0) ** 0.5)
    _ic_gap_oos = _test_rank_ic - _be_ic

    # --- Phase 0 anti-bias diagnostic: beta-residualized IC + effective-breadth breakeven ---
    # Observation-only (gate unchanged). Resolves whether raw dense_ranker_ic is tradable
    # market-neutral alpha or a beta/size factor tilt hedged away by the L/S book.
    # See docs/specs/ml_system_integrity_and_evaluation.md §2 (C1 + C2).
    _resid_decomp: dict[str, float] = {}
    _be_eff = float("nan")
    _n_eff = float("nan")
    _gap_resid_eff = float("nan")
    if _oos_ge5_mask.any():
        _beta_oos = (
            labels.beta_2d[_oos_ge5_mask].astype(np.float64)
            if labels.beta_2d is not None
            else None
        )
        _mkt_fwd_oos = (
            labels.market_fwd_ret[_oos_ge5_mask].astype(np.float64)
            if labels.market_fwd_ret is not None
            else None
        )
        _resid_decomp = diagnose_alpha_ic_decomposition(
            pred_dense_2d=_score_grid_f[_oos_ge5_mask],
            realized_raw_2d=_ic_target[_oos_ge5_mask],
            beta_2d=_beta_oos,
            market_fwd_1d=_mkt_fwd_oos,
            horizon_bars=int(ml_cfg.label_horizon_bars),
        )
        # Effective (correlation-adjusted) breadth from realized cross-section comovement.
        _n_eff = effective_breadth_corr(_ic_target[_oos_ge5_mask])
        _tgt_oos_flat = _ic_target[_oos_ge5_mask]
        _tgt_oos_finite_vals = _tgt_oos_flat[np.isfinite(_tgt_oos_flat)]
        _sigma_r_bps_oos = (
            float(np.nanstd(_tgt_oos_finite_vals)) * 1e4
            if _tgt_oos_finite_vals.size > 0
            else 400.0
        )
        _cost_bps_oos = round_trip_cost_bps()
        _be_eff = _cost_bps_oos / (
            max(_sigma_r_bps_oos, 1e-6) * max(_n_eff, 1.0) ** 0.5
        )
        _resid_ic = float(_resid_decomp.get("dense_c1_resid_ic", float("nan")))
        _gap_resid_eff = _resid_ic - _be_eff
        _logger.info(
            "🔬 [RESID-IC] raw=%.4f resid=%.4f resid_hit=%.3f",
            float(_resid_decomp.get("dense_c1_raw_ic", float("nan"))),
            _resid_ic,
            float(_resid_decomp.get("dense_c1_resid_hit", float("nan"))),
        )
        _logger.info(
            "🔬 [BE-EFF] N_raw=%.1f N_eff=%.1f sigma_r=%.1fbps be_raw=%.4f be_eff=%.4f"
            " gap_resid_eff=%+.4f",
            float(_cofinite_p50),
            float(_n_eff),
            float(_sigma_r_bps_oos),
            float(_be_ic),
            float(_be_eff),
            float(_gap_resid_eff),
        )
    panel.attrs["oos_resid_ic_decomp"] = {
        **_resid_decomp,
        "n_eff": float(_n_eff),
        "be_eff": float(_be_eff),
        "gap_resid_eff": float(_gap_resid_eff),
    }
    _feature_ic_audit: list[dict[str, float | str]] = []
    if _oos_ge5_mask.any():
        _feature_ic_audit = feature_cs_ic_audit(
            features.values[_oos_ge5_mask].astype(np.float64),
            tuple(features.feature_names),
            _ic_target[_oos_ge5_mask],
            breakeven_ic=float(_be_eff) if math.isfinite(_be_eff) else float(_be_ic),
            horizon_bars=int(ml_cfg.label_horizon_bars),
            top_k=15,
        )
    panel.attrs["feature_ic_audit"] = _feature_ic_audit
    if _feature_ic_audit:
        _feature_ic_msg = " | ".join(
            f"{row['name']}:ic={float(row['mean_ic']):.4f},gap={float(row['gap']):.4f}"
            for row in _feature_ic_audit
        )
        _logger.info("🔬 [FEATURE-IC] %s", _feature_ic_msg)
    else:
        _logger.info("🔬 [FEATURE-IC] no_oos_bars_with_cofinite_ge5")
    if not ml_cfg.ranker_enabled:
        _decision = "not_measured"
    elif (
        _test_rank_ic >= 0.015
        and float(_oos_ic_stats.get("t_stat", 0.0)) >= 2.0
        and _retention_ratio >= 0.50
    ):
        _decision = "continue"
    elif _test_rank_ic < 0.005 or float(_oos_ic_stats.get("t_stat", 0.0)) < 1.0:
        _decision = "no_edge"
    else:
        _decision = "continue"
    _be_ic_for_report = float(_be_eff) if math.isfinite(_be_eff) else float(_be_ic)
    panel.attrs["generalization_report"] = {
        "is_rank_ic": _is_rank_ic,
        "valid_rank_ic": _valid_rank_ic,
        "test_rank_ic": _test_rank_ic,
        "oos_rank_ic": _test_rank_ic,
        "retention_ratio": _retention_ratio,
        "decision": _decision,
        "be_ic": _be_ic_for_report,
        "ic_gap": _test_rank_ic - _be_ic_for_report,
    }
    if not ml_cfg.ranker_enabled:
        # NDCG is N/A without ranker; mark as passing to avoid false gate failure
        quality_report["ranker_valid_ndcg_at_5"] = 1.0
    valid_stack_long = (
        np.concatenate(valid_ev_long_all)
        if valid_ev_long_all
        else np.zeros((0,), dtype=np.float32)
    )
    valid_stack_short = (
        np.concatenate(valid_ev_short_all)
        if valid_ev_short_all
        else np.zeros((0,), dtype=np.float32)
    )
    valid_alpha = (
        np.concatenate([valid_stack_long, valid_stack_short])
        if (valid_stack_long.size + valid_stack_short.size) > 0
        else np.zeros((0,), dtype=np.float32)
    )
    if "in_fold_valid_alpha_p95_bps" not in quality_report:
        quality_report["in_fold_valid_alpha_p95_bps"] = (
            float(np.percentile(np.abs(valid_alpha) * 1e4, 95)) if valid_alpha.size > 0 else 0.0
        )
    cost_floor = (
        round_trip_cost_bps() + float(default_ev_hurdle_bps(OPT_FUTURES_CONFIG))
    ) / 10000.0
    xs_long_proxy = np.where(alpha_long_final >= cost_floor, alpha_long_final, 0.0)
    xs_short_proxy = np.where(alpha_short_final >= cost_floor, alpha_short_final, 0.0)
    xs_long_preservation = preservation_ratio(alpha_long_final, xs_long_proxy)
    xs_short_preservation = preservation_ratio(alpha_short_final, xs_short_proxy)
    quality_report.update(
        side_alpha_tail_metrics(
            alpha_long_final,
            alpha_short_final,
            cost_floor=cost_floor,
        )
    )
    quality_report["alpha_full_matrix_p95_bps"] = float(quality_report.get("alpha_p95_bps", 0.0))
    quality_report["xs_long_preservation_ratio"] = xs_long_preservation
    quality_report["xs_short_preservation_ratio"] = xs_short_preservation
    _friction_bps = round_trip_cost_bps()
    _hurdle_default_bps = float(default_ev_hurdle_bps(OPT_FUTURES_CONFIG))
    alpha_diag = alpha_gate_diagnostics(
        alpha_p95_bps=float(quality_report.get("alpha_p95_bps", 0.0)),
        friction_bps=float(_friction_bps),
        hurdle_bps=float(_hurdle_default_bps),
        long_nz=float(quality_report.get("alpha_long_non_zero_ratio", 0.0)),
        short_nz=float(quality_report.get("alpha_short_non_zero_ratio", 0.0)),
        xs_long_preservation_ratio=xs_long_preservation,
        xs_short_preservation_ratio=xs_short_preservation,
        min_long_nz=ml_cfg.alpha_gate_min_long_nz,
        min_short_nz=ml_cfg.alpha_gate_min_short_nz,
        min_xs_preservation=ml_cfg.alpha_gate_min_xs_preservation,
        cost_wall_tolerance_bps=ml_cfg.alpha_gate_cost_wall_tolerance_bps,
        active_alpha_p95_bps=float(quality_report.get("alpha_active_p95_bps", 0.0)),
        tradable_long_nz=float(quality_report.get("alpha_long_tradable_nz", 0.0)),
        tradable_short_nz=float(quality_report.get("alpha_short_tradable_nz", 0.0)),
        min_tradable_long_nz=ml_cfg.alpha_gate_min_tradable_long_nz,
        min_tradable_short_nz=ml_cfg.alpha_gate_min_tradable_short_nz,
    )
    quality_report.update(alpha_diag)
    panel.attrs["quality_report"] = quality_report
    if not bool(quality_report.get("alpha_gate_pass", False)):
        raise RuntimeError(
            "strategy ml alpha gate failed: "
            f"reasons={quality_report.get('alpha_gate_fail_reasons', [])} "
            f"alpha_gate_metric_bps={quality_report.get('alpha_gate_metric_bps', 0.0):.2f} "
            f"alpha_full_matrix_p95_bps={quality_report.get('alpha_full_matrix_p95_bps', 0.0):.2f} "
            f"floor_bps={quality_report.get('alpha_gate_floor_bps', 0.0):.2f} "
            f"long_nz={quality_report.get('alpha_long_non_zero_ratio', 0.0):.4f} "
            f"short_nz={quality_report.get('alpha_short_non_zero_ratio', 0.0):.4f} "
            f"xs_long_preservation={quality_report.get('xs_long_preservation_ratio', 0.0):.4f} "
            f"xs_short_preservation={quality_report.get('xs_short_preservation_ratio', 0.0):.4f}"
        )
    if not passes_quality_gate(quality_report):
        failed_keys = {
            k: v
            for k, v in quality_report.items()
            if (k == "feature_finite_ratio" and v < 0.990)
            or (k == "label_valid_ratio" and v <= 0.0)
            or (k == "ranker_valid_ndcg_at_5" and v <= 0.0)
            or (k == "spearman_rank_ic" and v < 0.0)
        }
        raise RuntimeError(
            f"strategy ml quality gate failed: reasons={failed_keys} full={quality_report}"
        )
    if float(np.count_nonzero(panel["alpha_long"].to_numpy(dtype=np.float64))) <= 0.0:
        raise RuntimeError("generated alpha_long is all zero")
    if float(np.count_nonzero(panel["alpha_short"].to_numpy(dtype=np.float64))) <= 0.0:
        raise RuntimeError("generated alpha_short is all zero")
    metrics = ml_alpha_metrics(
        panel["alpha_long"].to_numpy(dtype=np.float64).reshape(-1, 1),
        panel["alpha_short"].to_numpy(dtype=np.float64).reshape(-1, 1),
    )
    _logger.info(
        "📊 ML_EVAL: nz(L=%.3f S=%.3f) ic=%.4f t=%.2f hit=%.3f obs=%d",
        metrics["long_nz"],
        metrics["short_nz"],
        quality_report.get("spearman_rank_ic", 0.0),
        quality_report.get("ic_t_stat", 0.0),
        quality_report.get("ic_hit_ratio", 0.0),
        int(quality_report.get("ic_n_obs", 0)),
    )
    t_grid_assembly += time.perf_counter() - t_grid_start

    # Print Granular Performance Profile (Detailed Profiler Output)
    t_total_elapsed = time.perf_counter() - t_start_total
    if not cfg.ml.horizon_experiment_enabled:
        _logger.debug("=" * 60)
        _logger.debug(" [ML-DETAILED-PROFILE] Granular Execution Performance Profiling")
        _logger.debug(
            "  1. Align & Feat/Label Gen: %7.2f ms (%5.1f%%)",
            t_align_feature_label * 1000.0,
            (t_align_feature_label / t_total_elapsed) * 100.0,
        )
        _logger.debug(
            "  2. Fold Split & Preprocess: %7.2f ms (%5.1f%%)",
            t_fold_split_preprocess * 1000.0,
            (t_fold_split_preprocess / t_total_elapsed) * 100.0,
        )
        _logger.debug(
            "  3. Matrix Dataset Build   : %7.2f ms (%5.1f%%)",
            t_dataset_build * 1000.0,
            (t_dataset_build / t_total_elapsed) * 100.0,
        )
        _logger.debug(
            "  4. LightGBM Fold Train/Prd: %7.2f ms (%5.1f%%)",
            t_fit_predict_fold * 1000.0,
            (t_fit_predict_fold / t_total_elapsed) * 100.0,
        )
        _logger.debug(
            "  5. OOS Grid & Assembly    : %7.2f ms (%5.1f%%)",
            t_grid_assembly * 1000.0,
            (t_grid_assembly / t_total_elapsed) * 100.0,
        )
        _logger.debug("  * Total Pipeline Execution: %7.2f ms", t_total_elapsed * 1000.0)
        _logger.debug("=" * 60)

    # Cost wall diagnosis: compact summary
    _floor_bps = _friction_bps + _hurdle_default_bps
    _gate_metric_bps = float(
        quality_report.get(
            "alpha_gate_metric_bps",
            quality_report.get("alpha_active_p95_bps", quality_report.get("alpha_p95_bps", 0.0)),
        )
    )
    _logger.info(
        "💰 ML_COST: gate=%.1fbps floor=%.1fbps pass=%s",
        _gate_metric_bps,
        _floor_bps,
        str(_gate_metric_bps >= _floor_bps).lower(),
    )
    # B4: IC gate — config-driven 임계값으로 통계적 유의성 검사
    _ic_pass = passes_ic_gate(
        quality_report,
        min_mean_ic=ml_cfg.ic_gate_min_mean_ic,
        min_t_stat=ml_cfg.ic_gate_min_t_stat,
        min_hit_ratio=ml_cfg.ic_gate_min_hit_ratio,
    )
    if not _ic_pass:
        if ml_cfg.ic_gate_warn_only:
            _logger.warning(
                "[ML-IC-GATE] IC gate WARN: mean_ic=%.4f t_stat=%.2f hit_ratio=%.3f",
                quality_report.get("spearman_rank_ic", 0.0),
                quality_report.get("ic_t_stat", 0.0),
                quality_report.get("ic_hit_ratio", 0.0),
            )
        else:
            raise RuntimeError(
                f"[ML-IC-GATE] IC gate failed: mean_ic="
                f"{quality_report.get('spearman_rank_ic', 0.0):.4f} "
                f"t_stat={quality_report.get('ic_t_stat', 0.0):.2f} "
                f"hit_ratio={quality_report.get('ic_hit_ratio', 0.0):.3f}"
            )
    _idx = pd.MultiIndex.from_product(
        [features.datetimes, features.symbols], names=["datetime", "symbol"]
    )
    if ml_cfg.regime_gate_enabled:
        _btc_ser = _btc_close_from_data_maps(data_maps, tf)
        alpha_long_final, alpha_short_final = apply_regime_gate(
            alpha_long_final, alpha_short_final, features.datetimes, _btc_ser, ml_cfg
        )
        panel.loc[:, "alpha_long"] = pd.Series(alpha_long_final.reshape(-1), index=_idx)
        panel.loc[:, "alpha_short"] = pd.Series(alpha_short_final.reshape(-1), index=_idx)
    # P1: trading_symbols 마스킹 — 학습 패널(training_panel) 중 Stage6 미포함 심볼 거래 차단
    _effective_trading = trading_symbols if trading_symbols else ml_cfg.trading_symbols
    if _effective_trading:
        _trading_set = set(_effective_trading)
        for _i, _sym in enumerate(features.symbols):
            if _sym not in _trading_set:
                alpha_long_final[:, _i] = 0.0
                alpha_short_final[:, _i] = 0.0
        panel.loc[:, "alpha_long"] = pd.Series(alpha_long_final.reshape(-1), index=_idx)
        panel.loc[:, "alpha_short"] = pd.Series(alpha_short_final.reshape(-1), index=_idx)
    return panel


def build_ml_strategy_alpha_anchored(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
    anchor_end_idx: int,
    target_start: int,
    target_end: int,
    precomputed_panels: AnchoredMLPrecomputedPanels | None = None,
    trading_symbols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Single anchored-pass ML alpha.

    Trains on [0, anchor_end_idx), infers on [target_start, target_end).
    """
    from dataclasses import replace

    from src.domain.futures.strategy.contracts import FoldSpec

    t0_total = time.perf_counter()
    ml_cfg = cfg.ml
    t_feature_label = time.perf_counter()
    if precomputed_panels is None:
        precomputed_panels = precompute_anchored_ml_panels(data_maps, symbols, tf, cfg)
    features = precomputed_panels.features
    labels = precomputed_panels.labels
    feature_label_elapsed = time.perf_counter() - t_feature_label

    t_size = features.values.shape[0]
    anchor_end = int(np.clip(anchor_end_idx, 0, t_size))
    tgt_start = int(np.clip(target_start, 0, t_size))
    tgt_end = int(np.clip(target_end, 0, t_size))
    if anchor_end < 32:
        raise RuntimeError(
            f"anchored refit: anchor_end={anchor_end} too small (< 32 bars); cannot train"
        )
    if tgt_end <= tgt_start:
        raise RuntimeError(f"anchored refit: empty target window [{tgt_start}, {tgt_end})")

    idx = pd.to_datetime(features.datetimes[:anchor_end])
    if idx.size > 1:
        total_anchor_months = max(1.0, (idx[-1] - idx[0]).days / 30.4375)
        bars_per_month = anchor_end / total_anchor_months
    else:
        bars_per_month = anchor_end / max(1, ml_cfg.train_months + ml_cfg.valid_months)
    valid_bars = max(8, int(ml_cfg.valid_months * bars_per_month))
    train_end = max(32, anchor_end - valid_bars)
    valid_start = train_end
    valid_end = anchor_end

    _purged_valid_start = min(valid_start + ml_cfg.purge_bars, max(valid_start, valid_end - 1))
    _embargoed_test_start = min(tgt_start + ml_cfg.embargo_bars, max(tgt_start, tgt_end - 1))
    fold = FoldSpec(
        fold_id=0,
        train_start=0,
        train_end=train_end,
        valid_start=_purged_valid_start,
        valid_end=valid_end,
        test_start=_embargoed_test_start,
        test_end=tgt_end,
        purge_bars=ml_cfg.purge_bars,
        embargo_bars=ml_cfg.embargo_bars,
    )
    _logger.info(
        "[ML-ANCHORED] anchor_end=%d train=[0,%d) valid=[%d,%d) target=[%d,%d)",
        anchor_end,
        train_end,
        valid_start,
        valid_end,
        tgt_start,
        tgt_end,
    )

    train_values = features.values[fold.train_start : fold.train_end].astype(np.float64, copy=False)
    bounds = fit_robust_bounds(train_values, clip_quantile=0.995)
    clipped_values = apply_robust_bounds(features.values.astype(np.float64, copy=False), bounds)
    imputer = fit_missing_value_imputer(train_values)
    normalized = apply_missing_value_imputer(clipped_values, imputer).astype(np.float32, copy=False)
    normalized_features = FeaturePanel(
        datetimes=features.datetimes,
        symbols=features.symbols,
        values=normalized,
        feature_names=features.feature_names,
        valid_mask=features.valid_mask,
        availability_masks=features.availability_masks,
        metadata={
            **features.metadata,
            "train_imputer_applied": True,
            "missing_imputer": "train_median",
        },
    )

    t_matrix = time.perf_counter()
    (
        long_rank_target,
        short_rank_target,
        long_rel,
        short_rel,
        long_mag,
        short_mag,
    ) = _resolve_side_targets(labels, ml_cfg)

    train_long = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.train_start,
        end=fold.train_end,
        fold=fold,
        split="train",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=long_rank_target,
        relevance_override=long_rel,
        ev_target_override=long_mag,
    )
    # [ML-UPGRADE] Thin-Data Guard (Dynamic Regularization)
    n_train_rows = int(train_long.X.shape[0])
    if n_train_rows < 20_000:
        ml_cfg = replace(
            ml_cfg,
            ranker_n_estimators=min(ml_cfg.ranker_n_estimators, 400),
            num_leaves=min(ml_cfg.num_leaves, 15),
            min_data_in_leaf=max(ml_cfg.min_data_in_leaf, 60),
        )
        _logger.info(
            "[ML-ANCHORED] Thin-data guard active: rows=%d -> trees=%d, leaves=%d, min_leaf=%d",
            n_train_rows,
            ml_cfg.ranker_n_estimators,
            ml_cfg.num_leaves,
            ml_cfg.min_data_in_leaf,
        )
    valid_long = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.valid_start,
        end=fold.valid_end,
        fold=fold,
        split="valid",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=long_rank_target,
        relevance_override=long_rel,
        ev_target_override=long_mag,
    )
    test_long = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.test_start,
        end=fold.test_end,
        fold=fold,
        split="test",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=long_rank_target,
        relevance_override=long_rel,
        ev_target_override=long_mag,
    )
    train_short = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.train_start,
        end=fold.train_end,
        fold=fold,
        split="train",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=short_rank_target,
        relevance_override=short_rel,
        ev_target_override=short_mag,
    )
    valid_short = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.valid_start,
        end=fold.valid_end,
        fold=fold,
        split="valid",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=short_rank_target,
        relevance_override=short_rel,
        ev_target_override=short_mag,
    )
    test_short = _build_side_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.test_start,
        end=fold.test_end,
        fold=fold,
        split="test",
        min_group_size=ml_cfg.min_group_size,
        rank_target_override=short_rank_target,
        relevance_override=short_rel,
        ev_target_override=short_mag,
    )
    validate_long_matrix(train_long)
    validate_long_matrix(valid_long)
    validate_long_matrix(test_long)
    validate_long_matrix(train_short)
    validate_long_matrix(valid_short)
    validate_long_matrix(test_short)
    matrix_elapsed = time.perf_counter() - t_matrix

    t_fit_predict = time.perf_counter()
    _awf_result = _fit_predict_fold_dual_side(
        train_long=train_long,
        valid_long=valid_long,
        test_long=test_long,
        train_short=train_short,
        valid_short=valid_short,
        test_short=test_short,
        ml_cfg=ml_cfg,
    )
    ev_test_long = _awf_result.ev_test_long
    ev_test_short = _awf_result.ev_test_short
    quant_test_long = _awf_result.quant_test_long
    quant_test_short = _awf_result.quant_test_short
    conf_test_long = _awf_result.conf_test_long
    conf_test_short = _awf_result.conf_test_short
    score_test_long = _awf_result.score_test_long
    score_test_short = _awf_result.score_test_short
    fit_predict_elapsed = time.perf_counter() - t_fit_predict
    calib_elapsed = 0.0  # Absorbed into _fit_predict_fold_dual_side
    _logger.info(
        "[ML-ANCHORED-RANKER] train_n=%d valid_n=%d test_n=%d",
        int(train_long.X.shape[0]),
        int(valid_long.X.shape[0]),
        int(test_long.X.shape[0]),
    )
    _logger.info(
        "[ML-ANCHORED-SCORE-SPLIT] ev_mean=%.6e ev_p10=%.6e ev_p90=%.6e",
        float(np.mean(ev_test_long, dtype=np.float32)) if ev_test_long.size > 0 else 0.0,
        float(np.percentile(ev_test_long, 10)) if ev_test_long.size > 0 else 0.0,
        float(np.percentile(ev_test_long, 90)) if ev_test_long.size > 0 else 0.0,
    )
    ev_long_grid = np.zeros((t_size, features.values.shape[1]), dtype=np.float32)
    ev_short_grid = np.zeros((t_size, features.values.shape[1]), dtype=np.float32)
    q10_long_grid = np.zeros_like(ev_long_grid)
    q50_long_grid = np.zeros_like(ev_long_grid)
    q90_long_grid = np.zeros_like(ev_long_grid)
    q10_short_grid = np.zeros_like(ev_long_grid)
    q50_short_grid = np.zeros_like(ev_long_grid)
    q90_short_grid = np.zeros_like(ev_long_grid)
    confidence_long_grid = np.zeros_like(ev_long_grid)
    confidence_short_grid = np.zeros_like(ev_long_grid)
    score_grid = np.full((t_size, features.values.shape[1]), np.nan, dtype=np.float32)
    rank_score_long_grid_awf = np.full_like(score_grid, np.nan, dtype=np.float32)
    rank_score_short_grid_awf = np.full_like(score_grid, np.nan, dtype=np.float32)
    for row, (t_idx, s_idx) in enumerate(test_long.index_map):
        q10_long_grid[int(t_idx), int(s_idx)] = quant_test_long.q10[row]
        q50_long_grid[int(t_idx), int(s_idx)] = quant_test_long.q50[row]
        q90_long_grid[int(t_idx), int(s_idx)] = quant_test_long.q90[row]
        confidence_long_grid[int(t_idx), int(s_idx)] = conf_test_long[row]
        ev_long_grid[int(t_idx), int(s_idx)] = np.float32(max(ev_test_long[row], 0.0))
        score_grid[int(t_idx), int(s_idx)] = score_test_long[row]
        rank_score_long_grid_awf[int(t_idx), int(s_idx)] = score_test_long[row]
    for row, (t_idx, s_idx) in enumerate(test_short.index_map):
        q10_short_grid[int(t_idx), int(s_idx)] = quant_test_short.q10[row]
        q50_short_grid[int(t_idx), int(s_idx)] = quant_test_short.q50[row]
        q90_short_grid[int(t_idx), int(s_idx)] = quant_test_short.q90[row]
        confidence_short_grid[int(t_idx), int(s_idx)] = conf_test_short[row]
        ev_short_grid[int(t_idx), int(s_idx)] = np.float32(max(ev_test_short[row], 0.0))
        rank_score_short_grid_awf[int(t_idx), int(s_idx)] = score_test_short[row]
        if np.isnan(score_grid[int(t_idx), int(s_idx)]):
            score_grid[int(t_idx), int(s_idx)] = score_test_short[row]

    clip_lim_awf = float(ml_cfg.alpha_clip_bps / 10000.0)
    eligible_2d_awf = labels.eligible_mask
    _emit_mode_awf = str(getattr(ml_cfg, "alpha_emit_mode", "rank_sized"))
    if _emit_mode_awf == "rank_sized":
        if bool(getattr(ml_cfg, "rank_policy_enabled", True)):
            valid_score_long_grid_awf = np.full_like(rank_score_long_grid_awf, np.nan, dtype=np.float64)
            valid_score_short_grid_awf = np.full_like(rank_score_short_grid_awf, np.nan, dtype=np.float64)
            for row, (t_idx, s_idx) in enumerate(valid_long.index_map):
                valid_score_long_grid_awf[int(t_idx), int(s_idx)] = _awf_result.score_valid_long[row]
            for row, (t_idx, s_idx) in enumerate(valid_short.index_map):
                valid_score_short_grid_awf[int(t_idx), int(s_idx)] = _awf_result.score_valid_short[row]
            valid_signed_awf = derive_signed_rank_signal(
                valid_score_long_grid_awf,
                valid_score_short_grid_awf,
            )
            hold_awf = int(ml_cfg.label_horizon_bars)
            if hold_awf not in tuple(int(h) for h in ml_cfg.rank_policy_holding_candidates):
                hold_awf = int(ml_cfg.rank_policy_holding_candidates[0])
            awf_policy = calibrate_rank_selection_policy(
                signed_score_2d=valid_signed_awf,
                realized_fwd_ret_2d=labels.exec_net_ret.astype(np.float64),
                eligible_2d=labels.eligible_mask.astype(bool),
                quantiles=tuple(float(q) for q in ml_cfg.rank_policy_quantiles),
                min_abs_z_grid=tuple(float(z) for z in ml_cfg.rank_policy_min_abs_z_grid),
                holding_bars=hold_awf,
                cost_bps=24.0,
                min_obs=int(ml_cfg.rank_policy_min_validation_obs),
                weight_k=float(ml_cfg.alpha_emit_weight_k),
                weighting=ml_cfg.rank_policy_weighting,
            )
            awf_signed = derive_signed_rank_signal(
                rank_score_long_grid_awf.astype(np.float64),
                rank_score_short_grid_awf.astype(np.float64),
            )
            alpha_long_final_awf, alpha_short_final_awf = apply_rank_selection_policy(
                signed_score_2d=awf_signed,
                eligible_2d=eligible_2d_awf,
                policy=awf_policy,
            )
        else:
            alpha_long_final_awf, alpha_short_final_awf = _emit_rank_sized_alpha(
                rank_score_long_grid_awf,
                rank_score_short_grid_awf,
                eligible_2d_awf,
                select_q=float(getattr(ml_cfg, "alpha_emit_select_q", 0.40)),
                weight_k=float(getattr(ml_cfg, "alpha_emit_weight_k", 3.0)),
                clip_lim=1.0,  # tanh weights가 (-1,+1) 범위이므로 continuous 보존
            )
    else:
        alpha_long_final_awf = np.where(
            eligible_2d_awf,
            np.clip(np.maximum(ev_long_grid, 0.0), 0.0, clip_lim_awf),
            0.0,
        ).astype(np.float32, copy=False)
        alpha_short_final_awf = np.where(
            eligible_2d_awf,
            np.clip(np.maximum(ev_short_grid, 0.0), 0.0, clip_lim_awf),
            0.0,
        ).astype(np.float32, copy=False)
    alpha_ic_score_awf = (alpha_long_final_awf - alpha_short_final_awf).astype(
        np.float64, copy=False
    )

    _idx = pd.MultiIndex.from_product(
        [features.datetimes, features.symbols], names=["datetime", "symbol"]
    )
    panel = pd.DataFrame(
        {
            "alpha_long": alpha_long_final_awf.reshape(-1),
            "alpha_short": alpha_short_final_awf.reshape(-1),
            "rank_score_long": rank_score_long_grid_awf.reshape(-1),
            "rank_score_short": rank_score_short_grid_awf.reshape(-1),
        },
        index=_idx,
    ).sort_index()
    from src.domain.futures.strategy.contracts import ALPHA_FORECAST_CONTRACT
    panel.attrs["forecast_contract_version"] = ALPHA_FORECAST_CONTRACT
    panel.attrs["alpha_forecast_metadata"] = {
        "q10_long": q10_long_grid.reshape(-1),
        "q50_long": q50_long_grid.reshape(-1),
        "q90_long": q90_long_grid.reshape(-1),
        "q10_short": q10_short_grid.reshape(-1),
        "q50_short": q50_short_grid.reshape(-1),
        "q90_short": q90_short_grid.reshape(-1),
        "confidence_long": confidence_long_grid.reshape(-1),
        "confidence_short": confidence_short_grid.reshape(-1),
        "rank_score_long": rank_score_long_grid_awf.reshape(-1),
        "rank_score_short": rank_score_short_grid_awf.reshape(-1),
    }
    panel.attrs["rank_score_contract"] = {
        "version": 1,
        "mode": "signed_single_ranker",
        "long_higher_is_better": True,
        "short_lower_is_better": True,
        "signed_signal_formula": "derive_signed_rank_signal(rank_score_long, rank_score_short)",
    }
    if (
        _emit_mode_awf == "rank_sized"
        and bool(getattr(ml_cfg, "rank_policy_enabled", True))
        and "awf_policy" in locals()
    ):
        panel.attrs["rank_selection_policy"] = policy_to_dict(awf_policy)
        panel.attrs["rank_selection_policy_by_fold"] = [policy_to_dict(awf_policy)]

    # AWF quality/IC gates — warn-only モード (Optuna 최적화 중단 방지)
    # WF 경로와 동일하게 build_quality_report()로 실제 IC/NDCG/alpha_p95 계산
    awf_quality_report: dict[str, Any] = build_quality_report(
        feature_values=normalized_features.values,
        feature_valid_mask=normalized_features.valid_mask,
        label_eligible_mask=labels.eligible_mask,
        score_2d=score_grid,
        signed_ret_2d=labels.signed_net_ret.astype(np.float64),
        relevance_2d=labels.relevance.astype(np.float64),
        alpha_long_2d=alpha_long_final_awf,
        alpha_short_2d=alpha_short_final_awf,
        ic_score_2d=alpha_ic_score_awf,
    )
    if not ml_cfg.ranker_enabled:
        # NDCG is N/A without ranker; mark as passing to avoid false gate failure
        awf_quality_report["ranker_valid_ndcg_at_5"] = 1.0
    _awf_valid_stack = (
        np.concatenate([_awf_result.ev_valid_long, _awf_result.ev_valid_short])
        if (_awf_result.ev_valid_long.size + _awf_result.ev_valid_short.size) > 0
        else np.zeros((0,), dtype=np.float32)
    )
    if "in_fold_valid_alpha_p95_bps" not in awf_quality_report:
        awf_quality_report["in_fold_valid_alpha_p95_bps"] = (
            float(np.percentile(np.abs(_awf_valid_stack) * 1e4, 95))
            if _awf_valid_stack.size > 0
            else 0.0
        )
    cost_floor_awf = (
        round_trip_cost_bps() + float(default_ev_hurdle_bps(OPT_FUTURES_CONFIG))
    ) / 10000.0
    xs_long_proxy_awf = np.where(
        alpha_long_final_awf >= cost_floor_awf,
        alpha_long_final_awf,
        0.0,
    )
    xs_short_proxy_awf = np.where(
        alpha_short_final_awf >= cost_floor_awf,
        alpha_short_final_awf,
        0.0,
    )
    xs_long_pres_awf = preservation_ratio(alpha_long_final_awf, xs_long_proxy_awf)
    xs_short_pres_awf = preservation_ratio(alpha_short_final_awf, xs_short_proxy_awf)
    awf_quality_report.update(
        side_alpha_tail_metrics(
            alpha_long_final_awf,
            alpha_short_final_awf,
            cost_floor=cost_floor_awf,
        )
    )
    awf_quality_report["alpha_full_matrix_p95_bps"] = float(
        awf_quality_report.get("alpha_p95_bps", 0.0)
    )
    awf_quality_report["xs_long_preservation_ratio"] = xs_long_pres_awf
    awf_quality_report["xs_short_preservation_ratio"] = xs_short_pres_awf
    _awf_friction_bps = round_trip_cost_bps()
    _awf_hurdle_bps = float(default_ev_hurdle_bps(OPT_FUTURES_CONFIG))
    alpha_diag_awf = alpha_gate_diagnostics(
        alpha_p95_bps=float(awf_quality_report.get("alpha_p95_bps", 0.0)),
        friction_bps=float(_awf_friction_bps),
        hurdle_bps=float(_awf_hurdle_bps),
        long_nz=float(awf_quality_report.get("alpha_long_non_zero_ratio", 0.0)),
        short_nz=float(awf_quality_report.get("alpha_short_non_zero_ratio", 0.0)),
        xs_long_preservation_ratio=float(
            awf_quality_report.get("xs_long_preservation_ratio", 0.0)
        ),
        xs_short_preservation_ratio=float(
            awf_quality_report.get("xs_short_preservation_ratio", 0.0)
        ),
        min_long_nz=ml_cfg.alpha_gate_min_long_nz,
        min_short_nz=ml_cfg.alpha_gate_min_short_nz,
        min_xs_preservation=ml_cfg.alpha_gate_min_xs_preservation,
        cost_wall_tolerance_bps=ml_cfg.alpha_gate_cost_wall_tolerance_bps,
        active_alpha_p95_bps=float(awf_quality_report.get("alpha_active_p95_bps", 0.0)),
        tradable_long_nz=float(awf_quality_report.get("alpha_long_tradable_nz", 0.0)),
        tradable_short_nz=float(awf_quality_report.get("alpha_short_tradable_nz", 0.0)),
        min_tradable_long_nz=ml_cfg.alpha_gate_min_tradable_long_nz,
        min_tradable_short_nz=ml_cfg.alpha_gate_min_tradable_short_nz,
    )
    awf_quality_report.update(alpha_diag_awf)
    if not bool(awf_quality_report.get("alpha_gate_pass", False)):
        _logger.warning(
            "[AWF-ALPHA-GATE] alpha gate WARN: reasons=%s "
            "long_nz=%.4f short_nz=%.4f "
            "xs_long_pres=%.4f xs_short_pres=%.4f",
            awf_quality_report.get("alpha_gate_fail_reasons", []),
            float(awf_quality_report.get("alpha_long_non_zero_ratio", 0.0)),
            float(awf_quality_report.get("alpha_short_non_zero_ratio", 0.0)),
            xs_long_pres_awf,
            xs_short_pres_awf,
        )
    if not passes_quality_gate(awf_quality_report):
        _failed_keys_awf = {
            k: v
            for k, v in awf_quality_report.items()
            if (k == "feature_finite_ratio" and v < 0.990)
            or (k == "label_valid_ratio" and v <= 0.0)
            or (k == "ranker_valid_ndcg_at_5" and v <= 0.0)
            or (k == "spearman_rank_ic" and v < 0.0)
        }
        _logger.warning(
            "[AWF-QUALITY-GATE] quality gate WARN: reasons=%s",
            _failed_keys_awf,
        )
    # IC gate: AWF는 항상 warn-only
    _awf_ic_pass = passes_ic_gate(
        awf_quality_report,
        min_mean_ic=ml_cfg.ic_gate_min_mean_ic,
        min_t_stat=ml_cfg.ic_gate_min_t_stat,
        min_hit_ratio=ml_cfg.ic_gate_min_hit_ratio,
    )
    if not _awf_ic_pass:
        _logger.warning(
            "[AWF-IC-GATE] IC gate WARN: mean_ic=%.4f t_stat=%.2f hit_ratio=%.3f",
            awf_quality_report.get("spearman_rank_ic", 0.0),
            awf_quality_report.get("ic_t_stat", 0.0),
            awf_quality_report.get("ic_hit_ratio", 0.0),
        )
    panel.attrs["quality_report"] = awf_quality_report

    from dataclasses import asdict

    from src.domain.futures.strategy.cache import build_manifest_hash

    total_elapsed = time.perf_counter() - t0_total
    _logger.info(
        (
            "[AWF-REFIT-PROF] total=%.2fs feature_label=%.2fs matrix=%.2fs "
            "fit_predict=%.2fs calibrator=%.2fs train_rows=%d valid_rows=%d "
            "test_rows=%d"
        ),
        total_elapsed,
        feature_label_elapsed,
        matrix_elapsed,
        fit_predict_elapsed,
        calib_elapsed,
        int(train_long.X.shape[0]),
        int(valid_long.X.shape[0]),
        int(test_long.X.shape[0]),
    )
    panel.attrs["strategy_name"] = cfg.name
    panel.attrs["feature_names"] = list(features.feature_names)
    panel.attrs["fold_count"] = 1
    panel.attrs["config_hash"] = build_manifest_hash(asdict(cfg.ml))
    panel.attrs["anchored"] = True
    panel.attrs["anchor_end_idx"] = anchor_end
    panel.attrs["target_range"] = (tgt_start, tgt_end)
    panel.attrs["model_family"] = "lightgbm_signed_lambdarank"
    panel.attrs["ranking_mode"] = "group_ndcg_signed"
    panel.attrs["alpha_contract"] = "return_unit_grinold_rank"
    panel.attrs["feature_groups_enabled"] = list(features.availability_masks.keys())
    _tgt_cells = max(1, (tgt_end - tgt_start) * features.values.shape[1])
    _tgt_long_slice = ev_long_grid[tgt_start:tgt_end]
    _tgt_short_slice = ev_short_grid[tgt_start:tgt_end]
    _logger.info(
        "[ML-ANCHORED] target_long_nz=%.4f target_short_nz=%.4f",
        float(np.count_nonzero(_tgt_long_slice) / _tgt_cells),
        float(np.count_nonzero(_tgt_short_slice) / _tgt_cells),
    )
    if ml_cfg.regime_gate_enabled:
        _btc_ser_awf = _btc_close_from_data_maps(data_maps, tf)
        alpha_long_final_awf, alpha_short_final_awf = apply_regime_gate(
            alpha_long_final_awf, alpha_short_final_awf, features.datetimes, _btc_ser_awf, ml_cfg
        )
        panel.loc[:, "alpha_long"] = alpha_long_final_awf.reshape(-1)
        panel.loc[:, "alpha_short"] = alpha_short_final_awf.reshape(-1)
    # P1: trading_symbols 마스킹 — 학습 패널(training_panel) 중 Stage6 미포함 심볼 거래 차단
    _effective_trading_awf = trading_symbols if trading_symbols else ml_cfg.trading_symbols
    if _effective_trading_awf:
        _trading_set_awf = set(_effective_trading_awf)
        for _i_awf, _sym_awf in enumerate(features.symbols):
            if _sym_awf not in _trading_set_awf:
                alpha_long_final_awf[:, _i_awf] = 0.0
                alpha_short_final_awf[:, _i_awf] = 0.0
        panel.loc[:, "alpha_long"] = alpha_long_final_awf.reshape(-1)
        panel.loc[:, "alpha_short"] = alpha_short_final_awf.reshape(-1)
    return panel
