from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from src.core.settings import round_trip_cost_bps
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.strategy.cache import build_manifest_hash
from src.domain.futures.strategy.calibrator import (
    fit_quantile_calibrators,
    predict_conservative_ev,
)
from src.domain.futures.strategy.common.alignment import align_data_maps
from src.domain.futures.strategy.common.normalization import (
    apply_robust_bounds,
    fit_robust_bounds,
)
from src.domain.futures.strategy.common.validation import (
    validate_feature_panel,
    validate_label_panel,
    validate_long_matrix,
)
from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy.contracts import FeaturePanel
from src.domain.futures.strategy.dataset import (
    build_long_matrix,
    make_walk_forward_folds,
)
from src.domain.futures.strategy.diagnostics import (
    build_quality_report,
    ml_alpha_metrics,
    passes_ic_gate,
    passes_quality_gate,
)
from src.domain.futures.strategy.features import build_feature_panel
from src.domain.futures.strategy.inference import (
    assemble_alpha_panel,
    infer_fold_alpha,
)
from src.domain.futures.strategy.labels import build_label_panel
from src.domain.futures.strategy.ranker import fit_ranker, predict_rank_score

_logger = logging.getLogger(__name__)


def build_ml_strategy_alpha(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """Build ML strategy alpha panel."""
    # [ML-UPGRADE] LightGBM 트리가 충분히 쪼개질 수 있도록
    # 런타임 규제를 완화하고 학습 능력을 제고합니다.
    from dataclasses import replace
    ml_cfg = replace(
        cfg.ml,
        lambda_l2=1.0,
        min_data_in_leaf=30,
        num_leaves=31,
    )
    aligned = align_data_maps(data_maps=data_maps, symbols=symbols, tf=tf)
    if len(aligned.symbols) < ml_cfg.min_group_size:
        raise ValueError(
            f"strategy needs >= {ml_cfg.min_group_size} symbols, got {len(aligned.symbols)}"
        )
    features = build_feature_panel(aligned, ml_cfg)
    labels = build_label_panel(aligned, ml_cfg)
    validate_feature_panel(features)
    validate_label_panel(labels, t=features.values.shape[0], n=features.values.shape[1])

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
                    "[ML-ADJUST] Dynamic train window adjustment: history covers %d months, "
                    "adjusting train_months from %d to %d (with 1-month safety margin) "
                    "to satisfy walk-forward layout.",
                    total_months, ml_cfg.train_months, adjusted_train
                )
                ml_cfg = replace(ml_cfg, train_months=adjusted_train)

    folds = make_walk_forward_folds(features.datetimes, ml_cfg)
    if not folds:
        raise RuntimeError("no walk-forward folds can be built")
    _logger.info(
        "[ML-FEATURE] rows=%d symbols=%d features=%d",
        features.values.shape[0],
        features.values.shape[1],
        features.values.shape[2],
    )
    _logger.info(
        "[ML-LABEL] eligible=%.4f sample_weight_mean=%.4f",
        float(np.mean(labels.eligible_mask)),
        (
            float(np.mean(labels.sample_weight[labels.eligible_mask]))
            if np.any(labels.eligible_mask)
            else 0.0
        ),
    )

    ev_grid = np.zeros((features.values.shape[0], features.values.shape[1]), dtype=np.float32)
    score_grid = np.full_like(ev_grid, np.nan, dtype=np.float32)
    for fold in folds:
        _logger.info(
            "[ML-FOLD] id=%d train=[%d,%d) valid=[%d,%d) test=[%d,%d)",
            fold.fold_id,
            fold.train_start,
            fold.train_end,
            fold.valid_start,
            fold.valid_end,
            fold.test_start,
            fold.test_end,
        )
        train_values = features.values[fold.train_start : fold.train_end].astype(
            np.float64, copy=False
        )
        bounds = fit_robust_bounds(train_values, clip_quantile=0.995)
        clipped_values = apply_robust_bounds(features.values.astype(np.float64, copy=False), bounds)
        train_medians = np.nanmedian(train_values.reshape(-1, train_values.shape[2]), axis=0)
        train_medians = np.where(np.isfinite(train_medians), train_medians, 0.0)
        normalized = np.where(
            np.isfinite(clipped_values),
            clipped_values,
            train_medians[np.newaxis, np.newaxis, :],
        ).astype(np.float32, copy=False)
        normalized_features = FeaturePanel(
            datetimes=features.datetimes,
            symbols=features.symbols,
            values=normalized,
            feature_names=features.feature_names,
            valid_mask=features.valid_mask,
            availability_masks=features.availability_masks,
            metadata={**features.metadata, "train_imputer_applied": True},
        )
        train = build_long_matrix(
            features=normalized_features,
            labels=labels,
            start=fold.train_start,
            end=fold.train_end,
            fold=fold,
            split="train",
            min_group_size=ml_cfg.min_group_size,
        )
        valid = build_long_matrix(
            features=normalized_features,
            labels=labels,
            start=fold.valid_start,
            end=fold.valid_end,
            fold=fold,
            split="valid",
            min_group_size=ml_cfg.min_group_size,
        )
        test = build_long_matrix(
            features=normalized_features,
            labels=labels,
            start=fold.test_start,
            end=fold.test_end,
            fold=fold,
            split="test",
            min_group_size=ml_cfg.min_group_size,
        )
        validate_long_matrix(train)
        validate_long_matrix(valid)
        validate_long_matrix(test)
        ranker = fit_ranker(train=train, valid=valid, cfg=ml_cfg)
        rank_train = predict_rank_score(ranker.model, train)
        rank_valid = predict_rank_score(ranker.model, valid)
        rank_test = predict_rank_score(ranker.model, test)
        _logger.info(
            "[ML-RANKER] fold=%d train_n=%d valid_n=%d test_n=%d train_mean=%.6f valid_mean=%.6f",
            fold.fold_id,
            int(train.X.shape[0]),
            int(valid.X.shape[0]),
            int(test.X.shape[0]),
            float(np.mean(rank_train, dtype=np.float32)) if rank_train.size > 0 else 0.0,
            float(np.mean(rank_valid, dtype=np.float32)) if rank_valid.size > 0 else 0.0,
        )
        calibrators = fit_quantile_calibrators(
            train=train,
            valid=valid,
            rank_score_train=rank_train,
            rank_score_valid=rank_valid,
            cfg=ml_cfg,
        )
        ev_test = predict_conservative_ev(calibrators, test, rank_test, ml_cfg)
        _logger.info(
            "[ML-CALIB] fold=%d ev_mean=%.6e ev_p10=%.6e ev_p90=%.6e",
            fold.fold_id,
            float(np.mean(ev_test, dtype=np.float32)) if ev_test.size > 0 else 0.0,
            float(np.percentile(ev_test, 10)) if ev_test.size > 0 else 0.0,
            float(np.percentile(ev_test, 90)) if ev_test.size > 0 else 0.0,
        )
        fold_alpha = infer_fold_alpha(
            fold=fold,
            test=test,
            ev_test=ev_test,
            t_size=features.values.shape[0],
            n_size=features.values.shape[1],
        )
        ev_grid += fold_alpha.ev_grid
        for row, (t_idx, s_idx) in enumerate(test.index_map):
            score_grid[int(t_idx), int(s_idx)] = rank_test[row]
        _logger.info(
            "[ML-OOS] fold=%d test_rows=%d alpha_nonzero=%.4f",
            fold.fold_id,
            int(test.X.shape[0]),
            float(np.count_nonzero(ev_test) / max(1, ev_test.size)),
        )

    # [ML-OOS-FILL] Check if there is an uncovered live/OOS window at the end of the timeline
    last_test_end = folds[-1].test_end
    total_bars = features.values.shape[0]
    if last_test_end < total_bars:
        _logger.info(
            "[ML-OOS-FILL] Uncovered OOS/live window detected: [%d, %d)",
            last_test_end, total_bars
        )
        # Construct virtual fold for the remaining live window
        v_size = folds[-1].valid_end - folds[-1].valid_start
        v_train_start = 0
        v_train_end = last_test_end - v_size
        v_valid_start = v_train_end
        v_valid_end = last_test_end
        v_test_start = last_test_end
        v_test_end = total_bars

        from src.domain.futures.strategy.contracts import FoldSpec
        v_fold = FoldSpec(
            fold_id=len(folds),
            train_start=v_train_start,
            train_end=v_train_end,
            valid_start=v_valid_start,
            valid_end=v_valid_end,
            test_start=v_test_start,
            test_end=v_test_end,
            purge_bars=ml_cfg.purge_bars,
            embargo_bars=ml_cfg.embargo_bars,
        )
        v_train_values = features.values[v_train_start:v_train_end].astype(np.float64, copy=False)
        v_bounds = fit_robust_bounds(v_train_values, clip_quantile=0.995)
        v_clipped_values = apply_robust_bounds(
            features.values.astype(np.float64, copy=False),
            v_bounds,
        )
        v_train_medians = np.nanmedian(
            v_train_values.reshape(-1, v_train_values.shape[2]),
            axis=0,
        )
        v_train_medians = np.where(np.isfinite(v_train_medians), v_train_medians, 0.0)
        v_normalized = np.where(
            np.isfinite(v_clipped_values),
            v_clipped_values,
            v_train_medians[np.newaxis, np.newaxis, :],
        ).astype(np.float32, copy=False)
        v_normalized_features = FeaturePanel(
            datetimes=features.datetimes,
            symbols=features.symbols,
            values=v_normalized,
            feature_names=features.feature_names,
            valid_mask=features.valid_mask,
            availability_masks=features.availability_masks,
            metadata={**features.metadata, "train_imputer_applied": True},
        )
        v_train = build_long_matrix(
            features=v_normalized_features,
            labels=labels,
            start=v_train_start,
            end=v_train_end,
            fold=v_fold,
            split="train",
            min_group_size=ml_cfg.min_group_size,
        )
        v_valid = build_long_matrix(
            features=v_normalized_features,
            labels=labels,
            start=v_valid_start,
            end=v_valid_end,
            fold=v_fold,
            split="valid",
            min_group_size=ml_cfg.min_group_size,
        )
        v_test = build_long_matrix(
            features=v_normalized_features,
            labels=labels,
            start=v_test_start,
            end=v_test_end,
            fold=v_fold,
            split="test",
            min_group_size=ml_cfg.min_group_size,
        )
        if v_test.X.shape[0] > 0:
            validate_long_matrix(v_train)
            validate_long_matrix(v_valid)
            validate_long_matrix(v_test)
            v_ranker = fit_ranker(train=v_train, valid=v_valid, cfg=ml_cfg)
            v_rank_train = predict_rank_score(v_ranker.model, v_train)
            v_rank_valid = predict_rank_score(v_ranker.model, v_valid)
            v_rank_test = predict_rank_score(v_ranker.model, v_test)
            v_calibrators = fit_quantile_calibrators(
                train=v_train,
                valid=v_valid,
                rank_score_train=v_rank_train,
                rank_score_valid=v_rank_valid,
                cfg=ml_cfg,
            )
            v_ev_test = predict_conservative_ev(v_calibrators, v_test, v_rank_test, ml_cfg)
            v_fold_alpha = infer_fold_alpha(
                fold=v_fold,
                test=v_test,
                ev_test=v_ev_test,
                t_size=total_bars,
                n_size=features.values.shape[1],
            )
            ev_grid += v_fold_alpha.ev_grid
            for row, (t_idx, s_idx) in enumerate(v_test.index_map):
                score_grid[int(t_idx), int(s_idx)] = v_rank_test[row]
            _logger.info(
                "[ML-OOS-FILL] Completed virtual refit fold. test_rows=%d alpha_nonzero=%.4f",
                int(v_test.X.shape[0]),
                float(np.count_nonzero(v_ev_test) / max(1, v_ev_test.size)),
            )

    panel = assemble_alpha_panel(
        datetimes=features.datetimes,
        symbols=features.symbols,
        ev_grid=ev_grid,
        clip_abs=float(ml_cfg.alpha_clip_bps / 10000.0),
        eligible_mask=labels.eligible_mask,
    )
    panel.attrs["strategy_name"] = cfg.name
    panel.attrs["feature_names"] = list(features.feature_names)
    panel.attrs["fold_count"] = len(folds)
    panel.attrs["config_hash"] = build_manifest_hash(asdict(cfg.ml))
    quality_report = build_quality_report(
        feature_values=features.values,
        feature_valid_mask=features.valid_mask,
        label_eligible_mask=labels.eligible_mask,
        score_2d=score_grid,
        signed_ret_2d=labels.signed_net_ret.astype(np.float64),
        relevance_2d=labels.relevance.astype(np.float64),
        alpha_long_2d=np.maximum(ev_grid, 0.0),
        alpha_short_2d=np.maximum(-ev_grid, 0.0),
    )
    panel.attrs["quality_report"] = quality_report
    if not passes_quality_gate(quality_report):
        failed_keys = {
            k: v for k, v in quality_report.items()
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
        "[ML-ALPHA] rows=%d symbols=%d long_nz=%.4f short_nz=%.4f "
        "long_p95=%.2fbps short_p95=%.2fbps",
        len(panel),
        len(features.symbols),
        metrics["long_nz"],
        metrics["short_nz"],
        metrics["long_p95_bps"],
        metrics["short_p95_bps"],
    )
    # IC quality summary
    _logger.info(
        "[ML-ALPHA-IC] mean_ic=%.4f icir=%.3f t_stat=%.2f hit_ratio=%.3f n_obs=%d",
        quality_report.get("spearman_rank_ic", 0.0),
        quality_report.get("ic_icir", 0.0),
        quality_report.get("ic_t_stat", 0.0),
        quality_report.get("ic_hit_ratio", 0.0),
        int(quality_report.get("ic_n_obs", 0)),
    )
    # Cost wall diagnosis: gross alpha vs effective cost floor
    _friction_bps = round_trip_cost_bps()
    _hurdle_default_bps = float(OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 10.0))
    _floor_bps = _friction_bps + _hurdle_default_bps
    _alpha_p95 = max(
        quality_report.get("alpha_p95_bps", 0.0),
        0.0,
    )
    _logger.info(
        "[ML-COST-WALL] alpha_p95=%.2fbps friction=%.1fbps hurdle_default=%.1fbps "
        "floor=%.1fbps signal_clears_floor=%s",
        _alpha_p95,
        _friction_bps,
        _hurdle_default_bps,
        _floor_bps,
        str(_alpha_p95 >= _floor_bps),
    )
    # B4: IC gate warning (enforced as warning; raise after B2 uplift confirmed)
    if not passes_ic_gate(
        quality_report,
        min_mean_ic=0.02,
        min_t_stat=2.0,
        min_hit_ratio=0.45,
    ):
        _logger.warning(
            "[ML-IC-GATE] IC gate not satisfied: mean_ic=%.4f t_stat=%.2f hit_ratio=%.3f — "
            "proceeding with warning (gate enforced after B2 uplift confirmed)",
            quality_report.get("spearman_rank_ic", 0.0),
            quality_report.get("ic_t_stat", 0.0),
            quality_report.get("ic_hit_ratio", 0.0),
        )
    return panel


def build_ml_strategy_alpha_anchored(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
    anchor_end_idx: int,
    target_start: int,
    target_end: int,
) -> pd.DataFrame:
    """Single anchored-pass ML alpha.

    Trains on [0, anchor_end_idx), infers on [target_start, target_end).
    """
    from dataclasses import replace

    from src.domain.futures.strategy.contracts import FoldSpec

    ml_cfg = replace(
        cfg.ml,
        lambda_l2=1.0,
        min_data_in_leaf=30,
        num_leaves=31,
    )
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

    t_size = features.values.shape[0]
    anchor_end = int(np.clip(anchor_end_idx, 0, t_size))
    tgt_start = int(np.clip(target_start, 0, t_size))
    tgt_end = int(np.clip(target_end, 0, t_size))
    if anchor_end < 32:
        raise RuntimeError(
            f"anchored refit: anchor_end={anchor_end} too small (< 32 bars); cannot train"
        )
    if tgt_end <= tgt_start:
        raise RuntimeError(
            f"anchored refit: empty target window [{tgt_start}, {tgt_end})"
        )

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

    fold = FoldSpec(
        fold_id=0,
        train_start=0,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=tgt_start,
        test_end=tgt_end,
        purge_bars=ml_cfg.purge_bars,
        embargo_bars=ml_cfg.embargo_bars,
    )
    _logger.info(
        "[ML-ANCHORED] anchor_end=%d train=[0,%d) valid=[%d,%d) target=[%d,%d)",
        anchor_end, train_end, valid_start, valid_end, tgt_start, tgt_end,
    )

    train_values = features.values[fold.train_start : fold.train_end].astype(np.float64, copy=False)
    bounds = fit_robust_bounds(train_values, clip_quantile=0.995)
    clipped_values = apply_robust_bounds(features.values.astype(np.float64, copy=False), bounds)
    train_medians = np.nanmedian(train_values.reshape(-1, train_values.shape[2]), axis=0)
    train_medians = np.where(np.isfinite(train_medians), train_medians, 0.0)
    normalized = np.where(
        np.isfinite(clipped_values),
        clipped_values,
        train_medians[np.newaxis, np.newaxis, :],
    ).astype(np.float32, copy=False)
    normalized_features = FeaturePanel(
        datetimes=features.datetimes,
        symbols=features.symbols,
        values=normalized,
        feature_names=features.feature_names,
        valid_mask=features.valid_mask,
        availability_masks=features.availability_masks,
        metadata={**features.metadata, "train_imputer_applied": True},
    )

    train = build_long_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.train_start,
        end=fold.train_end,
        fold=fold,
        split="train",
        min_group_size=ml_cfg.min_group_size,
    )
    # [ML-UPGRADE] Thin-Data Guard (Dynamic Regularization)
    n_train_rows = int(train.X.shape[0])
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
    valid = build_long_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.valid_start,
        end=fold.valid_end,
        fold=fold,
        split="valid",
        min_group_size=ml_cfg.min_group_size,
    )
    test = build_long_matrix(
        features=normalized_features,
        labels=labels,
        start=fold.test_start,
        end=fold.test_end,
        fold=fold,
        split="test",
        min_group_size=ml_cfg.min_group_size,
    )
    validate_long_matrix(train)
    validate_long_matrix(valid)
    validate_long_matrix(test)

    ranker = fit_ranker(train=train, valid=valid, cfg=ml_cfg)
    rank_train = predict_rank_score(ranker.model, train)
    rank_valid = predict_rank_score(ranker.model, valid)
    rank_test = predict_rank_score(ranker.model, test)
    _logger.info(
        "[ML-ANCHORED-RANKER] train_n=%d valid_n=%d test_n=%d train_mean=%.6f valid_mean=%.6f",
        int(train.X.shape[0]),
        int(valid.X.shape[0]),
        int(test.X.shape[0]),
        float(np.mean(rank_train, dtype=np.float32)) if rank_train.size > 0 else 0.0,
        float(np.mean(rank_valid, dtype=np.float32)) if rank_valid.size > 0 else 0.0,
    )

    calibrators = fit_quantile_calibrators(
        train=train,
        valid=valid,
        rank_score_train=rank_train,
        rank_score_valid=rank_valid,
        cfg=ml_cfg,
    )
    ev_test = predict_conservative_ev(calibrators, test, rank_test, ml_cfg)
    _logger.info(
        "[ML-ANCHORED-CALIB] ev_mean=%.6e ev_p10=%.6e ev_p90=%.6e",
        float(np.mean(ev_test, dtype=np.float32)) if ev_test.size > 0 else 0.0,
        float(np.percentile(ev_test, 10)) if ev_test.size > 0 else 0.0,
        float(np.percentile(ev_test, 90)) if ev_test.size > 0 else 0.0,
    )

    fold_alpha = infer_fold_alpha(
        fold=fold,
        test=test,
        ev_test=ev_test,
        t_size=t_size,
        n_size=features.values.shape[1],
    )
    ev_grid = fold_alpha.ev_grid
    score_grid = np.full((t_size, features.values.shape[1]), np.nan, dtype=np.float32)
    for row, (t_idx, s_idx) in enumerate(test.index_map):
        score_grid[int(t_idx), int(s_idx)] = rank_test[row]

    panel = assemble_alpha_panel(
        datetimes=features.datetimes,
        symbols=features.symbols,
        ev_grid=ev_grid,
        clip_abs=float(ml_cfg.alpha_clip_bps / 10000.0),
        eligible_mask=labels.eligible_mask,
    )
    from dataclasses import asdict

    from src.domain.futures.strategy.cache import build_manifest_hash
    panel.attrs["strategy_name"] = cfg.name
    panel.attrs["feature_names"] = list(features.feature_names)
    panel.attrs["fold_count"] = 1
    panel.attrs["config_hash"] = build_manifest_hash(asdict(cfg.ml))
    panel.attrs["anchored"] = True
    panel.attrs["anchor_end_idx"] = anchor_end
    panel.attrs["target_range"] = (tgt_start, tgt_end)
    _tgt_cells = max(1, (tgt_end - tgt_start) * features.values.shape[1])
    _tgt_slice = ev_grid[tgt_start:tgt_end]
    _logger.info(
        "[ML-ANCHORED] target_long_nz=%.4f target_short_nz=%.4f",
        float(np.count_nonzero(_tgt_slice) / _tgt_cells),
        float(np.count_nonzero(np.maximum(-_tgt_slice, 0)) / _tgt_cells),
    )
    return panel
