from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import (
    FeaturePanel,
    FoldSpec,
    LabelPanel,
    LongMatrixDataset,
)


def make_walk_forward_folds(datetimes: np.ndarray, cfg: StrategyMLConfig) -> list[FoldSpec]:
    """Build chronological month-based walk-forward folds."""
    n = int(datetimes.shape[0])
    if n < 180:
        return []
    idx = pd.to_datetime(datetimes)
    start = pd.Timestamp(idx[0])
    end = pd.Timestamp(idx[-1])
    folds: list[FoldSpec] = []
    fold_id = 0
    cursor = start + pd.DateOffset(months=cfg.train_months + cfg.valid_months)
    while cursor + pd.DateOffset(months=cfg.test_months) <= end:
        train_start_ts = cursor - pd.DateOffset(months=cfg.train_months + cfg.valid_months)
        train_end_ts = cursor - pd.DateOffset(months=cfg.valid_months)
        valid_start_ts = train_end_ts
        valid_end_ts = cursor
        test_start_ts = cursor
        test_end_ts = cursor + pd.DateOffset(months=cfg.test_months)
        train_start = int(np.searchsorted(idx.values, train_start_ts.to_datetime64(), side="left"))
        train_end = int(np.searchsorted(idx.values, train_end_ts.to_datetime64(), side="left"))
        valid_start = int(np.searchsorted(idx.values, valid_start_ts.to_datetime64(), side="left"))
        valid_end = int(np.searchsorted(idx.values, valid_end_ts.to_datetime64(), side="left"))
        test_start = int(np.searchsorted(idx.values, test_start_ts.to_datetime64(), side="left"))
        test_end = int(np.searchsorted(idx.values, test_end_ts.to_datetime64(), side="left"))
        enough_train = train_end - train_start >= 32
        enough_valid = valid_end - valid_start >= 8
        enough_test = test_end - test_start >= 8
        if enough_train and enough_valid and enough_test:
            folds.append(
                FoldSpec(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=min(valid_start + cfg.purge_bars, max(valid_start, valid_end - 1)),
                    valid_end=valid_end,
                    test_start=min(test_start + cfg.embargo_bars, max(test_start, test_end - 1)),
                    test_end=test_end,
                    purge_bars=cfg.purge_bars,
                    embargo_bars=cfg.embargo_bars,
                )
            )
            fold_id += 1
        cursor = cursor + pd.DateOffset(months=cfg.test_months)
    return folds


def build_long_matrix(
    features: FeaturePanel,
    labels: LabelPanel,
    start: int | None = None,
    end: int | None = None,
    min_group_size: int = 1,
    *,
    fold: FoldSpec | None = None,
    split: Literal["train", "valid", "test"] | None = None,
    rank_target_override: np.ndarray | None = None,
    relevance_override: np.ndarray | None = None,
    ev_target_override: np.ndarray | None = None,
) -> LongMatrixDataset:
    """Flatten [T, N, F] tensors to LightGBM-ready matrix.

    Time complexity: O(M) where M is the number of valid samples.
    Space complexity: O(M * F) for the flattened feature matrix X.
    """
    if fold is not None or split is not None:
        if fold is None or split is None:
            raise ValueError("fold and split must be provided together")
        if split == "train":
            start = fold.train_start
            end = fold.train_end
        elif split == "valid":
            start = fold.valid_start
            end = fold.valid_end
        else:
            start = fold.test_start
            end = fold.test_end
    if start is None or end is None:
        raise ValueError("start/end or fold/split must be provided")

    t_slice = end - start
    if t_slice <= 0:
        empty_x = np.zeros((0, features.values.shape[2]), dtype=np.float32)
        empty_i = np.zeros((0, 2), dtype=np.int64)
        return LongMatrixDataset(
            X=empty_x,
            y_rank=np.zeros((0,), dtype=np.int32),
            y_ev=np.zeros((0,), dtype=np.float32),
            group=np.zeros((0,), dtype=np.int32),
            sample_weight=np.zeros((0,), dtype=np.float32),
            index_map=empty_i,
            feature_names=features.feature_names,
        )

    # 1. 2D 마스크 결합 (Timeframe slice 추출)
    valid_mask = features.valid_mask[start:end]
    eligible_mask = labels.eligible_mask[start:end]
    full_mask_2d = valid_mask & eligible_mask  # [t_slice, N]

    # 2. Finite Feature Mask 생성
    # LightGBM handles NaN natively — only require that the symbol row is not ALL-NaN
    # to avoid passing completely empty feature rows.
    feature_slice = features.values[start:end]  # [t_slice, N, F]
    # Drop only rows where ALL features are NaN (completely missing symbol at this bar).
    # Partial NaN features are acceptable — LightGBM uses them natively.
    any_finite_feat_mask = np.any(np.isfinite(feature_slice), axis=2)  # [t_slice, N]

    # 3. 최종 유효 2D 마스크 결합
    final_mask = full_mask_2d & any_finite_feat_mask  # [t_slice, N]

    # 4. 행별 그룹 크기 필터링 (min_group_size 조건 충족 행 필터링)
    group_counts = final_mask.sum(axis=1)  # [t_slice]
    valid_rows_mask = group_counts >= min_group_size  # [t_slice]

    # 유효하지 않은 행의 마스크는 False로 오프셋 차단
    final_mask = final_mask.copy()
    final_mask[~valid_rows_mask, :] = False

    # 5. C-level에서 2D 마스크 좌표 추출
    t_indices, col_indices = np.where(final_mask)

    if t_indices.size == 0:
        empty_x = np.zeros((0, features.values.shape[2]), dtype=np.float32)
        empty_i = np.zeros((0, 2), dtype=np.int64)
        return LongMatrixDataset(
            X=empty_x,
            y_rank=np.zeros((0,), dtype=np.int32),
            y_ev=np.zeros((0,), dtype=np.float32),
            group=np.zeros((0,), dtype=np.int32),
            sample_weight=np.zeros((0,), dtype=np.float32),
            index_map=empty_i,
            feature_names=features.feature_names,
        )

    # 6. Fancy Indexing을 통한 고속 2D 매핑 (Zero-Loop)
    x = feature_slice[t_indices, col_indices].astype(np.float32, copy=False)

    # y_rank 조립
    if rank_target_override is not None:
        rank_override_slice = rank_target_override[start:end]
        rank_vals = rank_override_slice[t_indices, col_indices]
        y_rank = np.where(rank_vals > 0.0, np.int32(4), np.int32(0))
    else:
        rank_src_slice = (
            relevance_override[start:end]
            if relevance_override is not None
            else labels.relevance[start:end]
        )
        y_rank = rank_src_slice[t_indices, col_indices].astype(np.int32)

    # y_ev 조립
    ev_source = (
        ev_target_override[start:end]
        if ev_target_override is not None
        else (
            labels.magnitude_target[start:end]
            if labels.magnitude_target is not None
            else labels.exec_net_ret[start:end]
        )
    )
    y_ev = ev_source[t_indices, col_indices].astype(np.float32)

    # sample_weight 및 index_map 조립
    sample_weight = labels.sample_weight[start:end][t_indices, col_indices].astype(np.float32)
    global_t_indices = t_indices + start
    index_map = np.column_stack((global_t_indices, col_indices)).astype(np.int64)

    # Group Size 리스트 조립
    group = group_counts[valid_rows_mask].astype(np.int32)

    return LongMatrixDataset(
        X=x,
        y_rank=y_rank,
        y_ev=y_ev,
        group=group,
        sample_weight=sample_weight,
        index_map=index_map,
        feature_names=features.feature_names,
    )


def append_rank_features_for_calibrator(
    dataset: LongMatrixDataset,
    rank_score: np.ndarray,
) -> LongMatrixDataset:
    """Append rank score and normalized rank score for calibrator input."""
    if dataset.X.shape[0] != rank_score.shape[0]:
        raise ValueError("rank_score length mismatch")
    if rank_score.shape[0] == 0:
        return dataset
    rank_score = np.asarray(rank_score, dtype=np.float32)
    r_mean = np.mean(rank_score, dtype=np.float32)
    r_std = np.std(rank_score, dtype=np.float32)
    rank_z = (rank_score - r_mean) / np.maximum(r_std, np.float32(1e-12))
    x_aug = np.column_stack([dataset.X, rank_score, rank_z]).astype(np.float32)
    names = (*dataset.feature_names, "rank_score", "rank_zscore")
    return LongMatrixDataset(
        X=x_aug,
        y_rank=dataset.y_rank,
        y_ev=dataset.y_ev,
        group=dataset.group,
        sample_weight=dataset.sample_weight,
        index_map=dataset.index_map,
        feature_names=names,
    )
