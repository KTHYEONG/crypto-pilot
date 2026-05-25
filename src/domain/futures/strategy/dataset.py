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
) -> LongMatrixDataset:
    """Flatten [T, N, F] tensors to LightGBM-ready matrix."""
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

    rows_x: list[np.ndarray] = []
    rows_rank: list[np.int32] = []
    rows_ev: list[np.float32] = []
    rows_w: list[np.float32] = []
    rows_idx: list[tuple[int, int]] = []
    groups: list[int] = []
    for t in range(start, end):
        mask_t = features.valid_mask[t] & labels.eligible_mask[t]
        idx = np.flatnonzero(mask_t)
        if idx.size < min_group_size:
            continue
        x_t: list[np.ndarray] = []
        rank_t: list[np.int32] = []
        ev_t: list[np.float32] = []
        w_t: list[np.float32] = []
        i_t: list[tuple[int, int]] = []
        group_count = 0
        for col in idx:
            feat = features.values[t, col]
            if not np.all(np.isfinite(feat)):
                continue
            # sample_weight from labels (liquidity * (1+2|y_ev|), computed in labels.py).
            # exec_net_ret: pre-CS-demean beta-residualized return — absolute EV for calibrator.
            # signed_net_ret (CS-demeaned) is consumed only by ranker via _cs_demean in ranker.py.
            ev_val = np.float32(labels.exec_net_ret[t, col])
            w = np.float32(labels.sample_weight[t, col])

            x_t.append(feat.astype(np.float32, copy=False))
            rank_t.append(np.int32(labels.relevance[t, col]))
            ev_t.append(ev_val)
            w_t.append(w)
            i_t.append((t, int(col)))
            group_count += 1
        if group_count >= min_group_size:
            rows_x.extend(x_t)
            rows_rank.extend(rank_t)
            rows_ev.extend(ev_t)
            rows_w.extend(w_t)
            rows_idx.extend(i_t)
            groups.append(group_count)
    if not rows_x:
        empty_x = np.zeros((0, features.values.shape[2]), dtype=np.float32)
        empty_i: np.ndarray = np.zeros((0, 2), dtype=np.int64)
        return LongMatrixDataset(
            X=empty_x,
            y_rank=np.zeros((0,), dtype=np.int32),
            y_ev=np.zeros((0,), dtype=np.float32),
            group=np.zeros((0,), dtype=np.int32),
            sample_weight=np.zeros((0,), dtype=np.float32),
            index_map=empty_i,
            feature_names=features.feature_names,
        )
    return LongMatrixDataset(
        X=np.vstack(rows_x).astype(np.float32, copy=False),
        y_rank=np.asarray(rows_rank, dtype=np.int32),
        y_ev=np.asarray(rows_ev, dtype=np.float32),
        group=np.asarray(groups, dtype=np.int32),
        sample_weight=np.asarray(rows_w, dtype=np.float32),
        index_map=np.asarray(rows_idx, dtype=np.int64),
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
