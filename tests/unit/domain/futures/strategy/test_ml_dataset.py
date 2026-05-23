from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import FeaturePanel, LabelPanel
from src.domain.futures.strategy.dataset import (
    append_rank_features_for_calibrator,
    build_long_matrix,
    make_walk_forward_folds,
)


def _panels(t: int = 240, n: int = 8, f: int = 4) -> tuple[FeaturePanel, LabelPanel]:
    dt = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(4 * i, "h") for i in range(t)],
        dtype="datetime64[ns]",
    )
    values = np.random.default_rng(42).normal(size=(t, n, f)).astype(np.float32)
    valid = np.ones((t, n), dtype=bool)
    fp = FeaturePanel(
        datetimes=dt,
        symbols=tuple(f"S{i}" for i in range(n)),
        values=values,
        feature_names=tuple(f"f{i}" for i in range(f)),
        valid_mask=valid,
    )
    signed = np.random.default_rng(7).normal(size=(t, n)).astype(np.float32) * 1e-3
    lp = LabelPanel(
        long_net_ret=signed.copy(),
        short_net_ret=(-signed).copy(),
        signed_net_ret=signed,
        relevance=np.full((t, n), 2, dtype=np.int32),
        sample_weight=np.ones((t, n), dtype=np.float32),
        eligible_mask=np.ones((t, n), dtype=bool),
    )
    return fp, lp


def test_make_walk_forward_folds_builds_non_empty() -> None:
    fp, _ = _panels(t=1200)
    cfg = StrategyMLConfig(train_months=1, valid_months=1, test_months=1)
    folds = make_walk_forward_folds(fp.datetimes, cfg)
    assert len(folds) >= 1


def test_append_rank_features_for_calibrator() -> None:
    fp, lp = _panels(t=100, n=8)
    ds = build_long_matrix(fp, lp, 0, 80, min_group_size=4)
    rank_score = np.linspace(-1.0, 1.0, ds.X.shape[0], dtype=np.float32)
    out = append_rank_features_for_calibrator(ds, rank_score)
    assert out.X.shape[1] == ds.X.shape[1] + 2


def test_build_long_matrix_supports_fold_split_contract() -> None:
    fp, lp = _panels(t=1200, n=8)
    cfg = StrategyMLConfig(train_months=1, valid_months=1, test_months=1)
    folds = make_walk_forward_folds(fp.datetimes, cfg)
    ds = build_long_matrix(
        features=fp,
        labels=lp,
        fold=folds[0],
        split="train",
        min_group_size=4,
    )
    assert ds.X.shape[0] > 0
    assert int(np.sum(ds.group)) == ds.X.shape[0]


def test_build_long_matrix_drops_partial_group_after_feature_filter() -> None:
    fp, lp = _panels(t=10, n=4, f=2)
    values = fp.values.copy()
    values[0, 2:, :] = np.nan
    fp2 = FeaturePanel(
        datetimes=fp.datetimes,
        symbols=fp.symbols,
        values=values,
        feature_names=fp.feature_names,
        valid_mask=fp.valid_mask,
    )
    ds = build_long_matrix(fp2, lp, start=0, end=1, min_group_size=3)
    assert ds.X.shape[0] == 0
    assert ds.group.shape[0] == 0
