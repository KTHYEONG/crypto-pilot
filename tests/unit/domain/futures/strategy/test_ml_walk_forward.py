from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.common.normalization import apply_robust_bounds, fit_robust_bounds
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.dataset import make_walk_forward_folds


def test_make_walk_forward_folds_are_chronological_and_purged() -> None:
    datetimes = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(4 * i, "h") for i in range(1800)],
        dtype="datetime64[ns]",
    )
    cfg = StrategyMLConfig(
        train_months=2,
        valid_months=1,
        test_months=1,
        purge_bars=3,
        embargo_bars=2,
        label_horizon_bars=2,
    )
    folds = make_walk_forward_folds(datetimes, cfg)
    assert folds
    for fold in folds:
        assert (
            fold.train_start
            < fold.train_end
            <= fold.valid_start
            < fold.valid_end
            <= fold.test_start
            < fold.test_end
        )
        assert fold.valid_start >= fold.train_end
        assert fold.test_start >= fold.valid_end


def test_train_only_normalization_bounds_do_not_use_future_rows() -> None:
    train = np.array([[[0.0], [1.0]], [[2.0], [3.0]]], dtype=np.float64)
    future = np.array([[[9999.0], [-9999.0]]], dtype=np.float64)
    bounds = fit_robust_bounds(train, clip_quantile=0.95)
    clipped_future = apply_robust_bounds(future, bounds)
    assert float(np.max(np.abs(clipped_future))) < 10.0
