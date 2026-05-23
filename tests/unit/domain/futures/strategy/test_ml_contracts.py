from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.common.validation import (
    validate_feature_panel,
    validate_long_matrix,
)
from src.domain.futures.strategy.contracts import FeaturePanel, LongMatrixDataset


def test_feature_panel_shape_validation_fails() -> None:
    panel = FeaturePanel(
        datetimes=np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        values=np.zeros((1, 1, 2), dtype=np.float32),
        feature_names=("a",),
        valid_mask=np.ones((1, 1), dtype=bool),
    )
    with pytest.raises(ValueError, match="feature names length mismatch"):
        validate_feature_panel(panel)


def test_long_matrix_group_sum_validation_fails() -> None:
    ds = LongMatrixDataset(
        X=np.zeros((3, 2), dtype=np.float32),
        y_rank=np.zeros((3,), dtype=np.int32),
        y_ev=np.zeros((3,), dtype=np.float32),
        group=np.array([2], dtype=np.int32),
        sample_weight=np.ones((3,), dtype=np.float32),
        index_map=np.zeros((3, 2), dtype=np.int64),
        feature_names=("f1", "f2"),
    )
    with pytest.raises(ValueError, match="sum\\(group\\)"):
        validate_long_matrix(ds)

