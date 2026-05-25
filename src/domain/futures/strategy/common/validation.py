from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.contracts import FeaturePanel, LabelPanel, LongMatrixDataset


def validate_feature_panel(panel: FeaturePanel) -> None:
    """Validate shape and ordering contracts for feature panel."""
    t, n, f = panel.values.shape
    if panel.datetimes.shape[0] != t:
        raise ValueError("feature datetimes length mismatch")
    if len(panel.symbols) != n:
        raise ValueError("feature symbols length mismatch")
    if len(panel.feature_names) != f:
        raise ValueError("feature names length mismatch")
    if panel.valid_mask.shape != (t, n):
        raise ValueError("feature valid_mask shape mismatch")
    if not np.all(panel.datetimes[1:] >= panel.datetimes[:-1]):
        raise ValueError("feature datetimes must be monotonic increasing")


def validate_label_panel(panel: LabelPanel, t: int, n: int) -> None:
    """Validate shape contracts for label panel."""
    expected = (t, n)
    for name, arr in (
        ("long_net_ret", panel.long_net_ret),
        ("short_net_ret", panel.short_net_ret),
        ("signed_net_ret", panel.signed_net_ret),
        ("exec_net_ret", panel.exec_net_ret),
        ("relevance", panel.relevance),
        ("sample_weight", panel.sample_weight),
        ("eligible_mask", panel.eligible_mask),
    ):
        if arr.shape != expected:
            raise ValueError(f"{name} shape mismatch")


def validate_long_matrix(ds: LongMatrixDataset) -> None:
    """Validate flattened training matrix contracts."""
    m = ds.X.shape[0]
    if ds.y_rank.shape[0] != m or ds.y_ev.shape[0] != m:
        raise ValueError("label vector length mismatch")
    if ds.sample_weight.shape[0] != m:
        raise ValueError("sample_weight length mismatch")
    if ds.index_map.shape != (m, 2):
        raise ValueError("index_map shape mismatch")
    if int(np.sum(ds.group)) != m:
        raise ValueError("sum(group) must equal row count")
