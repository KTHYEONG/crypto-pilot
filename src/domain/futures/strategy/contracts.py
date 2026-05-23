from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class FeaturePanel:
    """Feature tensor contract."""

    datetimes: np.ndarray
    symbols: tuple[str, ...]
    values: np.ndarray
    feature_names: tuple[str, ...]
    valid_mask: np.ndarray
    availability_masks: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LabelPanel:
    """Label tensor contract."""

    long_net_ret: np.ndarray
    short_net_ret: np.ndarray
    signed_net_ret: np.ndarray
    relevance: np.ndarray
    sample_weight: np.ndarray
    eligible_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LongMatrixDataset:
    """Flattened matrix contract for LightGBM."""

    X: np.ndarray
    y_rank: np.ndarray
    y_ev: np.ndarray
    group: np.ndarray
    sample_weight: np.ndarray
    index_map: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class FoldSpec:
    """Chronological fold bounds (half-open intervals)."""

    fold_id: int
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int
    test_start: int
    test_end: int
    purge_bars: int
    embargo_bars: int
