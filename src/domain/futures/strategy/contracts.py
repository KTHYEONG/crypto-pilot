from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ALPHA_FORECAST_CONTRACT = "metadata"
ALPHA_FORECAST_ATTR_KEY = "alpha_forecast_metadata"


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
    exec_net_ret: np.ndarray  # pre-CS-demean beta-residualized return; for calibrator absolute EV
    relevance: np.ndarray
    sample_weight: np.ndarray
    eligible_mask: np.ndarray  # eligible & finite_long (학습에 사용되는 마스크)
    # finite_long 적용 전 pure eligibility — 무결성 진단 tautology 방지용
    raw_eligible_mask: np.ndarray | None = None
    rank_target: np.ndarray | None = None
    magnitude_target: np.ndarray | None = None
    rank_target_long: np.ndarray | None = None
    rank_target_short: np.ndarray | None = None
    magnitude_target_long: np.ndarray | None = None
    magnitude_target_short: np.ndarray | None = None
    relevance_long: np.ndarray | None = None
    relevance_short: np.ndarray | None = None
    forward_gross_ret: np.ndarray | None = None
    forward_gross_rank_target: np.ndarray | None = None
    forward_gross_relevance: np.ndarray | None = None
    dynamic_cost_bps_2d: np.ndarray | None = None  # per-symbol dynamic cost for EV gate
    # Leak-free trailing OLS beta [T, N] and equal-weight market forward return [T].
    # Exposed for beta-residualized OOS IC decomposition (anti-bias diagnostic).
    beta_2d: np.ndarray | None = None
    market_fwd_ret: np.ndarray | None = None
    forward_return_by_horizon: dict[int, np.ndarray] | None = None
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


@dataclass(slots=True, frozen=True)
class CandidateSignalPanel:
    """Candidate signal matrix contract."""

    datetimes: np.ndarray
    symbols: tuple[str, ...]
    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CandidatePortfolioPanelMeta:
    """Candidate portfolio panel metadata contract."""

    datetimes: np.ndarray
    symbols: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
