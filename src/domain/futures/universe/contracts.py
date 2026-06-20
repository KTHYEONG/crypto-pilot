"""Immutable universe contract and eligibility types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "DataConfidence",
    "EligibilityCode",
    "EligibilityReason",
    "EligibilitySnapshot",
    "ExecutionEligibility",
    "ExecutionRules",
    "InstrumentRecord",
    "MarketObservation",
    "StrategyReadinessCube",
    "StrategyRequirement",
    "UniverseStateCube",
]


class DataConfidence(StrEnum):
    """Confidence labels for observed and reconstructed data."""

    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"
    UNKNOWN = "unknown"


class EligibilityCode(StrEnum):
    """Canonical execution eligibility codes."""

    ELIGIBLE = "ELIGIBLE"
    NOT_ONBOARDED = "NOT_ONBOARDED"
    STATUS_NOT_TRADING = "STATUS_NOT_TRADING"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    MISSING_RULES = "MISSING_RULES"
    ORDER_TOO_SMALL = "ORDER_TOO_SMALL"
    COST_TOO_HIGH = "COST_TOO_HIGH"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    DATA_CONFIDENCE_LOW = "DATA_CONFIDENCE_LOW"
    DATA_INTEGRITY_FAIL = "DATA_INTEGRITY_FAIL"
    LEVERAGED_TOKEN = "LEVERAGED_TOKEN"  # noqa: S105
    ADV_FLOOR_FAIL = "ADV_FLOOR_FAIL"


@dataclass(frozen=True, slots=True)
class InstrumentRecord:
    """Append-only contract lifecycle record."""

    instrument_id: str
    symbol: str
    pair: str
    quote_asset: str
    margin_asset: str
    contract_type: Literal["PERPETUAL"]
    onboard_at: datetime
    status: str
    state_valid_from: datetime
    available_at: datetime
    confidence: DataConfidence


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """Point-in-time market observation."""

    instrument_id: str
    metric: str
    observed_at: datetime
    available_at: datetime
    value: float
    source: str
    confidence: DataConfidence


@dataclass(frozen=True, slots=True)
class ExecutionRules:
    """Point-in-time execution rules for a single instrument."""

    instrument_id: str
    decision_at: datetime
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float
    taker_fee_bps: float
    tick_size_confidence: DataConfidence
    step_size_confidence: DataConfidence
    min_qty_confidence: DataConfidence
    min_notional_confidence: DataConfidence
    taker_fee_confidence: DataConfidence


@dataclass(frozen=True, slots=True)
class EligibilityReason:
    """Single eligibility reason with observation metadata."""

    code: EligibilityCode
    hard: bool
    observed_value: float | None
    threshold: float | None
    source: str
    confidence: DataConfidence


@dataclass(frozen=True, slots=True)
class ExecutionEligibility:
    """Execution decision and associated risk metadata."""

    instrument_id: str
    decision_at: datetime
    eligible: bool
    code: EligibilityCode
    reasons: tuple[EligibilityReason, ...] = field(default_factory=tuple)
    intended_notional_usdt: float = 0.0
    rounded_notional_usdt: float = 0.0
    capacity_usdt: float = 0.0
    risk_scale: float = 1.0
    cost_bps: float = 0.0
    confidence: DataConfidence = DataConfidence.UNKNOWN


@dataclass(frozen=True, slots=True)
class EligibilitySnapshot:
    """Point-in-time execution snapshot for all tracked instruments."""

    decision_at: datetime
    eligibilities: tuple[ExecutionEligibility, ...]
    instrument_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UniverseStateCube:
    """Dense eligibility cube indexed by calendar and instrument."""

    calendar: pd.DatetimeIndex
    instrument_ids: tuple[str, ...]
    eligible: NDArray[np.bool_]
    entry_block: NDArray[np.bool_]
    exit_required: NDArray[np.bool_]
    capacity_usdt: NDArray[np.float64]
    risk_scale: NDArray[np.float64]
    cost_bps: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StrategyRequirement:
    """Per-strategy readiness requirements."""

    strategy: str
    required_lookback_bars: int
    required_features: tuple[str, ...] = field(default_factory=tuple)
    requires_funding: bool = False
    requires_open_interest: bool = False
    min_training_events: int = 0
    min_data_confidence: DataConfidence = DataConfidence.UNKNOWN


@dataclass(frozen=True, slots=True)
class StrategyReadinessCube:
    """Dense readiness cube indexed by strategy, calendar, and instrument."""

    strategies: tuple[str, ...]
    calendar: pd.DatetimeIndex
    instrument_ids: tuple[str, ...]
    ready: NDArray[np.bool_]
    reason_code: NDArray[np.object_]
