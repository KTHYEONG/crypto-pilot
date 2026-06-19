from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.domain.futures.universe.contracts import (
    DataConfidence,
    EligibilityCode,
    EligibilityReason,
    EligibilitySnapshot,
    ExecutionEligibility,
    ExecutionRules,
    InstrumentRecord,
    MarketObservation,
    StrategyReadinessCube,
    StrategyRequirement,
    UniverseStateCube,
)


def test_contract_enums_expose_expected_values() -> None:
    """Enums must preserve the contract values from the spec."""

    assert DataConfidence.OBSERVED.value == "observed"
    assert DataConfidence.RECONSTRUCTED.value == "reconstructed"
    assert DataConfidence.UNKNOWN.value == "unknown"

    assert tuple(code.value for code in EligibilityCode) == (
        "ELIGIBLE",
        "NOT_ONBOARDED",
        "STATUS_NOT_TRADING",
        "STALE_MARKET_DATA",
        "MISSING_RULES",
        "ORDER_TOO_SMALL",
        "COST_TOO_HIGH",
        "INSUFFICIENT_OBSERVATIONS",
        "DATA_CONFIDENCE_LOW",
    )


def test_instrument_and_observation_contracts_are_frozen_and_slotted() -> None:
    """Core record contracts should be immutable value objects."""

    instrument = InstrumentRecord(
        instrument_id="binance:usdt_perpetual:BTCUSDT:2020-01-01T00:00:00Z",
        symbol="BTCUSDT",
        pair="BTCUSDT",
        quote_asset="USDT",
        margin_asset="USDT",
        contract_type="PERPETUAL",
        onboard_at=datetime(2020, 1, 1, tzinfo=UTC),
        status="TRADING",
        state_valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        available_at=datetime(2020, 1, 1, tzinfo=UTC),
        confidence=DataConfidence.OBSERVED,
    )
    observation = MarketObservation(
        instrument_id=instrument.instrument_id,
        metric="tick_size",
        observed_at=datetime(2020, 1, 1, tzinfo=UTC),
        available_at=datetime(2020, 1, 1, tzinfo=UTC),
        value=0.01,
        source="exchangeInfo",
        confidence=DataConfidence.OBSERVED,
    )

    assert instrument.contract_type == "PERPETUAL"
    assert observation.metric == "tick_size"
    assert getattr(InstrumentRecord, "__dataclass_params__").frozen is True
    assert getattr(MarketObservation, "__dataclass_params__").frozen is True
    assert hasattr(InstrumentRecord, "__slots__")
    assert hasattr(MarketObservation, "__slots__")
    assert [field.name for field in fields(InstrumentRecord)] == [
        "instrument_id",
        "symbol",
        "pair",
        "quote_asset",
        "margin_asset",
        "contract_type",
        "onboard_at",
        "status",
        "state_valid_from",
        "available_at",
        "confidence",
    ]
    assert [field.name for field in fields(MarketObservation)] == [
        "instrument_id",
        "metric",
        "observed_at",
        "available_at",
        "value",
        "source",
        "confidence",
    ]


def test_execution_rules_and_reason_contracts_keep_all_metadata_fields() -> None:
    """Execution contract types must retain confidence and reason metadata."""

    rules = ExecutionRules(
        instrument_id="BTCUSDT",
        decision_at=datetime(2025, 1, 1, tzinfo=UTC),
        tick_size=0.01,
        step_size=0.001,
        min_qty=0.001,
        min_notional=5.0,
        taker_fee_bps=5.0,
        tick_size_confidence=DataConfidence.OBSERVED,
        step_size_confidence=DataConfidence.RECONSTRUCTED,
        min_qty_confidence=DataConfidence.OBSERVED,
        min_notional_confidence=DataConfidence.UNKNOWN,
        taker_fee_confidence=DataConfidence.OBSERVED,
    )
    reason = EligibilityReason(
        code=EligibilityCode.ORDER_TOO_SMALL,
        hard=True,
        observed_value=4.5,
        threshold=5.0,
        source="exchange rules",
        confidence=DataConfidence.RECONSTRUCTED,
    )

    assert rules.taker_fee_bps == 5.0
    assert rules.min_notional_confidence is DataConfidence.UNKNOWN
    assert reason.hard is True
    assert reason.code is EligibilityCode.ORDER_TOO_SMALL
    assert [field.name for field in fields(ExecutionRules)] == [
        "instrument_id",
        "decision_at",
        "tick_size",
        "step_size",
        "min_qty",
        "min_notional",
        "taker_fee_bps",
        "tick_size_confidence",
        "step_size_confidence",
        "min_qty_confidence",
        "min_notional_confidence",
        "taker_fee_confidence",
    ]
    assert [field.name for field in fields(EligibilityReason)] == [
        "code",
        "hard",
        "observed_value",
        "threshold",
        "source",
        "confidence",
    ]
    assert [field.name for field in fields(ExecutionEligibility)] == [
        "instrument_id",
        "decision_at",
        "eligible",
        "code",
        "reasons",
        "intended_notional_usdt",
        "rounded_notional_usdt",
        "capacity_usdt",
        "risk_scale",
        "cost_bps",
        "confidence",
    ]
    assert [field.name for field in fields(EligibilitySnapshot)] == [
        "decision_at",
        "eligibilities",
        "instrument_ids",
        "metadata",
    ]


def test_eligibility_snapshot_shapes_are_dense_and_indexed() -> None:
    """Cube contracts must expose explicit calendar and instrument axes."""

    calendar = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T04:00:00Z")]
    )
    instrument_ids = ("BTCUSDT", "ETHUSDT")
    eligible = np.array([[True, False], [True, True]], dtype=bool)
    entry_block = np.array([[False, True], [False, False]], dtype=bool)
    exit_required = np.array([[False, False], [True, False]], dtype=bool)
    capacity_usdt = np.array([[1000.0, 0.0], [1500.0, 800.0]], dtype=np.float64)
    risk_scale = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float64)
    cost_bps = np.array([[12.5, 0.0], [10.0, 8.0]], dtype=np.float64)

    cube = UniverseStateCube(
        calendar=calendar,
        instrument_ids=instrument_ids,
        eligible=eligible,
        entry_block=entry_block,
        exit_required=exit_required,
        capacity_usdt=capacity_usdt,
        risk_scale=risk_scale,
        cost_bps=cost_bps,
    )
    snapshot = EligibilitySnapshot(
        decision_at=datetime(2025, 1, 1, tzinfo=UTC),
        eligibilities=(
            ExecutionEligibility(
                instrument_id="BTCUSDT",
                decision_at=datetime(2025, 1, 1, tzinfo=UTC),
                eligible=True,
                code=EligibilityCode.ELIGIBLE,
            ),
        ),
        instrument_ids=("BTCUSDT",),
        metadata={"source": "unit-test"},
    )

    assert cube.calendar.equals(calendar)
    assert cube.instrument_ids == instrument_ids
    assert cube.eligible.shape == (2, 2)
    assert cube.capacity_usdt.dtype == np.float64
    assert bool(cube.exit_required[1, 0]) is True
    assert snapshot.decision_at == datetime(2025, 1, 1, tzinfo=UTC)
    assert snapshot.eligibilities[0].eligible is True
    assert [field.name for field in fields(UniverseStateCube)] == [
        "calendar",
        "instrument_ids",
        "eligible",
        "entry_block",
        "exit_required",
        "capacity_usdt",
        "risk_scale",
        "cost_bps",
    ]


def test_strategy_readiness_contract_keeps_strategy_axis_and_reason_codes() -> None:
    """Strategy readiness cubes must carry the full 3D readiness surface."""

    calendar = pd.DatetimeIndex([pd.Timestamp("2025-01-01T00:00:00Z")])
    strategies = ("trend_ma", "mean_reversion")
    instrument_ids = ("BTCUSDT",)
    ready = np.array([[[True]], [[False]]], dtype=bool)
    reason_code = np.array([[["READY"]], [["INSUFFICIENT_HISTORY"]]], dtype=object)

    requirement = StrategyRequirement(
        strategy="trend_ma",
        required_lookback_bars=120,
        required_features=("close", "volume"),
        requires_funding=True,
        requires_open_interest=False,
        min_training_events=5,
        min_data_confidence=DataConfidence.OBSERVED,
    )
    cube = StrategyReadinessCube(
        strategies=strategies,
        calendar=calendar,
        instrument_ids=instrument_ids,
        ready=ready,
        reason_code=reason_code,
    )

    assert requirement.required_lookback_bars == 120
    assert requirement.requires_funding is True
    assert cube.strategies == strategies
    assert cube.ready.shape == (2, 1, 1)
    assert cube.reason_code.dtype == object
    assert [field.name for field in fields(StrategyRequirement)] == [
        "strategy",
        "required_lookback_bars",
        "required_features",
        "requires_funding",
        "requires_open_interest",
        "min_training_events",
        "min_data_confidence",
    ]
    assert [field.name for field in fields(StrategyReadinessCube)] == [
        "strategies",
        "calendar",
        "instrument_ids",
        "ready",
        "reason_code",
    ]
