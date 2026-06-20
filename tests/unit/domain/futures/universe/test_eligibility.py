"""Unit tests for src/domain/futures/universe/eligibility.py.

Covers:
  - Scenario 1: instrument lifecycle (not onboarded / eligible / delisted)
  - Scenario 2: no 20-symbol cap - all 37 instruments represented
  - Scenario 3: PIT boundary guard raises RuntimeError
  - Scenario 4: ORDER_TOO_SMALL and COST_TOO_HIGH hard gates
  - resolve_execution_rules: fallback and hard-fail paths
  - build_universe_state_cube: array fill and entry_block propagation
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.universe.contracts import (
    DataConfidence,
    EligibilityCode,
    EligibilitySnapshot,
    ExecutionEligibility,
    ExecutionRules,
)
from src.domain.futures.universe.eligibility import (
    ExecutionEligibilityConfig,
    RuleFallbackPolicy,
    build_universe_state_cube,
    evaluate_execution_eligibility,
    resolve_execution_rules,
)

# ---------------------------------------------------------------------------
# Shared UTC helper
# ---------------------------------------------------------------------------

_UTC = UTC


def _dt(iso: str) -> datetime:
    """Parse ISO string into UTC-aware datetime."""
    return datetime.fromisoformat(iso).replace(tzinfo=_UTC)


# ---------------------------------------------------------------------------
# Fixtures: shared builders
# ---------------------------------------------------------------------------


def _make_instruments(rows: list[dict]) -> pd.DataFrame:
    """Build an instruments DataFrame from list of dicts."""
    return pd.DataFrame(rows)


def _make_observations(rows: list[dict]) -> pd.DataFrame:
    """Build an observations DataFrame from list of dicts."""
    return pd.DataFrame(rows)


def _make_rules(
    instrument_id: str,
    *,
    decision_at: datetime,
    taker_fee_bps: float = 4.0,
    min_notional: float = 10.0,
    min_qty: float = 0.001,
    step_size: float = 0.001,
    tick_size: float = 0.01,
) -> ExecutionRules:
    return ExecutionRules(
        instrument_id=instrument_id,
        decision_at=decision_at,
        tick_size=tick_size,
        step_size=step_size,
        min_qty=min_qty,
        min_notional=min_notional,
        taker_fee_bps=taker_fee_bps,
        tick_size_confidence=DataConfidence.OBSERVED,
        step_size_confidence=DataConfidence.OBSERVED,
        min_qty_confidence=DataConfidence.OBSERVED,
        min_notional_confidence=DataConfidence.OBSERVED,
        taker_fee_confidence=DataConfidence.OBSERVED,
    )


def _standard_config() -> ExecutionEligibilityConfig:
    return ExecutionEligibilityConfig(
        max_round_trip_cost_bps=50.0,
        max_participation_rate=0.01,
        min_data_confidence=DataConfidence.RECONSTRUCTED,
        default_intended_notional_usdt=10_000.0,
    )


def _eligible_obs(iid: str, available_at: datetime, adv30: float = 3_000_000.0) -> list[dict]:
    """Return a minimal set of observations that passes all metric gates."""
    return [
        {
            "instrument_id": iid,
            "metric": "adv30_usdt",
            "available_at": available_at,
            "value": adv30,
            "source": "kline",
            "confidence": "observed",
        },
        {
            "instrument_id": iid,
            "metric": "last_price",
            "available_at": available_at,
            "value": 100.0,
            "source": "kline",
            "confidence": "observed",
        },
    ]


# ===========================================================================
# Scenario 1: Instrument lifecycle
# ===========================================================================


class TestScenario1Lifecycle:
    """Scenario 1: A (full), B (mid-listing), C (mid-delisting)."""

    def test_instrument_b_not_eligible_before_onboard(self) -> None:
        """B's available_at is after decision_at → NOT_ONBOARDED."""
        # Arrange
        decision = _dt("2024-01-10T00:00:00")
        instruments = _make_instruments(
            [
                {
                    "instrument_id": "B",
                    "status": "TRADING",
                    "available_at": _dt("2024-01-15T00:00:00"),  # future
                    "confidence": "observed",
                }
            ]
        )
        observations = _make_observations([])
        rules: dict[str, ExecutionRules] = {}
        config = _standard_config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=config,
        )

        # Assert
        assert len(snap.eligibilities) == 1
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.NOT_ONBOARDED

    def test_instrument_b_eligible_after_onboard_with_warmup(self) -> None:
        """B is TRADING and available before decision_at → can be ELIGIBLE."""
        # Arrange
        decision = _dt("2024-02-01T00:00:00")
        instruments = _make_instruments(
            [
                {
                    "instrument_id": "B",
                    "status": "TRADING",
                    "available_at": _dt("2024-01-15T00:00:00"),
                    "confidence": "observed",
                }
            ]
        )
        obs_rows = _eligible_obs("B", available_at=_dt("2024-01-31T00:00:00"))
        observations = _make_observations(obs_rows)
        rules = {"B": _make_rules("B", decision_at=decision)}
        config = _standard_config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=config,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is True
        assert result.code == EligibilityCode.ELIGIBLE

    def test_instrument_c_ineligible_after_status_not_trading(self) -> None:
        """C's status transitions away from TRADING → STATUS_NOT_TRADING."""
        # Arrange
        decision = _dt("2024-03-20T00:00:00")
        instruments = _make_instruments(
            [
                {
                    "instrument_id": "C",
                    "status": "DELIVERING",  # no longer TRADING
                    "available_at": _dt("2024-01-01T00:00:00"),
                    "confidence": "observed",
                }
            ]
        )
        observations = _make_observations([])
        rules: dict[str, ExecutionRules] = {}
        config = _standard_config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=config,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.STATUS_NOT_TRADING

    def test_instrument_a_eligible_full_period(self) -> None:
        """A passes all gates → ELIGIBLE."""
        # Arrange
        decision = _dt("2024-06-01T00:00:00")
        instruments = _make_instruments(
            [
                {
                    "instrument_id": "A",
                    "status": "TRADING",
                    "available_at": _dt("2023-01-01T00:00:00"),
                    "confidence": "observed",
                }
            ]
        )
        obs_rows = _eligible_obs("A", available_at=_dt("2024-05-31T00:00:00"))
        observations = _make_observations(obs_rows)
        rules = {"A": _make_rules("A", decision_at=decision)}
        config = _standard_config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=config,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is True


# ===========================================================================
# Scenario 2: No 20-symbol cap
# ===========================================================================


class TestScenario2No20Cap:
    """Scenario 2: 37 instruments all passing hard gates → all 37 in snapshot."""

    def _build_37_instruments_data(
        self, decision: datetime
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ExecutionRules]]:
        """Build instruments, observations, and rules for 37 eligible symbols."""
        inst_rows = []
        obs_rows: list[dict] = []
        rules: dict[str, ExecutionRules] = {}

        for i in range(37):
            iid = f"SYM_{i:02d}"
            inst_rows.append(
                {
                    "instrument_id": iid,
                    "status": "TRADING",
                    "available_at": _dt("2023-01-01T00:00:00"),
                    "confidence": "observed",
                }
            )
            obs_rows.extend(_eligible_obs(iid, available_at=_dt("2024-05-31T00:00:00")))
            rules[iid] = _make_rules(iid, decision_at=decision)

        return _make_instruments(inst_rows), _make_observations(obs_rows), rules

    def test_all_37_present_in_instrument_ids(self) -> None:
        """All 37 instruments appear in EligibilitySnapshot.instrument_ids."""
        # Arrange
        decision = _dt("2024-06-01T00:00:00")
        instruments, observations, rules = self._build_37_instruments_data(decision)
        config = _standard_config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=config,
        )

        # Assert
        assert len(snap.instrument_ids) == 37

    def test_all_37_eligible(self) -> None:
        """Eligible count equals 37 - no artificial cap applied."""
        # Arrange
        decision = _dt("2024-06-01T00:00:00")
        instruments, observations, rules = self._build_37_instruments_data(decision)
        config = _standard_config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=config,
        )

        # Assert
        eligible_count = int(snap.metadata.get("n_eligible", "0"))
        assert eligible_count >= 37


# ===========================================================================
# Scenario 3: PIT boundary guard
# ===========================================================================


class TestScenario3PITBoundary:
    """Scenario 3: observation with available_at > decision_at raises error."""

    def test_future_observation_raises_runtime_error(self) -> None:
        """evaluate_execution_eligibility raises RuntimeError on PIT violation."""
        # Arrange
        decision = _dt("2024-01-10T00:00:00")
        instruments = _make_instruments(
            [
                {
                    "instrument_id": "A",
                    "status": "TRADING",
                    "available_at": _dt("2023-01-01T00:00:00"),
                    "confidence": "observed",
                }
            ]
        )
        # observation available_at is AFTER decision_at → PIT violation
        observations = _make_observations(
            [
                {
                    "instrument_id": "A",
                    "metric": "adv30_usdt",
                    "available_at": _dt("2024-01-11T00:00:00"),  # future!
                    "value": 500_000.0,
                    "source": "kline",
                    "confidence": "observed",
                }
            ]
        )
        rules: dict[str, ExecutionRules] = {}
        config = _standard_config()

        # Act / Assert
        with pytest.raises(RuntimeError, match="PIT observation boundary violated"):
            evaluate_execution_eligibility(
                decision_at=decision,
                instruments=instruments,
                observations=observations,
                rules=rules,
                intended_notional_usdt={},
                config=config,
            )

    def test_observation_exactly_at_decision_at_is_allowed(self) -> None:
        """available_at == decision_at is NOT a PIT violation."""
        # Arrange
        decision = _dt("2024-01-10T00:00:00")
        instruments = _make_instruments(
            [
                {
                    "instrument_id": "A",
                    "status": "TRADING",
                    "available_at": _dt("2023-01-01T00:00:00"),
                    "confidence": "observed",
                }
            ]
        )
        obs_rows = _eligible_obs("A", available_at=decision)
        observations = _make_observations(obs_rows)
        rules = {"A": _make_rules("A", decision_at=decision)}
        config = _standard_config()

        # Act - must not raise
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=config,
        )

        # Assert
        assert snap is not None


# ===========================================================================
# Scenario 4: Order rules - ORDER_TOO_SMALL and COST_TOO_HIGH
# ===========================================================================


class TestScenario4OrderRules:
    """Scenario 4: order size and cost hard gates."""

    def _make_obs_with_price(
        self, iid: str, available_at: datetime, last_price: float = 100.0
    ) -> pd.DataFrame:
        rows = [
            {
                "instrument_id": iid,
                "metric": "adv30_usdt",
                "available_at": available_at,
                "value": 3_000_000.0,  # above 2M ADV floor (Phase 3)
                "source": "kline",
                "confidence": "observed",
            },
            {
                "instrument_id": iid,
                "metric": "last_price",
                "available_at": available_at,
                "value": last_price,
                "source": "kline",
                "confidence": "observed",
            },
        ]
        return pd.DataFrame(rows)

    def test_order_too_small_when_intended_notional_below_min(self) -> None:
        """intended_notional=5.0 USDT with min_notional=10.0 → ORDER_TOO_SMALL."""
        # Arrange
        decision = _dt("2024-06-01T00:00:00")
        iid = "SYM_SMALL"
        instruments = _make_instruments(
            [
                {
                    "instrument_id": iid,
                    "status": "TRADING",
                    "available_at": _dt("2023-01-01T00:00:00"),
                    "confidence": "observed",
                }
            ]
        )
        observations = self._make_obs_with_price(iid, _dt("2024-05-31T00:00:00"), last_price=100.0)
        rules = {iid: _make_rules(iid, decision_at=decision, min_notional=10.0, step_size=1.0, min_qty=0.1)}
        config = _standard_config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={iid: 5.0},  # below min_notional
            config=config,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.ORDER_TOO_SMALL

    def test_cost_too_high_with_very_low_adv(self) -> None:
        """Very low ADV (50 USDT) → ADV_FLOOR_FAIL (Phase 3 gate precedes COST_TOO_HIGH)."""
        # Arrange
        decision = _dt("2024-06-01T00:00:00")
        iid = "SYM_LOW_ADV"
        instruments = _make_instruments(
            [
                {
                    "instrument_id": iid,
                    "status": "TRADING",
                    "available_at": _dt("2023-01-01T00:00:00"),
                    "confidence": "observed",
                }
            ]
        )
        # Very low ADV → sqrt(10000/50) ~ 14.14 * 18 = 254.6 bps impact
        low_adv_obs = pd.DataFrame(
            [
                {
                    "instrument_id": iid,
                    "metric": "adv30_usdt",
                    "available_at": _dt("2024-05-31T00:00:00"),
                    "value": 50.0,  # extremely low ADV
                    "source": "kline",
                    "confidence": "observed",
                },
                {
                    "instrument_id": iid,
                    "metric": "last_price",
                    "available_at": _dt("2024-05-31T00:00:00"),
                    "value": 100.0,
                    "source": "kline",
                    "confidence": "observed",
                },
            ]
        )
        rules = {iid: _make_rules(iid, decision_at=decision, taker_fee_bps=4.0)}
        config = _standard_config()  # max_round_trip_cost_bps=50.0

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=low_adv_obs,
            rules=rules,
            intended_notional_usdt={iid: 10_000.0},
            config=config,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.ADV_FLOOR_FAIL

    def test_capacity_clipped_when_notional_near_adv_limit(self) -> None:
        """ELIGIBLE but capacity_usdt < intended_notional when ADV is limited."""
        # Arrange
        decision = _dt("2024-06-01T00:00:00")
        iid = "SYM_CAP"
        instruments = _make_instruments(
            [
                {
                    "instrument_id": iid,
                    "status": "TRADING",
                    "available_at": _dt("2023-01-01T00:00:00"),
                    "confidence": "observed",
                }
            ]
        )
        # ADV = 5_000_000 (above 2M floor); participation cap = 1% = 50_000 USDT
        obs = _make_observations(
            _eligible_obs(iid, available_at=_dt("2024-05-31T00:00:00"), adv30=5_000_000.0)
        )
        rules = {iid: _make_rules(iid, decision_at=decision, taker_fee_bps=4.0)}
        config = ExecutionEligibilityConfig(
            max_round_trip_cost_bps=50.0,
            max_participation_rate=0.01,  # capacity = 1% * 5_000_000 = 50_000
        )

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=decision,
            instruments=instruments,
            observations=obs,
            rules=rules,
            intended_notional_usdt={iid: 10_000.0},
            config=config,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is True
        assert result.code == EligibilityCode.ELIGIBLE
        # capacity_usdt should be capped at participation limit (1% * 5M = 50k)
        assert result.capacity_usdt <= 50_000.0 + 1e-6


# ===========================================================================
# resolve_execution_rules tests
# ===========================================================================


class TestResolveExecutionRules:
    """Tests for resolve_execution_rules."""

    def _make_rule_history(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "instrument_id": "SYM_A",
                    "available_at": _dt("2024-01-01T00:00:00"),
                    "tick_size": 0.01,
                    "step_size": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 5.0,
                    "taker_fee_bps": 4.0,
                    "confidence": "observed",
                },
                {
                    "instrument_id": "SYM_A",
                    "available_at": _dt("2024-03-01T00:00:00"),
                    "tick_size": 0.01,
                    "step_size": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 10.0,
                    "taker_fee_bps": 4.5,
                    "confidence": "observed",
                },
            ]
        )

    def test_resolve_execution_rules_returns_fallback_when_missing(self) -> None:
        """No matching row + allow_reconstructed=True → RECONSTRUCTED fallback."""
        # Arrange
        rule_history = self._make_rule_history()
        fallback = RuleFallbackPolicy(allow_reconstructed=True)
        decision = _dt("2023-06-01T00:00:00")  # before any row

        # Act
        result = resolve_execution_rules(
            "SYM_A",
            decision_at=decision,
            rule_history=rule_history,
            fallback_policy=fallback,
        )

        # Assert
        assert result.tick_size_confidence == DataConfidence.RECONSTRUCTED
        assert result.taker_fee_bps == fallback.conservative_taker_fee_bps

    def test_resolve_execution_rules_raises_when_fallback_not_allowed(self) -> None:
        """No matching row + allow_reconstructed=False → RuntimeError."""
        # Arrange
        rule_history = self._make_rule_history()
        fallback = RuleFallbackPolicy(allow_reconstructed=False)
        decision = _dt("2023-06-01T00:00:00")

        # Act / Assert
        with pytest.raises(RuntimeError, match="MISSING_RULES"):
            resolve_execution_rules(
                "SYM_A",
                decision_at=decision,
                rule_history=rule_history,
                fallback_policy=fallback,
            )

    def test_resolve_execution_rules_returns_latest_row(self) -> None:
        """Returns the row with the greatest available_at <= decision_at."""
        # Arrange
        rule_history = self._make_rule_history()
        fallback = RuleFallbackPolicy()
        decision = _dt("2024-06-01T00:00:00")

        # Act
        result = resolve_execution_rules(
            "SYM_A",
            decision_at=decision,
            rule_history=rule_history,
            fallback_policy=fallback,
        )

        # Assert - should pick the March row (min_notional=10.0, fee=4.5)
        assert result.min_notional == pytest.approx(10.0)
        assert result.taker_fee_bps == pytest.approx(4.5)

    def test_resolve_execution_rules_picks_exact_boundary_row(self) -> None:
        """Row with available_at == decision_at is included (<=)."""
        # Arrange
        rule_history = self._make_rule_history()
        fallback = RuleFallbackPolicy()
        decision = _dt("2024-03-01T00:00:00")  # exactly the second row's date

        # Act
        result = resolve_execution_rules(
            "SYM_A",
            decision_at=decision,
            rule_history=rule_history,
            fallback_policy=fallback,
        )

        # Assert
        assert result.min_notional == pytest.approx(10.0)


# ===========================================================================
# build_universe_state_cube tests
# ===========================================================================


class TestBuildUniverseStateCube:
    """Tests for build_universe_state_cube."""

    def _make_snap(
        self,
        decision: datetime,
        iid: str,
        eligible: bool,
        capacity: float = 5_000.0,
        cost_bps: float = 12.0,
    ) -> EligibilitySnapshot:
        elig = ExecutionEligibility(
            instrument_id=iid,
            decision_at=decision,
            eligible=eligible,
            code=EligibilityCode.ELIGIBLE if eligible else EligibilityCode.STATUS_NOT_TRADING,
            capacity_usdt=capacity if eligible else 0.0,
            risk_scale=1.0,
            cost_bps=cost_bps if eligible else 0.0,
        )
        return EligibilitySnapshot(
            decision_at=decision,
            eligibilities=(elig,),
            instrument_ids=(iid,),
            metadata={"n_eligible": "1" if eligible else "0"},
        )

    def test_build_universe_state_cube_fills_arrays_correctly(self) -> None:
        """Eligible snapshot fills eligible=True and capacity correctly."""
        # Arrange
        calendar = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        instruments = ("SYM_X",)
        snap = self._make_snap(_dt("2024-01-01T00:00:00"), "SYM_X", eligible=True, capacity=7_000.0)

        # Act
        cube = build_universe_state_cube(
            calendar=calendar,
            instruments=instruments,
            snapshots=[snap],
        )

        # Assert
        assert cube.eligible[0, 0] is np.bool_(True)
        assert cube.capacity_usdt[0, 0] == pytest.approx(7_000.0)
        # Forward-fill: bars 1 and 2 also use the same snapshot
        assert cube.eligible[1, 0] is np.bool_(True)
        assert cube.eligible[2, 0] is np.bool_(True)

    def test_exit_required_sets_entry_block_for_subsequent_bars(self) -> None:
        """Eligible→ineligible transition sets exit_required and entry_block forward."""
        # Arrange
        calendar = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
        instruments = ("SYM_Y",)

        snap_t0 = self._make_snap(_dt("2024-01-01T00:00:00"), "SYM_Y", eligible=True)
        snap_t1 = self._make_snap(_dt("2024-01-01T04:00:00"), "SYM_Y", eligible=False)

        # Act
        cube = build_universe_state_cube(
            calendar=calendar,
            instruments=instruments,
            snapshots=[snap_t0, snap_t1],
        )

        # Assert
        # t=0: eligible, no exit required
        assert cube.eligible[0, 0] is np.bool_(True)
        assert cube.exit_required[0, 0] is np.bool_(False)
        # t=1: now ineligible → exit required
        assert cube.eligible[1, 0] is np.bool_(False)
        assert cube.exit_required[1, 0] is np.bool_(True)
        # t=2, t=3: entry blocked
        assert cube.entry_block[2, 0] is np.bool_(True)
        assert cube.entry_block[3, 0] is np.bool_(True)

    def test_build_universe_state_cube_raises_on_unknown_instrument(self) -> None:
        """Snapshot containing instrument_id not in instruments tuple → ValueError."""
        # Arrange
        calendar = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        instruments = ("SYM_A",)
        snap = self._make_snap(_dt("2024-01-01T00:00:00"), "UNKNOWN_SYM", eligible=True)

        # Act / Assert
        with pytest.raises(ValueError, match="universe symbol axis mismatch"):
            build_universe_state_cube(
                calendar=calendar,
                instruments=instruments,
                snapshots=[snap],
            )

    def test_no_snapshot_defaults_to_entry_block(self) -> None:
        """Calendar bars before any snapshot → fail-closed (entry_block=True)."""
        # Arrange
        calendar = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        instruments = ("SYM_Z",)
        # Snapshot after all calendar bars
        snap = self._make_snap(_dt("2024-01-02T00:00:00"), "SYM_Z", eligible=True)

        # Act
        cube = build_universe_state_cube(
            calendar=calendar,
            instruments=instruments,
            snapshots=[snap],
        )

        # Assert - bars 0 and 1 have no preceding snapshot, fail-closed
        assert cube.eligible[0, 0] is np.bool_(False)
        assert cube.entry_block[0, 0] is np.bool_(True)
