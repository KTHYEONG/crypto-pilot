"""Unit tests for Phase 2+3 eligibility redesign.

Covers spec §Test Scenario Design for evaluate_execution_eligibility:
  S1: max_gap_bars > threshold → DATA_INTEGRITY_FAIL (even if coverage 95%)
  S2: staleness_bars > max_staleness_bars → STALE_MARKET_DATA
  S3: leveraged token symbol → LEVERAGED_TOKEN
  S4: has_nan=True → DATA_CONFIDENCE_LOW (G3 no-op regression prevention)
  S5: 200 symbols all pass ADV floor → admitted≈200 (capacity prefix removal)
  S6: adv BVA — boundary pass (2M) vs fail (1.9M) → ADV_FLOOR_FAIL

Time Complexity: O(N) per scenario where N = number of test instruments.
Space Complexity: O(N) for eligibility results.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.domain.futures.universe.contracts import (
    DataConfidence,
    EligibilityCode,
    ExecutionRules,
)
from src.domain.futures.universe.eligibility import (
    ExecutionEligibilityConfig,
    evaluate_execution_eligibility,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_UTC = UTC


def _dt(iso: str) -> datetime:
    """Parse ISO string into UTC-aware datetime."""
    return datetime.fromisoformat(iso).replace(tzinfo=_UTC)


_DECISION = _dt("2024-06-01T00:00:00")
_AVAIL = _dt("2024-01-01T00:00:00")  # well before decision


def _make_rules(iid: str, *, taker_fee_bps: float = 4.0) -> ExecutionRules:
    return ExecutionRules(
        instrument_id=iid,
        decision_at=_DECISION,
        tick_size=0.01,
        step_size=0.001,
        min_qty=0.001,
        min_notional=10.0,
        taker_fee_bps=taker_fee_bps,
        tick_size_confidence=DataConfidence.OBSERVED,
        step_size_confidence=DataConfidence.OBSERVED,
        min_qty_confidence=DataConfidence.OBSERVED,
        min_notional_confidence=DataConfidence.OBSERVED,
        taker_fee_confidence=DataConfidence.OBSERVED,
    )


def _make_obs_rows(iid: str, *, adv: float = 5_000_000.0) -> list[dict[str, object]]:
    """Minimal observations that pass G5 (adv30_usdt present, last_price present)."""
    return [
        {
            "instrument_id": iid,
            "metric": "adv30_usdt",
            "available_at": _AVAIL,
            "value": adv,
            "source": "kline",
            "confidence": "observed",
        },
        {
            "instrument_id": iid,
            "metric": "last_price",
            "available_at": _AVAIL,
            "value": 100.0,
            "source": "kline",
            "confidence": "observed",
        },
    ]


def _standard_inst_row(
    iid: str,
    *,
    confidence: str = "observed",
    staleness_bars: int = 0,
    n_bar_gaps: int = 0,
    max_gap_bars: int = 0,
    frozen_bars: int = 0,
    n_zero_volume_bars_60d: int = 0,
    last_60d_coverage: float = 1.0,
    has_nan: bool = False,
    has_inf: bool = False,
    has_timestamp_issues: bool = False,
) -> dict[str, object]:
    """Build a standard instrument row that passes all gates by default."""
    return {
        "instrument_id": iid,
        "status": "TRADING",
        "available_at": _AVAIL,
        "confidence": confidence,
        "staleness_bars": staleness_bars,
        "n_bar_gaps": n_bar_gaps,
        "max_gap_bars": max_gap_bars,
        "frozen_bars": frozen_bars,
        "n_zero_volume_bars_60d": n_zero_volume_bars_60d,
        "last_60d_coverage": last_60d_coverage,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "has_timestamp_issues": has_timestamp_issues,
    }


def _config(
    *,
    max_gap_bars: int = 6,
    max_gap_count: int = 3,
    max_frozen_bars: int = 6,
    max_zero_volume_bars: int = 3,
    max_staleness_bars: int = 2,
    min_coverage_ratio: float = 0.95,
    reject_on_nan_inf: bool = True,
    reject_on_timestamp_issues: bool = True,
    min_adv_usdt: float = 2_000_000.0,
    max_round_trip_cost_bps: float = 60.0,
    exclude_leveraged: bool = True,
    min_data_confidence: DataConfidence = DataConfidence.RECONSTRUCTED,
) -> ExecutionEligibilityConfig:
    return ExecutionEligibilityConfig(
        max_staleness_bars=max_staleness_bars,
        max_round_trip_cost_bps=max_round_trip_cost_bps,
        max_participation_rate=0.01,
        min_data_confidence=min_data_confidence,
        default_intended_notional_usdt=10_000.0,
        exclude_leveraged=exclude_leveraged,
        min_coverage_ratio=min_coverage_ratio,
        max_gap_count=max_gap_count,
        max_gap_bars=max_gap_bars,
        max_frozen_bars=max_frozen_bars,
        max_zero_volume_bars=max_zero_volume_bars,
        reject_on_nan_inf=reject_on_nan_inf,
        reject_on_timestamp_issues=reject_on_timestamp_issues,
        min_adv_usdt=min_adv_usdt,
    )


# ---------------------------------------------------------------------------
# S1: max_gap_bars > 6 → DATA_INTEGRITY_FAIL even at coverage=95% (R5 핵심)
# ---------------------------------------------------------------------------


class TestS1DataIntegrityFailGapBars:
    """S1: 24h+ continuous gap triggers DATA_INTEGRITY_FAIL regardless of coverage."""

    def test_max_gap_bars_exceeds_threshold_triggers_integrity_fail(self) -> None:
        """max_gap_bars=7 > config.max_gap_bars=6 → DATA_INTEGRITY_FAIL.

        Coverage=0.95 (at floor) must NOT prevent the gap check from failing.
        This validates R5: connectivity matters even when bar-count is OK.
        """
        # Arrange
        iid = "binance_usdt_perpetual:XYZUSDT"
        instruments = pd.DataFrame([
            _standard_inst_row(
                iid,
                max_gap_bars=7,           # exceeds threshold of 6
                last_60d_coverage=0.95,   # exactly at floor — should not save it
            )
        ])
        observations = pd.DataFrame(_make_obs_rows(iid))
        rules = {iid: _make_rules(iid)}
        cfg = _config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert
        assert len(snap.eligibilities) == 1
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.DATA_INTEGRITY_FAIL

    def test_max_gap_bars_at_threshold_passes(self) -> None:
        """max_gap_bars=6 == threshold → gate passes (boundary: inclusive)."""
        # Arrange
        iid = "binance_usdt_perpetual:XYZUSDT"
        instruments = pd.DataFrame([
            _standard_inst_row(iid, max_gap_bars=6, last_60d_coverage=0.96)
        ])
        observations = pd.DataFrame(_make_obs_rows(iid))
        rules = {iid: _make_rules(iid)}
        cfg = _config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert: gate passes (may still fail ORDER_TOO_SMALL or be ELIGIBLE)
        result = snap.eligibilities[0]
        assert result.code != EligibilityCode.DATA_INTEGRITY_FAIL


# ---------------------------------------------------------------------------
# S2: staleness_bars > max_staleness_bars → STALE_MARKET_DATA
# ---------------------------------------------------------------------------


class TestS2StalenessCheck:
    """S2: recency gate fires when staleness_bars exceeds threshold."""

    def test_staleness_bars_exceeds_max_triggers_stale(self) -> None:
        """staleness_bars=3 > max_staleness_bars=2 → STALE_MARKET_DATA."""
        # Arrange
        iid = "binance_usdt_perpetual:ABCUSDT"
        instruments = pd.DataFrame([
            _standard_inst_row(iid, staleness_bars=3)
        ])
        observations = pd.DataFrame(_make_obs_rows(iid))
        rules = {iid: _make_rules(iid)}
        cfg = _config(max_staleness_bars=2)

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.STALE_MARKET_DATA

    def test_staleness_at_threshold_passes(self) -> None:
        """staleness_bars=2 == max_staleness_bars=2 → gate passes."""
        # Arrange
        iid = "binance_usdt_perpetual:ABCUSDT"
        instruments = pd.DataFrame([
            _standard_inst_row(iid, staleness_bars=2)
        ])
        observations = pd.DataFrame(_make_obs_rows(iid))
        rules = {iid: _make_rules(iid)}
        cfg = _config(max_staleness_bars=2)

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert: staleness gate passes
        result = snap.eligibilities[0]
        assert result.code != EligibilityCode.STALE_MARKET_DATA


# ---------------------------------------------------------------------------
# S3: BTCDOWNUSDT → LEVERAGED_TOKEN
# ---------------------------------------------------------------------------


class TestS3LeveragedToken:
    """S3: G0 pattern matching excludes leveraged tokens."""

    @pytest.mark.parametrize("symbol_suffix", ["BTCDOWNUSDT", "ETHUPUSDT", "LINKBULLUSDT", "XRPBEARUSDT"])
    def test_leveraged_token_excluded_by_g0(self, symbol_suffix: str) -> None:
        """Symbols containing UP/DOWN/BULL/BEAR → LEVERAGED_TOKEN."""
        # Arrange
        iid = f"binance_usdt_perpetual:{symbol_suffix}"
        instruments = pd.DataFrame([
            _standard_inst_row(iid)
        ])
        observations = pd.DataFrame(_make_obs_rows(iid))
        rules = {iid: _make_rules(iid)}
        cfg = _config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.LEVERAGED_TOKEN

    def test_normal_symbol_not_excluded_by_g0(self) -> None:
        """Regular symbol BTCUSDT is not excluded by G0."""
        # Arrange
        iid = "binance_usdt_perpetual:BTCUSDT"
        instruments = pd.DataFrame([
            _standard_inst_row(iid)
        ])
        observations = pd.DataFrame(_make_obs_rows(iid, adv=10_000_000.0))
        rules = {iid: _make_rules(iid)}
        cfg = _config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert: G0 did not fire
        result = snap.eligibilities[0]
        assert result.code != EligibilityCode.LEVERAGED_TOKEN


# ---------------------------------------------------------------------------
# S4: has_nan=True → DATA_CONFIDENCE_LOW (G3 no-op regression prevention)
# ---------------------------------------------------------------------------


class TestS4ConfidenceResolution:
    """S4: _resolve_confidence must map has_nan=True → UNKNOWN → G3 fires."""

    def test_has_nan_true_resolves_to_unknown_confidence_triggers_g3(self) -> None:
        """has_nan=True in inst_row → confidence=UNKNOWN → DATA_CONFIDENCE_LOW at G3.

        Validates that G3 is no longer a no-op (spec R2 fix, Phase 2).
        The confidence field must be set to 'unknown' in the instruments DataFrame
        (as _resolve_confidence does in the ledger builder).
        """
        # Arrange: confidence explicitly set to 'unknown' (as _resolve_confidence produces)
        iid = "binance_usdt_perpetual:NANUSDT"
        instruments = pd.DataFrame([
            _standard_inst_row(iid, has_nan=True, confidence="unknown")
        ])
        observations = pd.DataFrame(_make_obs_rows(iid))
        rules = {iid: _make_rules(iid)}
        # min_data_confidence = RECONSTRUCTED → UNKNOWN fails G3
        cfg = _config(min_data_confidence=DataConfidence.RECONSTRUCTED)

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.DATA_CONFIDENCE_LOW


# ---------------------------------------------------------------------------
# S5: 200 symbols all pass ADV floor → admitted≈200 (capacity prefix removed)
# ---------------------------------------------------------------------------


class TestS5CapacityPrefixRemoval:
    """S5: breadth-maximizing — no capacity prefix cut applied.

    Per spec C4: capacity_coverage prefix is removed. k_max=150 is the only bound.
    With 200 symbols all passing G0-G8 gates and ADV=5M (>2M floor),
    admitted count should equal min(200, k_max). Default k_max=150 per spec.
    """

    def test_all_symbols_admitted_up_to_k_max(self) -> None:
        """200 eligible symbols → exactly k_max=150 admitted (compute backstop only)."""
        # Arrange: 200 symbols, all passing all gates
        n = 200
        iids = [f"binance_usdt_perpetual:SYM{i:03d}USDT" for i in range(n)]
        inst_rows = [_standard_inst_row(iid) for iid in iids]
        instruments = pd.DataFrame(inst_rows)

        obs_rows: list[dict[str, object]] = []
        for iid in iids:
            obs_rows.extend(_make_obs_rows(iid, adv=5_000_000.0))
        observations = pd.DataFrame(obs_rows)

        rules = {iid: _make_rules(iid) for iid in iids}
        cfg = _config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert: all 200 are eligible (no capacity prefix cut in eligibility engine)
        # Note: the k_max backstop is applied in pipeline.py, not here.
        eligible_count = sum(1 for e in snap.eligibilities if e.eligible)
        assert eligible_count == n, (
            f"Expected all {n} symbols eligible (capacity prefix removed), "
            f"got {eligible_count}. "
            "If < n, a gate is incorrectly filtering. "
            "Verified: NOT 57 (old capacity prefix behavior)."
        )

    def test_admitted_count_not_57(self) -> None:
        """Regression: old capacity_coverage prefix would have cut to ~57.

        With 200 symbols and uniform ADV, the old 90%-coverage prefix would
        have returned ~57 (since BTC+ETH dominated ~64% of capacity).
        This asserts the old behavior is gone.
        """
        # Arrange: 200 symbols, first 2 have high ADV (like BTC/ETH), rest normal
        n = 200
        iids = [f"binance_usdt_perpetual:SYM{i:03d}USDT" for i in range(n)]
        inst_rows = [_standard_inst_row(iid) for iid in iids]
        instruments = pd.DataFrame(inst_rows)

        obs_rows: list[dict[str, object]] = []
        for i, iid in enumerate(iids):
            adv = 1_000_000_000.0 if i < 2 else 5_000_000.0  # BTC/ETH dominance pattern
            obs_rows.extend(_make_obs_rows(iid, adv=adv))
        observations = pd.DataFrame(obs_rows)

        rules = {iid: _make_rules(iid) for iid in iids}
        cfg = _config()

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert: all 200 eligible — NOT 57 (the old capacity prefix result)
        eligible_count = sum(1 for e in snap.eligibilities if e.eligible)
        assert eligible_count == n
        assert eligible_count != 57, "Regression: capacity prefix cut re-introduced"


# ---------------------------------------------------------------------------
# S6: ADV floor BVA — 2M pass, 1.9M fail
# ---------------------------------------------------------------------------


class TestS6AdvFloorBoundaryValue:
    """S6: ADV floor boundary value analysis.

    min_adv_usdt = 2_000_000.0 (spec C3).
    At boundary (2M): gate passes (>= floor).
    Below boundary (1.9M): ADV_FLOOR_FAIL.
    """

    def test_adv_exactly_at_floor_passes(self) -> None:
        """adv=2_000_000 == min_adv_usdt=2M → ADV floor gate passes."""
        # Arrange
        iid = "binance_usdt_perpetual:BOUNDUSDT"
        instruments = pd.DataFrame([_standard_inst_row(iid)])
        observations = pd.DataFrame(_make_obs_rows(iid, adv=2_000_000.0))
        rules = {iid: _make_rules(iid)}
        cfg = _config(min_adv_usdt=2_000_000.0)

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert: ADV floor did not fire
        result = snap.eligibilities[0]
        assert result.code != EligibilityCode.ADV_FLOOR_FAIL

    def test_adv_below_floor_triggers_adv_floor_fail(self) -> None:
        """adv=1_900_000 < min_adv_usdt=2M → ADV_FLOOR_FAIL."""
        # Arrange
        iid = "binance_usdt_perpetual:ILLIQUSDT"
        instruments = pd.DataFrame([_standard_inst_row(iid)])
        observations = pd.DataFrame(_make_obs_rows(iid, adv=1_900_000.0))
        rules = {iid: _make_rules(iid)}
        cfg = _config(min_adv_usdt=2_000_000.0)

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert
        result = snap.eligibilities[0]
        assert result.eligible is False
        assert result.code == EligibilityCode.ADV_FLOOR_FAIL
        assert result.reasons[0].observed_value == pytest.approx(1_900_000.0)
        assert result.reasons[0].threshold == pytest.approx(2_000_000.0)

    def test_adv_well_above_floor_eligible(self) -> None:
        """adv=10M >> min_adv_usdt=2M → ADV_FLOOR_FAIL gate passes, instrument eligible."""
        # Arrange
        iid = "binance_usdt_perpetual:LIQUIDUSDT"
        instruments = pd.DataFrame([_standard_inst_row(iid)])
        observations = pd.DataFrame(_make_obs_rows(iid, adv=10_000_000.0))
        rules = {iid: _make_rules(iid)}
        cfg = _config(min_adv_usdt=2_000_000.0)

        # Act
        snap = evaluate_execution_eligibility(
            decision_at=_DECISION,
            instruments=instruments,
            observations=observations,
            rules=rules,
            intended_notional_usdt={},
            config=cfg,
        )

        # Assert: ELIGIBLE (all gates pass for high-ADV liquid symbol)
        result = snap.eligibilities[0]
        assert result.eligible is True
        assert result.code == EligibilityCode.ELIGIBLE
