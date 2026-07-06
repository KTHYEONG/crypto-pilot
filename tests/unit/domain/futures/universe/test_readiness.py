"""Unit tests for evaluate_strategy_readiness (Scenario 6 coverage).

Validates the PIT strategy readiness cube under:
- Price-only strategies (no auxiliary data requirements)
- Funding-required strategies without funding data
- Lookback sufficiency gates
- Not-eligible instruments
- Output shape contract [S, T, N]
- reason_code values for each gate

All tests use synthetic in-memory data; no network or filesystem I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.universe.contracts import (
    StrategyRequirement,
    UniverseStateCube,
)
from src.domain.futures.universe.readiness import (
    _REASON_INSUFFICIENT_FUNDING,
    _REASON_INSUFFICIENT_LOOKBACK,
    _REASON_MISSING_FUNDING,
    _REASON_NOT_ELIGIBLE,
    _REASON_READY,
    evaluate_strategy_readiness,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOOKBACK_BARS: int = 10
_T: int = 30
_N: int = 3


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_eligibility_cube(n_bars: int, n_instr: int, *, eligible: bool = True) -> UniverseStateCube:
    """Build a minimal UniverseStateCube with uniform eligibility.

    Args:
        T: Number of bars in the calendar.
        N: Number of instruments.
        eligible: Whether all cells are eligible (default True).

    Returns:
        UniverseStateCube with the requested shape and eligibility mask.
    """
    calendar = pd.date_range("2024-01-01", periods=n_bars, freq="4h", tz="UTC")
    instruments: tuple[str, ...] = tuple(f"SYM{i}" for i in range(n_instr))
    return UniverseStateCube(
        calendar=calendar,
        instrument_ids=instruments,
        eligible=np.full((n_bars, n_instr), eligible, dtype=np.bool_),
        entry_block=np.zeros((n_bars, n_instr), dtype=np.bool_),
        exit_required=np.zeros((n_bars, n_instr), dtype=np.bool_),
        capacity_usdt=np.zeros((n_bars, n_instr), dtype=np.float64),
        risk_scale=np.ones((n_bars, n_instr), dtype=np.float64),
        cost_bps=np.zeros((n_bars, n_instr), dtype=np.float64),
    )


class _AlignedStub:
    """Minimal aligned market data stub for test injection."""

    def __init__(
        self,
        close: np.ndarray,
        funding_rate: np.ndarray | None = None,
        open_interest: np.ndarray | None = None,
    ) -> None:
        self.close = close
        if funding_rate is not None:
            self.funding_rate = funding_rate
        if open_interest is not None:
            self.open_interest = open_interest


def _make_price_only_requirement(
    strategy: str = "price_strategy",
    lookback: int = _LOOKBACK_BARS,
) -> StrategyRequirement:
    """Build a price-only StrategyRequirement."""
    return StrategyRequirement(
        strategy=strategy,
        required_lookback_bars=lookback,
        requires_funding=False,
        requires_open_interest=False,
    )


def _make_funding_requirement(
    strategy: str = "funding_strategy",
    lookback: int = _LOOKBACK_BARS,
) -> StrategyRequirement:
    """Build a StrategyRequirement that requires funding data."""
    return StrategyRequirement(
        strategy=strategy,
        required_lookback_bars=lookback,
        requires_funding=True,
        requires_open_interest=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReadinessCubeShape:
    """Verify output shape contract [S, T, N]."""

    def test_readiness_cube_shape_matches_inputs(self) -> None:
        """Readiness cube arrays have shape (S, T, N) matching inputs."""
        # Arrange
        T, N = _T, _N
        close = np.full((T, N), 100.0, dtype=np.float64)
        aligned = _AlignedStub(close=close)
        eligibility = _make_eligibility_cube(T, N)
        requirements = {
            "strat_a": _make_price_only_requirement("strat_a", lookback=5),
            "strat_b": _make_price_only_requirement("strat_b", lookback=8),
        }

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert
        S = len(requirements)
        assert cube.ready.shape == (S, T, N)
        assert cube.reason_code.shape == (S, T, N)
        assert cube.ready.dtype == np.bool_
        assert cube.reason_code.dtype == object
        assert cube.strategies == ("strat_a", "strat_b")


class TestPriceOnlyStrategyReadiness:
    """Scenario 6: price-only strategy is ready after its own lookback."""

    def test_strategy_requiring_only_price_ready_after_lookback(self) -> None:
        """Price strategy becomes ready exactly at bar index lookback-1."""
        # Arrange
        LOOKBACK = _LOOKBACK_BARS
        T, N = _T, _N
        close = np.full((T, N), 50.0, dtype=np.float64)
        aligned = _AlignedStub(close=close)
        eligibility = _make_eligibility_cube(T, N)
        requirements = {"strat": _make_price_only_requirement(lookback=LOOKBACK)}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert: first LOOKBACK bars have insufficient finite count
        assert not cube.ready[0, : LOOKBACK - 1, :].any(), "Bars before lookback-1 must not be ready"
        # From bar LOOKBACK-1 onwards, all instruments must be ready
        assert cube.ready[0, LOOKBACK - 1 :, :].all(), "All bars at or after lookback must be ready"

    def test_global_history_gate_does_not_block_price_only_strategy(self) -> None:
        """Price strategy with short lookback is unaffected by large T window."""
        # Arrange: T=100 bars, lookback=10 — ready from bar index 9 onward
        T, N = 100, 2
        LOOKBACK = 10
        close = np.full((T, N), 30.0, dtype=np.float64)
        aligned = _AlignedStub(close=close)
        eligibility = _make_eligibility_cube(T, N)
        requirements = {"price_short": _make_price_only_requirement(lookback=LOOKBACK)}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert: no global gate blocks bars in [LOOKBACK-1 : T]
        ready_from = cube.ready[0, LOOKBACK - 1 :, :]
        assert ready_from.all(), "Global history (T=100) must not impose additional gate beyond lookback=10"


class TestInsufficientLookback:
    """Verify insufficient_lookback reason_code assignment."""

    def test_insufficient_lookback_reason_code(self) -> None:
        """Bars before lookback carry reason_code='insufficient_lookback'."""
        # Arrange
        LOOKBACK = _LOOKBACK_BARS
        T, N = _T, 1
        close = np.full((T, N), 20.0, dtype=np.float64)
        aligned = _AlignedStub(close=close)
        eligibility = _make_eligibility_cube(T, N)
        requirements = {"strat": _make_price_only_requirement(lookback=LOOKBACK)}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert: bars [0, LOOKBACK-2] have insufficient_lookback
        pre_ready_reasons = cube.reason_code[0, : LOOKBACK - 1, 0]
        assert (pre_ready_reasons == _REASON_INSUFFICIENT_LOOKBACK).all(), (
            f"Expected '{_REASON_INSUFFICIENT_LOOKBACK}' for pre-lookback bars, got: {np.unique(pre_ready_reasons)}"
        )
        # Bar LOOKBACK-1 and beyond must be ready
        post_reasons = cube.reason_code[0, LOOKBACK - 1 :, 0]
        assert (post_reasons == _REASON_READY).all()


class TestFundingRequirement:
    """Scenario 6: funding strategy not ready without funding data."""

    def test_strategy_requiring_funding_not_ready_without_funding(self) -> None:
        """Funding strategy returns 'missing_funding' when funding array absent."""
        # Arrange
        T, N = _T, _N
        close = np.full((T, N), 100.0, dtype=np.float64)
        # Deliberately omit funding_rate from the stub
        aligned = _AlignedStub(close=close)
        eligibility = _make_eligibility_cube(T, N)
        requirements = {"funding_strat": _make_funding_requirement()}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert: no bar is ready
        assert not cube.ready[0].any(), "Funding strategy must not be ready without funding data"

        # Bars that passed lookback gate must carry missing_funding reason
        # (bars before lookback carry insufficient_lookback which is correct)
        lookback = _LOOKBACK_BARS
        eligible_mask = eligibility.eligible  # [T, N]
        post_lookback_mask = np.zeros((T, N), dtype=np.bool_)
        post_lookback_mask[lookback - 1 :, :] = True
        check_mask = eligible_mask & post_lookback_mask
        if check_mask.any():
            checked_reasons = cube.reason_code[0][check_mask]
            assert (checked_reasons == _REASON_MISSING_FUNDING).all(), (
                f"Expected '{_REASON_MISSING_FUNDING}', got: {np.unique(checked_reasons)}"
            )

    def test_strategy_requiring_funding_ready_when_funding_sufficient(self) -> None:
        """Funding strategy becomes ready when both close and funding have sufficient bars."""
        # Arrange
        LOOKBACK = _LOOKBACK_BARS
        T, N = _T, _N
        close = np.full((T, N), 100.0, dtype=np.float64)
        funding = np.full((T, N), 0.01, dtype=np.float64)
        aligned = _AlignedStub(close=close, funding_rate=funding)
        eligibility = _make_eligibility_cube(T, N)
        requirements = {"funding_strat": _make_funding_requirement(lookback=LOOKBACK)}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert: from LOOKBACK-1 onward, all instruments ready
        assert cube.ready[0, LOOKBACK - 1 :, :].all()

    def test_strategy_requiring_funding_insufficient_funding_reason_code(self) -> None:
        """Funding strategy with NaN-filled funding array returns insufficient_funding."""
        # Arrange
        LOOKBACK = _LOOKBACK_BARS
        T, N = _T, 1
        close = np.full((T, N), 100.0, dtype=np.float64)
        # All NaN funding — rolling count will always be 0
        funding = np.full((T, N), np.nan, dtype=np.float64)
        aligned = _AlignedStub(close=close, funding_rate=funding)
        eligibility = _make_eligibility_cube(T, N)
        requirements = {"funding_strat": _make_funding_requirement(lookback=LOOKBACK)}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert: bars past lookback that are eligible carry insufficient_funding
        # (bars before lookback carry insufficient_lookback which is expected)
        eligible_mask = eligibility.eligible  # [T, N]
        post_lookback_mask = np.zeros((T, N), dtype=np.bool_)
        post_lookback_mask[LOOKBACK - 1 :, :] = True
        check_mask = eligible_mask & post_lookback_mask
        if check_mask.any():
            checked_reasons = cube.reason_code[0][check_mask]
            assert (checked_reasons == _REASON_INSUFFICIENT_FUNDING).all(), (
                f"Expected '{_REASON_INSUFFICIENT_FUNDING}', got: {np.unique(checked_reasons)}"
            )


class TestNotEligibleInstruments:
    """Verify not_eligible instruments are always marked not_ready."""

    def test_not_eligible_instruments_always_not_ready(self) -> None:
        """Ineligible cells have reason_code='not_eligible' regardless of close data."""
        # Arrange
        T, N = _T, _N
        close = np.full((T, N), 100.0, dtype=np.float64)
        aligned = _AlignedStub(close=close)
        # All instruments ineligible
        eligibility = _make_eligibility_cube(T, N, eligible=False)
        requirements = {"strat": _make_price_only_requirement(lookback=5)}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert: no cell is ready
        assert not cube.ready[0].any(), "Ineligible cells must never be ready"
        # All cells carry not_eligible reason
        assert (cube.reason_code[0] == _REASON_NOT_ELIGIBLE).all(), (
            f"Expected '{_REASON_NOT_ELIGIBLE}', got: {np.unique(cube.reason_code[0])}"
        )

    def test_mixed_eligibility_ready_only_where_eligible(self) -> None:
        """Ready cells appear only in eligible instruments after lookback is satisfied."""
        # Arrange
        LOOKBACK = 5
        T, N = 20, 2
        close = np.full((T, N), 80.0, dtype=np.float64)
        aligned = _AlignedStub(close=close)

        # SYM0 eligible, SYM1 ineligible
        eligible_mask = np.array([[True, False]] * T, dtype=np.bool_)
        calendar = pd.date_range("2024-01-01", periods=T, freq="4h", tz="UTC")
        eligibility = UniverseStateCube(
            calendar=calendar,
            instrument_ids=("SYM0", "SYM1"),
            eligible=eligible_mask,
            entry_block=np.zeros((T, N), dtype=np.bool_),
            exit_required=np.zeros((T, N), dtype=np.bool_),
            capacity_usdt=np.zeros((T, N), dtype=np.float64),
            risk_scale=np.ones((T, N), dtype=np.float64),
            cost_bps=np.zeros((T, N), dtype=np.float64),
        )
        requirements = {"strat": _make_price_only_requirement(lookback=LOOKBACK)}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert SYM0 (col 0) ready from bar LOOKBACK-1 onward
        assert cube.ready[0, LOOKBACK - 1 :, 0].all()
        # Assert SYM1 (col 1) never ready
        assert not cube.ready[0, :, 1].any()
        # Assert SYM1 reason is not_eligible everywhere
        assert (cube.reason_code[0, :, 1] == _REASON_NOT_ELIGIBLE).all()


class TestNaNCloseHandling:
    """Verify NaN close values are counted as non-finite (lookback gate applies)."""

    def test_nan_close_bars_do_not_count_toward_lookback(self) -> None:
        """A window with LOOKBACK-1 finite and 1 NaN close is not ready."""
        # Arrange
        LOOKBACK = 5
        T, N = 10, 1
        close = np.full((T, N), 50.0, dtype=np.float64)
        # Inject NaN at bar 2 — so in the window [0:5], only 4 are finite
        close[2, 0] = np.nan
        aligned = _AlignedStub(close=close)
        eligibility = _make_eligibility_cube(T, N)
        requirements = {"strat": _make_price_only_requirement(lookback=LOOKBACK)}

        # Act
        cube = evaluate_strategy_readiness(
            aligned=aligned,
            requirements=requirements,
            eligibility=eligibility,
        )

        # Assert: bar index 4 (window [0:5]) has only 4 finite values → not ready
        assert not cube.ready[0, 4, 0], "Bar 4 window contains 1 NaN so finite count < LOOKBACK=5; must not be ready"
        # Bar 5 (window [1:6]) — includes 5 full finite bars [1,3,4,5] + ... check
        # bar 9 must definitely be ready (NaN at bar 2 is no longer in window [5:10])
        assert cube.ready[0, 9, 0], "Bar 9 must be ready (NaN bar is outside trailing window)"
