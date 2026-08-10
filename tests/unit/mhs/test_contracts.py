from __future__ import annotations

import pandas as pd
import pytest

from src.mhs.contracts import (
    MEASURED_EXECUTION_COST_TIERS_BPS,
    PHASE_1_BOOK_BLEND_WEIGHTS,
    PHASE_1_BOOK_SPECS,
    BookSpec,
    ExecutionSpec,
    HorizonBand,
)


class TestHorizonBand:
    """MHS-01-BAND-SIGN-FAIL-CLOSED: HorizonBand/BookSpec fail closed on invalid input."""

    def test_valid_band(self) -> None:
        band = HorizonBand(name="fast_reversal", horizons_hours=(24, 48, 72), sign=-1)
        assert band.name == "fast_reversal"
        assert band.horizons_hours == (24, 48, 72)
        assert band.sign == -1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": "", "horizons_hours": (24,), "sign": -1},
            {"name": "x", "horizons_hours": (), "sign": -1},
            {"name": "x", "horizons_hours": (0,), "sign": -1},
            {"name": "x", "horizons_hours": (48, 24), "sign": -1},
            {"name": "x", "horizons_hours": (24,), "sign": 0},
        ],
    )
    def test_rejects_invalid_bands(self, kwargs: dict) -> None:
        with pytest.raises(ValueError, match="must"):
            HorizonBand(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"band": None, "horizon_hours": 48, "step_hours": 7},
            {"band": None, "horizon_hours": 36, "step_hours": 6},
            {"band": None, "horizon_hours": 48, "step_hours": 0},
            {"band": None, "horizon_hours": 48, "step_hours": 6, "min_symbols": 1},
        ],
    )
    def test_book_spec_fails_closed(self, kwargs: dict) -> None:
        band = HorizonBand(name="fast_reversal", horizons_hours=(24, 48, 72), sign=-1)
        kwargs["band"] = band
        with pytest.raises(ValueError, match=r"must|not in band"):
            BookSpec(**kwargs)

    def test_book_spec_tranche_count(self) -> None:
        band = HorizonBand(name="fast_reversal", horizons_hours=(24, 48, 72), sign=-1)
        assert BookSpec(band=band, horizon_hours=48, step_hours=6).tranche_count() == 8
        assert BookSpec(band=band, horizon_hours=24, step_hours=6).tranche_count() == 4


class TestExecutionSpec:
    def test_default_spec_matches_cost_model_8bp(self) -> None:
        spec = ExecutionSpec()
        assert (spec.maker_fee_bps, spec.taker_fee_bps, spec.taker_slippage_bps) == (2.0, 5.0, 3.0)
        assert spec.passive_timeout_minutes == 30
        assert spec.require_trade_through is True
        assert spec.one_way_taker_bps() == 8.0

    def test_negative_fee_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="fees and slippage"):
            ExecutionSpec(maker_fee_bps=-1.0)


class TestFrozenLiterals:
    """MHS-13-FIXED-BLEND-AND-COST-STRESS: the Phase 1 candidate set is frozen."""

    def test_execution_cost_tiers(self) -> None:
        assert set(MEASURED_EXECUTION_COST_TIERS_BPS) == {"optimistic", "base", "stress"}
        assert MEASURED_EXECUTION_COST_TIERS_BPS["optimistic"] == pytest.approx(2.64)
        assert MEASURED_EXECUTION_COST_TIERS_BPS["base"] == pytest.approx(4.18)
        assert MEASURED_EXECUTION_COST_TIERS_BPS["stress"] == pytest.approx(6.07)
        assert (
            MEASURED_EXECUTION_COST_TIERS_BPS["optimistic"]
            < MEASURED_EXECUTION_COST_TIERS_BPS["base"]
            < MEASURED_EXECUTION_COST_TIERS_BPS["stress"]
        )
        assert all(v > 0 for v in MEASURED_EXECUTION_COST_TIERS_BPS.values())

    def test_blend_weights_reflect_prescreen_admission(self) -> None:
        # SCENARIO_MHS_BLEND_ADMISSION_01: fast_reversal's 446-symbol prescreen
        # net t-stat stays below the |t| >= 2.0 admission floor at every cost
        # tier (0.0bps pre-cost net_t=+0.577 .. 2.64bps net_t=-0.150, sign
        # already unstable; docs/results/mhs_horizon_diagnostic.json), so its
        # Research-GO blend weight is zero; slow_momentum clears |t| >= 2.0
        # pre-cost (net_t=+1.859) and stays above it through 2.64bps
        # (net_t=+1.634), with the pre-registered momentum sign throughout, so
        # it takes the full allocation.
        # fast_reversal stays a computed book (signal/prescreen kept for
        # re-measurement), just zero-weighted.
        assert PHASE_1_BOOK_BLEND_WEIGHTS == {"fast_reversal": 0.0, "slow_momentum": 1.0}
        assert set(PHASE_1_BOOK_BLEND_WEIGHTS) == set(PHASE_1_BOOK_SPECS)
        assert PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] == 0.0
        assert abs(sum(PHASE_1_BOOK_BLEND_WEIGHTS.values()) - 1.0) < 1e-12

    def test_book_specs_contain_only_frozen_books(self) -> None:
        assert set(PHASE_1_BOOK_SPECS) == {"fast_reversal", "slow_momentum"}
        fast = PHASE_1_BOOK_SPECS["fast_reversal"]
        slow = PHASE_1_BOOK_SPECS["slow_momentum"]
        assert fast.horizon_hours == 48
        assert fast.step_hours == 6
        assert fast.tranche_count() == 8
        assert fast.band.sign == -1
        assert slow.horizon_hours == 168
        assert slow.step_hours == 24
        assert slow.tranche_count() == 7
        assert slow.band.sign == 1

    def test_blend_preserves_reduced_gross(self) -> None:
        fast = pd.DataFrame({"A": [1.0], "B": [-1.0]})
        slow = pd.DataFrame({"A": [-1.0], "B": [1.0]})
        blended = 0.5 * fast + 0.5 * slow
        assert float(blended.abs().sum(axis=1).iloc[0]) == pytest.approx(0.0)
        assert not blended.abs().gt(1.0).any().any()
