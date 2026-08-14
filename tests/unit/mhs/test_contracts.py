from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src.mhs.contracts import (
    MEASURED_EXECUTION_COST_TIERS_BPS,
    MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    MHS_TREND_SLEEVE_HORIZONS_HOURS,
    MOMENTUM_HORIZON_CANDIDATES_HOURS,
    PHASE_1_BOOK_BLEND_WEIGHTS,
    PHASE_1_BOOK_SPECS,
    REVERSAL_HORIZON_CANDIDATES_HOURS,
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

    def test_ladder_tranches_default_and_validation(self) -> None:
        assert ExecutionSpec().ladder_tranches == 4
        assert ExecutionSpec(ladder_tranches=1).ladder_tranches == 1
        with pytest.raises(ValueError, match="ladder_tranches"):
            ExecutionSpec(ladder_tranches=0)


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

    def test_widened_band_accepts_fold_selected_candidates(self) -> None:
        # SCENARIO_MHS_FOLD_SAFE_HORIZON_04_WIDENED_BAND_ACCEPTS_CANDIDATES:
        # the slow band's allowed set is the full measured momentum candidate
        # grid (all 19 horizons, 72..504 step 24) so a fold-selected horizon
        # passes BookSpec.__post_init__'s band check, while the frozen 168h
        # default is unchanged and out-of-band values still fail closed.
        slow = PHASE_1_BOOK_SPECS["slow_momentum"]
        assert slow.band.horizons_hours == MOMENTUM_HORIZON_CANDIDATES_HOURS
        assert len(MOMENTUM_HORIZON_CANDIDATES_HOURS) == 19
        assert MOMENTUM_HORIZON_CANDIDATES_HOURS[0] == 72
        assert MOMENTUM_HORIZON_CANDIDATES_HOURS[-1] == 504
        assert tuple(range(72, 504 + 1, 24)) == MOMENTUM_HORIZON_CANDIDATES_HOURS
        assert slow.horizon_hours == 168
        fast = PHASE_1_BOOK_SPECS["fast_reversal"]
        assert fast.band.horizons_hours == REVERSAL_HORIZON_CANDIDATES_HOURS
        assert len(REVERSAL_HORIZON_CANDIDATES_HOURS) == 7
        assert REVERSAL_HORIZON_CANDIDATES_HOURS == (24, 48, 72, 96, 120, 144, 168)
        assert fast.horizon_hours == 48
        assert fast.step_hours == 6
        assert fast.tranche_count() == 8
        # SCENARIO_FAST_BAND_GRID_WIDENED_DEFAULT_UNCHANGED: widening the fast
        # band to the 7-candidate reversal grid leaves the frozen 48h default
        # byte-identical (BookSpec.__post_init__ still accepts it), while an
        # out-of-band horizon still fails closed.
        assert dataclasses.replace(fast, horizon_hours=120).horizon_hours == 120
        with pytest.raises(ValueError, match="not in band"):
            dataclasses.replace(fast, horizon_hours=100)
        widened = dataclasses.replace(slow, horizon_hours=360)
        assert widened.horizon_hours == 360
        assert widened.step_hours == slow.step_hours
        assert widened.min_symbols == slow.min_symbols
        with pytest.raises(ValueError, match="not in band"):
            dataclasses.replace(slow, horizon_hours=100)

    def test_blend_preserves_reduced_gross(self) -> None:
        fast = pd.DataFrame({"A": [1.0], "B": [-1.0]})
        slow = pd.DataFrame({"A": [-1.0], "B": [1.0]})
        blended = 0.5 * fast + 0.5 * slow
        assert float(blended.abs().sum(axis=1).iloc[0]) == pytest.approx(0.0)
        assert not blended.abs().gt(1.0).any().any()

    def test_mhs_discovery_start_single_sourced_in_folds(self) -> None:
        # SCENARIO_MHS_GAP_HARDENING_04: MHS_DISCOVERY_START is the domain
        # single source and all three folds derive train_start from it -- a
        # regression guard against the constant re-diverging into independent
        # literals.
        from src.mhs.contracts import MHS_DISCOVERY_START
        from src.mhs.evaluation import phase_1_anchored_purged_folds

        assert pd.Timestamp("2021-01-01", tz="UTC") == MHS_DISCOVERY_START
        folds = phase_1_anchored_purged_folds()
        assert len(folds) == 3
        for fold in folds:
            assert fold.train_start == MHS_DISCOVERY_START
            assert fold.train_start is MHS_DISCOVERY_START

    def test_funding_carry_grid_no_capital_allocated(self) -> None:
        # SCENARIO_MHS_NO_CAPITAL_ALLOCATED_06: the funding-carry lookback grid
        # is a measured candidate grid -- not a frozen BookSpec -- following the
        # same governance pattern as the horizon grids, and 'funding_carry'
        # appears in neither PHASE_1_BOOK_SPECS nor PHASE_1_BOOK_BLEND_WEIGHTS.
        # This is the explicit guard against scope creep into P1's
        # capital-allocation territory.
        assert len(MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS) >= 3
        assert all(h > 0 for h in MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS)
        assert tuple(sorted(set(MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS))) == tuple(
            MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS
        )
        assert set(PHASE_1_BOOK_SPECS) == {"fast_reversal", "slow_momentum"}
        assert set(PHASE_1_BOOK_BLEND_WEIGHTS) == {"fast_reversal", "slow_momentum"}
        assert "funding_carry" not in PHASE_1_BOOK_SPECS
        assert "funding_carry" not in PHASE_1_BOOK_BLEND_WEIGHTS
        assert abs(sum(PHASE_1_BOOK_BLEND_WEIGHTS.values()) - 1.0) < 1e-12

    def test_trend_sleeve_horizons_are_frozen_measured_band(self) -> None:
        # The trend sleeve's slow band is the frozen measured 6-horizon ensemble
        # (docs/specs/mhs_directional_trend_sleeve.md §1.4); individual net
        # Sharpe at the base tier ranges -0.058..+0.282, so any single-horizon
        # selection is overfitting and the constant is never edited inline.
        assert MHS_TREND_SLEEVE_HORIZONS_HOURS == (336, 480, 600, 720, 1080, 1440)
        assert len(MHS_TREND_SLEEVE_HORIZONS_HOURS) == 6
        assert tuple(sorted(MHS_TREND_SLEEVE_HORIZONS_HOURS)) == MHS_TREND_SLEEVE_HORIZONS_HOURS
        assert all(h > 0 for h in MHS_TREND_SLEEVE_HORIZONS_HOURS)
