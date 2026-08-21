from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src.mhs.types import (
    MEASURED_EXECUTION_COST_TIERS_BPS,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    TREND_SLEEVE_HORIZONS_HOURS,
    MOMENTUM_HORIZON_CANDIDATES_HOURS,
    BOOK_BLEND_WEIGHTS,
    BOOK_SPECS,
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
        assert BOOK_BLEND_WEIGHTS == {"fast_reversal": 0.0, "slow_momentum": 1.0}
        assert set(BOOK_BLEND_WEIGHTS) == set(BOOK_SPECS)
        assert BOOK_BLEND_WEIGHTS["fast_reversal"] == 0.0
        assert abs(sum(BOOK_BLEND_WEIGHTS.values()) - 1.0) < 1e-12

    def test_book_specs_contain_only_frozen_books(self) -> None:
        assert set(BOOK_SPECS) == {"fast_reversal", "slow_momentum"}
        fast = BOOK_SPECS["fast_reversal"]
        slow = BOOK_SPECS["slow_momentum"]
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
        slow = BOOK_SPECS["slow_momentum"]
        assert slow.band.horizons_hours == MOMENTUM_HORIZON_CANDIDATES_HOURS
        assert len(MOMENTUM_HORIZON_CANDIDATES_HOURS) == 19
        assert MOMENTUM_HORIZON_CANDIDATES_HOURS[0] == 72
        assert MOMENTUM_HORIZON_CANDIDATES_HOURS[-1] == 504
        assert tuple(range(72, 504 + 1, 24)) == MOMENTUM_HORIZON_CANDIDATES_HOURS
        assert slow.horizon_hours == 168
        fast = BOOK_SPECS["fast_reversal"]
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
        # SCENARIO_MHS_GAP_HARDENING_04: DISCOVERY_START is the domain
        # single source and all four folds derive train_start from it -- a
        # regression guard against the constant re-diverging into independent
        # literals.
        from src.mhs.types import DISCOVERY_START
        from src.mhs.evidence import phase_1_anchored_purged_folds

        assert pd.Timestamp("2021-01-01", tz="UTC") == DISCOVERY_START
        folds = phase_1_anchored_purged_folds()
        assert len(folds) == 4
        for fold in folds:
            assert fold.train_start == DISCOVERY_START
            assert fold.train_start is DISCOVERY_START

    def test_funding_carry_grid_no_capital_allocated(self) -> None:
        # SCENARIO_MHS_NO_CAPITAL_ALLOCATED_06: the funding-carry lookback grid
        # is a measured candidate grid -- not a frozen BookSpec -- following the
        # same governance pattern as the horizon grids, and 'funding_carry'
        # appears in neither BOOK_SPECS nor BOOK_BLEND_WEIGHTS.
        # This is the explicit guard against scope creep into P1's
        # capital-allocation territory.
        assert len(FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS) >= 3
        assert all(h > 0 for h in FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS)
        assert tuple(sorted(set(FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS))) == tuple(
            FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS
        )
        assert set(BOOK_SPECS) == {"fast_reversal", "slow_momentum"}
        assert set(BOOK_BLEND_WEIGHTS) == {"fast_reversal", "slow_momentum"}
        assert "funding_carry" not in BOOK_SPECS
        assert "funding_carry" not in BOOK_BLEND_WEIGHTS
        assert abs(sum(BOOK_BLEND_WEIGHTS.values()) - 1.0) < 1e-12

    def test_trend_sleeve_horizons_are_frozen_measured_band(self) -> None:
        # The trend sleeve's slow band is the frozen measured 6-horizon ensemble.
        assert TREND_SLEEVE_HORIZONS_HOURS == (336, 480, 600, 720, 1080, 1440)
        assert len(TREND_SLEEVE_HORIZONS_HOURS) == 6
        assert tuple(sorted(TREND_SLEEVE_HORIZONS_HOURS)) == TREND_SLEEVE_HORIZONS_HOURS
        assert all(h > 0 for h in TREND_SLEEVE_HORIZONS_HOURS)

    def test_committee_members_and_target_vol_are_frozen(self) -> None:
        # SCENARIO_MHS_COMMITTEE_LITERALS_FROZEN and
        # SCENARIO_MHS_COMMITTEE_MEMBERS_K5_FROZEN: the k=5 committee is declared
        # by economic family (order flow x2, cross-sectional trend x2,
        # higher-moment x1; xs_mom_720h removed as a rank-invariant no-op), its
        # 15% target volatility and 720h purge are frozen contract values, and
        # the purge matches the longest 720h lookbacks so no overlapping-label
        # information leaks across a walk-forward boundary.
        from src.mhs.types import (
            COMMITTEE_MEMBERS,
            COMMITTEE_OOS_START,
            COMMITTEE_PURGE_HOURS,
            COMMITTEE_TARGET_VOL,
        )

        assert COMMITTEE_MEMBERS == (
            "flow_imb_720h",
            "flow_imb_168h",
            "xs_mom_336h",
            "xs_idio_mom_336h",
            "mom3_skew_168h",
        )
        assert len(COMMITTEE_MEMBERS) == 5
        assert len(set(COMMITTEE_MEMBERS)) == 5
        assert "xs_mom_720h" not in COMMITTEE_MEMBERS
        assert pytest.approx(0.15) == COMMITTEE_TARGET_VOL
        assert COMMITTEE_PURGE_HOURS == 720
        assert pd.Timestamp("2023-01-01", tz="UTC") == COMMITTEE_OOS_START
        assert COMMITTEE_OOS_START.tzinfo is not None

    def test_committee_growth_diagnostic_constants_are_frozen(self) -> None:
        # SCENARIO_COMMITTEE_GROWTH_CONTRACTS_FROZEN: the discovery-window
        # growth-optimal risk-grid multipliers and constraint anchors are frozen
        # contract values with a strictly ascending, 1.0-containing grid.
        from src.mhs.types import (
            COMMITTEE_GROWTH_BARS_PER_YEAR,
            COMMITTEE_GROWTH_HORIZON_YEARS,
            COMMITTEE_GROWTH_MAX_DRAWDOWN,
            COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
            COMMITTEE_GROWTH_MAX_RUIN_PROB,
            COMMITTEE_GROWTH_N_PATHS,
            COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS,
            COMMITTEE_GROWTH_RUIN_FRACTION,
        )

        assert COMMITTEE_GROWTH_BARS_PER_YEAR == 365
        assert COMMITTEE_GROWTH_HORIZON_YEARS == 3.0
        assert COMMITTEE_GROWTH_N_PATHS == 2000
        assert COMMITTEE_GROWTH_MAX_DRAWDOWN == 0.25
        assert COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB == 0.10
        assert COMMITTEE_GROWTH_RUIN_FRACTION == 0.60
        assert COMMITTEE_GROWTH_MAX_RUIN_PROB == 0.01
        grid = COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS
        assert len(grid) >= 2
        assert tuple(sorted(grid)) == grid
        assert all(g > 0 for g in grid)
        assert 1.0 in grid

    def test_committee_purge_hours_matches_longest_member_lookback(self) -> None:
        # SCENARIO_COMMITTEE_PURGE_HOURS_MATCHES_LONGEST_MEMBER_LOOKBACK (B2):
        # the purge gap must always cover the longest committee feature lookback
        # (720h, still carried by flow_imb_720h after the k=5 change) so no
        # overlapping-label information leaks across a walk-forward boundary. A
        # future member with a longer lookback than the purge fails this static
        # test loudly instead of silently recurring the doc/value mismatch the
        # B2 fix corrected.
        from src.mhs.types import COMMITTEE_MEMBERS, COMMITTEE_PURGE_HOURS
        from src.mhs.features import FEATURE_REGISTRY

        registry = {spec.name: spec for spec in FEATURE_REGISTRY}
        assert COMMITTEE_PURGE_HOURS >= 720
        for name in COMMITTEE_MEMBERS:
            assert name in registry, f"committee member {name} not in registry"
        assert "flow_imb_720h" in COMMITTEE_MEMBERS


class TestFillMarkParityGateConstants:
    """SCENARIO_MHS_FILL_MARK_PARITY_02: contract constants for parity gate."""

    def test_price_protection_band(self) -> None:
        from src.mhs.types import FILL_MARK_PRICE_PROTECTION_BAND

        assert FILL_MARK_PRICE_PROTECTION_BAND == 0.05

    def test_max_log_divergence(self) -> None:
        import math

        from src.mhs.types import FILL_MARK_MAX_LOG_DIVERGENCE, FILL_MARK_PRICE_PROTECTION_BAND

        assert pytest.approx(math.log1p(FILL_MARK_PRICE_PROTECTION_BAND)) == FILL_MARK_MAX_LOG_DIVERGENCE
        assert 0.048 < FILL_MARK_MAX_LOG_DIVERGENCE < 0.049

    def test_vol_target_max_scale(self) -> None:
        from src.mhs.types import (
            COMMITTEE_TARGET_GROSS,
            PNL_VOL_TARGET_MAX_SCALE,
        )

        assert pytest.approx(1.0, abs=1e-12) == PNL_VOL_TARGET_MAX_SCALE * COMMITTEE_TARGET_GROSS
        assert PNL_VOL_TARGET_MAX_SCALE > 1.0


class TestFrozenLiteralsCommitteeTiming:
    """Split out of TestFrozenLiterals so TestFillMarkParityGateConstants stays
    scoped to the parity-gate scenario only (SCENARIO_MHS_FILL_MARK_PARITY_02)."""

    def test_ram_guard_constants_are_frozen_with_sane_bounds(self) -> None:
        # SCENARIO_MHS_RAM_GUARD_CONSTANTS: the automatic RAM-guard tuning
        # constants are frozen contract values with sane bounds (budget/reserve
        # fractions in (0,1), floor a positive power-of-two MiB count).
        from src.mhs.types import (
            RAM_BUDGET_FRACTION,
            RAM_RESERVE_FLOOR_BYTES,
            RAM_RESERVE_FRACTION,
        )

        assert RAM_BUDGET_FRACTION == 0.85
        assert 0.0 < RAM_BUDGET_FRACTION < 1.0
        assert RAM_RESERVE_FRACTION == 0.05
        assert 0.0 < RAM_RESERVE_FRACTION < 1.0
        assert RAM_RESERVE_FLOOR_BYTES == 268435456
        assert RAM_RESERVE_FLOOR_BYTES > 0
        assert RAM_RESERVE_FLOOR_BYTES % (2**20) == 0

    def test_committee_tranche_count_frozen(self) -> None:
        # SCENARIO_MHS_COMMITTEE_TRANCHE_COUNT_FROZEN: the committee decision
        # cadence smoothing is a frozen structural constant (24h grid x 3 =
        # effective 72h signal life) with a valid tranche count.
        from src.mhs.types import COMMITTEE_TRANCHE_COUNT

        assert isinstance(COMMITTEE_TRANCHE_COUNT, int)
        assert COMMITTEE_TRANCHE_COUNT >= 1
        assert COMMITTEE_TRANCHE_COUNT == 3

    def test_committee_regime_adaptive_window_frozen(self) -> None:
        # SCENARIO_MHS_COMMITTEE_REGIME_ADAPTIVE_WINDOW_FROZEN: the regime-
        # adaptive tranche gating window is a frozen constant confirmed via
        # real 3m replay to sit inside a plateau (15-25 all pass every
        # anchored fold), not a single fitted point -- windows 10 and 90 both
        # trigger CAPITAL_INVARIANT_BREACH.
        from src.mhs.types import COMMITTEE_REGIME_ADAPTIVE_WINDOW

        assert isinstance(COMMITTEE_REGIME_ADAPTIVE_WINDOW, int)
        assert COMMITTEE_REGIME_ADAPTIVE_WINDOW >= 3
        assert COMMITTEE_REGIME_ADAPTIVE_WINDOW == 15

    def test_registered_target_gross_default(self) -> None:
        """SCENARIO_MHS_REGISTERED_TARGET_GROSS_DEFAULT: the registered
        committee exposure is 0.92, the largest replay-certified gross inside
        the registered drawdown budget."""
        from src.mhs.types import COMMITTEE_TARGET_GROSS

        assert 0.0 < COMMITTEE_TARGET_GROSS <= 2.0
        assert COMMITTEE_TARGET_GROSS > 0.7950

        import dataclasses

        from src.application.research.mhs.evaluation import (
            MhsDiagnosticRequest,
        )
        from src.application.research.mhs.research_go import (
            _resolved_committee_target_gross,
        )

        default_request = MhsDiagnosticRequest(committee_capital=True)
        assert _resolved_committee_target_gross(default_request) == COMMITTEE_TARGET_GROSS
        assert MhsDiagnosticRequest(committee_capital=True, committee_target_gross=None).committee_target_gross is None
        with pytest.raises(ValueError, match="committee_target_gross"):
            MhsDiagnosticRequest(committee_capital=True, committee_target_gross=0.0)

        # Regression: dataclasses.replace() on an unset (implicit-default)
        # request must not resolve the sentinel into the field, or a copy
        # dropping committee_capital would wrongly see an "explicit" gross and
        # raise -- even though no caller ever set committee_target_gross.
        copied = dataclasses.replace(default_request, committee_capital=False)
        assert copied.committee_capital is False
        assert _resolved_committee_target_gross(copied) == COMMITTEE_TARGET_GROSS


class TestCompoundingGrowthContractConstants:
    """SCENARIO_CONTRACT_CONSTANTS_REGISTERED: new risk-budget constants are registered."""

    def test_constants_registered(self) -> None:
        from src.mhs.types import (
            FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS,
            FUNDING_CARRY_SLEEVE_WEIGHT,
            PNL_TARGET_ANNUAL_VOL,
            PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
        )

        assert PNL_TARGET_ANNUAL_VOL == 0.20
        assert 0.0 < PNL_TARGET_ANNUAL_VOL <= 1.0

        assert PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS == 20
        assert PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS >= 1

        assert FUNDING_CARRY_SLEEVE_WEIGHT == 0.30
        assert 0.0 <= FUNDING_CARRY_SLEEVE_WEIGHT < 1.0

        assert FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS == 168
        assert FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS in FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS


class TestCompoundingAlphaAxesContract:
    """SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_01: committee member set contracts."""

    def test_member_sets_have_exactly_two_keys(self) -> None:
        from src.mhs.params import COMMITTEE_MEMBER_SETS
        assert set(COMMITTEE_MEMBER_SETS.keys()) == {"flow_momentum", "risk_premia"}

    def test_default_member_set_is_flow_momentum(self) -> None:
        # risk_premia was measured non-adopted: a full 3m replay (2021-2025)
        # breached the registered drawdown budget (MDD -31.4% vs -25% budget)
        # and turned two folds STRESS_SHARPE_NOT_POSITIVE. See
        # ADR_20260820_MHS_COMPOUNDING_ALPHA_AXES.
        from src.mhs.params import COMMITTEE_DEFAULT_MEMBER_SET, COMMITTEE_MEMBERS, COMMITTEE_MEMBER_SETS
        assert COMMITTEE_DEFAULT_MEMBER_SET == "flow_momentum"
        assert COMMITTEE_MEMBER_SETS["flow_momentum"] == COMMITTEE_MEMBERS

    def test_risk_premia_members(self) -> None:
        from src.mhs.params import COMMITTEE_MEMBER_SETS
        assert COMMITTEE_MEMBER_SETS["risk_premia"] == (
            "flow_imb_720h", "flow_imb_168h", "mom3_skew_168h", "lowvol_168h", "rev_24h",
        )

    def test_flow_momentum_members(self) -> None:
        from src.mhs.params import COMMITTEE_MEMBER_SETS
        assert COMMITTEE_MEMBER_SETS["flow_momentum"] == (
            "flow_imb_720h", "flow_imb_168h", "xs_mom_336h", "xs_idio_mom_336h", "mom3_skew_168h",
        )

    def test_all_member_names_in_registry(self) -> None:
        from src.mhs.params import COMMITTEE_MEMBER_SETS
        from src.mhs.features import FEATURE_REGISTRY
        registry_names = {s.name for s in FEATURE_REGISTRY}
        for members in COMMITTEE_MEMBER_SETS.values():
            for name in members:
                assert name in registry_names, f"{name} not in FEATURE_REGISTRY"

    def test_growth_budget_annual_vol_exists(self) -> None:
        from src.mhs.committee import growth_budget_annual_vol
        assert callable(growth_budget_annual_vol)

    def test_resolved_committee_members(self) -> None:
        from src.application.research.mhs.research_go import _resolved_committee_members
        from src.application.research.mhs.contracts import MhsDiagnosticRequest
        from src.mhs.params import COMMITTEE_MEMBER_SETS

        req_v2 = MhsDiagnosticRequest(committee_capital=True, committee_member_set="risk_premia")
        assert _resolved_committee_members(req_v2) == COMMITTEE_MEMBER_SETS["risk_premia"]

        req_v1 = MhsDiagnosticRequest(committee_capital=True, committee_member_set="flow_momentum")
        assert _resolved_committee_members(req_v1) == COMMITTEE_MEMBER_SETS["flow_momentum"]

        req_bad = MhsDiagnosticRequest(committee_capital=True, committee_member_set="risk_premia")
        # Simulate an unregistered key by replacing the field (bypassing validation)
        object.__setattr__(req_bad, "committee_member_set", "unregistered")
        with pytest.raises(ValueError, match="unknown committee_member_set"):
            _resolved_committee_members(req_bad)
