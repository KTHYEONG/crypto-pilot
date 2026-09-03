"""Cross-sectional expert contract tests (split by behavioral domain)."""

from __future__ import annotations




"""Contract scenarios XSC-01..XSC-05, XSC-07, XSA-02, and XSV3-01 for the cross-sectional module.

XSC-01-NO-TRADE-BAND-STATEFUL, XSC-02-WEIGHTS-DOLLAR-NEUTRAL,
XSC-03-SPEC-FROZEN-BOUNDS, XSC-04-LEDGER-EXECUTION-LAG,
XSC-05-ADMISSION-SCALE-INVARIANT, XSC-07-COMPOSITE-BEATS-SINGLE-FAMILY,
XSA-02-COMPOSITE-PRESERVATION, XSV3-01-FAMILY-SUM,
SCENARIO_XSV5_01_DUAL_FAMILY_EXCLUDES_FUNDING,
SCENARIO_XSV6_01_CAUSAL_VOL_WEIGHTS_EXCLUDE_CURRENT_BAR,
SCENARIO_XSV6_02_VOL_WEIGHTED_MATCHES_MANUAL_RECOMPUTE,
SCENARIO_XS_POSITIONING_WEIGHTS_01,
SCENARIO_XSV6SIZE_01_DISCOVERY_ONLY_SIZING_NO_LEAKAGE,
SCENARIO_XSV6SIZE_02_INFEASIBLE_SIZING_FAILS_CLOSED, and
SCENARIO_COSTFIX_01..07 (honest turnover-cost repricing of the vol-target
overlay stack).
"""
import numpy as np
import pandas as pd
import pytest
from src.common.errors import DataIntegrityError
from src.quant.technical_experts.cross_sectional import (
    XsCompositeSpec,
    run_xs_composite_ledger,
)

class TestCompositeLedger:
    def _ledger_inputs(self, bars: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        index = pd.date_range("2024-01-01", periods=bars, freq="4h", tz="UTC")
        opens = pd.DataFrame(
            {
                "A": np.linspace(100.0, 130.0, bars),
                "B": np.linspace(100.0, 70.0, bars),
            },
            index=index,
        )
        weights = pd.DataFrame({"A": [0.5] * bars, "B": [-0.5] * bars}, index=index)
        funding = pd.DataFrame(0.0, index=index, columns=["A", "B"])
        return weights, opens, funding

    def test_xsc_04_execution_lag_leaves_first_bars_flat(self) -> None:
        weights, opens, funding = self._ledger_inputs()
        equity, turnover = run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())
        assert equity.index.equals(opens.index)
        assert turnover.index.equals(opens.index)
        assert float(turnover.iloc[:2].sum()) == 0.0
        assert float(turnover.sum()) > 0.0

    def test_xsc_04_long_riser_short_faller_profits(self) -> None:
        weights, opens, funding = self._ledger_inputs()
        equity, _turnover = run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())
        assert bool((equity > 0).all())
        assert float(equity.iloc[-1]) > float(equity.iloc[0])

    def test_xsc_04_mismatched_index_raises(self) -> None:
        weights, opens, funding = self._ledger_inputs()
        shifted = opens.iloc[1:].copy()
        with pytest.raises(DataIntegrityError):
            run_xs_composite_ledger(weights, shifted, funding, XsCompositeSpec())

    def test_xsc_04_mismatched_columns_raise(self) -> None:
        weights, opens, funding = self._ledger_inputs()
        opens = opens.rename(columns={"B": "C"})
        with pytest.raises(DataIntegrityError):
            run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())

    def test_xsc_04_equity_would_reach_zero_raises(self) -> None:
        index = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
        opens = pd.DataFrame({"A": [100.0] * 5}, index=index)
        weights = pd.DataFrame({"A": [1.0] * 5}, index=index)
        funding = pd.DataFrame(0.0, index=index, columns=["A"])
        opens.iloc[2] = 0.0
        with pytest.raises(DataIntegrityError):
            run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())

    def test_xsc_04_funding_debits_long_credits_short(self) -> None:
        index = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
        opens = pd.DataFrame({"A": [100.0] * 8, "B": [100.0] * 8}, index=index)
        weights = pd.DataFrame({"A": [0.5] * 8, "B": [-0.5] * 8}, index=index)
        funding = pd.DataFrame(0.001, index=index, columns=["A", "B"])
        no_funding = pd.DataFrame(0.0, index=index, columns=["A", "B"])
        with_funding, _t = run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())
        without, _t = run_xs_composite_ledger(weights, opens, no_funding, XsCompositeSpec())
        # A long pays positive funding, a short is credited: net funding PnL is
        # -w_long*f + -w_short*f = -(0.5 - 0.5) = 0 here, so the two ledgers must
        # only differ by the turnover cost term, which is identical.
        assert float(with_funding.iloc[-1]) == pytest.approx(float(without.iloc[-1]))

    def test_xsc_04_inactive_lifecycle_nan_is_ignored(self) -> None:
        weights, opens, funding = self._ledger_inputs(8)
        weights["B"] = 0.0
        opens.loc[opens.index[3:], "B"] = np.nan
        funding.loc[funding.index[3:], "B"] = np.nan
        equity, _turnover = run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())
        assert bool(np.isfinite(equity.to_numpy()).all())

    def test_xsc_04_active_lifecycle_nan_fails_closed(self) -> None:
        weights, opens, funding = self._ledger_inputs(8)
        opens.loc[opens.index[3], "A"] = np.nan
        with pytest.raises(DataIntegrityError, match="active ledger cell"):
            run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())

    def test_xsc_04_gap_carry_absorbs_internal_gap_carries_flat(self) -> None:
        # SCENARIO_MHS_GAP_HARDENING_01: with gap_carry=True the same fixture
        # that crashes the default fails-closed path is absorbed -- the held
        # position carries flat at its last valid open across the gap bar.
        weights, opens, funding = self._ledger_inputs(8)
        weights["B"] = 0.0
        opens.loc[opens.index[3], "A"] = np.nan
        equity, _turnover = run_xs_composite_ledger(
            weights, opens, funding, XsCompositeSpec(gap_carry=True),
        )
        assert bool(np.isfinite(equity.to_numpy()).all())
        assert float(equity.iloc[-1]) > 0.0
        net = equity.pct_change().fillna(0.0)
        assert float(net.iloc[3]) == pytest.approx(0.0, abs=1e-12)
        resumed = float(opens.iloc[4]["A"] / opens.iloc[2]["A"] - 1.0)
        assert float(net.iloc[4]) == pytest.approx(0.5 * resumed, rel=1e-9)

    def test_xsc_04_gap_carry_never_fills_pre_listing_leading_nan(self) -> None:
        # SCENARIO_MHS_GAP_HARDENING_02: forward-fill never fills leading NaN,
        # so a genuinely-invalid pre-listing position still fails closed even
        # with gap_carry=True -- the safety net is unweakened.
        weights, opens, funding = self._ledger_inputs(8)
        weights["B"] = 0.0
        opens.loc[opens.index[:3], "A"] = np.nan
        with pytest.raises(DataIntegrityError, match="active ledger cell"):
            run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec(gap_carry=True))
