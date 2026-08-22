"""Execution replay engine contract tests (split by behavioral domain)."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.common.errors import DataIntegrityError
from src.mhs.execution import (
    bar_funding_panel,
    mhs_ledger_pnl,
    mhs_ledger_pnl_multi_tier,
    simulated_inventory_ledger,
)
from src.research.baseline.backtest import _align_funding_rates
from src.research.technical_experts.cross_sectional import XsCompositeSpec, run_xs_composite_ledger

class TestBarFundingPanel:
    """MHS-12-LEDGER-REUSES-PRODUCTION-FORMULA: funding aligns via _align_funding_rates."""

    def test_matches_align_funding_rates(self) -> None:
        grid = pd.date_range("2021-01-01", periods=8, freq="1h", tz="UTC")
        rates = pd.Series(
            [0.0001],
            index=[pd.Timestamp("2021-01-01 01:30", tz="UTC")],
        )
        panel = bar_funding_panel({"AAAUSDT": rates}, grid)
        assert list(panel.columns) == ["AAAUSDT"]
        assert panel.index.equals(grid)
        reference = pd.Series(
            _align_funding_rates(rates, grid), index=grid,
        )
        assert (panel["AAAUSDT"] - reference).abs().max() < 1e-12

    def test_excludes_symbol_with_unalignable_funding(self) -> None:
        grid = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
        bad = pd.Series(
            [0.0001],
            index=[pd.Timestamp("1999-01-01", tz="UTC")],
        )
        panel = bar_funding_panel({"AAAUSDT": bad}, grid)
        assert panel.empty

class TestMhsLedgerPnl:
    """MHS-12-LEDGER-REUSES-PRODUCTION-FORMULA: identical to run_xs_composite_ledger."""

    def test_reproduces_legacy_ledger_exactly(self) -> None:
        weights = pd.DataFrame({"A": [0.5, 0.5, -0.5], "B": [-0.5, -0.5, 0.5]})
        opens = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 49.0, 48.0]})
        funding = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        net, turnover = mhs_ledger_pnl(weights, opens, funding, one_way_bps=8.0)
        equity_ref, turnover_ref = run_xs_composite_ledger(
            weights, opens, funding,
            XsCompositeSpec(
                halflife_bars=0, no_trade_band=0.0, execution_delay_bars=1,
                fee_rate=0.0004, slippage_rate=0.0004, gap_carry=True,
            ),
        )
        assert turnover.equals(turnover_ref)
        assert len(net) == len(equity_ref) - 1
        assert np.allclose(net.to_numpy(), equity_ref.pct_change().dropna().to_numpy())

    def test_gap_carry_default_true_absorbs_internal_gap(self) -> None:
        # SCENARIO_MHS_GAP_HARDENING_03: mhs_ledger_pnl carries a held weight
        # across a 3-bar internal NaN open gap instead of failing closed --
        # gap_carry defaults to True on this MHS-only pre-screen entrypoint.
        index = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
        opens = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan,
                      106.0, 107.0, 108.0, 109.0],
            },
            index=index,
        )
        weights = pd.DataFrame({"A": [0.5] * len(index)}, index=index)
        funding = pd.DataFrame(0.0, index=index, columns=["A"])
        net, turnover = mhs_ledger_pnl(weights, opens, funding, one_way_bps=8.0)
        assert bool(np.isfinite(net.to_numpy()).all())
        assert turnover.index.equals(index)

    def test_multi_tier_bit_identical_to_per_tier_calls(self) -> None:
        # SCENARIO_MHS_LEDGER_MULTI_TIER_BIT_IDENTICAL: the single-pass shared-array
        # multi-tier ledger must equal per-tier mhs_ledger_pnl calls exactly (net
        # and turnover, check_exact=True) for the same one-way bps list -- the
        # property that makes the committee/multi-feature streaming rewrites
        # bit-identical.
        rng = np.random.default_rng(42)
        index = pd.date_range("2021-01-01", periods=2400, freq="1h", tz="UTC")
        symbols = [f"S{i:03d}" for i in range(8)]
        opens = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, (len(index), len(symbols))), axis=0)),
            index=index, columns=symbols,
        )
        funding = pd.DataFrame(
            rng.normal(1e-5, 1e-6, (len(index), len(symbols))),
            index=index, columns=symbols,
        )
        step_index = index[::24]
        step = pd.DataFrame(
            rng.normal(0.0, 0.05, (len(step_index), len(symbols))),
            index=step_index, columns=symbols,
        )
        weights = step.reindex(index, method="ffill").fillna(0.0)

        bps_list = [2.64, 4.18, 6.07]
        singles = [
            mhs_ledger_pnl(weights, opens, funding, bps) for bps in bps_list
        ]
        multi = mhs_ledger_pnl_multi_tier(weights, opens, funding, bps_list)
        assert len(multi) == len(bps_list)
        for (net_s, tc_s), (net_m, tc_m) in zip(singles, multi, strict=True):
            pd.testing.assert_series_equal(net_s, net_m, check_exact=True)
            pd.testing.assert_series_equal(tc_s, tc_m, check_exact=True)

    def test_multi_tier_fails_closed(self) -> None:
        # SCENARIO_MHS_LEDGER_MULTI_TIER_FAIL_CLOSED: empty bps list and negative
        # bps raise ValueError; an index mismatch raises the same DataIntegrityError
        # message as the single call.
        weights = pd.DataFrame({"A": [0.5, 0.5, -0.5], "B": [-0.5, -0.5, 0.5]})
        opens = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 49.0, 48.0]})
        funding = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        with pytest.raises(ValueError, match="must not be empty"):
            mhs_ledger_pnl_multi_tier(weights, opens, funding, [])
        with pytest.raises(ValueError, match=">= 0"):
            mhs_ledger_pnl_multi_tier(weights, opens, funding, [2.64, -1.0])
        mismatched = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [50.0, 49.0, 48.0]},
            index=pd.DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-03"]),
        )
        with pytest.raises(DataIntegrityError, match="identical index"):
            mhs_ledger_pnl_multi_tier(weights, opens, mismatched, [2.64])

class TestSimulatedInventoryLedger:
    """MHS-15-INVENTORY-DRIFT-NO-FREE-REBALANCE: fixed contracts drift; no free rebalance."""

    def test_fixed_contracts_return_10_then_909(self) -> None:
        marks = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]},
            index=pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC"),
        )
        fills = pd.DataFrame(
            [
                {"timestamp": marks.index[0], "symbol": "A", "quantity_delta": 0.005,
                 "fill_price": 100.0, "fee_bps": 0.0, "reason": "passive_fill"},
                {"timestamp": marks.index[0], "symbol": "B", "quantity_delta": -0.005,
                 "fill_price": 100.0, "fee_bps": 0.0, "reason": "passive_fill"},
            ],
        )
        result = simulated_inventory_ledger(
            fills, marks, pd.DataFrame(0.0, index=marks.index, columns=marks.columns),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        assert abs(result.net_returns.iloc[0] - 0.10) < 1e-12
        assert abs(result.net_returns.iloc[1] - (0.10 / 1.10)) < 1e-12
        assert result.fill_turnover.iloc[0] == pytest.approx(1.0)
        assert (result.fill_source, result.mark_source) == ("OHLCV_STRICT_PROXY", "MARK_PRICE")

    def test_target_weight_ledger_is_a_separate_prescreen(self) -> None:
        weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]})
        opens = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]})
        funding = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        net, _ = mhs_ledger_pnl(weights, opens, funding, 8.0)
        # Target-weight implicit rebalancing is a screening proxy, not inventory.
        assert len(net) == 2

class TestFlatMarkNanLedger:
    """MHS-27-FLAT-MARK-NAN-LEDGER: an unavailable mark is zero only when flat.

    Leading unavailable marks with zero units leave equity exactly at the
    initial equity; a held position at an unavailable mark remains
    primary-invalid instead of leaking ``0 * NaN`` into cash equity.
    """

    def test_leading_nan_marks_flat_stay_at_initial_equity(self) -> None:
        idx = pd.date_range("2021-01-01", periods=5, freq="1h", tz="UTC")
        marks = pd.DataFrame(
            {"A": [np.nan, np.nan, 100.0, 101.0, 102.0]}, index=idx,
        )
        result = simulated_inventory_ledger(
            pd.DataFrame(),
            marks,
            pd.DataFrame(0.0, index=idx, columns=["A"]),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        assert np.isfinite(result.equity.to_numpy()).all()
        assert result.equity.iloc[0] == pytest.approx(1.0)
        assert result.equity.iloc[1] == pytest.approx(1.0)
        assert result.primary_valid is True
        assert result.net_returns.isna().sum() == 0

    def test_held_unavailable_mark_is_primary_invalid(self) -> None:
        idx = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
        marks = pd.DataFrame(
            {"A": [100.0, 100.0, np.nan, 100.0]}, index=idx,
        )
        fills = pd.DataFrame(
            [
                {"timestamp": idx[0], "symbol": "A", "quantity_delta": 0.01,
                 "fill_price": 100.0, "fee_bps": 2.0, "reason": "passive_fill"},
            ],
        )
        result = simulated_inventory_ledger(
            fills,
            marks,
            pd.DataFrame(0.0, index=idx, columns=["A"]),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        assert result.primary_valid is False
        assert "MISSING_DATA" in result.invalid_reasons
        assert np.isfinite(result.equity.to_numpy()).all()
        assert (result.equity > 0).all()

    def test_flat_position_at_nan_never_raises_on_equity(self) -> None:
        idx = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        marks = pd.DataFrame(
            {"A": [np.nan, np.nan, np.nan]}, index=idx,
        )
        result = simulated_inventory_ledger(
            pd.DataFrame(),
            marks,
            pd.DataFrame(0.0, index=idx, columns=["A"]),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        assert (result.equity == 1.0).all()
