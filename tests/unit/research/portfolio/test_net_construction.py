from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.contracts import CostModel
from src.research.portfolio.defaults import STRESS_FEE_MULT, STRESS_SLIPPAGE_MULT
from src.research.portfolio.net_construction import (
    NetConstructionSpec,
    NetReturnStream,
    apply_no_trade_band,
    compute_net_return_stream,
)


def _index(n: int = 4, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="4h", tz="UTC")


class TestNetConstructionSpec:
    def test_defaults(self) -> None:
        spec = NetConstructionSpec()
        assert spec.rebalance_bars == 1
        assert spec.no_trade_band == 0.0
        assert abs(spec.costs.round_trip_bps() - 16.0) < 1e-9

    def test_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="rebalance_bars"):
            NetConstructionSpec(rebalance_bars=0)
        with pytest.raises(ValueError, match="no_trade_band"):
            NetConstructionSpec(no_trade_band=-0.1)

    def test_reuses_cost_model(self) -> None:
        costs = CostModel(fee_rate=0.001, slippage_rate=0.0002)
        spec = NetConstructionSpec(costs=costs)
        assert spec.costs is costs


class TestApplyNoTradeBand:
    def test_hysteresis(self) -> None:
        out = apply_no_trade_band(np.array([0.1, 0.5]), np.array([0.12, 0.0]), 0.05)
        assert np.allclose(out, np.array([0.12, 0.5]))

    def test_zero_band_always_trades_to_target(self) -> None:
        out = apply_no_trade_band(np.array([0.1, 0.5]), np.array([0.12, 0.0]), 0.0)
        assert np.allclose(out, np.array([0.1, 0.5]))

    def test_does_not_mutate_inputs(self) -> None:
        target = np.array([0.1, 0.5])
        held = np.array([0.12, 0.0])
        apply_no_trade_band(target, held, 0.05)
        assert np.allclose(target, np.array([0.1, 0.5]))
        assert np.allclose(held, np.array([0.12, 0.0]))

    def test_rejects_negative_band(self) -> None:
        with pytest.raises(ValueError, match="band"):
            apply_no_trade_band(np.array([0.1]), np.array([0.1]), -0.1)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            apply_no_trade_band(np.array([0.1]), np.array([0.1, 0.2]), 0.05)


class TestNetReturnStream:
    def test_series_share_index(self) -> None:
        idx = _index(2)
        z = pd.Series([0.0, 0.0], index=idx)
        stream = NetReturnStream(z, z, z, z, pd.DataFrame({"A": [0.0, 0.0]}, index=idx))
        assert list(stream.net.index) == list(idx)


class TestComputeNetReturnStream:
    def test_turnover_and_cost_contract(self) -> None:
        idx = _index(4)
        tw = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0]}, index=idx)
        fr = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01]}, index=idx)
        stream = compute_net_return_stream(tw, fr, NetConstructionSpec())
        assert abs(float(stream.turnover.sum()) - 2.0) < 1e-9
        assert float(stream.net.sum()) < float(stream.gross.sum())
        assert abs(float(stream.gross.iloc[0]) - 0.01) < 1e-12

    # GEV2-07-NET-TURNOVER
    def test_higher_rebalance_bars_reduces_turnover_and_cost(self) -> None:
        idx = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
        # Signal toggles every 5 bars so intra-month turnover is visible.
        targets = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        tw = pd.DataFrame({"A": [targets[i % len(targets)] for i in range(40)]}, index=idx)
        fr = pd.DataFrame({"A": [0.001] * 40}, index=idx)

        base = compute_net_return_stream(tw, fr, NetConstructionSpec(rebalance_bars=1))
        slow = compute_net_return_stream(tw, fr, NetConstructionSpec(rebalance_bars=4))
        assert slow.turnover.sum() < base.turnover.sum()
        assert slow.cost.sum() < base.cost.sum()

    def test_no_trade_band_reduces_turnover_and_cost(self) -> None:
        idx = _index(40)
        rng = np.random.default_rng(0)
        targets = np.clip(0.5 + 0.2 * np.sin(np.arange(40) / 3.0) + 0.05 * rng.standard_normal(40), 0.0, 1.0)
        tw = pd.DataFrame({"A": targets}, index=idx)
        fr = pd.DataFrame({"A": [0.001] * 40}, index=idx)

        base = compute_net_return_stream(tw, fr, NetConstructionSpec(no_trade_band=0.0))
        banded = compute_net_return_stream(tw, fr, NetConstructionSpec(no_trade_band=0.03))
        assert banded.turnover.sum() < base.turnover.sum()
        assert banded.cost.sum() < base.cost.sum()

    def test_zero_band_never_changes_rebalance_bar_weights(self) -> None:
        idx = _index(6)
        tw = pd.DataFrame({"A": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}, index=idx)
        fr = pd.DataFrame({"A": [0.001] * 6}, index=idx)
        stream = compute_net_return_stream(tw, fr, NetConstructionSpec(rebalance_bars=1, no_trade_band=0.0))
        assert np.allclose(stream.realized_weights["A"].to_numpy(), tw["A"].to_numpy())

    # GEV2-08-COST-REUSE
    def test_charges_fee_plus_slippage_per_one_way_turnover(self) -> None:
        idx = _index(2)
        tw = pd.DataFrame({"A": [1.0, 0.0]}, index=idx)
        fr = pd.DataFrame({"A": [0.0, 0.0]}, index=idx)
        spec = NetConstructionSpec()
        stream = compute_net_return_stream(tw, fr, spec)
        expected_rate = spec.costs.fee_rate + spec.costs.slippage_rate
        assert np.allclose(stream.cost.to_numpy(), np.array([expected_rate, expected_rate]))

    def test_stress_cost_model_scales_cost_proportionally(self) -> None:
        idx = _index(40)
        rng = np.random.default_rng(1)
        targets = np.clip(0.5 + 0.3 * np.sin(np.arange(40) / 2.0), 0.0, 1.0)
        tw = pd.DataFrame({"A": targets}, index=idx)
        fr = pd.DataFrame({"A": [0.001] * 40}, index=idx)

        base = compute_net_return_stream(tw, fr, NetConstructionSpec())
        stressed = compute_net_return_stream(
            tw,
            fr,
            NetConstructionSpec(
                costs=CostModel(
                    fee_rate=0.0005 * STRESS_FEE_MULT,
                    slippage_rate=0.0003 * STRESS_SLIPPAGE_MULT,
                )
            ),
        )
        base_rate = 0.0005 + 0.0003
        stress_rate = 0.0005 * STRESS_FEE_MULT + 0.0003 * STRESS_SLIPPAGE_MULT
        ratio = stress_rate / base_rate
        assert np.allclose(stressed.cost.to_numpy(), base.cost.to_numpy() * ratio)

    def test_rejects_mismatched_index(self) -> None:
        idx = _index(4)
        tw = pd.DataFrame({"A": [0.0] * 4}, index=idx)
        fr = pd.DataFrame({"A": [0.0] * 4}, index=_index(4, start="2024-02-01"))
        with pytest.raises(DataIntegrityError):
            compute_net_return_stream(tw, fr, NetConstructionSpec())

    def test_rejects_non_datetime_index(self) -> None:
        idx = pd.RangeIndex(4)
        tw = pd.DataFrame({"A": [0.0] * 4}, index=idx)
        fr = pd.DataFrame({"A": [0.0] * 4}, index=idx)
        with pytest.raises(DataIntegrityError, match="DatetimeIndex"):
            compute_net_return_stream(tw, fr, NetConstructionSpec())

    def test_rejects_tz_naive_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="4h")
        tw = pd.DataFrame({"A": [0.0] * 4}, index=idx)
        fr = pd.DataFrame({"A": [0.0] * 4}, index=idx)
        with pytest.raises(DataIntegrityError):
            compute_net_return_stream(tw, fr, NetConstructionSpec())

    def test_rejects_non_monotonic_index(self) -> None:
        idx = _index(4)
        shuffled = pd.DatetimeIndex([idx[2], idx[0], idx[3], idx[1]])
        tw = pd.DataFrame({"A": [0.0] * 4}, index=shuffled)
        fr = pd.DataFrame({"A": [0.0] * 4}, index=shuffled)
        with pytest.raises(DataIntegrityError):
            compute_net_return_stream(tw, fr, NetConstructionSpec())

    def test_rejects_column_mismatch(self) -> None:
        idx = _index(4)
        tw = pd.DataFrame({"A": [0.0] * 4}, index=idx)
        fr = pd.DataFrame({"B": [0.0] * 4}, index=idx)
        with pytest.raises(DataIntegrityError):
            compute_net_return_stream(tw, fr, NetConstructionSpec())
