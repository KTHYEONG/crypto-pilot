from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import DenseSimConfig
from src.domain.futures.compound.contracts import ExecutionLedger, TimeframeBarCube
from src.domain.futures.compound.dense_simulator import simulate_dense_portfolio


class TestSimulateDensePortfolio:
    @pytest.fixture
    def two_asset_bars(self):
        n_bars = 500
        n_syms = 2
        close = np.cumprod(
            1.0 + np.tile(np.array([[0.001, -0.0005]]), (n_bars, 1)),
            axis=0,
        ).astype(np.float32) * 100.0
        return TimeframeBarCube(
            timeframe="4h",
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * 14_400_000_000_000,
            symbols=("ASSET_A", "ASSET_B"),
            open_2d=close.copy(),
            high_2d=close * 1.005,
            low_2d=close * 0.995,
            close_2d=close,
            quote_volume_2d=np.ones((n_bars, n_syms), dtype=np.float32) * 1_000_000,
            complete_2d=np.ones(n_bars, dtype=np.bool_),
        )

    def _cfg(self):
        return DenseSimConfig(bars_per_year=2190.0)

    def test_simulator_one_bar_execution_delay(self, two_asset_bars):
        n_bars = 500
        n_syms = 2
        w = np.zeros((n_bars, n_syms), dtype=np.float64)
        w[100:, 0] = 0.5

        ledger = simulate_dense_portfolio(
            two_asset_bars, w,
            funding_1h_2d=np.zeros((n_bars * 4, n_syms), dtype=np.float32),
            cost_bps=8.0, config=self._cfg(),
        )

        assert isinstance(ledger, ExecutionLedger)
        assert ledger.net_returns_1d[100] == pytest.approx(0.0)
        assert ledger.net_returns_1d[101] != 0.0

    def test_simulator_cost_charged_on_turnover(self, two_asset_bars):
        n_bars = 500
        n_syms = 2
        w = np.zeros((n_bars, n_syms), dtype=np.float64)
        w[100:, 0] = 0.5

        ledger = simulate_dense_portfolio(
            two_asset_bars, w,
            funding_1h_2d=np.zeros((n_bars * 4, n_syms), dtype=np.float32),
            cost_bps=8.0, config=self._cfg(),
        )

        expected_cost = 0.5 * 8.0 * 1e-4
        gross_101 = 0.5 * 0.001
        assert ledger.net_returns_1d[101] == pytest.approx(gross_101 - expected_cost, abs=1e-6)

    def test_funding_cost_applied(self, two_asset_bars):
        n_bars = 500
        n_syms = 2
        w = np.zeros((n_bars, n_syms), dtype=np.float64)
        w[100:, 0] = 0.5
        funding = np.ones((n_bars * 4, n_syms), dtype=np.float32) * 0.0001

        ledger = simulate_dense_portfolio(
            two_asset_bars, w,
            funding_1h_2d=funding,
            cost_bps=0.0, config=self._cfg(),
        )

        expected_funding_4h = 4 * 0.0001
        assert ledger.funding_returns_1d[101] == pytest.approx(-0.5 * expected_funding_4h, rel=1e-6)

    def test_shape_mismatch_raises(self, two_asset_bars):
        n_bars = two_asset_bars.timestamps_ns.size
        n_syms = len(two_asset_bars.symbols)
        with pytest.raises(ValueError, match="target_weights_2d shape"):
            simulate_dense_portfolio(
                two_asset_bars,
                np.ones((n_bars + 1, n_syms), dtype=np.float64),
                np.zeros((n_bars * 4, n_syms), dtype=np.float32),
                8.0, self._cfg(),
            )

    def test_equity_curve_monotonic_in_cash(self, two_asset_bars):
        n_bars = 500
        n_syms = 2
        w = np.zeros((n_bars, n_syms), dtype=np.float64)

        ledger = simulate_dense_portfolio(
            two_asset_bars, w,
            funding_1h_2d=np.zeros((n_bars * 4, n_syms), dtype=np.float32),
            cost_bps=0.0, config=self._cfg(),
        )
        assert np.allclose(ledger.equity_1d, 1.0)

    def test_cost_bps_as_array(self, two_asset_bars):
        n_bars = 500
        n_syms = 2
        w = np.zeros((n_bars, n_syms), dtype=np.float64)
        w[100:, 0] = 0.5
        cost_arr = np.full((n_bars, n_syms), 8.0, dtype=np.float32)

        ledger = simulate_dense_portfolio(
            two_asset_bars, w,
            funding_1h_2d=np.zeros((n_bars * 4, n_syms), dtype=np.float32),
            cost_bps=cost_arr, config=self._cfg(),
        )

        expected_cost = 0.5 * 8.0 * 1e-4
        gross_101 = 0.5 * 0.001
        assert ledger.net_returns_1d[101] == pytest.approx(gross_101 - expected_cost, abs=1e-6)

    def test_simple_return_accounting(self, two_asset_bars):
        n_bars = len(two_asset_bars.timestamps_ns)
        n_syms = len(two_asset_bars.symbols)
        w = np.zeros((n_bars, n_syms), dtype=np.float64)
        w[100:, 0] = 0.5

        ledger = simulate_dense_portfolio(
            two_asset_bars, w,
            funding_1h_2d=np.zeros((n_bars * 4, n_syms), dtype=np.float32),
            cost_bps=0.0, config=self._cfg(),
        )

        close_t = two_asset_bars.close_2d[101, 0]
        close_prev = two_asset_bars.close_2d[100, 0]
        simple_ret = float(close_t / close_prev - 1.0)
        expected = 0.5 * simple_ret
        assert ledger.net_returns_1d[101] == pytest.approx(expected, abs=1e-7)
