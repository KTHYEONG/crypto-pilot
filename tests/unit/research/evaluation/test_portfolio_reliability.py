from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.evaluation.reliability import (
    ReliabilityGateConfig,
    compute_portfolio_reliability_gate,
    derive_block_size,
)


@pytest.fixture
def serial_portfolio_equity() -> pd.Series:
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.0005, 1000)
    idx = pd.date_range("2024-01-01", periods=1000, freq="4h", tz="UTC")
    return pd.Series(10000.0 * np.cumprod(1.0 + 0.001 + noise), index=idx)


class TestComputePortfolioReliabilityGate:
    def test_bootstrap_uses_marked_total_equity_returns(
        self, serial_portfolio_equity: pd.Series,
    ) -> None:
        # SC-PORT-06: the gate bootstraps the marked total-equity 4h return stream
        # and retains the frozen 15% hurdle, -25% MDD, 2.0 t-stat, and 30-close limits.
        config = ReliabilityGateConfig()
        gate = compute_portfolio_reliability_gate(serial_portfolio_equity, closed_trade_count=120)
        returns = serial_portfolio_equity.pct_change().dropna().to_numpy(dtype=np.float64)
        assert gate.block_size_used == derive_block_size(returns)
        assert gate.trade_count == 120
        assert gate.verdict == "PASS"
        assert gate.lcb90_cagr > config.hurdle_rate
        assert gate.t_stat > config.t_stat_floor
        assert gate.point_cagr > 0.0

    def test_portfolio_gate_determinism(self, serial_portfolio_equity: pd.Series) -> None:
        a = compute_portfolio_reliability_gate(serial_portfolio_equity, closed_trade_count=120)
        b = compute_portfolio_reliability_gate(serial_portfolio_equity, closed_trade_count=120)
        assert a == b

    def test_weak_returns_fail_not_pass(self) -> None:
        rng = np.random.default_rng(11)
        noise = rng.normal(0.0, 0.0005, 1000)
        idx = pd.date_range("2024-01-01", periods=1000, freq="4h", tz="UTC")
        equity = pd.Series(10000.0 * np.cumprod(1.0 + 0.00001 + noise), index=idx)
        gate = compute_portfolio_reliability_gate(equity, closed_trade_count=120)
        assert gate.verdict == "FAIL"
        assert gate.lcb90_cagr < ReliabilityGateConfig().hurdle_rate

    def test_pending_below_min_closed_trades(self, serial_portfolio_equity: pd.Series) -> None:
        gate = compute_portfolio_reliability_gate(serial_portfolio_equity, closed_trade_count=10)
        assert gate.verdict == "PENDING"

    def test_invalid_equity_raises(self, serial_portfolio_equity: pd.Series) -> None:
        with pytest.raises(ValueError, match="monotonic"):
            compute_portfolio_reliability_gate(
                serial_portfolio_equity.sort_index(ascending=False), 100,
            )
        with pytest.raises(ValueError, match="strictly positive"):
            compute_portfolio_reliability_gate(
                pd.Series(
                    [10000.0, -1.0, 20000.0],
                    index=pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC"),
                ), 100,
            )
        with pytest.raises(ValueError, match="finite"):
            compute_portfolio_reliability_gate(
                pd.Series(
                    [10000.0, np.nan],
                    index=pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC"),
                ), 100,
            )
        with pytest.raises(ValueError, match="at least 2"):
            compute_portfolio_reliability_gate(
                pd.Series(
                    [10000.0],
                    index=pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC"),
                ), 100,
            )
        with pytest.raises(ValueError, match="closed_trade_count"):
            compute_portfolio_reliability_gate(serial_portfolio_equity, -1)
