from __future__ import annotations

import pandas as pd

from src.quant.evaluation.metrics import compute_metrics


class TestMetrics:
    def test_known_curve_and_degenerate_inputs(self) -> None:
        eq = pd.Series(
            [100.0, 120.0, 90.0, 180.0],
            index=pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"], utc=True),
        )
        m = compute_metrics(eq, pd.DataFrame(columns=["pnl"]))
        assert abs(m.mdd - (-0.25)) < 1e-12
        assert abs(m.cagr - 0.21638603540817) < 1e-9
        assert m.trade_count == 0

    def test_all_losing_trades(self) -> None:
        eq = pd.Series(
            [100.0, 180.0],
            index=pd.to_datetime(["2022-01-01", "2025-01-01"], utc=True),
        )
        trades = pd.DataFrame({"pnl": [-10.0, -20.0, -30.0]})
        m = compute_metrics(eq, trades)
        assert m.profit_factor == 0.0
        assert m.win_rate == 0.0
