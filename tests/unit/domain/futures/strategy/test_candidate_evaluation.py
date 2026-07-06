from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_evaluation import evaluate_compound_backtest
from src.domain.futures.strategy.config import CandidateStrategyConfig


def test_evaluate_compound_backtest_calculates_correct_metrics() -> None:
    # 10 bars, upward compounding equity
    equity_curve = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0], dtype=np.float64)
    trades = pd.DataFrame(
        {
            "pnl": [2.0, 2.0, 2.0, 2.0, 2.0],
            "fee": [0.1, 0.1, 0.1, 0.1, 0.1],
            "funding": [0.0, 0.0, 0.0, 0.0, 0.0],
            "size": [10.0, 10.0, 10.0, 10.0, 10.0],
            "is_liquidation": [0, 0, 0, 0, 0],
        }
    )

    cfg = CandidateStrategyConfig(timeframe="4h", gross_cap=1.2)
    report = evaluate_compound_backtest(trades=trades, equity_curve=equity_curve, cfg=cfg)

    assert report.mean_log_growth > 0.0
    assert report.cagr > 0.0
    assert report.max_drawdown == 0.0
    assert report.net_pnl == 10.0
    assert report.fees == 0.5
    assert report.liquidation_count == 0


def test_evaluate_compound_backtest_fails_on_liquidation_or_high_drawdown() -> None:
    # Liquidation case
    equity_curve = np.array([100.0, 50.0, 10.0, 100.0], dtype=np.float64)
    trades = pd.DataFrame(
        {
            "pnl": [-50.0, -40.0, 90.0],
            "fee": [0.1, 0.1, 0.1],
            "is_liquidation": [0, 1, 0],
        }
    )

    cfg = CandidateStrategyConfig(timeframe="4h", gross_cap=0.1)  # tiny gross cap, drawdown will fail!
    report = evaluate_compound_backtest(trades=trades, equity_curve=equity_curve, cfg=cfg)

    assert not report.pass_compound_gate
    assert "liquidation occurred during simulation" in report.fail_reasons
    assert any("drawdown" in reason for reason in report.fail_reasons)
