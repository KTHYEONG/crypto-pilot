from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.optimization.metrics import _bars_per_year_for_tf
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


# =============================================================================
# Fix 2: cost_amortize_by_holding field removed
# =============================================================================


def test_cost_amortize_by_holding_field_removed() -> None:
    field_names = {f.name for f in dataclasses.fields(CandidateStrategyConfig)}
    assert "cost_amortize_by_holding" not in field_names


# =============================================================================
# Fix 3: TF-generic bars_per_year
# =============================================================================


def test_cagr_calculation_uses_tf_generic_bars_per_year_for_6h() -> None:
    cfg = CandidateStrategyConfig(timeframe="6h")
    equity_curve = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
    trades = pd.DataFrame({
        "pnl": [1.0, 1.0, 1.0],
        "fee": [0.1, 0.1, 0.1],
        "funding": [0.0, 0.0, 0.0],
        "size": [10.0, 10.0, 10.0],
        "is_liquidation": [0, 0, 0],
    })

    report = evaluate_compound_backtest(trades=trades, equity_curve=equity_curve, cfg=cfg)

    expected_bpy = _bars_per_year_for_tf("6h")
    assert expected_bpy == pytest.approx(1460.0, rel=1e-6)
    assert report.cagr != 0.0


@pytest.mark.parametrize(
    ("tf", "expected_bpy"),
    [
        ("4h", 2190.0),
        ("6h", 1460.0),
        ("8h", 1095.0),
        ("12h", 730.0),
        ("1h", 8760.0),
        ("1d", 365.0),
    ],
)
def test_bars_per_year_for_tf_matches_across_all_supported_tfs(tf: str, expected_bpy: float) -> None:
    assert _bars_per_year_for_tf(tf) == pytest.approx(expected_bpy, rel=1e-6)


def test_candidate_evaluation_unsupported_tf_falls_back_to_4h_constant_not_crash() -> None:
    result = _bars_per_year_for_tf("unknown_tf_string")
    assert result == pytest.approx(2190.0, rel=1e-6)
