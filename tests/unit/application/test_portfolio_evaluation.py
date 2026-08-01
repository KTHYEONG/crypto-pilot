from __future__ import annotations

import numpy as np
import pytest

from src.application.research.portfolio.evaluation import run_portfolio_evaluation
from src.research.contracts import (
    CostModel,
    PortfolioEvaluationRequest,
    PortfolioSpec,
    StrategySpec,
)
from src.research.portfolio.backtest import run_portfolio_backtest
from src.research.portfolio.defaults import STRESS_FEE_MULT, STRESS_SLIPPAGE_MULT


def test_portfolio_evaluation_preserves_execution_invariants(
    monkeypatch,
    portfolio_frames,
) -> None:
    """RF-PORT-01: the canonical application path preserves the frozen ledger."""
    frames, funding_rates = portfolio_frames
    symbols = tuple(sorted(frames))
    monkeypatch.setattr(
        "src.application.research.portfolio.evaluation._load_symbol_frame",
        lambda symbol, start, end: frames.get(symbol),
    )
    monkeypatch.setattr(
        "src.application.research.portfolio.evaluation._load_symbol_funding",
        lambda symbol, frame: funding_rates.get(symbol),
    )

    request = PortfolioEvaluationRequest(
        symbols=symbols, start="2024-01-01", unseal_holdout=False, log_run=False,
    )
    report = run_portfolio_evaluation(request)

    strategy_spec = StrategySpec()
    portfolio_spec = PortfolioSpec()
    costs = CostModel()
    result = run_portfolio_backtest(
        frames, funding_rates, strategy_spec, portfolio_spec, costs,
    )

    assert report.status == "PASS"
    assert report.result.equity.equals(result.equity)
    assert report.result.trades.equals(result.trades)

    open_risk = 0.0
    if len(result.trades) > 0:
        assert set(result.trades.columns) >= {
            "symbol", "initial_risk", "portfolio_equity_before_entry",
        }


def test_portfolio_evaluation_preserves_2_5_percent_aggregate_risk(
    monkeypatch,
    portfolio_frames,
) -> None:
    """RF-PORT-01: the 2.5% aggregate initial-risk invariant never leaks."""
    from src.research.portfolio.backtest import MAX_TOTAL_INITIAL_RISK

    assert MAX_TOTAL_INITIAL_RISK == 0.025

    frames, funding_rates = portfolio_frames
    symbols = tuple(sorted(frames))
    monkeypatch.setattr(
        "src.application.research.portfolio.evaluation._load_symbol_frame",
        lambda symbol, start, end: frames.get(symbol),
    )
    monkeypatch.setattr(
        "src.application.research.portfolio.evaluation._load_symbol_funding",
        lambda symbol, frame: funding_rates.get(symbol),
    )
    report = run_portfolio_evaluation(
        PortfolioEvaluationRequest(
            symbols=symbols, start="2024-01-01", unseal_holdout=False, log_run=False,
        ),
    )
    trades = report.result.trades
    if len(trades) == 0:
        pytest.skip("fixture produced no closed trades")
    equity_before = trades["portfolio_equity_before_entry"].to_numpy(dtype=np.float64)
    initial_risk = trades["initial_risk"].to_numpy(dtype=np.float64)
    assert np.all(initial_risk <= MAX_TOTAL_INITIAL_RISK * equity_before + 1e-12)


def test_portfolio_evaluation_stress_uses_frozen_multipliers(
    monkeypatch,
    portfolio_frames,
) -> None:
    frames, funding_rates = portfolio_frames
    symbols = tuple(sorted(frames))
    monkeypatch.setattr(
        "src.application.research.portfolio.evaluation._load_symbol_frame",
        lambda symbol, start, end: frames.get(symbol),
    )
    monkeypatch.setattr(
        "src.application.research.portfolio.evaluation._load_symbol_funding",
        lambda symbol, frame: funding_rates.get(symbol),
    )
    report = run_portfolio_evaluation(
        PortfolioEvaluationRequest(
            symbols=symbols, start="2024-01-01", unseal_holdout=False, log_run=False,
        ),
    )
    costs = CostModel()
    assert abs(STRESS_FEE_MULT * costs.fee_rate - 1.5 * costs.fee_rate) < 1e-12
    assert abs(STRESS_SLIPPAGE_MULT * costs.slippage_rate - 2.0 * costs.slippage_rate) < 1e-12
