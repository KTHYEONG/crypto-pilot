from __future__ import annotations

import argparse

from src.application.research.portfolio import evaluation as evaluation_module
from src.research.contracts import PortfolioEvaluationRequest
from src.research.portfolio.defaults import DEFAULT_SYMBOLS


def _run_portfolio(args: argparse.Namespace) -> None:
    request = PortfolioEvaluationRequest(
        symbols=tuple(args.symbols),
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    evaluation_module.run_portfolio_evaluation(request)


def add_portfolio_multi_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``research run portfolio multi`` subcommand."""
    portfolio = run_sub.add_parser(
        "multi", help="Run the causal liquidity portfolio backtest",
    )
    portfolio.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    portfolio.add_argument("--start", default=None)
    portfolio.add_argument("--end", default=None)
    portfolio.add_argument("--initial-equity", type=float, default=10_000.0)
    portfolio.add_argument("--unseal-holdout", action="store_true", default=False)
    portfolio.add_argument("--no-log-run", action="store_true", default=False)
    portfolio.set_defaults(handler=_run_portfolio)
