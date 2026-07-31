from __future__ import annotations

import argparse
import logging

from src.application.portfolio_evaluation import run_portfolio_evaluation
from src.research.contracts import PortfolioEvaluationRequest
from src.research.portfolio.defaults import DEFAULT_SYMBOLS

_logger = logging.getLogger("PortfolioBacktestRunner")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal liquidity portfolio v2 backtest")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--unseal-holdout", action="store_true", default=False)
    parser.add_argument("--no-log-run", action="store_true", default=False)
    args = parser.parse_args()

    request = PortfolioEvaluationRequest(
        symbols=tuple(args.symbols),
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_portfolio_evaluation(request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
