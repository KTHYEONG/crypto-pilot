from __future__ import annotations

import argparse
import logging

from src.application.expert_portfolio_evaluation import run_expert_portfolio_evaluation
from src.research.expert_portfolio.contracts import ExpertPortfolioEvaluationRequest

_logger = logging.getLogger("ExpertPortfolioBacktestRunner")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a pre-registered expert portfolio backtest",
    )
    parser.add_argument("--library-id", required=True, help="registered expert library id")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--unseal-holdout", action="store_true", default=False)
    parser.add_argument("--no-log-run", action="store_true", default=False)
    args = parser.parse_args()

    request = ExpertPortfolioEvaluationRequest(
        library_id=args.library_id,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_expert_portfolio_evaluation(request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
