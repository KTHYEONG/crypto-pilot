from __future__ import annotations

import argparse
import logging
import os

from src.application.cash_carry_evaluation import run_cash_carry_evaluation
from src.research.contracts import CashCarryEvaluationRequest

_logger = logging.getLogger("CashCarryBacktestRunner")


def _run(args: argparse.Namespace) -> None:
    request = CashCarryEvaluationRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_cash_carry_evaluation(request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sealed cash-and-carry research runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a sealed cash-and-carry evaluation")
    run_p.add_argument("--symbol", default="BTCUSDT")
    run_p.add_argument("--start", default=None)
    run_p.add_argument("--end", default=None)
    run_p.add_argument("--initial-equity", type=float, default=10_000.0)
    run_p.add_argument("--unseal-holdout", action="store_true", default=False)
    run_p.add_argument("--no-log-run", action="store_true", default=False)
    run_p.set_defaults(func=_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    configured_level = getattr(
        logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO,
    )
    logging.basicConfig(level=configured_level)
    main()
