from __future__ import annotations

import argparse

from src.application.research.cash_carry import evaluation as evaluation_module
from src.research.contracts import CashCarryEvaluationRequest


def _run_cash_carry(args: argparse.Namespace) -> None:
    request = CashCarryEvaluationRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    evaluation_module.run_cash_carry_evaluation(request)


def add_cash_carry_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``research run cash-carry`` subcommand."""
    cash_carry = run_sub.add_parser(
        "cash-carry", help="Run a sealed cash-and-carry evaluation",
    )
    cash_carry.add_argument("--symbol", default="BTCUSDT")
    cash_carry.add_argument("--start", default=None)
    cash_carry.add_argument("--end", default=None)
    cash_carry.add_argument("--initial-equity", type=float, default=10_000.0)
    cash_carry.add_argument("--unseal-holdout", action="store_true", default=False)
    cash_carry.add_argument("--no-log-run", action="store_true", default=False)
    cash_carry.set_defaults(handler=_run_cash_carry)
