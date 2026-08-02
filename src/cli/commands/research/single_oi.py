from __future__ import annotations

import argparse

from src.application.research.oi import evaluation as evaluation_module
from src.research.contracts import OIDeleveragingEvaluationRequest


def _run_oi_deleveraging(args: argparse.Namespace) -> None:
    request = OIDeleveragingEvaluationRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    evaluation_module.run_oi_deleveraging_evaluation(request)


def add_single_oi_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``research run single oi`` subcommand."""
    oi = run_sub.add_parser(
        "oi", help="Run the sealed open-interest deleveraging screen",
    )
    oi.add_argument("--symbol", default="BTCUSDT")
    oi.add_argument("--start", default=None)
    oi.add_argument("--end", default=None)
    oi.add_argument("--unseal-holdout", action="store_true", default=False)
    oi.add_argument("--no-log-run", action="store_true", default=False)
    oi.set_defaults(handler=_run_oi_deleveraging)
