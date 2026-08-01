from __future__ import annotations

import argparse

from src.application.research.baseline import evaluation as evaluation_module
from src.research.contracts import BaselineEvaluationRequest


def _run_baseline(args: argparse.Namespace) -> None:
    request = BaselineEvaluationRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        min_taker_buy_ratio=args.min_taker_buy_ratio,
        funding_path=args.funding_path,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    evaluation_module.run_baseline_evaluation(request)


def add_baseline_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``research run baseline`` subcommand."""
    baseline = run_sub.add_parser("baseline", help="Run the v1 Donchian baseline backtest")
    baseline.add_argument("--symbol", default="BTCUSDT")
    baseline.add_argument("--start", default=None)
    baseline.add_argument("--end", default=None)
    baseline.add_argument("--initial-equity", type=float, default=10_000.0)
    baseline.add_argument("--min-taker-buy-ratio", type=float, default=None)
    baseline.add_argument("--funding-path", default=None)
    baseline.add_argument("--unseal-holdout", action="store_true", default=False)
    baseline.add_argument("--no-log-run", action="store_true", default=False)
    baseline.set_defaults(handler=_run_baseline)
