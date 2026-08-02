from __future__ import annotations

import argparse

from src.application.research.blend import evaluation as evaluation_module
from src.research.contracts import SleeveBlendEvaluationRequest

_SLEEVE_DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT")


def _run_sleeve_blend(args: argparse.Namespace) -> None:
    request = SleeveBlendEvaluationRequest(
        symbols=tuple(args.symbols),
        mdd_budget_fraction=args.mdd_budget_fraction,
        candidate_kind=args.candidate_kind,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    evaluation_module.run_sleeve_blend_evaluation(request)


def add_portfolio_blend_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``research run portfolio blend`` subcommand."""
    sleeve = run_sub.add_parser("blend", help="Run a sleeve-blend evaluation")
    sleeve.add_argument("--symbols", nargs="+", default=list(_SLEEVE_DEFAULT_SYMBOLS))
    sleeve.add_argument("--mdd-budget-fraction", type=float, default=0.85)
    sleeve.add_argument(
        "--candidate-kind", default="fixed_long_only_v1",
        choices=["fixed_long_only_v1", "funding_signed_directional_v1"],
    )
    sleeve.add_argument("--start", default=None)
    sleeve.add_argument("--end", default=None)
    sleeve.add_argument("--initial-equity", type=float, default=10_000.0)
    sleeve.add_argument("--unseal-holdout", action="store_true", default=False)
    sleeve.add_argument("--no-log-run", action="store_true", default=False)
    sleeve.set_defaults(handler=_run_sleeve_blend)
