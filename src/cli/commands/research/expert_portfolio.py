from __future__ import annotations

import argparse
import json
import sys

from src.application.research.expert_portfolio import admission as admission_module
from src.application.research.expert_portfolio import (
    admission_backtest as admission_backtest_module,
)
from src.application.research.expert_portfolio import evaluation as evaluation_module
from src.research.expert_portfolio.admission_types import (
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionBacktestRequest,
    TechnicalLibraryAdmissionRequest,
    expert_ids_from_admission_proposal_id,
)
from src.research.expert_portfolio.models import (
    ContextualRouterSpec,
    ExpertPortfolioEvaluationRequest,
)


def _run_expert_portfolio(args: argparse.Namespace) -> None:
    request = ExpertPortfolioEvaluationRequest(
        library_id=args.library_id,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    evaluation_module.run_expert_portfolio_evaluation(request)


def _run_library_admission(args: argparse.Namespace) -> None:
    request = TechnicalLibraryAdmissionRequest(
        candidate_sources=tuple(args.candidate_source),
        symbols=tuple(args.symbols),
        router=ContextualRouterSpec(
            context_symbol=args.router_context_symbol,
            trend_lookback_bars=args.router_trend_lookback_bars,
            volatility_lookback_bars=args.router_volatility_lookback_bars,
            min_context_history_bars=args.router_min_context_history_bars,
        ),
        admission=LibraryAdmissionConfig(
            min_experts=args.min_experts,
            max_experts=args.max_experts,
            min_closed_trades=args.min_closed_trades,
            min_active_return_bars=args.min_active_return_bars,
            max_abs_pairwise_log_return_correlation=(
                args.max_abs_pairwise_log_return_correlation
            ),
            max_joint_negative_return_rate=args.max_joint_negative_return_rate,
            min_context_covered_states=args.min_context_covered_states,
            max_combinations=args.max_combinations,
            max_workers=args.max_workers,
        ),
        start=args.start,
        end=args.end,
    )
    report = admission_module.run_technical_library_admission(request)
    sys.stdout.write(
        json.dumps(report.to_report_dict(), sort_keys=True, indent=2, default=str)
        + "\n"
    )


def _run_library_admission_backtest(args: argparse.Namespace) -> None:
    if args.proposal_id is not None:
        expert_ids = expert_ids_from_admission_proposal_id(args.proposal_id)
    else:
        expert_ids = tuple(args.expert_id)
    request = TechnicalLibraryAdmissionBacktestRequest(
        expert_ids=expert_ids,
        router=ContextualRouterSpec(
            context_symbol=args.router_context_symbol,
            trend_lookback_bars=args.router_trend_lookback_bars,
            volatility_lookback_bars=args.router_volatility_lookback_bars,
            min_context_history_bars=args.router_min_context_history_bars,
        ),
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        max_workers=args.max_workers,
        log_run=not args.no_log_run,
    )
    report = admission_backtest_module.run_technical_library_admission_backtest(request)
    sys.stdout.write(
        json.dumps(report.to_report_dict(), sort_keys=True, indent=2, default=str)
        + "\n"
    )


def add_expert_portfolio_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the expert-portfolio feature leaf commands to ``research run``."""
    expert = run_sub.add_parser(
        "expert-portfolio", help="Run a registered expert portfolio evaluation",
    )
    expert.add_argument("--library-id", required=True)
    expert.add_argument("--start", default=None)
    expert.add_argument("--end", default=None)
    expert.add_argument("--initial-equity", type=float, default=10_000.0)
    expert.add_argument("--unseal-holdout", action="store_true", default=False)
    expert.add_argument("--no-log-run", action="store_true", default=False)
    expert.set_defaults(handler=_run_expert_portfolio)

    admission = run_sub.add_parser(
        "library-admission",
        help="Run a sealed library admission diagnostic over a candidate universe",
    )
    admission.add_argument(
        "--candidate-source", action="append", required=True,
        help="Frozen technical return source; repeatable",
    )
    admission.add_argument("--symbols", nargs="+", required=True)
    admission.add_argument("--router-context-symbol", required=True)
    admission.add_argument("--router-trend-lookback-bars", type=int, required=True)
    admission.add_argument("--router-volatility-lookback-bars", type=int, required=True)
    admission.add_argument("--router-min-context-history-bars", type=int, required=True)
    admission.add_argument("--min-experts", type=int, required=True)
    admission.add_argument("--max-experts", type=int, required=True)
    admission.add_argument("--min-closed-trades", type=int, required=True)
    admission.add_argument("--min-active-return-bars", type=int, required=True)
    admission.add_argument(
        "--max-abs-pairwise-log-return-correlation", type=float, required=True,
    )
    admission.add_argument("--max-joint-negative-return-rate", type=float, required=True)
    admission.add_argument("--min-context-covered-states", type=int, required=True)
    admission.add_argument("--max-combinations", type=int, required=True)
    admission.add_argument("--max-workers", type=int, default=None)
    admission.add_argument("--start", default=None)
    admission.add_argument("--end", default=None)
    admission.set_defaults(handler=_run_library_admission)

    admission_backtest = run_sub.add_parser(
        "library-admission-backtest",
        help="Backtest one admission proposal without registration or catalog mutation",
    )
    proposal = admission_backtest.add_mutually_exclusive_group(required=True)
    proposal.add_argument(
        "--proposal-id",
        help="Reversible proposal_id emitted by library-admission",
    )
    proposal.add_argument(
        "--expert-id",
        action="append",
        help="Selected '<return_source>:<symbol>' expert id; repeatable",
    )
    admission_backtest.add_argument("--router-context-symbol", required=True)
    admission_backtest.add_argument("--router-trend-lookback-bars", type=int, required=True)
    admission_backtest.add_argument(
        "--router-volatility-lookback-bars", type=int, required=True,
    )
    admission_backtest.add_argument(
        "--router-min-context-history-bars", type=int, required=True,
    )
    admission_backtest.add_argument("--initial-equity", type=float, default=10_000.0)
    admission_backtest.add_argument("--max-workers", type=int, default=None)
    admission_backtest.add_argument("--start", default=None)
    admission_backtest.add_argument("--end", default=None)
    admission_backtest.add_argument("--no-log-run", action="store_true", default=False)
    admission_backtest.set_defaults(handler=_run_library_admission_backtest)
