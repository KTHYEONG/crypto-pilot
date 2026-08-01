from __future__ import annotations

import argparse
import json
import sys

from src.application.baseline_evaluation import run_baseline_evaluation
from src.application.cash_carry_evaluation import run_cash_carry_evaluation
from src.application.expert_portfolio_evaluation import run_expert_portfolio_evaluation
from src.application.oi_deleveraging_evaluation import run_oi_deleveraging_evaluation
from src.application.portfolio_evaluation import run_portfolio_evaluation
from src.application.sleeve_blend_evaluation import run_sleeve_blend_evaluation
from src.application.technical_expert_evaluation import run_technical_expert_evaluation
from src.application.technical_library_admission import run_technical_library_admission
from src.application.technical_library_admission_backtest import (
    run_technical_library_admission_backtest,
)
from src.research.contracts import (
    BaselineEvaluationRequest,
    CashCarryEvaluationRequest,
    OIDeleveragingEvaluationRequest,
    PortfolioEvaluationRequest,
    SleeveBlendEvaluationRequest,
    TechnicalExpertEvaluationRequest,
)
from src.research.expert_portfolio.contracts import (
    ContextualRouterSpec,
    ExpertPortfolioEvaluationRequest,
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionBacktestRequest,
    TechnicalLibraryAdmissionRequest,
    expert_ids_from_admission_proposal_id,
)
from src.research.portfolio.defaults import DEFAULT_SYMBOLS

_SLEEVE_DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT")


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
    run_baseline_evaluation(request)


def _run_portfolio(args: argparse.Namespace) -> None:
    request = PortfolioEvaluationRequest(
        symbols=tuple(args.symbols),
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_portfolio_evaluation(request)


def _run_cash_carry(args: argparse.Namespace) -> None:
    request = CashCarryEvaluationRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_cash_carry_evaluation(request)


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
    run_sleeve_blend_evaluation(request)


def _run_expert_portfolio(args: argparse.Namespace) -> None:
    request = ExpertPortfolioEvaluationRequest(
        library_id=args.library_id,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_expert_portfolio_evaluation(request)


def _run_oi_deleveraging(args: argparse.Namespace) -> None:
    request = OIDeleveragingEvaluationRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_oi_deleveraging_evaluation(request)


def _run_technical_expert(args: argparse.Namespace) -> None:
    request = TechnicalExpertEvaluationRequest(
        candidate_id=args.candidate_id,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_technical_expert_evaluation(request)


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
    report = run_technical_library_admission(request)
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
    report = run_technical_library_admission_backtest(request)
    sys.stdout.write(
        json.dumps(report.to_report_dict(), sort_keys=True, indent=2, default=str)
        + "\n"
    )


def add_research_commands(research_parser: argparse.ArgumentParser) -> None:
    """Attach the ``research run <evaluation>`` group to the root parser."""
    sub = research_parser.add_subparsers(dest="research_command", required=True)
    run = sub.add_parser("run", help="Run one sealed research evaluation")
    run_sub = run.add_subparsers(dest="run_command", required=True)

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

    portfolio = run_sub.add_parser(
        "portfolio", help="Run the causal liquidity portfolio backtest",
    )
    portfolio.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    portfolio.add_argument("--start", default=None)
    portfolio.add_argument("--end", default=None)
    portfolio.add_argument("--initial-equity", type=float, default=10_000.0)
    portfolio.add_argument("--unseal-holdout", action="store_true", default=False)
    portfolio.add_argument("--no-log-run", action="store_true", default=False)
    portfolio.set_defaults(handler=_run_portfolio)

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

    sleeve = run_sub.add_parser("sleeve-blend", help="Run a sleeve-blend evaluation")
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

    oi = run_sub.add_parser(
        "oi-deleveraging", help="Run the sealed open-interest deleveraging screen",
    )
    oi.add_argument("--symbol", default="BTCUSDT")
    oi.add_argument("--start", default=None)
    oi.add_argument("--end", default=None)
    oi.add_argument("--unseal-holdout", action="store_true", default=False)
    oi.add_argument("--no-log-run", action="store_true", default=False)
    oi.set_defaults(handler=_run_oi_deleveraging)

    technical = run_sub.add_parser(
        "technical-expert", help="Run one sealed technical-expert candidate screen",
    )
    technical.add_argument("--candidate-id", required=True)
    technical.add_argument("--symbol", default="BTCUSDT")
    technical.add_argument("--start", default=None)
    technical.add_argument("--end", default=None)
    technical.add_argument("--initial-equity", type=float, default=10_000.0)
    technical.add_argument("--unseal-holdout", action="store_true", default=False)
    technical.add_argument("--no-log-run", action="store_true", default=False)
    technical.set_defaults(handler=_run_technical_expert)

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
