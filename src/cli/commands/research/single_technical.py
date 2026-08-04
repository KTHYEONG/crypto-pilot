from __future__ import annotations

import argparse
import logging

from src.application.research.technical import evaluation as evaluation_module
from src.research.contracts import TechnicalExpertEvaluationRequest

_logger = logging.getLogger("XsScreen")


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
    evaluation_module.run_technical_expert_evaluation(request)


def _run_trend_screen(args: argparse.Namespace) -> None:
    from src.application.research.technical.trend_screen import (
        TREND_SCREEN_PROFILE_ID,
        persist_trend_screen_report,
        run_trend_screen,
        trend_screen_report_path,
    )

    profile = args.profile or TREND_SCREEN_PROFILE_ID
    if profile != TREND_SCREEN_PROFILE_ID:
        raise ValueError(
            f"unknown trend-screen profile '{profile}'; the source-controlled "
            f"profile is '{TREND_SCREEN_PROFILE_ID}'"
        )
    report = run_trend_screen(start=args.start, end=args.end)
    persist_trend_screen_report(report, trend_screen_report_path())


def _run_xs_trend_screen(args: argparse.Namespace) -> None:
    from src.application.research.technical.xs_trend_screen import (
        XS_ALPHA_PROFILE_ID,
        XS_NEUTRAL_PROFILE_ID,
        persist_xs_screen_report,
        run_xs_trend_screen,
        xs_screen_report_path,
    )

    profile = args.profile or XS_NEUTRAL_PROFILE_ID
    if profile not in (XS_NEUTRAL_PROFILE_ID, XS_ALPHA_PROFILE_ID):
        raise ValueError(
            f"unknown xs screen profile '{profile}'; the source-controlled "
            f"profiles are '{XS_NEUTRAL_PROFILE_ID}' and '{XS_ALPHA_PROFILE_ID}'"
        )
    report = run_xs_trend_screen(
        start=args.start, end=args.end, unseal_holdout=args.unseal_holdout,
        profile=profile,
    )
    persist_xs_screen_report(report, xs_screen_report_path(profile))
    _logger.info(
        "xs-screen profile=%s discovery_admitted=%s qualification_admitted=%s "
        "qualification_sharpe=%.4f binding_constraint=%s",
        report.profile,
        report.discovery.admitted,
        report.qualification.admitted,
        report.qualification.sharpe,
        report.qualification.binding_constraint,
    )


def add_single_technical_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``research run single technical`` subcommands."""
    technical = run_sub.add_parser(
        "technical", help="Run one sealed technical-expert candidate screen",
    )
    technical.add_argument("--candidate-id", required=True)
    technical.add_argument("--symbol", default="BTCUSDT")
    technical.add_argument("--start", default=None)
    technical.add_argument("--end", default=None)
    technical.add_argument("--initial-equity", type=float, default=10_000.0)
    technical.add_argument("--unseal-holdout", action="store_true", default=False)
    technical.add_argument("--no-log-run", action="store_true", default=False)
    technical.set_defaults(handler=_run_technical_expert)

    trend_screen = run_sub.add_parser(
        "trend-screen", help="Run the pre-registered 450-cell baseline-gate trend screen",
    )
    trend_screen.add_argument("--profile", default=None)
    trend_screen.add_argument("--start", default=None)
    trend_screen.add_argument("--end", default=None)
    trend_screen.add_argument("--no-log-run", action="store_true", default=False)
    trend_screen.set_defaults(handler=_run_trend_screen)

    xs_screen = run_sub.add_parser(
        "xs-screen",
        help="Run the XS dollar-neutral composite trend screen (xs_neutral_composite_v1)",
    )
    xs_screen.add_argument("--profile", default=None)
    xs_screen.add_argument("--start", default=None)
    xs_screen.add_argument("--end", default=None)
    xs_screen.add_argument("--unseal-holdout", action="store_true", default=False)
    xs_screen.add_argument("--no-log-run", action="store_true", default=False)
    xs_screen.set_defaults(handler=_run_xs_trend_screen)
