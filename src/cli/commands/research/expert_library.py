from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from src.application.research.expert import admission as admission_module
from src.application.research.expert import (
    admission_backtest as admission_backtest_module,
)
from src.application.research.expert import (
    admission_pipeline as admission_pipeline_module,
)
from src.application.research.expert import evaluation as evaluation_module
from src.application.research.expert import exit_sweep as exit_sweep_module
from src.application.research.expert import (
    rolling_admission as rolling_admission_module,
)
from src.research.expert_portfolio.admission_types import (
    LIBRARY_ADMISSION_PROFILES,
    ROLLING_LIBRARY_ADMISSION_PROFILES,
    LibraryAdmissionConfig,
    TechnicalExpertExitSweepRequest,
    TechnicalLibraryAdmissionBacktestRequest,
    TechnicalLibraryAdmissionPipelineRequest,
    TechnicalLibraryAdmissionRequest,
    expert_ids_from_admission_proposal_id,
    resolve_library_admission_profile,
    resolve_rolling_library_admission_profile,
)
from src.research.expert_portfolio.models import (
    ContextualRouterSpec,
    ExpertPortfolioEvaluationRequest,
)
from src.research.expert_portfolio.rolling import rolling_admission_config_for_profile


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
        timeframe=args.timeframe,
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
        timeframe=args.timeframe,
        stop_loss_mode=args.stop_loss_mode, stop_loss_value=args.stop_loss_value, atr_period=args.atr_period, trailing_stop=args.trailing_stop,
    )
    report = admission_backtest_module.run_technical_library_admission_backtest(request)
    sys.stdout.write(
        json.dumps(report.to_report_dict(), sort_keys=True, indent=2, default=str)
        + "\n"
    )


def _run_exit_sweep(args: argparse.Namespace) -> None:
    request = TechnicalExpertExitSweepRequest(
        candidate_sources=tuple(args.candidate_source),
        symbols=tuple(args.symbols),
        timeframes=tuple(args.timeframes),
        fixed_pct_values=tuple(args.fixed_pct_values),
        atr_multiple_values=tuple(args.atr_multiple_values),
        atr_period=args.atr_period,
        include_baseline=not args.no_baseline,
        start=args.start,
        end=args.end,
        max_workers=args.max_workers,
    )
    report = exit_sweep_module.run_technical_expert_exit_sweep(request)
    sys.stdout.write(
        json.dumps(report.to_report_dict(), sort_keys=True, indent=2, default=str)
        + "\n"
    )


def _run_library_admission_pipeline(args: argparse.Namespace) -> None:
    selection_request = dataclasses.replace(
        resolve_library_admission_profile(args.profile), timeframe=args.timeframe,
    )
    request = TechnicalLibraryAdmissionPipelineRequest(
        selection=selection_request,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        max_backtest_proposals=args.max_backtest_proposals,
        initial_equity=args.initial_equity,
    )
    report = admission_pipeline_module.run_technical_library_admission_pipeline(request)
    sys.stdout.write(
        json.dumps(report.to_report_dict(), sort_keys=True, indent=2, default=str)
        + "\n"
    )


def _run_rolling_library_admission(args: argparse.Namespace) -> None:
    profile = resolve_rolling_library_admission_profile(args.profile)
    config = rolling_admission_config_for_profile(
        args.profile, profile.symbols, timeframe=args.timeframe,
    )
    request = rolling_admission_module.RollingLibraryAdmissionRequest(
        profile=profile,
        as_of=args.as_of,
        config=config,
        mode=args.mode,
        log_run=not args.no_log_run,
        require_complete_history=args.require_complete_history,
    )
    report = rolling_admission_module.run_rolling_library_admission(request)
    sys.stdout.write(
        json.dumps(report.to_report_dict(), sort_keys=True, indent=2, default=str)
        + "\n"
    )


def add_expert_library_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the expert-library lifecycle leaf commands to ``research run``."""
    expert = run_sub.add_parser(
        "eval", help="Run a registered expert portfolio evaluation",
    )
    expert.add_argument("--library-id", required=True)
    expert.add_argument("--start", default=None)
    expert.add_argument("--end", default=None)
    expert.add_argument("--initial-equity", type=float, default=10_000.0)
    expert.add_argument("--unseal-holdout", action="store_true", default=False)
    expert.add_argument("--no-log-run", action="store_true", default=False)
    expert.set_defaults(handler=_run_expert_portfolio)

    admission = run_sub.add_parser(
        "admission",
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
    admission.add_argument(
        "--timeframe", default="4h",
        help="Research timeframe (1h/2h/4h/6h/8h/12h/1d); default 4h preserves existing behavior",
    )
    admission.set_defaults(handler=_run_library_admission)

    admission_backtest = run_sub.add_parser(
        "backtest",
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
    admission_backtest.add_argument(
        "--timeframe", default="4h",
        help="Research timeframe (1h/2h/4h/6h/8h/12h/1d); default 4h preserves existing behavior",
    )
    admission_backtest.add_argument("--no-log-run", action="store_true", default=False)
    admission_backtest.add_argument(
        "--stop-loss-mode", choices=["fixed_pct", "atr_multiple"], default=None,
        help="Opt-in causal stop-loss engine. Flags: --stop-loss-mode, --stop-loss-value, --atr-period, --trailing-stop. None preserves pre-existing behavior",
    )
    admission_backtest.add_argument(
        "--stop-loss-value", type=float, default=None,
        help="Stop distance: fixed fraction of entry price, or ATR multiple",
    )
    admission_backtest.add_argument("--atr-period", type=int, default=14)
    admission_backtest.add_argument(
        "--trailing-stop", action="store_true", default=False,
    )
    admission_backtest.set_defaults(handler=_run_library_admission_backtest)

    exit_sweep = run_sub.add_parser(
        "exit-sweep",
        help="Sweep stop-loss exit settings for one or more technical candidates in-process, without the portfolio/router/stress path",
    )
    exit_sweep.add_argument(
        "--candidate-source", action="append", required=True,
        help="Frozen technical return source; repeatable",
    )
    exit_sweep.add_argument("--symbols", nargs="+", required=True)
    exit_sweep.add_argument(
        "--timeframes", nargs="+", default=["4h"],
        help="Research timeframes to sweep (1h/2h/4h/6h/8h/12h/1d)",
    )
    exit_sweep.add_argument("--start", default=None)
    exit_sweep.add_argument("--end", default=None)
    exit_sweep.add_argument(
        "--fixed-pct-values", nargs="*", type=float, default=[0.03, 0.05, 0.08],
        help="Fixed-percent stop distances crossed with static/trailing anchors",
    )
    exit_sweep.add_argument(
        "--atr-multiple-values", nargs="*", type=float, default=[1.5, 2.5, 4.0],
        help="ATR-multiple stop distances crossed with static/trailing anchors",
    )
    exit_sweep.add_argument("--atr-period", type=int, default=14)
    exit_sweep.add_argument(
        "--no-baseline", action="store_true", default=False,
        help="Omit the stop_loss_mode=None baseline cell from the grid",
    )
    exit_sweep.add_argument("--max-workers", type=int, default=None)
    exit_sweep.set_defaults(handler=_run_exit_sweep)

    pipeline = run_sub.add_parser(
        "pipeline",
        help="Run the frozen profile: candidate discovery plus OOS proposal backtests in one execution",
    )
    pipeline.add_argument(
        "--profile", required=True, choices=sorted(LIBRARY_ADMISSION_PROFILES),
        help="Frozen library-admission profile (e.g. technical-5symbol-2022-v1)",
    )
    pipeline.add_argument("--evaluation-start", default="2025-01-01")
    pipeline.add_argument("--evaluation-end", default="2025-12-31 20:00")
    pipeline.add_argument("--max-backtest-proposals", type=int, default=24)
    pipeline.add_argument("--initial-equity", type=float, default=10_000.0)
    pipeline.add_argument(
        "--timeframe", default="4h",
        help="Research timeframe (1h/2h/4h/6h/8h/12h/1d); default 4h preserves existing behavior",
    )
    pipeline.set_defaults(handler=_run_library_admission_pipeline)

    rolling = run_sub.add_parser(
        "rolling",
        help="Replay the quarterly rolling library admission through as_of and stitch closed quarters",
    )
    rolling.add_argument(
        "--profile", required=True, choices=sorted(ROLLING_LIBRARY_ADMISSION_PROFILES),
        help="Frozen rolling library admission profile (e.g. technical-5symbol-rolling)",
    )
    rolling.add_argument(
        "--as-of", required=True,
        help="Frozen data snapshot; the only temporal source, e.g. 2026-07-07 20:00+00:00",
    )
    rolling.add_argument("--mode", choices=("paper", "live"), default="paper")
    rolling.add_argument(
        "--require-complete-history", action="store_true", default=False,
        help="Reject the request when the newest deployment quarter is incomplete",
    )
    rolling.add_argument(
        "--timeframe", default="4h",
        help="Research timeframe (1h/2h/4h/6h/8h/12h/1d); default 4h preserves existing behavior",
    )
    rolling.add_argument("--no-log-run", action="store_true", default=False)
    rolling.set_defaults(handler=_run_rolling_library_admission)
