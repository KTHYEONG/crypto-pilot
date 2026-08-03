from __future__ import annotations

import argparse
import logging

from src.application.research.growth import evaluation as growth_evaluation_module
from src.research.contracts import GrowthEngineEvaluationRequest
from src.research.portfolio.net_construction import NetConstructionSpec
from src.research.universe.pit_universe import PitUniverseSpec

_logger = logging.getLogger("PortfolioGrowthCli")


def _run_portfolio_growth(args: argparse.Namespace) -> None:
    request = GrowthEngineEvaluationRequest(
        universe=PitUniverseSpec(
            universe_size=args.universe_size,
            max_positions=args.max_positions,
        ),
        construction=NetConstructionSpec(
            rebalance_bars=args.rebalance_bars,
            no_trade_band=args.no_trade_band,
        ),
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        symbol_scope=args.symbol_scope,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    report = growth_evaluation_module.run_growth_engine_evaluation(request)
    _logger.info(
        "[EVAL] growth engine status=%s start=%s bars=%d trades=%d selected_risk=%s",
        report.status,
        report.start,
        len(report.equity),
        len(report.trades),
        report.sizing.selected_risk if report.sizing is not None else None,
    )


def add_portfolio_growth_commands(
    run_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Attach the ``research run portfolio growth`` subcommand."""
    growth = run_sub.add_parser(
        "growth", help="Run the growth-engine (constraint-first) universe evaluation",
    )
    growth.add_argument("--universe-size", type=int, default=20)
    growth.add_argument("--max-positions", type=int, default=5)
    growth.add_argument("--rebalance-bars", type=int, default=1)
    growth.add_argument("--no-trade-band", type=float, default=0.0)
    growth.add_argument("--symbol-scope", choices=("dev", "holdout", "all"), default="dev")
    growth.add_argument("--start", default=None)
    growth.add_argument("--end", default=None)
    growth.add_argument("--initial-equity", type=float, default=10_000.0)
    growth.add_argument("--unseal-holdout", action="store_true", default=False)
    growth.add_argument("--no-log-run", action="store_true", default=False)
    growth.set_defaults(handler=_run_portfolio_growth)
