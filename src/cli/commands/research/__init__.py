from __future__ import annotations

import argparse

from src.cli.commands.research.expert_library import add_expert_library_commands
from src.cli.commands.research.portfolio_blend import add_portfolio_blend_commands
from src.cli.commands.research.portfolio_growth import add_portfolio_growth_commands
from src.cli.commands.research.portfolio_multi import add_portfolio_multi_commands
from src.cli.commands.research.single_baseline import add_single_baseline_commands
from src.cli.commands.research.single_carry import add_single_carry_commands
from src.cli.commands.research.single_oi import add_single_oi_commands
from src.cli.commands.research.single_technical import add_single_technical_commands


def add_research_commands(research_parser: argparse.ArgumentParser) -> None:
    """Attach the ``research run <tier> <evaluation>`` group to the root parser.

    Tiers classify evaluations by scope: ``single`` strategy screens, ``portfolio``
    multi-asset sleeves, and ``expert`` portfolio lifecycle. Each leaf module owns
    its ``argparse`` declaration and conversion to its typed request; this module
    only composes the tier registry.
    """
    sub = research_parser.add_subparsers(dest="research_command", required=True)
    run = sub.add_parser("run", help="Run one sealed research evaluation")
    run_sub = run.add_subparsers(dest="run_command", required=True)

    single = run_sub.add_parser("single", help="Run one sealed single-strategy screen")
    single_sub = single.add_subparsers(dest="single_command", required=True)
    add_single_baseline_commands(single_sub)
    add_single_technical_commands(single_sub)
    add_single_carry_commands(single_sub)
    add_single_oi_commands(single_sub)

    portfolio = run_sub.add_parser(
        "portfolio", help="Run a sealed multi-asset or sleeve portfolio evaluation",
    )
    portfolio_sub = portfolio.add_subparsers(dest="portfolio_command", required=True)
    add_portfolio_multi_commands(portfolio_sub)
    add_portfolio_blend_commands(portfolio_sub)
    add_portfolio_growth_commands(portfolio_sub)

    expert = run_sub.add_parser("expert", help="Run one sealed expert-library lifecycle step")
    expert_sub = expert.add_subparsers(dest="expert_command", required=True)
    add_expert_library_commands(expert_sub)
