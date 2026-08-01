from __future__ import annotations

import argparse

from src.cli.commands.research.baseline import add_baseline_commands
from src.cli.commands.research.cash_carry import add_cash_carry_commands
from src.cli.commands.research.expert_portfolio import add_expert_portfolio_commands
from src.cli.commands.research.oi_deleveraging import add_oi_deleveraging_commands
from src.cli.commands.research.portfolio import add_portfolio_commands
from src.cli.commands.research.sleeve_blend import add_sleeve_blend_commands
from src.cli.commands.research.technical_experts import add_technical_experts_commands


def add_research_commands(research_parser: argparse.ArgumentParser) -> None:
    """Attach the ``research run <evaluation>`` group to the root parser.

    Each leaf module owns its ``argparse`` declaration and the conversion to
    its existing typed request; this module only composes the registry.
    """
    sub = research_parser.add_subparsers(dest="research_command", required=True)
    run = sub.add_parser("run", help="Run one sealed research evaluation")
    run_sub = run.add_subparsers(dest="run_command", required=True)

    add_baseline_commands(run_sub)
    add_portfolio_commands(run_sub)
    add_cash_carry_commands(run_sub)
    add_sleeve_blend_commands(run_sub)
    add_oi_deleveraging_commands(run_sub)
    add_technical_experts_commands(run_sub)
    add_expert_portfolio_commands(run_sub)
