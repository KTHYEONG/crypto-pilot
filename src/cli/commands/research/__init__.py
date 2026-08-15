from __future__ import annotations

import argparse


def add_research_commands(research_parser: argparse.ArgumentParser) -> None:
    """Attach the ``research run portfolio <evaluation>`` group to the root parser."""
    # Keep leaf-module imports lazy so coverage and command discovery can load
    # one command without initializing every research backend first.
    from src.cli.commands.research.mhs import add_mhs_commands

    sub = research_parser.add_subparsers(dest="research_command", required=True)
    run = sub.add_parser("run", help="Run one sealed research evaluation")
    run_sub = run.add_subparsers(dest="run_command", required=True)

    portfolio = run_sub.add_parser(
        "portfolio", help="Run a sealed multi-asset or sleeve portfolio evaluation",
    )
    portfolio_sub = portfolio.add_subparsers(dest="portfolio_command", required=True)
    add_mhs_commands(portfolio_sub)
