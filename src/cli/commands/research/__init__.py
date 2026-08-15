from __future__ import annotations

import argparse


def add_research_commands(research_parser: argparse.ArgumentParser) -> None:
    """Attach the ``research run portfolio <evaluation>`` group to the root parser.

    The MHS evaluation is the only research leaf retained after the legacy
    isolation refactor (docs/specs/mhs_refactor.md §3.2): the ``single``/``expert``
    tiers and the non-MHS ``portfolio`` leaves moved to ``legacy/``. This module
    only composes the MHS leaf; the leaf module owns its ``argparse`` declaration
    and conversion to its typed request.
    """
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
