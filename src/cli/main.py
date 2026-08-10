from __future__ import annotations

import argparse
import logging

from src.cli.commands.data import add_data_commands
from src.cli.commands.provenance import add_provenance_commands
from src.cli.commands.research import add_research_commands


def build_root_parser() -> argparse.ArgumentParser:
    """Compose the single documented CLI entry point with three command groups.

    Top-level groups are ``data``, ``research``, and ``provenance``; all new run
    commands are children of ``research run``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.main",
        description="Consolidated crypto-pilot command line",
    )
    subparsers = parser.add_subparsers(dest="group", required=True)
    add_data_commands(subparsers.add_parser("data", help="Collect and manage market data"))
    add_research_commands(subparsers.add_parser("research", help="Run sealed research evaluations"))
    add_provenance_commands(subparsers.add_parser("provenance", help="Registration and run provenance"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (defaults to ``sys.argv[1:]``) and dispatch the handler."""
    parser = build_root_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
