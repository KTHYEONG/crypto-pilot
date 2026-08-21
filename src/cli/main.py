from __future__ import annotations

import argparse
import logging

from src.cli.commands.data import add_data_commands
from src.cli.commands.research import add_research_commands

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def configure_logging(*, level: int = logging.INFO, debug_streams: bool = False) -> None:
    """Configure root logging with the given level. debug_streams is reserved for P4."""
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def build_root_parser() -> argparse.ArgumentParser:
    """Compose the single documented CLI entry point with two command groups.

    Top-level groups are ``data`` and ``research``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.main",
        description="Consolidated crypto-pilot command line",
    )
    parser.add_argument(
        "--log-level",
        choices=list(_LOG_LEVELS),
        default="INFO",
        help="Set the root logging level (default: INFO).",
    )
    parser.add_argument(
        "--debug-streams",
        action="store_true",
        default=False,
        help="Enable JSONL sidecar streams for MHS stage telemetry (P4).",
    )
    subparsers = parser.add_subparsers(dest="group", required=True)
    add_data_commands(subparsers.add_parser("data", help="Collect and manage market data"))
    add_research_commands(subparsers.add_parser("research", help="Run sealed research evaluations"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (defaults to ``sys.argv[1:]``) and dispatch the handler."""
    parser = build_root_parser()
    args = parser.parse_args(argv)
    configure_logging(level=_LOG_LEVELS[args.log_level], debug_streams=args.debug_streams)
    args.handler(args)


if __name__ == "__main__":
    main()
