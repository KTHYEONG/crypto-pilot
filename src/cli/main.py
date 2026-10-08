from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import TypeVar, overload

from src.cli.commands.backtest import add_backtest_commands
from src.cli.commands.data import add_data_commands
from src.cli.commands.live import add_live_commands
from src.cli.commands.ops import add_ops_commands

_LAB_HELP = (
    "Exploratory research. Results here never deploy by themselves; "
    "a strategy reaches live trading only as a `StrategyRelease` "
    "promoted by `evaluate promote`"
)

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# Global options taking a separate value token; the lab-selection scan skips
# the value so a directory literally named "lab" never triggers registration.
_VALUE_OPTIONS = frozenset({"--log-level"})
_Namespace = TypeVar("_Namespace")


def configure_logging(*, level: int = logging.INFO, debug_streams: bool = False) -> None:
    """Configure root logging with the given level. debug_streams is reserved for P4."""
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def _lab_group_selected(argv: Sequence[str]) -> bool:
    """Detect ``lab`` as the selected top-level group without building its leaves.

    The research stack is registered only when selected, so ``live``, ``data``
    and ``backtest`` startup never imports ``src.lab``. The first bare token
    (after skipping global flags and their values) is the group under the root
    parser contract.
    """
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in _VALUE_OPTIONS:
            skip_next = True
            continue
        if arg.startswith("-") and arg != "-":
            continue
        return arg == "lab"
    return False


class _CommandParser(argparse.ArgumentParser):
    """Register research arguments when argparse dispatches to the lab group."""

    @overload
    def parse_known_args(
        self, args: Sequence[str] | None = None, namespace: None = None,
    ) -> tuple[argparse.Namespace, list[str]]: ...

    @overload
    def parse_known_args(
        self, args: Sequence[str] | None, namespace: _Namespace,
    ) -> tuple[_Namespace, list[str]]: ...

    @overload
    def parse_known_args(self, *, namespace: _Namespace) -> tuple[_Namespace, list[str]]: ...

    def parse_known_args(
        self, args: Sequence[str] | None = None, namespace: _Namespace | None = None,
    ) -> tuple[argparse.Namespace | _Namespace, list[str]]:
        if self.prog.endswith(" lab") and self._subparsers is None:
            from src.cli.commands.lab import add_lab_commands

            add_lab_commands(self)
        return super().parse_known_args(args, namespace)


def build_root_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    """Compose the single documented CLI entry point with five command groups.

    Top-level groups are ``data``, ``backtest``, ``lab``, ``live`` and ``ops``.
    ``argv`` can pre-register the lab leaves for command discovery. Otherwise
    argparse registers them when parsing the lab group, so the parser remains
    reusable and other command groups start without importing ``src.lab``.
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
    subparsers = parser.add_subparsers(dest="group", required=True, parser_class=_CommandParser)
    add_data_commands(subparsers.add_parser("data", help="Collect and manage market data"))
    add_backtest_commands(subparsers.add_parser("backtest", help="Run three-minute inventory backtests"))
    lab_parser = subparsers.add_parser("lab", help=_LAB_HELP, description=_LAB_HELP)
    probe = list(argv) if argv is not None else sys.argv[1:]
    if _lab_group_selected(probe):
        from src.cli.commands.lab import add_lab_commands

        add_lab_commands(lab_parser)
    add_live_commands(subparsers.add_parser("live", help="Live/shadow exchange execution"))
    add_ops_commands(subparsers.add_parser("ops", help="Operational provisioning for the VPS runtime"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (defaults to ``sys.argv[1:]``) and dispatch the handler."""
    parser = build_root_parser(argv)
    args = parser.parse_args(argv)
    configure_logging(level=_LOG_LEVELS[args.log_level], debug_streams=args.debug_streams)
    args.handler(args)


if __name__ == "__main__":
    main()
