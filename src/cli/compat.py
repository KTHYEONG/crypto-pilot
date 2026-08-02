"""Temporary forwarding adapters for the legacy per-run CLI entry points.

Each legacy module now forwards to the consolidated ``src.cli.main`` parser.
These adapters contain no argument parsing or business logic of their own;
they only map the legacy invocation shape onto a grouped command for the one
compatibility release.
"""

from __future__ import annotations

import sys

from src.cli.main import main as root_main
from src.common.config import BASE_DIR
from src.research.provenance.registration import migrate_legacy_candidate_registry

_COLLECT_RENAME = {"ohlcv": "futures-ohlcv"}


def _forward(argv: list[str]) -> None:
    root_main(argv)


def run_collect_data(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        args[0] = _COLLECT_RENAME.get(args[0], args[0])
    _forward(["data", "collect", *args])


def run_backtest(argv: list[str] | None = None) -> None:
    _forward(["research", "run", "single", "baseline", *list(sys.argv[1:] if argv is None else argv)])


def run_cash_carry_backtest(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "run":
        args = args[1:]
    _forward(["research", "run", "single", "carry", *args])


def run_portfolio_backtest(argv: list[str] | None = None) -> None:
    _forward(["research", "run", "portfolio", "multi", *list(sys.argv[1:] if argv is None else argv)])


def run_sleeve_blend_backtest(argv: list[str] | None = None) -> None:
    _forward(["research", "run", "portfolio", "blend", *list(sys.argv[1:] if argv is None else argv)])


def run_expert_portfolio_backtest(argv: list[str] | None = None) -> None:
    _forward(["research", "run", "expert", "eval", *list(sys.argv[1:] if argv is None else argv)])


def compare_runs(argv: list[str] | None = None) -> None:
    _forward(["provenance", "compare-runs", *list(sys.argv[1:] if argv is None else argv)])


def register_directional_candidate(argv: list[str] | None = None) -> None:
    """Forward the legacy registration to the idempotent RETIRED migration.

    The directional anti-pattern can never become ACTIVE; once the legacy
    registry is migrated this becomes a verified no-op.
    """
    del argv
    migrate_legacy_candidate_registry(
        BASE_DIR / "docs" / "results" / "candidate_registry.json",
    )
