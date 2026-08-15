from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.research.expert_portfolio.catalog import default_catalog
from src.research.provenance.ledger import RUNS_LOG_PATH, load_evaluation_runs
from src.research.provenance.registration import register_expert_library

_SUMMARY_COLS = [
    "ts", "git_sha", "git_dirty", "symbol", "end",
    "metrics.trade_count", "metrics.cagr", "metrics.mdd",
    "metrics.sharpe", "metrics.profit_factor", "metrics.win_rate",
    "reliability.observation.verdict", "reliability.observation.lcb90_cagr",
    "reliability.fold_distribution.max_period_contribution",
    "reliability.stress_test.verdict",
]


def register_expert_library_command(args: argparse.Namespace) -> None:
    """Fingerprint and register one expert library, printing its registration id."""
    registration = register_expert_library(args.library_id, catalog=default_catalog())
    print(registration.registration_id)


def compare_runs_command(args: argparse.Namespace) -> None:
    """Render the filtered evaluation-run comparison (registrations excluded)."""
    ledger_path = Path(args.ledger_path) if args.ledger_path else RUNS_LOG_PATH
    df = load_evaluation_runs(ledger_path=ledger_path)
    if df.empty:
        print(f"No evaluation runs recorded yet at {ledger_path}")
        return

    if not args.full:
        cols = [c for c in _SUMMARY_COLS if c in df.columns]
        df = df[cols]

    df = df.sort_values(args.sort_by).tail(args.last)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


def add_provenance_commands(provenance_parser: argparse.ArgumentParser) -> None:
    """Attach the ``provenance`` registration and comparison group."""
    sub = provenance_parser.add_subparsers(dest="provenance_command", required=True)

    register = sub.add_parser("register", help="Register an immutable expert library")
    register_sub = register.add_subparsers(dest="register_command", required=True)
    expert = register_sub.add_parser(
        "expert-library", help="Fingerprint and register an expert library as ACTIVE",
    )
    expert.add_argument("--library-id", required=True)
    expert.set_defaults(handler=register_expert_library_command)

    compare = sub.add_parser("compare-runs", help="Compare recorded evaluation runs")
    compare.add_argument("--last", type=int, default=10, help="Show only the most recent N runs")
    compare.add_argument("--sort-by", default="ts", help="Column to sort by, e.g. metrics.sharpe")
    compare.add_argument("--full", action="store_true", help="Show every column instead of the summary set")
    compare.add_argument("--ledger-path", default=None, help="Override the provenance ledger path")
    compare.set_defaults(handler=compare_runs_command)
