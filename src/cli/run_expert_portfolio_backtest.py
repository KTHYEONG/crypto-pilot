"""Compatibility adapter for the legacy ``run_expert_portfolio_backtest`` invocation.

New canonical invocation: ``uv run python -m src.cli.main research run expert-portfolio --library-id <id>``.
"""

from __future__ import annotations

import logging

from src.cli import compat


def main() -> None:
    compat.run_expert_portfolio_backtest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
