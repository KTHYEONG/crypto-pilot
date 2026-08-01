"""Compatibility adapter for the legacy ``run_cash_carry_backtest`` invocation.

New canonical invocation: ``uv run python -m src.cli.main research run cash-carry ...``.
"""

from __future__ import annotations

import logging

from src.cli import compat


def main() -> None:
    compat.run_cash_carry_backtest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
