"""Compatibility adapter for the legacy ``run_backtest`` invocation.

New canonical invocation: ``uv run python -m src.cli.main research run baseline ...``.
"""

from __future__ import annotations

import logging

from src.cli import compat
from src.research.evaluation.policy import HOLDOUT_CUTOFF

__all__ = ["HOLDOUT_CUTOFF"]


def main() -> None:
    compat.run_backtest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
