"""Compatibility adapter for the legacy ``run_sleeve_blend_backtest`` invocation.

New canonical invocation: ``uv run python -m src.cli.main research run sleeve-blend ...``.
"""

from __future__ import annotations

import logging

from src.cli import compat


def main() -> None:
    compat.run_sleeve_blend_backtest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
