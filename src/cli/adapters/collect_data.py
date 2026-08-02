"""Compatibility adapter for the legacy ``collect_data`` invocation.

New canonical invocation: ``uv run python -m src.cli.main data collect ...``.
"""

from __future__ import annotations

import logging

from src.cli import compat


def main() -> None:
    compat.run_collect_data()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
