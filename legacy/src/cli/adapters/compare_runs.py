"""Compatibility adapter for the legacy ``compare_runs`` invocation.

New canonical invocation: ``uv run python -m src.cli.main provenance compare-runs ...``.
"""

from __future__ import annotations

import logging

from src.cli import compat


def main() -> None:
    compat.compare_runs()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
