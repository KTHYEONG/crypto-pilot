"""Compatibility adapter for the legacy ``register_directional_candidate`` invocation.

The funding-gated directional candidate is an anti-pattern: it can never become
ACTIVE. The adapter forwards to the idempotent RETIRED migration of the legacy
registry; after the registry is deleted this is a verified no-op.
"""

from __future__ import annotations

import logging

from src.cli import compat


def main() -> None:
    compat.register_directional_candidate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
