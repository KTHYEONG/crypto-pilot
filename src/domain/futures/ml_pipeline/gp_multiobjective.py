"""
Tier-3 NSGA-II style multi-objective GP (true Pareto over program trees) is not bundled:
it requires DEAP (or PySR) + symbolic regression wiring separate from gplearn SymbolicTransformer.

This module only exposes availability checks for optional future spikes.
"""

from __future__ import annotations


def is_deap_available() -> bool:
    try:
        import deap  # noqa: F401

        _ = deap
        return True
    except ImportError:
        return False
