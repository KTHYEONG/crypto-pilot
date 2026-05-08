"""Availability checks for optional multi-objective GP spikes.

Tier-3 NSGA-II style multi-objective GP (true Pareto over program trees) is not bundled:
it requires DEAP (or PySR) + symbolic regression wiring separate from gplearn
SymbolicTransformer.
"""

from __future__ import annotations


def is_deap_available() -> bool:
    """Check if the DEAP library is installed.

    Returns:
        True if DEAP is available.

    """
    try:
        import deap

        _ = deap
        return True
    except ImportError:
        return False
