"""Compatibility re-exports for the split sleeve-blend execution modules.

The frozen public surface lives in four canonical modules — ``common.py``,
``fixed.py``, ``weights.py``, and ``directional.py`` — and this facade only
re-exports their public objects so the ``src.research.sleeve_blend.backtest``
import path and object identity are preserved. Private helpers are not
compatibility API.
"""

from __future__ import annotations

from src.research.sleeve_blend.directional import (
    run_directional_sleeve_portfolio,
    run_directional_sleeve_portfolio_fixed_weights,
    run_directional_sleeve_portfolio_with_weights,
)
from src.research.sleeve_blend.fixed import (
    apply_leverage_schedule,
    build_causal_leverage_schedule,
    run_fixed_sleeve_portfolio,
    run_fixed_sleeve_portfolio_calibrated,
    run_fixed_sleeve_portfolio_with_leverage,
    run_fixed_sleeve_portfolio_with_schedule,
)
from src.research.sleeve_blend.weights import (
    component_labels,
    compute_causal_risk_weights,
    symbol_of_component,
)

__all__ = [
    "apply_leverage_schedule",
    "build_causal_leverage_schedule",
    "component_labels",
    "compute_causal_risk_weights",
    "run_directional_sleeve_portfolio",
    "run_directional_sleeve_portfolio_fixed_weights",
    "run_directional_sleeve_portfolio_with_weights",
    "run_fixed_sleeve_portfolio",
    "run_fixed_sleeve_portfolio_calibrated",
    "run_fixed_sleeve_portfolio_with_leverage",
    "run_fixed_sleeve_portfolio_with_schedule",
    "symbol_of_component",
]


def _check_contract() -> None:
    """Executable import-identity assertions locking the facade to its canon."""
    assert apply_leverage_schedule.__module__ == "src.research.sleeve_blend.fixed"
    assert build_causal_leverage_schedule.__module__ == "src.research.sleeve_blend.fixed"
    assert run_fixed_sleeve_portfolio.__module__ == "src.research.sleeve_blend.fixed"
    assert run_fixed_sleeve_portfolio_calibrated.__module__ == (
        "src.research.sleeve_blend.fixed"
    )
    assert run_fixed_sleeve_portfolio_with_leverage.__module__ == (
        "src.research.sleeve_blend.fixed"
    )
    assert run_fixed_sleeve_portfolio_with_schedule.__module__ == (
        "src.research.sleeve_blend.fixed"
    )
    assert compute_causal_risk_weights.__module__ == "src.research.sleeve_blend.weights"
    assert component_labels.__module__ == "src.research.sleeve_blend.weights"
    assert symbol_of_component.__module__ == "src.research.sleeve_blend.weights"
    assert run_directional_sleeve_portfolio.__module__ == (
        "src.research.sleeve_blend.directional"
    )
    assert run_directional_sleeve_portfolio_with_weights.__module__ == (
        "src.research.sleeve_blend.directional"
    )
    assert run_directional_sleeve_portfolio_fixed_weights.__module__ == (
        "src.research.sleeve_blend.directional"
    )


_check_contract()
