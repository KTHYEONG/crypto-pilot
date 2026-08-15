"""Compatibility façade for the canonical ``src.application.research.expert.evaluation``.

The public surface is preserved so the legacy import path resolves to the same
objects. Identity is verified by ``tests/contract/test_legacy_imports.py``.
"""

from __future__ import annotations

from src.application.research.expert.evaluation import (
    build_component_panel,
    build_library_decision_context,
    run_expert_portfolio_evaluation,
)

__all__ = [
    "build_component_panel",
    "build_library_decision_context",
    "run_expert_portfolio_evaluation",
]
