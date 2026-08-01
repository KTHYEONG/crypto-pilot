"""Compatibility façade for the canonical ``src.application.library_evaluation``.

The public surface is preserved so the legacy import path resolves to the same
object. Identity is verified by ``tests/contract/test_legacy_imports.py``.
"""

from __future__ import annotations

from src.application.library_evaluation import run_expert_portfolio_evaluation

__all__ = ["run_expert_portfolio_evaluation"]
