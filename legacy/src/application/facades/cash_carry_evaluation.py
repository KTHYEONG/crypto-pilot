"""Compatibility façade for the canonical ``src.application.research.carry.evaluation``.

The public surface is preserved so the legacy import path resolves to the same
object. Identity is verified by ``tests/contract/test_legacy_imports.py``.
"""

from __future__ import annotations

from src.application.research.carry.evaluation import run_cash_carry_evaluation

__all__ = ["run_cash_carry_evaluation"]
