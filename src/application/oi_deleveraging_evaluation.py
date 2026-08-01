"""Compatibility façade for the canonical ``src.application.research.oi_deleveraging.evaluation``.

The public surface is preserved so the legacy import path resolves to the same
object. Identity is verified by ``tests/contract/test_legacy_imports.py``.
"""

from __future__ import annotations

from src.application.research.oi_deleveraging.evaluation import run_oi_deleveraging_evaluation

__all__ = ["run_oi_deleveraging_evaluation"]
