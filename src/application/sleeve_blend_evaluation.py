"""Compatibility façade for the canonical ``src.application.research.sleeve_blend.evaluation``.

The public surface is preserved so the legacy import path resolves to the same
object. Identity is verified by ``tests/contract/test_legacy_imports.py``.
"""

from __future__ import annotations

from src.application.research.sleeve_blend.evaluation import run_sleeve_blend_evaluation

__all__ = ["run_sleeve_blend_evaluation"]
