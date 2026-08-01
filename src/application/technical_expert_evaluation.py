"""Compatibility façade for the canonical ``src.application.expert_evaluation``.

The public surface is preserved so the legacy import path resolves to the same
object. Identity is verified by ``tests/contract/test_legacy_imports.py``.
"""

from __future__ import annotations

from src.application.expert_evaluation import run_technical_expert_evaluation

__all__ = ["run_technical_expert_evaluation"]
