"""Compatibility façade for the canonical ``src.application.research.expert_portfolio.admission``.

The public surface is preserved so the legacy import path resolves to the same
object. Identity is verified by ``tests/contract/test_legacy_imports.py``.
"""

from __future__ import annotations

from src.application.research.expert_portfolio.admission import run_technical_library_admission

__all__ = ["run_technical_library_admission"]
