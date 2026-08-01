"""Compatibility façade for the canonical ``src.application.data.collection``.

The public surface is preserved so the legacy import path resolves to the same
objects. Identity is verified by ``tests/contract/test_legacy_imports.py``.
"""

from __future__ import annotations

from src.application.data.collection import (
    collect_borrow,
    collect_funding,
    collect_metrics,
    collect_ohlcv,
    collect_spot_ohlcv,
    import_borrow,
    repair_spot_gap,
)

__all__ = [
    "collect_borrow",
    "collect_funding",
    "collect_metrics",
    "collect_ohlcv",
    "collect_spot_ohlcv",
    "import_borrow",
    "repair_spot_gap",
]
