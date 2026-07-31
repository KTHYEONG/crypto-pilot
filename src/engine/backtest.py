"""Compatibility façade for the canonical ``src.research.baseline.backtest``.

RF-COMPAT-01 compatibility of the legacy ``src.engine.backtest`` surface is
verified by ``tests/contract/test_legacy_imports.py::legacy_import_contract``.
"""

from __future__ import annotations

from src.research.baseline.backtest import (
    BacktestResult,
    TradeRecord,
    calculate_position_size,
    run_backtest,
)

__all__ = [
    "BacktestResult",
    "TradeRecord",
    "calculate_position_size",
    "run_backtest",
]
