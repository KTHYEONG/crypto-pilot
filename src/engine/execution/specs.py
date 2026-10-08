"""Stress cost execution spec owned by the execution core."""

from __future__ import annotations

import dataclasses

from src.core.params import SETTLEMENT_PRICE_STRESS_HAIRCUT_BPS, STRESS_COST_MULTIPLIER
from src.core.types import ExecutionSpec


def _stress_cost_execution_spec(base: ExecutionSpec | None = None) -> ExecutionSpec:
    """Apply the registered stress cost multiplier without changing fill mechanics. Args: optional existing base costs. Returns: ExecutionSpec with multiplied fee/slippage and unchanged timeout/drift controls. Raises: existing ExecutionSpec validation errors for invalid costs."""
    resolved = ExecutionSpec() if base is None else base
    return dataclasses.replace(
        resolved,
        maker_fee_bps=resolved.maker_fee_bps * STRESS_COST_MULTIPLIER,
        taker_fee_bps=resolved.taker_fee_bps * STRESS_COST_MULTIPLIER,
        taker_slippage_bps=resolved.taker_slippage_bps * STRESS_COST_MULTIPLIER,
        settlement_price_haircut_bps=SETTLEMENT_PRICE_STRESS_HAIRCUT_BPS,
    )


__all__ = ["_stress_cost_execution_spec"]
