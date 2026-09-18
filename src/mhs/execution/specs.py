"""Stress cost execution spec owned by the execution core."""

from __future__ import annotations

from src.mhs.params import STRESS_COST_MULTIPLIER
from src.mhs.types import ExecutionSpec


def _stress_cost_execution_spec(base: ExecutionSpec | None = None) -> ExecutionSpec:
    """Apply the registered stress cost multiplier without changing fill mechanics. Args: optional existing base costs. Returns: ExecutionSpec with multiplied fee/slippage and unchanged timeout/drift controls. Raises: existing ExecutionSpec validation errors for invalid costs."""
    resolved = ExecutionSpec() if base is None else base
    return ExecutionSpec(
        maker_fee_bps=resolved.maker_fee_bps * STRESS_COST_MULTIPLIER,
        taker_fee_bps=resolved.taker_fee_bps * STRESS_COST_MULTIPLIER,
        taker_slippage_bps=resolved.taker_slippage_bps * STRESS_COST_MULTIPLIER,
        passive_timeout_minutes=resolved.passive_timeout_minutes,
        name_drift_trim_max_weight=resolved.name_drift_trim_max_weight,
        name_drift_trim_interval_hours=resolved.name_drift_trim_interval_hours,
    )


__all__ = ["_stress_cost_execution_spec"]
