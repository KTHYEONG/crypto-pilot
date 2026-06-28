from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)


def assert_selection_replay_parity(
    *,
    replay_evaluation: Any,
    final_evaluation: Any,
    tolerance: float = 1e-8,
) -> bool:
    """replay/final parity diagnostic. Returns True if within tolerance.

    No longer raises ValueError — logs warning on mismatch and returns False.
    Caller decides whether to gate on parity.
    """
    metric_names = (
        ("cagr_hybrid", "cagr"),
        ("mdd_hybrid", "mdd"),
        ("fold_pass_ratio", "fold_pass"),
        ("trade_count", "trade_count"),
    )
    mismatches: list[str] = []
    details: list[str] = []
    for attr, label in metric_names:
        replay_value = getattr(replay_evaluation, attr, None)
        final_value = getattr(final_evaluation, attr, None)
        if replay_value is None or final_value is None:
            details.append(f"{label}: missing on {'replay' if replay_value is None else 'final'}")
            continue
        replay_f = float(replay_value)
        final_f = float(final_value)
        delta = abs(replay_f - final_f)
        details.append(f"{label} replay={replay_f:.8f} final={final_f:.8f} delta={delta:.8f}")
        if delta > float(tolerance):
            mismatches.append(f"{label} replay={replay_f:.8f} final={final_f:.8f}")
    # extra: deploy_leverage, sharpe, sortino for deeper diagnosis
    for attr, label in (
        ("deploy_leverage", "L*"),
        ("sharpe_hac_hybrid", "sharpe_hac"),
        ("sortino_hybrid", "sortino"),
        ("constraint_values", "constraints"),
    ):
        replay_v = getattr(replay_evaluation, attr, None)
        final_v = getattr(final_evaluation, attr, None)
        if replay_v is not None and final_v is not None:
            details.append(f"{label} replay={replay_v!r} final={final_v!r}")

    if mismatches:
        _logger.warning(
            "[L2-PARITY-DIAG] replay/final parity mismatch (tolerance=%s): %s | %s",
            tolerance,
            "; ".join(mismatches),
            " | ".join(details),
        )
        return False
    return True
