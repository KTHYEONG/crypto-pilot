from __future__ import annotations

from typing import Any


def assert_selection_replay_parity(
    *,
    replay_evaluation: Any,
    final_evaluation: Any,
    tolerance: float = 1e-8,
) -> None:
    metric_names = (
        ("cagr_hybrid", "cagr"),
        ("mdd_hybrid", "mdd"),
        ("fold_pass_ratio", "fold_pass"),
        ("trade_count", "trade_count"),
    )
    mismatches: list[str] = []
    for attr, label in metric_names:
        replay_value = getattr(replay_evaluation, attr, None)
        final_value = getattr(final_evaluation, attr, None)
        if replay_value is None or final_value is None:
            continue
        replay_f = float(replay_value)
        final_f = float(final_value)
        if abs(replay_f - final_f) > float(tolerance):
            mismatches.append(f"{label} replay={replay_f:.8f} final={final_f:.8f}")
    if mismatches:
        raise ValueError(f"replay/final parity failed: {', '.join(mismatches)}")
