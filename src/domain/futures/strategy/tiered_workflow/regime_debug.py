from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.tiered_workflow.dataclasses import RegimeRoutingDiagnostics


def replace_selected_regime_debug_diagnostics(
    *,
    routing_diag: RegimeRoutingDiagnostics,
    selected_return_sum_bps: NDArray[np.float64],
    selected_bar_count: NDArray[np.int64],
) -> RegimeRoutingDiagnostics:
    """Replace debug payload with realized selected-book regime returns."""
    debug_diag = routing_diag.debug_diagnostics
    if debug_diag is None:
        return routing_diag

    state_count = len(routing_diag.active_state_names)
    if state_count <= 0:
        return routing_diag

    mean_returns = tuple(
        float(selected_return_sum_bps[idx] / max(int(selected_bar_count[idx]), 1))
        for idx in range(state_count)
    )
    bar_counts = tuple(int(selected_bar_count[idx]) for idx in range(state_count))
    updated_debug_diag = replace(
        debug_diag,
        selected_regime_return_bps=mean_returns,
        selected_regime_bar_count=bar_counts,
    )
    return replace(routing_diag, debug_diagnostics=updated_debug_diag)
