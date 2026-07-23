from __future__ import annotations

import numpy as np

from src.domain.futures.compound.contracts import (
    ActiveForecastState,
    AlphaEventTape,
)


def build_active_forecast_state(
    *, tape: AlphaEventTape, decision_time_ns: int, symbols: tuple[str, ...]
) -> ActiveForecastState:
    n_syms = len(symbols)
    alpha_rate = np.zeros(n_syms, dtype=np.float64)
    epistemic_var = np.zeros(n_syms, dtype=np.float64)
    active_ids: list[str] = []

    for row in tape.events.to_pylist():
        if row["expiry_time_ns"] <= decision_time_ns:
            continue
        if row["decision_time_ns"] > decision_time_ns:
            continue

        symbol_idx = None
        for i, s in enumerate(symbols):
            if s == row["symbol"]:
                symbol_idx = i
                break
        if symbol_idx is None:
            continue

        alpha_rate[symbol_idx] += row["alpha_rate_per_hour"] * row["combination_weight"]
        epistemic_var[symbol_idx] += row["mean_edge_variance"]
        active_ids.append(row["recipe_id"])

    return ActiveForecastState(
        decision_time_ns=decision_time_ns,
        symbols=symbols,
        alpha_rate_1d=alpha_rate,
        epistemic_variance_1d=epistemic_var,
        active_event_ids=tuple(active_ids),
    )
