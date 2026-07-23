from __future__ import annotations

import pyarrow as pa
import pytest

from src.domain.futures.compound.alpha_events import build_active_forecast_state
from src.domain.futures.compound.contracts import AlphaEventTape


def _tape() -> AlphaEventTape:
    events = pa.table(
        {
            "recipe_id": ["r1", "future", "expired", "unknown"],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "NOPE"],
            "decision_time_ns": [10, 200, 10, 10],
            "expiry_time_ns": [200, 300, 50, 200],
            "alpha_rate_per_hour": [0.02, 0.5, 0.5, 0.1],
            "mean_edge_variance": [0.01, 0.01, 0.01, 0.01],
            "combination_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    return AlphaEventTape(
        events=events,
        recipe_definitions=(),
        evidence=(),
        active_recipe_ids=("r1",),
        model_version="v1",
        data_manifest_hash="m1",
        fold_manifest_hash="f1",
    )


def test_active_state_ignores_future_expired_and_unknown_events() -> None:
    state = build_active_forecast_state(
        tape=_tape(), decision_time_ns=100, symbols=("BTCUSDT",)
    )

    assert state.alpha_rate_1d.tolist() == pytest.approx([0.02])
    assert state.epistemic_variance_1d.tolist() == pytest.approx([0.01])
    assert state.active_event_ids == ("r1",)
