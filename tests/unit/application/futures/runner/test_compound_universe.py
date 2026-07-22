from __future__ import annotations

from datetime import UTC, datetime

from src.application.futures.runner.compound_universe import build_pit_universe_state


def test_pit_universe_state_has_requested_calendar() -> None:
    state = build_pit_universe_state(("BTCUSDT",), datetime(2026, 1, 1, tzinfo=UTC), bars=4)
    assert state.eligible.shape == (4, 1)
