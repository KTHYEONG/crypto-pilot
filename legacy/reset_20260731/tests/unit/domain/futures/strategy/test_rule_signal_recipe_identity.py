from __future__ import annotations

import numpy as np

from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.rule_signals import candidate_panels_to_events


def test_recipe_identity_always_emitted() -> None:
    panel = CandidateSignalPanel(
        family="trend_donchian",
        variant="lb20",
        params={"lookback": 20},
        datetimes=np.array(["2026-01-01T00:00", "2026-01-01T04:00"], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        signed_score_2d=np.array([[1.5], [2.0]], dtype=np.float64),
        side_hint_2d=np.array([[1], [1]], dtype=np.int8),
        expected_holding_bars=2,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        turnover_proxy_2d=np.ones((2, 1), dtype=np.float64),
        valid_mask_2d=np.ones((2, 1), dtype=np.bool_),
        metadata={"recipe_id": "r1", "native_tf": "4h"},
        archetype="trend",
    )

    events = candidate_panels_to_events(panels=(panel,), min_abs_score=0.5)

    assert "l0_recipe_id" in events.columns
    assert (events["l0_recipe_id"] == "r1").all()


def test_recipe_identity_empty_id_for_legacy_panels() -> None:
    panel = CandidateSignalPanel(
        family="trend_donchian",
        variant="lb20",
        params={"lookback": 20},
        datetimes=np.array(["2026-01-01T00:00", "2026-01-01T04:00"], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        signed_score_2d=np.array([[1.5], [2.0]], dtype=np.float64),
        side_hint_2d=np.array([[1], [1]], dtype=np.int8),
        expected_holding_bars=2,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        turnover_proxy_2d=np.ones((2, 1), dtype=np.float64),
        valid_mask_2d=np.ones((2, 1), dtype=np.bool_),
        metadata={},
        archetype="trend",
    )

    events = candidate_panels_to_events(panels=(panel,), min_abs_score=0.5)

    assert "l0_recipe_id" in events.columns
    assert (events["l0_recipe_id"] == "").all()
