from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_foundry.bridge_helpers import build_causal_l0_panel_views
from src.domain.futures.signals.contracts import CandidateSignalPanel


def _panel(recipe_id: str, tf: str = "4h") -> CandidateSignalPanel:
    n = 12
    return CandidateSignalPanel(
        family="trend_donchian",
        variant="lb20",
        params={"lookback": 20},
        datetimes=np.arange(
            np.datetime64("2026-01-01T00:00", "ns"),
            np.datetime64("2026-01-03T00:00", "ns"),
            np.timedelta64(4, "h"),
        ),
        symbols=("BTCUSDT",),
        signed_score_2d=np.ones((n, 1), dtype=np.float64),
        side_hint_2d=np.ones((n, 1), dtype=np.int8),
        expected_holding_bars=2,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        turnover_proxy_2d=np.ones((n, 1), dtype=np.float64),
        valid_mask_2d=np.ones((n, 1), dtype=np.bool_),
        metadata={"recipe_id": recipe_id, "native_tf": tf},
        archetype="trend",
    )


def test_causal_mask_is_exclusive_and_immutable() -> None:
    source = _panel("r1")
    original_mask = source.valid_mask_2d.copy()
    original_scores = source.signed_score_2d
    cutoff = int(np.datetime64("2026-01-02T00:00", "ns").astype(np.int64))

    (masked,) = build_causal_l0_panel_views(panels=(source,), evidence_end_ns=cutoff)

    assert np.array_equal(source.valid_mask_2d, original_mask)
    assert np.shares_memory(masked.signed_score_2d, original_scores)
    cutoff_idx = int(np.searchsorted(source.datetimes, np.datetime64(cutoff, "ns"), side="left"))
    assert not masked.valid_mask_2d[cutoff_idx:, :].any()


def test_causal_mask_excludes_equal_timestamp() -> None:
    source = _panel("r1")
    cutoff = int(np.datetime64("2026-01-02T00:00", "ns").astype(np.int64))
    cutoff_idx = int(np.searchsorted(source.datetimes, np.datetime64(cutoff, "ns"), side="left"))

    (masked,) = build_causal_l0_panel_views(panels=(source,), evidence_end_ns=cutoff)

    assert not masked.valid_mask_2d[cutoff_idx:, :].any()


def test_causal_mask_effect_limits_evidence_window() -> None:
    source = _panel("r1")
    cutoff = int(np.datetime64("2026-01-01T12:00", "ns").astype(np.int64))
    h_p = source.expected_holding_bars
    c_tf = int(np.searchsorted(source.datetimes, np.datetime64(cutoff, "ns"), side="left"))

    (masked,) = build_causal_l0_panel_views(panels=(source,), evidence_end_ns=cutoff)

    max_valid_t = c_tf - 1 - h_p
    if max_valid_t >= 0:
        assert not masked.valid_mask_2d[max_valid_t + 1 :, :].any()
