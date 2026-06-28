from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.tiered_workflow.entry_cooldown import apply_entry_cooldown


def test_apply_entry_cooldown_blocks_newly_active_symbol() -> None:
    active_mask = np.zeros((6, 1), dtype=bool)
    active_mask[2:, 0] = True
    tradeable = np.array([True], dtype=bool)

    blocked = apply_entry_cooldown(
        tradeable=tradeable,
        active_mask_2d=active_mask,
        t=4,
        cooldown_bars=3,
    )
    released = apply_entry_cooldown(
        tradeable=tradeable,
        active_mask_2d=active_mask,
        t=5,
        cooldown_bars=3,
    )

    assert bool(blocked[0]) is False
    assert bool(released[0]) is True


def test_apply_entry_cooldown_returns_input_when_disabled() -> None:
    tradeable = np.array([True, False], dtype=bool)

    result = apply_entry_cooldown(
        tradeable=tradeable,
        active_mask_2d=None,
        t=3,
        cooldown_bars=0,
    )

    np.testing.assert_array_equal(result, tradeable)

