from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.domain.futures.universe.membership import (
    build_membership_mask_bundle,
    canonical_symbol,
)


def test_membership_mask_symbol_canonicalization() -> None:
    timeline = {date(2025, 1, 1): frozenset({"BTCUSDT"})}
    dt = pd.Series(
        pd.to_datetime(
            ["2025-01-01T00:00:00Z", "2025-01-01T04:00:00Z", "2025-01-01T08:00:00Z"],
            utc=True,
        )
    )
    bundle = build_membership_mask_bundle(
        datetimes=dt,
        symbol="BTC/USDT",
        timeline=timeline,
        warmup_bars_required=1,
        raw_kill_signal=np.zeros(len(dt), dtype=np.float64),
    )
    assert canonical_symbol("BTC/USDT") == "BTCUSDT"
    assert np.all(bundle.universe_active_mask == 1.0)
    assert np.all(bundle.universe_entry_warm_mask == 1.0)
    assert np.all(bundle.entry_block_mask == 0.0)


def test_membership_kill_and_entry_warm_masks() -> None:
    timeline = {
        date(2025, 1, 1): frozenset({"BTCUSDT"}),
        date(2025, 4, 1): frozenset(),
    }
    dt = pd.Series(
        pd.to_datetime(
            [
                "2025-03-31T20:00:00Z",
                "2025-04-01T00:00:00Z",
                "2025-04-01T04:00:00Z",
            ],
            utc=True,
        )
    )
    bundle = build_membership_mask_bundle(
        datetimes=dt,
        symbol="BTCUSDT",
        timeline=timeline,
        warmup_bars_required=2,
        raw_kill_signal=np.zeros(len(dt), dtype=np.float64),
    )
    np.testing.assert_array_equal(bundle.universe_active_mask, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(bundle.membership_kill_signal, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(bundle.kill_signal, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(bundle.universe_entry_warm_mask, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(bundle.entry_block_mask, np.array([1.0, 1.0, 1.0]))
