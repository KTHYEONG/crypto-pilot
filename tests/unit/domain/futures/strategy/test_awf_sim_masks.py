from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.tiered_workflow.awf_sim import _resolve_tradeable_mask


def _make_aligned_for_tradeable_masks() -> AlignedMarketData:
    t = 3
    n = 2
    close = np.full((t, n), 100.0, dtype=np.float64)
    return AlignedMarketData(
        datetimes=np.array(
            [
                np.datetime64("2025-01-01T00"),
                np.datetime64("2025-01-01T04"),
                np.datetime64("2025-01-01T08"),
            ]
        ),
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close.copy(),
        low_2d=close.copy(),
        close_2d=close.copy(),
        volume_2d=np.ones((t, n), dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
    )


def test_resolve_tradeable_mask_combines_optional_masks() -> None:
    aligned = _make_aligned_for_tradeable_masks()
    execution_eligibility_mask = np.array(
        [
            [True, True],
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )
    strategy_readiness_mask = np.array(
        [
            [True, True],
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )
    promotion_active_mask = np.array(
        [
            [True, True],
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )
    aligned = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=aligned.close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=aligned.active_mask,
        warm_mask=aligned.warm_mask,
        entry_block_mask=aligned.entry_block_mask,
        kill_mask=aligned.kill_mask,
        execution_eligibility_mask=execution_eligibility_mask,
        strategy_readiness_mask=strategy_readiness_mask,
        promotion_active_mask=promotion_active_mask,
    )

    row0 = _resolve_tradeable_mask(aligned=aligned, t=0, n_sym=2)
    row1 = _resolve_tradeable_mask(aligned=aligned, t=1, n_sym=2)
    row2 = _resolve_tradeable_mask(aligned=aligned, t=2, n_sym=2)

    np.testing.assert_array_equal(row0, np.array([True, True]))
    np.testing.assert_array_equal(row1, np.array([False, False]))
    np.testing.assert_array_equal(row2, np.array([False, True]))
