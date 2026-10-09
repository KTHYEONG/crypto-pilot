"""assemble_fill_flow aligns window fills to the grid with fee attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.execution.fill_flow import assemble_fill_flow


def test_empty_window_yields_empty_aligned_arrays() -> None:
    """No fills produce empty index arrays and zero flow series."""
    grid_ns = pd.date_range("2025-01-01", periods=6, freq="3min", tz="UTC").asi8
    chunk = assemble_fill_flow(
        grid_ns, 6, ["AAA"], 0, 0, [], [], [], [], [], [],
    )
    assert chunk.positions.size == 0
    assert chunk.columns.size == 0
    assert (chunk.flow == 0.0).all()
    assert (chunk.fee_by_ts == 0.0).all()


def test_flow_and_fees_match_fill_economics() -> None:
    """Flow nets principal plus fee while fee_by_ts carries only the fee leg."""
    grid_ns = pd.date_range("2025-01-01", periods=6, freq="3min", tz="UTC").asi8
    chunk = assemble_fill_flow(
        grid_ns, 6, ["AAA", "BBB"], 2, 0,
        [int(grid_ns[1]), int(grid_ns[3])], ["AAA", "BBB"], [2.0, -1.0],
        [100.0, 50.0], [8.0, 8.0], [2.0, -1.0],
    )
    assert chunk.positions.tolist() == [1, 3]
    assert chunk.columns.tolist() == [0, 1]
    assert chunk.quantities.tolist() == [2.0, -1.0]
    fee_amt = np.array([8.0, 8.0]) / 1e4 * np.abs([2.0, -1.0]) * np.array([100.0, 50.0])
    np.testing.assert_allclose(chunk.fee_by_ts[[1, 3]], fee_amt)
    net = -(np.array([2.0, -1.0]) * np.array([100.0, 50.0]) + fee_amt)
    np.testing.assert_allclose(chunk.flow[[1, 3]], net)
    assert chunk.flow[[0, 2, 4, 5]].sum() == 0.0
