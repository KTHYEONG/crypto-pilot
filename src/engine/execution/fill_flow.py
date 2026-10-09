"""Window fill-flow assembly for the streamed ledger chunk."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class FillFlowChunk:
    """Fill arrays booked while consuming one window, aligned to its grid."""

    positions: npt.NDArray[np.intp]
    columns: npt.NDArray[np.intp]
    quantities: npt.NDArray[np.float64]
    post_units: npt.NDArray[np.float64]
    prices: npt.NDArray[np.float64]
    flow: npt.NDArray[np.float64]
    fee_by_ts: npt.NDArray[np.float64]


def assemble_fill_flow(
    grid_ns: npt.NDArray[np.int64],
    n_grid: int,
    local_cols: Sequence[str],
    n_fill: int,
    fill_start: int,
    fill_bar_ns: Sequence[int],
    fill_symbol: Sequence[str],
    fill_qty: Sequence[float],
    fill_price: Sequence[float],
    fill_fee_bps: Sequence[float],
    fill_post_units: Sequence[float],
) -> FillFlowChunk:
    """Assemble grid-aligned fill flow and fee series for one window chunk."""
    sym_to_local = {s: j for j, s in enumerate(local_cols)}
    if n_fill:
        positions = np.searchsorted(
            grid_ns, np.asarray(fill_bar_ns[fill_start:], dtype="int64"), side="left",
        )
        columns = np.asarray(
            [sym_to_local[s] for s in fill_symbol[fill_start:]],
            dtype=np.intp,
        )
        quantities = np.asarray(fill_qty[fill_start:], dtype="float64")
        prices = np.asarray(fill_price[fill_start:], dtype="float64")
        fee = np.asarray(fill_fee_bps[fill_start:], dtype="float64")
        post_units = np.asarray(fill_post_units[fill_start:], dtype="float64")
        fee_amt = fee / 1e4 * np.abs(quantities) * prices
        flow = np.zeros(n_grid, dtype="float64")
        fee_by_ts = np.zeros(n_grid, dtype="float64")
        np.add.at(flow, positions, -(quantities * prices + fee_amt))
        np.add.at(fee_by_ts, positions, fee_amt)
    else:
        positions = np.empty(0, dtype=np.intp)
        columns = np.empty(0, dtype=np.intp)
        quantities = np.empty(0, dtype="float64")
        post_units = np.empty(0, dtype="float64")
        prices = np.empty(0, dtype="float64")
        flow = np.zeros(n_grid, dtype="float64")
        fee_by_ts = np.zeros(n_grid, dtype="float64")
    return FillFlowChunk(
        positions=positions,
        columns=columns,
        quantities=quantities,
        post_units=post_units,
        prices=prices,
        flow=flow,
        fee_by_ts=fee_by_ts,
    )
