"""Target-weight pre-screen PnL proxies (never Research-GO sources)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.quant.technical_experts.cross_sectional import (
    XsCompositeSpec,
    run_xs_composite_ledger,
    run_xs_composite_ledger_multi_tier,
)


def mhs_ledger_pnl(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    one_way_bps: float,
    execution_delay_bars: int = 1,
    gap_carry: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Pinned target-weight pre-screen proxy delegating to ``run_xs_composite_ledger``.

    ``XsCompositeSpec.halflife_bars`` and ``no_trade_band`` are inert passthrough
    constants here (they only affect weight-construction call sites this module
    never calls). The rebalances target notional implicitly, so it must never be
    used for Research GO, OOS, capital metrics, or capacity claims.
    """
    half = one_way_bps / 2.0 * 1e-4
    spec = XsCompositeSpec(
        halflife_bars=0,
        no_trade_band=0.0,
        execution_delay_bars=execution_delay_bars,
        fee_rate=half,
        slippage_rate=half,
        gap_carry=gap_carry,
    )
    equity, turnover = run_xs_composite_ledger(weights, opens, bar_funding, spec)
    net = equity.pct_change().dropna()
    return net, turnover


def mhs_ledger_pnl_multi_tier(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    one_way_bps_list: Sequence[float],
    execution_delay_bars: int = 1,
    gap_carry: bool = True,
) -> list[tuple[pd.Series, pd.Series]]:
    """Single-pass multi-tier pre-screen proxy sharing the ledger arrays.

    Mirrors ``mhs_ledger_pnl`` exactly for each entry in ``one_way_bps_list``:
    the spec is built with ``fee_rate = slippage_rate = bps / 2.0 * 1e-4`` and
    the frozen round-trip rate ``half + half`` (IEEE doubling is exact, so it
    equals the single call's ``round_trip_cost_rate()`` bit-for-bit). The shared
    array construction means element ``i``'s ``(net, turnover)`` is bit-identical
    to ``mhs_ledger_pnl(weights, opens, bar_funding, bps_i)`` for the same
    index. Like ``mhs_ledger_pnl`` this is a pinned pre-screen proxy -- never
    Research GO, OOS, capital metrics, or capacity claims. Raises ``ValueError``
    on an empty list or any negative bps.
    """
    if not one_way_bps_list:
        raise ValueError("one_way_bps_list must not be empty")
    for bps in one_way_bps_list:
        if bps < 0.0:
            raise ValueError(f"one_way_bps must be >= 0, got {bps}")

    base_spec = XsCompositeSpec(
        halflife_bars=0,
        no_trade_band=0.0,
        execution_delay_bars=execution_delay_bars,
        fee_rate=0.0,
        slippage_rate=0.0,
        gap_carry=gap_carry,
    )
    cost_rates = [bps / 2.0 * 1e-4 + bps / 2.0 * 1e-4 for bps in one_way_bps_list]
    results = run_xs_composite_ledger_multi_tier(
        weights, opens, bar_funding, base_spec, cost_rates,
    )
    return [(equity.pct_change().dropna(), turnover) for equity, turnover in results]


def _column_order_row_sum(matrix: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Left-to-right column accumulation == ``matrix.cumsum(axis=1)[:, -1]``.

    Bit-identical to the cumsum last column by construction: cumsum along
    axis=1 is exactly the same left-to-right per-row order (``np.add.reduce``
    is FORBIDDEN here -- pairwise summation, verified NOT bit-identical).
    Accumulates through a transposed (column-major) view so the running total
    stays hot and the auxiliary allocation is O(n_grid), down from the
    O(n_grid * n_local) cumsum temporary.
    """
    if out is None:
        out = np.zeros(matrix.shape[0], dtype="float64")
    else:
        out[...] = 0.0
    view = matrix.T
    for col in range(view.shape[0]):
        out += view[col]
    return out
