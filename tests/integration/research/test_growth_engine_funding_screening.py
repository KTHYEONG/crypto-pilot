"""Integration: ``funding_contrarian_v1`` screens successfully on the sealed
dev-discovery schedule once the union-NaN (C1) and scheduled-span scoping (C2)
fixes are in place -- see docs/specs/growth_engine_gate_diagnostics.md.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.application.research.growth import evaluation as ge
from src.research.contracts import GrowthEngineEvaluationRequest
from src.research.evaluation.policy import resolve_evaluation_end
from src.research.portfolio.growth_strategy_library import screen_growth_strategy_weights
from src.research.portfolio.net_construction import NetConstructionSpec
from src.research.universe.pit_universe import (
    PitUniverseSpec,
    build_universe_schedule,
    earliest_admissible_start,
    symbol_partition,
)


def _sealed_dev_discovery_inputs(
) -> tuple[dict[pd.Timestamp, tuple[str, ...]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    request = GrowthEngineEvaluationRequest(
        universe=PitUniverseSpec(universe_size=20, max_positions=5),
        construction=NetConstructionSpec(rebalance_bars=3, no_trade_band=0.0),
        start=None, end=None, initial_equity=10_000.0,
        symbol_scope="dev", unseal_holdout=False, log_run=False,
    )
    end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    end_ts = end.tz_convert("UTC") if isinstance(end, pd.Timestamp) else pd.Timestamp(end, tz="UTC")
    coverage, liquidity, frames = ge._build_universe_inputs(
        ge._list_symbols(), request.universe, request.start, end,
    )
    data_start = min(cov.first_bar for cov in coverage)
    data_last = max(cov.last_bar for cov in coverage)
    rebalance_dates = ge._build_rebalance_dates(data_start, min(end_ts, data_last))
    start = earliest_admissible_start(coverage, rebalance_dates, request.universe)
    full_schedule = build_universe_schedule(coverage, liquidity, rebalance_dates, request.universe)
    full_schedule = {d: r for d, r in full_schedule.items() if d >= start}
    all_symbols = sorted({s for r in full_schedule.values() for s in r})
    px, _fwd, taker = ge._build_price_panel(all_symbols, frames, start, end_ts)
    dev_members = {
        sym for sym in all_symbols
        if symbol_partition(sym, request.universe.dev_fraction) == "dev"
    }
    dev_schedule = ge._subset_schedule(full_schedule, dev_members)
    discovery_schedule, _ = ge._split_dev_schedule(dev_schedule)
    settled_funding = ge._build_settled_funding(all_symbols, px.index)
    return discovery_schedule, px, taker, settled_funding


@pytest.mark.slow
def test_funding_contrarian_v1_screens_successfully_on_dev_discovery_schedule() -> None:
    schedule, px, taker, settled_funding = _sealed_dev_discovery_inputs()
    for window in (42, 84, 168):
        screen = screen_growth_strategy_weights(
            "funding_contrarian_v1", window, schedule, px, taker, settled_funding,
        )
        assert screen.status == "SCREENED", screen.reason
