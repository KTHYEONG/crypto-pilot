"""Account input assembly preserves release anchors and observable liquidity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.engine.account_sources import assemble_account_inputs
from src.strategy.targets import FLOW_MOM_TOP20, StrategyTargets
from src.engine.strategy_backtest import StrategySourceContext
from src.core.resources import resolve_mhs_memory_budget


def test_account_inputs_share_entry_labels_without_using_entry_day_liquidity(tmp_path: Path) -> None:
    entries = pd.date_range("2021-02-01", periods=2, tz="UTC")
    releases = entries - pd.Timedelta(hours=1)
    weights = pd.DataFrame({"HELDUSDT": [0.5, -0.5], "IDLEUSDT": [0.0, 0.0]}, index=entries)
    candidate = StrategyTargets(
        target_weights=weights, signal_available_at=releases, strategy=FLOW_MOM_TOP20,
    )
    grid = pd.date_range(releases[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    marks = tmp_path / "3m"
    marks.mkdir()
    pd.DataFrame({
        "timestamp": grid.as_unit("ns").asi8 // 1_000_000,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    }).to_parquet(marks / "HELDUSDT.parquet")
    daily_index = pd.date_range("2021-01-01", "2021-02-02", tz="UTC")
    close = pd.DataFrame(100.0, index=daily_index, columns=weights.columns)
    quote = pd.DataFrame(2e6, index=daily_index, columns=weights.columns)
    quote.loc[entries] = 2e8
    funding = pd.Series(
        [0.001, 0.002], index=pd.DatetimeIndex([releases[0] + pd.Timedelta(minutes=3), releases[1]]),
    )
    context = StrategySourceContext(
        census=tuple(weights.columns), root=str(tmp_path),
        funding_by_symbol={"HELDUSDT": funding}, funding_failures={},
        budget=resolve_mhs_memory_budget(None), daily_close=close, daily_quote_volume=quote,
    )
    unit, prices, cumulative, adv, sigma, anchors = assemble_account_inputs(candidate, context)
    pd.testing.assert_frame_equal(unit, weights[["HELDUSDT"]])
    pd.testing.assert_index_equal(anchors, releases)
    assert prices.close.index[0] == releases[0]
    assert list(cumulative["HELDUSDT"]) == [0.0, 0.003]
    assert list(adv["HELDUSDT"]) == [2e6, 2e6]
    assert list(sigma["HELDUSDT"]) == [0.0, 0.0]
    for frame in (cumulative, adv, sigma):
        pd.testing.assert_index_equal(frame.index, entries)
