"""Per-symbol funding attribution reconciles to the ledger total."""

from __future__ import annotations

import pandas as pd

from src.engine.execution.ledger import simulated_inventory_ledger


def _ledger_with_funding() -> tuple:
    grid = pd.date_range("2021-06-01", periods=8, freq="h", tz="UTC")
    cols = ["AAA", "BBB"]
    marks = pd.DataFrame(100.0, index=grid, columns=cols, dtype="float64")
    rates = pd.DataFrame(0.0, index=grid, columns=cols, dtype="float64")
    rates.loc[grid[2:5], "AAA"] = -0.001
    rates.loc[grid[2:5], "BBB"] = 0.0005
    fills = pd.DataFrame([
        {
            "timestamp": grid[1], "symbol": "AAA", "quantity_delta": 10.0,
            "fill_price": 100.0, "fee_bps": 0.0, "reason": "test",
        },
        {
            "timestamp": grid[1], "symbol": "BBB", "quantity_delta": -10.0,
            "fill_price": 100.0, "fee_bps": 0.0, "reason": "test",
        },
    ])
    result = simulated_inventory_ledger(
        fills, marks, rates, 100000.0, "OHLCV_IMMEDIATE_TAKER", "MARK_PRICE",
    )
    return result, grid


def test_per_symbol_funding_sums_to_total() -> None:
    """Symbol attributions preserve signs and reconcile to the aggregate charge."""
    result, _ = _ledger_with_funding()
    by_symbol = dict(result.funding_by_symbol)
    assert set(by_symbol) == {"AAA", "BBB"}
    assert by_symbol["AAA"] < 0.0
    total = float(result.funding_charge.sum())
    assert sum(by_symbol.values()) == total
    assert abs(sum(by_symbol.values()) - total) <= 1e-6 * 100000.0
    daily = result.funding_by_symbol_daily
    assert daily is not None
    assert abs(float(daily.to_numpy().sum()) - total) <= 1e-6 * 100000.0
