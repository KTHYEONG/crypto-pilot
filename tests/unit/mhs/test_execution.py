from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution import (
    _column_order_row_sum,
    ExecutionReplayWindow,
    ExecutionSpec,
    _BoundExecutionReplayAccumulator,
)

SPEC = ExecutionSpec()






















def _partition_windows(
    grid: pd.DatetimeIndex,
    weights: pd.DataFrame,
    signals: pd.DatetimeIndex,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    marks: pd.DataFrame,
    funding: pd.DataFrame,
    spec: ExecutionSpec,
    n_windows: int = 2,
) -> list[ExecutionReplayWindow]:
    """Split a full fixture into contiguous execution windows exactly like the
    application planner: grid start at the previous window's last decision,
    grid end at the final order's strict timeout bar (last window covers the
    full grid)."""
    full_ns = np.asarray(grid, dtype="datetime64[ns]").astype("int64")
    timeout = spec.passive_timeout_minutes * 60_000_000_000
    sig_ns = np.asarray(signals, dtype="datetime64[ns]").astype("int64")
    spos = np.searchsorted(full_ns, sig_ns, side="right")
    resolve = [None] * len(weights)
    for i in range(len(weights)):
        if spos[i] >= len(full_ns):
            continue
        tns = full_ns[spos[i]] + timeout
        tpos = int(np.searchsorted(full_ns, tns, side="left"))
        if tpos < len(full_ns) and full_ns[tpos] == tns:
            resolve[i] = pd.Timestamp(tns, unit="ns", tz="UTC")
    bounds = np.array_split(np.arange(len(weights)), n_windows)
    out: list[ExecutionReplayWindow] = []
    prev_last: pd.Timestamp | None = None
    for bi, idxs in enumerate(bounds):
        is_last = bi == len(bounds) - 1
        ws = weights.iloc[idxs]
        sg = signals[idxs]
        grid_start = grid[0] if prev_last is None else prev_last
        if is_last:
            grid_end = grid[-1]
        else:
            grid_end = max((resolve[i] for i in idxs if resolve[i] is not None), default=ws.index[-1] + pd.Timedelta(hours=2))
        wgrid = pd.date_range(grid_start, grid_end, freq="5min", tz="UTC")
        out.append(
            ExecutionReplayWindow(
                window_start=grid_start,
                window_end=grid_end,
                columns=tuple(weights.columns),
                symbols=tuple(weights.columns),
                minute_grid=wgrid,
                highs=highs.loc[wgrid],
                lows=lows.loc[wgrid],
                closes=closes.loc[wgrid],
                marks=marks.loc[wgrid],
                bar_funding=funding.loc[wgrid],
                target_weights=ws,
                signal_available_at=sg,
            )
        )
        prev_last = ws.index[-1]
    return out


def _assert_replay_equivalent(oracle, windowed) -> None:
    fill_o = oracle.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    fill_w = windowed.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    assert len(fill_o) == len(fill_w)
    for col in ("timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"):
        assert fill_o[col].tolist() == fill_w[col].tolist()
    for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
        np.testing.assert_allclose(
            getattr(oracle.ledger, field).to_numpy(),
            getattr(windowed.ledger, field).to_numpy(),
            rtol=1e-12, atol=1e-12,
        )
    assert oracle.ledger.primary_valid == windowed.ledger.primary_valid
    assert oracle.ledger.invalid_reasons == windowed.ledger.invalid_reasons
    assert dict(oracle.termination_counts) == dict(windowed.termination_counts)
    assert oracle.fill_count == windowed.fill_count
    assert oracle.unfilled_count == windowed.unfilled_count
    assert oracle.fallback_count == windowed.fallback_count
    assert list(oracle.simulated_units.columns) == list(windowed.simulated_units.columns)
    assert len(oracle.simulated_units) == len(windowed.simulated_units)


def _assert_pair_equivalent(independent, paired, label: str) -> None:
    """MHS-MEM-PAIR-01: the paired fan-out result equals the independent
    single-bound call in fills, the six ledger series, gaps, counters,
    snapshots, and terminal state."""
    fill_o = independent.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    fill_p = paired.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    assert len(fill_o) == len(fill_p), label
    for col in ("timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"):
        assert fill_o[col].tolist() == fill_p[col].tolist(), (label, col)
    np.testing.assert_allclose(
        fill_o["pre_trade_equity"].to_numpy(dtype="float64"),
        fill_p["pre_trade_equity"].to_numpy(dtype="float64"),
        rtol=1e-12, atol=1e-12, err_msg=f"{label}: pre_trade_equity",
    )
    for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
        np.testing.assert_allclose(
            getattr(independent.ledger, field).to_numpy(),
            getattr(paired.ledger, field).to_numpy(),
            rtol=1e-12, atol=1e-12, err_msg=f"{label}: {field}",
        )
    assert independent.ledger.primary_valid == paired.ledger.primary_valid
    assert independent.ledger.invalid_reasons == paired.ledger.invalid_reasons
    assert independent.data_gaps == paired.data_gaps
    assert dict(independent.termination_counts) == dict(paired.termination_counts)
    assert independent.fill_count == paired.fill_count
    assert independent.unfilled_count == paired.unfilled_count
    assert independent.fallback_count == paired.fallback_count
    assert independent.forced_exit_count == paired.forced_exit_count
    assert independent.forced_exit_notional == paired.forced_exit_notional
    assert independent.submit_times.tolist() == paired.submit_times.tolist()
    assert independent.fill_times.tolist() == paired.fill_times.tolist()
    assert independent.all_intent_shortfall_bps == paired.all_intent_shortfall_bps
    assert independent.fill_source == paired.fill_source
    assert independent.mark_source == paired.mark_source
    assert independent.event_snapshots_retained == paired.event_snapshots_retained
    assert independent.simulated_units.equals(paired.simulated_units)
    assert independent.simulated_notional_weights.equals(paired.simulated_notional_weights)





def _assert_full_equivalence(enabled, disabled) -> None:
    """MHS-MEM-01: fills, six ledger series, validity, gaps, counters, and
    terminal state are identical between snapshot-disabled and snapshot-enabled
    replay at rtol=atol=1e-12."""
    fill_e = enabled.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    fill_d = disabled.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    assert len(fill_e) == len(fill_d)
    for col in ("timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"):
        assert fill_e[col].tolist() == fill_d[col].tolist()
    for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
        np.testing.assert_allclose(
            getattr(enabled.ledger, field).to_numpy(),
            getattr(disabled.ledger, field).to_numpy(),
            rtol=1e-12, atol=1e-12,
        )
    assert enabled.ledger.primary_valid == disabled.ledger.primary_valid
    assert enabled.ledger.invalid_reasons == disabled.ledger.invalid_reasons
    assert enabled.ledger.data_gaps == disabled.ledger.data_gaps
    assert enabled.data_gaps == disabled.data_gaps
    assert dict(enabled.termination_counts) == dict(disabled.termination_counts)
    assert enabled.fill_count == disabled.fill_count
    assert enabled.unfilled_count == disabled.unfilled_count
    assert enabled.fallback_count == disabled.fallback_count
    assert enabled.forced_exit_count == disabled.forced_exit_count
    assert enabled.forced_exit_notional == disabled.forced_exit_notional
    assert enabled.submit_times.tolist() == disabled.submit_times.tolist()
    assert enabled.fill_times.tolist() == disabled.fill_times.tolist()
    assert enabled.all_intent_shortfall_bps == disabled.all_intent_shortfall_bps
    assert list(enabled.simulated_units.columns) == list(disabled.simulated_units.columns)
    assert list(enabled.simulated_notional_weights.columns) == list(
        disabled.simulated_notional_weights.columns
    )















def test_column_order_row_sum_matches_cumsum_last_column() -> None:
    """SCENARIO_MHS_PERF_P2_02_LEDGER_REDUCER_BIT_IDENTICAL: bit-identical to
    X.cumsum(axis=1)[:, -1]."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((15032, 45))
    assert np.array_equal(_column_order_row_sum(X), X.cumsum(axis=1)[:, -1])


def test_column_order_row_sum_all_nan_column_propagates() -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((64, 4))
    X[:, 2] = np.nan
    assert np.array_equal(
        _column_order_row_sum(X), X.cumsum(axis=1)[:, -1], equal_nan=True,
    )


def test_column_order_row_sum_zero_columns_yields_zeros() -> None:
    """n_local == 0 yields a zero contribution series."""
    X = np.zeros((16, 0), dtype="float64")
    out = _column_order_row_sum(X)
    assert out.shape == (16,)
    assert np.array_equal(out, np.zeros(16))


def test_column_order_row_sum_single_column_is_identity() -> None:
    """n_local == 1 is the identity."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((32, 1))
    assert np.array_equal(_column_order_row_sum(X), X[:, 0])


def test_column_order_row_sum_rejects_pairwise_add_reduce() -> None:
    """np.add.reduce is pairwise and documented FORBIDDEN (not array-equal)."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((4096, 128)) * 1e8
    reference = X.cumsum(axis=1)[:, -1]
    assert not np.array_equal(np.add.reduce(X, axis=1), reference)


def test_column_order_row_sum_out_buffer_reset_n_grid_allocation() -> None:
    """The out buffer is reused and reset; auxiliary space is n_grid floats."""
    X = np.ones((100, 10))
    buf = np.empty(100, dtype="float64")
    out = _column_order_row_sum(X, out=buf)
    assert out is buf
    _column_order_row_sum(X, out=buf)
    assert np.array_equal(buf, np.full(100, 10.0))


class _EquityAtHarness:
    @staticmethod
    def build(nan_mask):
        grid = pd.date_range("2021-01-01", periods=8, freq="1min", tz="UTC")
        symbols = ("AAAUSDT", "BBBUSDT")
        closes = pd.DataFrame(
            {s: np.linspace(100.0, 107.0, 8) for s in symbols}, index=grid,
        )
        window = ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1],
            columns=symbols, symbols=symbols, minute_grid=grid,
            highs=closes * 1.01, lows=closes * 0.99, closes=closes, marks=closes,
            bar_funding=pd.DataFrame(0.0, index=grid, columns=symbols),
            target_weights=pd.DataFrame(0.0, index=grid[1:], columns=symbols),
            signal_available_at=grid[1:],
        )
        acc = _BoundExecutionReplayAccumulator(
            window, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False,
        )
        acc.last_prices_arr[:] = np.linspace(90.0, 120.0, 2)
        acc.last_prices_arr[nan_mask] = np.nan
        acc.units_arr[:] = [0.5, -0.25]
        return acc


def test_equity_at_bit_identical_to_nan_to_num_over_1000_draws() -> None:
    """SCENARIO_MHS_PERF_P2_03_EQUITY_AT_BIT_IDENTICAL: exactly equal to the
    pre-change form."""
    rng = np.random.default_rng(5)
    for _ in range(1000):
        mask = rng.random(2) < 0.5
        acc = _EquityAtHarness.build(mask)
        units, prices, cash = acc.units_arr, acc.last_prices_arr, acc.cash
        expected = cash + float(np.sum(units * np.nan_to_num(prices, nan=0.0)))
        assert acc._equity_at() == expected
        gpos = np.array([0])
        expected_g = cash + float(
            np.sum(units[gpos] * np.nan_to_num(prices[gpos], nan=0.0))
        )
        assert acc._equity_at(gpos) == expected_g
        assert isinstance(acc._equity_at(), float)


def test_equity_at_infinite_price_raises_data_integrity_error() -> None:
    """+/-inf fails closed instead of silently substituting finfo.max."""
    acc = _EquityAtHarness.build(np.array([False, False]))
    acc.last_prices_arr[1] = np.inf
    with pytest.raises(DataIntegrityError, match="infinite"):
        acc._equity_at()
    with pytest.raises(DataIntegrityError):
        acc._equity_at(np.array([1]))
    acc.last_prices_arr[1] = -np.inf
    with pytest.raises(DataIntegrityError):
        acc._equity_at()
