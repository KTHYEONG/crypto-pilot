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


def test_replay_is_invariant_to_future_tail_mark() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=5, freq='3min', tz='UTC')
    weights = pd.DataFrame({'BTCUSDT': [1.0]}, index=pd.DatetimeIndex([grid[0]]))
    def run(last: float):
        price = pd.DataFrame({'BTCUSDT': [100.0, 100.0, 100.0, 100.0, last]}, index=grid)
        window = ExecutionReplayWindow(window_start=grid[0], window_end=grid[-1], columns=('BTCUSDT',), symbols=('BTCUSDT',), minute_grid=grid, highs=price, lows=price, closes=price, marks=price, bar_funding=price*0.0, target_weights=weights, signal_available_at=pd.DatetimeIndex([grid[0]]), quote_volumes=price*0.0+1000.0, funding_known=price.notna(), bar_available_at=grid+pd.Timedelta(minutes=3))
        return replay_execution_windows((window,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    a, b = run(100.0), run(200.0)
    cols = ['timestamp', 'quantity_delta', 'fill_price']
    pd.testing.assert_frame_equal(a.simulated_fills[cols], b.simulated_fills[cols])


def test_replay_does_not_fabricate_terminal_exit() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=4, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    w = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, px*0.0, pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]), pd.DatetimeIndex([grid[0]]), quote_volumes=px*0.0+1.0, funding_known=px.notna(), bar_available_at=grid+pd.Timedelta(minutes=3))
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    assert result.forced_exit_count == 0
    assert 'forced_exit' not in set(result.simulated_fills['reason'])
    assert 'delist_settlement' not in set(result.simulated_fills['reason'])
    assert result.ledger.primary_valid
    assert not any(g.code == 'UNKNOWN_TERMINATION' for g in result.ledger.data_gaps)
    assert [p.status for p in result.terminal_positions] == ['open_marked']
    assert float(result.terminal_positions[0].quantity) == pytest.approx(10.0)


def test_zero_volume_bar_cannot_fill() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=4, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    w = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, px*0.0, pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]), pd.DatetimeIndex([grid[0]]), quote_volumes=px*float('nan'), funding_known=px.notna(), bar_available_at=grid+pd.Timedelta(minutes=3))
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    assert result.fill_count == 0
    assert any(g.code == 'ZERO_OR_UNKNOWN_VOLUME' for g in result.ledger.data_gaps)


def test_fill_effective_time_is_not_before_bar_availability() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=4, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    available = grid + pd.Timedelta(minutes=3)
    w = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, px*0.0, pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]), pd.DatetimeIndex([grid[0]]), quote_volumes=px*0.0+1.0, funding_known=px.notna(), bar_available_at=available)
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    assert (pd.to_datetime(result.simulated_fills['timestamp'], utc=True) >= available[1]).all()


def test_strategy_replay_delegates_to_causal_window_engine(monkeypatch) -> None:
    import src.mhs.execution.strategy_replay as module
    sentinel = object()
    calls = []
    monkeypatch.setattr(module, 'replay_execution_windows', lambda *args, **kwargs: calls.append((args, kwargs)) or sentinel)
    result = module._delegate_single_panel_window(object(), initial_equity=1.0, execution_bound='OHLCV_IMMEDIATE_TAKER', spec=object(), min_equity_fraction=None)
    assert result is sentinel
    assert len(calls) == 1


def test_replay_flags_future_mark_reference() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=4, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    w = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, px*0.0, pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]), pd.DatetimeIndex([grid[0]]), quote_volumes=px*0.0+1.0, funding_known=px.notna(), bar_available_at=grid+pd.Timedelta(hours=1))
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    assert not result.ledger.primary_valid
    assert any(g.code == 'FUTURE_DATA_REFERENCE' for g in result.ledger.data_gaps)


def test_replay_blocks_fill_after_bar_availability() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=4, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    w = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, px*0.0, pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]), pd.DatetimeIndex([grid[0]]), quote_volumes=px*0.0+1.0, funding_known=px.notna(), bar_available_at=grid-pd.Timedelta(hours=1))
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    assert result.fill_count == 0
    assert any(g.code == 'CAUSAL_TIMING_VIOLATION' for g in result.ledger.data_gaps)


def test_replay_flags_unknown_funding_on_held_and_active() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=5, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    known = px.notna()
    known.iloc[2, 0] = False
    w = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, px*0.0, pd.DataFrame({'BTCUSDT': [1.0, 1.0]}, index=[grid[0], grid[2]]), pd.DatetimeIndex([grid[0], grid[2]]), quote_volumes=px*0.0+1.0, funding_known=known, bar_available_at=grid+pd.Timedelta(minutes=3))
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    assert not result.ledger.primary_valid
    assert any(g.code == 'MISSING_HELD_FUNDING' for g in result.ledger.data_gaps)
    # 2026-09-15 mhs_symbol_lifespan_pit_roster 후속: 펀딩 unknown으로 막힌 신규
    # 체결 시도는 보유 리스크가 없어(체결 전) KNOWN_ZERO_VOLUME과 동일하게 무효화
    # 없이 미체결·재시도된다(NO_FUNDING_UNFILLED).
    blocked_known = px.notna()
    blocked_known.iloc[1, 0] = False
    wb = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, px*0.0, pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]), pd.DatetimeIndex([grid[0]]), quote_volumes=px*0.0+1.0, funding_known=blocked_known, bar_available_at=grid+pd.Timedelta(minutes=3))
    blocked = replay_execution_windows((wb,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    assert blocked.fill_count == 0
    assert blocked.unfilled_count == 1
    assert blocked.termination_counts['NO_FUNDING_UNFILLED'] == 1
    assert not any(g.code == 'MISSING_ACTIVE_FUNDING' for g in blocked.ledger.data_gaps)
    assert blocked.ledger.primary_valid


def test_replay_ledger_and_causal_mirror_agree_when_unknown_bar_carries_a_nonzero_rate() -> None:
    """A held-but-unknown-funding bar can still carry a real, non-zero aligned
    rate (e.g. a settlement bucketed just inside a coverage-start bar). The
    vectorized ledger must zero that charge exactly like the causal mirror,
    or ``reconcile_causal_state`` diverges (regression for the real-data bug
    found 2026-09-15: AIAUSDT/OMNIUSDT funding coverage gaps)."""
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=5, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    known = px.notna()
    known.iloc[2, 0] = False
    funding = px * 0.0
    funding.iloc[2, 0] = 0.01  # nonzero rate aligned to the very bar that is unknown
    w = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, funding, pd.DataFrame({'BTCUSDT': [1.0, 1.0]}, index=[grid[0], grid[2]]), pd.DatetimeIndex([grid[0], grid[2]]), quote_volumes=px*0.0+1.0, funding_known=known, bar_available_at=grid+pd.Timedelta(minutes=3))

    # This must not raise DataIntegrityError("causal accounting diverged from ledger").
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())

    assert not result.ledger.primary_valid
    assert any(g.code == 'MISSING_HELD_FUNDING' for g in result.ledger.data_gaps)
    # The unknown bar must not have charged the nonzero rate into the ledger.
    assert float(result.ledger.funding_charge.iloc[2]) == 0.0


def test_laddered_proxy_blocks_zero_volume_fill() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01', periods=12, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    w = ExecutionReplayWindow(grid[0], grid[-1], ('BTCUSDT',), ('BTCUSDT',), grid, px, px, px, px, px*0.0, pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]), pd.DatetimeIndex([grid[0]]), quote_volumes=px*float('nan'), funding_known=px.notna(), bar_available_at=grid+pd.Timedelta(minutes=3))
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_LADDERED_PROXY', ExecutionSpec())
    assert result.fill_count == 0
    assert any(g.code == 'ZERO_OR_UNKNOWN_VOLUME' for g in result.ledger.data_gaps)


def test_peg_chase_proxy_books_effective_time_fill() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01 12:00', periods=40, freq='5min', tz='UTC')
    px = pd.DataFrame({'A': [100.20] * len(grid)}, index=grid)
    marks = px.copy()
    marks.iloc[0, 0] = 100.0
    target = pd.DataFrame({'A': [0.01]}, index=pd.DatetimeIndex([pd.Timestamp('2025-01-01 12:00', tz='UTC')]))
    signal_at = pd.DatetimeIndex([pd.Timestamp('2025-01-01 13:00', tz='UTC')])
    w = ExecutionReplayWindow(grid[0], grid[-1], ('A',), ('A',), grid, px, px, px, marks, px*0.0, target, signal_at, quote_volumes=px*0.0+1000.0, funding_known=px.notna(), bar_available_at=grid+pd.Timedelta(minutes=5))
    result = replay_execution_windows((w,), 1.0, 'OHLCV_PEG_CHASE_PROXY', ExecutionSpec())
    assert not result.simulated_fills.empty


def test_peg_chase_proxy_blocks_zero_volume_fill() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    grid = pd.date_range('2025-01-01 12:00', periods=40, freq='5min', tz='UTC')
    px = pd.DataFrame({'A': [100.20] * len(grid)}, index=grid)
    marks = px.copy()
    marks.iloc[0, 0] = 100.0
    target = pd.DataFrame({'A': [0.01]}, index=pd.DatetimeIndex([pd.Timestamp('2025-01-01 12:00', tz='UTC')]))
    signal_at = pd.DatetimeIndex([pd.Timestamp('2025-01-01 13:00', tz='UTC')])
    w = ExecutionReplayWindow(grid[0], grid[-1], ('A',), ('A',), grid, px, px, px, marks, px*0.0, target, signal_at, quote_volumes=px*float('nan'), funding_known=px.notna(), bar_available_at=grid+pd.Timedelta(minutes=5))
    result = replay_execution_windows((w,), 1.0, 'OHLCV_PEG_CHASE_PROXY', ExecutionSpec())
    assert result.fill_count == 0
    assert any(g.code == 'ZERO_OR_UNKNOWN_VOLUME' for g in result.ledger.data_gaps)


def test_zero_volume_known_bar_is_unfilled_without_invalidating_ledger() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    # Given: a known zero-volume fill bar (exchange halt / delisting tail), finite marks
    grid = pd.date_range('2025-01-01', periods=4, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    w = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=('BTCUSDT',), symbols=('BTCUSDT',),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]),
        signal_available_at=pd.DatetimeIndex([grid[0]]), quote_volumes=px * 0.0,
        funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(minutes=3),
    )
    # When
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    # Then: no fill, no data gap, ledger stays valid, the miss is counted as unfilled
    assert result.simulated_fills.empty
    assert result.unfilled_count == 1
    assert result.termination_counts['NO_VOLUME_UNFILLED'] == 1
    assert not any(g.code in ('ZERO_OR_UNKNOWN_VOLUME', 'KNOWN_ZERO_VOLUME') for g in result.ledger.data_gaps)
    assert result.ledger.primary_valid


def test_unknown_volume_bar_remains_execution_data_gap() -> None:
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
    # Given: the fill bar's volume is unknown (NaN), i.e. a real data gap
    grid = pd.date_range('2025-01-01', periods=4, freq='3min', tz='UTC')
    px = pd.DataFrame({'BTCUSDT': 100.0}, index=grid)
    w = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=('BTCUSDT',), symbols=('BTCUSDT',),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=pd.DataFrame({'BTCUSDT': [1.0]}, index=[grid[0]]),
        signal_available_at=pd.DatetimeIndex([grid[0]]), quote_volumes=px * float('nan'),
        funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(minutes=3),
    )
    # When
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    # Then: fail-closed gap contract unchanged
    assert result.simulated_fills.empty
    assert any(g.code == 'ZERO_OR_UNKNOWN_VOLUME' for g in result.ledger.data_gaps)
    assert not result.ledger.primary_valid
    assert result.termination_counts.get('NO_VOLUME_UNFILLED', 0) == 0


def test_idle_held_position_never_settles_without_evidenced_event() -> None:

    import numpy as np
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow

    def _window(grid, qv_values, mark_values, decisions, weights):
        px = pd.DataFrame({'A': mark_values}, index=grid)
        return ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1], columns=('A',), symbols=('A',),
            minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
            target_weights=pd.DataFrame({'A': weights}, index=pd.DatetimeIndex(decisions)),
            signal_available_at=pd.DatetimeIndex(decisions),
            quote_volumes=pd.DataFrame({'A': qv_values}, index=grid),
            funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(hours=1),
        )

    from src.mhs.execution import ExecutionSpec, replay_execution_windows
    # Given: 1h grid; A trades (qv>0) only on bars 0..3, then a flat zero-volume tail at mark 80
    grid = pd.date_range('2025-01-01', periods=40, freq='1h', tz='UTC')
    qv = np.where(np.arange(40) <= 3, 1000.0, 0.0)
    marks = np.where(np.arange(40) <= 3, 100.0, 80.0)
    # buy at bar 1, then a flat target 27h after the last liquid bar (grid[3])
    w = _window(grid, qv, marks, [grid[0], grid[30]], [0.5, 0.0])
    # When
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    # Then: idle never settles without an evidenced event; the exit is unfilled,
    # inventory stays open and marked, and only the entry fee is charged
    fills = result.simulated_fills
    assert 'delist_settlement' not in set(fills['reason'])
    assert result.termination_counts.get('DELIST_SETTLEMENT', 0) == 0
    assert [p.status for p in result.terminal_positions] == ['open_marked']
    assert float(result.terminal_positions[0].quantity) == pytest.approx(5.0)
    assert float(result.ledger.fee_charge.sum()) == pytest.approx(0.4)
    assert float(result.ledger.equity.iloc[-1]) == pytest.approx(499.6 + 5.0 * 80.0, rel=1e-12)


def test_idle_settlement_not_triggered_before_24h() -> None:

    import numpy as np
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow

    def _window(grid, qv_values, mark_values, decisions, weights):
        px = pd.DataFrame({'A': mark_values}, index=grid)
        return ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1], columns=('A',), symbols=('A',),
            minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
            target_weights=pd.DataFrame({'A': weights}, index=pd.DatetimeIndex(decisions)),
            signal_available_at=pd.DatetimeIndex(decisions),
            quote_volumes=pd.DataFrame({'A': qv_values}, index=grid),
            funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(hours=1),
        )

    from src.mhs.execution import ExecutionSpec, replay_execution_windows
    # Given: the flat decision is only 17h after the last liquid bar (grid[3])
    grid = pd.date_range('2025-01-01', periods=40, freq='1h', tz='UTC')
    qv = np.where(np.arange(40) <= 3, 1000.0, 0.0)
    marks = np.full(40, 100.0)
    w = _window(grid, qv, marks, [grid[0], grid[20]], [0.5, 0.0])
    # When
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    # Then: no settlement; the exit on a zero-volume bar is unfilled and inventory stays open and marked
    assert 'delist_settlement' not in set(result.simulated_fills['reason'])
    assert result.termination_counts.get('DELIST_SETTLEMENT', 0) == 0
    assert result.termination_counts['NO_VOLUME_UNFILLED'] >= 1
    assert [p.status for p in result.terminal_positions] == ['open_marked']
    # A blocked exit keeps exposure on, so the ledger is invalid with the volume cause retained.
    assert not result.ledger.primary_valid
    assert any(g.code == 'KNOWN_ZERO_VOLUME' for g in result.ledger.data_gaps)


def test_idle_settlement_not_triggered_by_unknown_volume_hole() -> None:

    import numpy as np
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow

    def _window(grid, qv_values, mark_values, decisions, weights):
        px = pd.DataFrame({'A': mark_values}, index=grid)
        return ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1], columns=('A',), symbols=('A',),
            minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
            target_weights=pd.DataFrame({'A': weights}, index=pd.DatetimeIndex(decisions)),
            signal_available_at=pd.DatetimeIndex(decisions),
            quote_volumes=pd.DataFrame({'A': qv_values}, index=grid),
            funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(hours=1),
        )

    from src.mhs.execution import ExecutionSpec, replay_execution_windows
    # Given: after bar 3 the volume is UNKNOWN (NaN data hole), not a known zero tail
    grid = pd.date_range('2025-01-01', periods=40, freq='1h', tz='UTC')
    qv = np.where(np.arange(40) <= 3, 1000.0, np.nan)
    marks = np.full(40, 100.0)
    w = _window(grid, qv, marks, [grid[0], grid[30]], [0.5, 0.0])
    # When
    result = replay_execution_windows((w,), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    # Then: a data hole is never treated as a delisting
    assert 'delist_settlement' not in set(result.simulated_fills['reason'])
    assert result.termination_counts.get('DELIST_SETTLEMENT', 0) == 0
    assert any(g.code == 'ZERO_OR_UNKNOWN_VOLUME' for g in result.ledger.data_gaps)
    assert not result.ledger.primary_valid


def test_idle_tail_across_windows_never_settles_without_event() -> None:

    import numpy as np
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow

    def _window(grid, qv_values, mark_values, decisions, weights):
        px = pd.DataFrame({'A': mark_values}, index=grid)
        return ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1], columns=('A',), symbols=('A',),
            minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
            target_weights=pd.DataFrame({'A': weights}, index=pd.DatetimeIndex(decisions)),
            signal_available_at=pd.DatetimeIndex(decisions),
            quote_volumes=pd.DataFrame({'A': qv_values}, index=grid),
            funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(hours=1),
        )

    from src.mhs.execution import ExecutionSpec, replay_execution_windows
    # Given: two overlapping windows shaped like the production generator
    # (window 2 starts at window 1's last decision). Liquid bars 0..3 exist only
    # in window 1; window 2 is an all-zero tail, so the idle clock must come
    # from the carried last-liquid timestamp.
    full = pd.date_range('2025-01-01', periods=46, freq='1h', tz='UTC')
    g1 = full[:21]
    g2 = full[12:]
    qv1 = np.where(np.arange(21) <= 3, 1000.0, 0.0)
    w1 = _window(g1, qv1, np.full(21, 100.0), [full[0], full[12]], [0.5, 0.5])
    w2 = _window(g2, np.zeros(len(g2)), np.full(len(g2), 100.0), [full[30]], [0.0])
    # When
    result = replay_execution_windows((w1, w2), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())
    # Then: an all-zero tail across windows never settles without an evidenced event
    assert result.termination_counts.get('DELIST_SETTLEMENT', 0) == 0
    assert 'delist_settlement' not in set(result.simulated_fills['reason'])
    assert [p.status for p in result.terminal_positions] == ['open_marked']
    assert float(result.terminal_positions[0].quantity) == pytest.approx(5.0)


def test_held_position_outside_window_roster_fails_closed() -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.execution import replay_execution_windows
    B_WEIGHT = 0.5

    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec

    full = pd.date_range('2025-01-01', periods=16, freq='1h', tz='UTC')
    g1 = full[:10]
    px1 = pd.DataFrame({'A': 100.0, 'B': 100.0}, index=g1)
    w1 = ExecutionReplayWindow(
        window_start=g1[0], window_end=g1[-1], columns=('A', 'B'), symbols=('A', 'B'),
        minute_grid=g1, highs=px1, lows=px1, closes=px1, marks=px1, bar_funding=px1 * 0.0,
        target_weights=pd.DataFrame({'A': [0.0], 'B': [B_WEIGHT]}, index=pd.DatetimeIndex([full[0]])),
        signal_available_at=pd.DatetimeIndex([full[0]]), quote_volumes=px1 * 0.0 + 1000.0,
        funding_known=px1.notna(), bar_available_at=g1 + pd.Timedelta(hours=1),
    )
    g2 = full[5:]
    px2 = pd.DataFrame({'A': 100.0}, index=g2)
    w2 = ExecutionReplayWindow(
        window_start=g2[0], window_end=g2[-1], columns=('A', 'B'), symbols=('A',),
        minute_grid=g2, highs=px2, lows=px2, closes=px2, marks=px2, bar_funding=px2 * 0.0,
        target_weights=pd.DataFrame({'A': [0.0]}, index=pd.DatetimeIndex([full[12]])),
        signal_available_at=pd.DatetimeIndex([full[12]]), quote_volumes=px2 * 0.0 + 1000.0,
        funding_known=px2.notna(), bar_available_at=g2 + pd.Timedelta(hours=1),
    )

    # When / Then: B is still held but window 2's roster omits it
    with pytest.raises(DataIntegrityError, match='outside execution window roster'):
        replay_execution_windows((w1, w2), 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec())


def test_dust_units_outside_window_roster_do_not_trip() -> None:
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator
    B_WEIGHT = 0.0

    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec

    full = pd.date_range('2025-01-01', periods=16, freq='1h', tz='UTC')
    g1 = full[:10]
    px1 = pd.DataFrame({'A': 100.0, 'B': 100.0}, index=g1)
    w1 = ExecutionReplayWindow(
        window_start=g1[0], window_end=g1[-1], columns=('A', 'B'), symbols=('A', 'B'),
        minute_grid=g1, highs=px1, lows=px1, closes=px1, marks=px1, bar_funding=px1 * 0.0,
        target_weights=pd.DataFrame({'A': [0.0], 'B': [B_WEIGHT]}, index=pd.DatetimeIndex([full[0]])),
        signal_available_at=pd.DatetimeIndex([full[0]]), quote_volumes=px1 * 0.0 + 1000.0,
        funding_known=px1.notna(), bar_available_at=g1 + pd.Timedelta(hours=1),
    )
    g2 = full[5:]
    px2 = pd.DataFrame({'A': 100.0}, index=g2)
    w2 = ExecutionReplayWindow(
        window_start=g2[0], window_end=g2[-1], columns=('A', 'B'), symbols=('A',),
        minute_grid=g2, highs=px2, lows=px2, closes=px2, marks=px2, bar_funding=px2 * 0.0,
        target_weights=pd.DataFrame({'A': [0.0]}, index=pd.DatetimeIndex([full[12]])),
        signal_available_at=pd.DatetimeIndex([full[12]]), quote_volumes=px2 * 0.0 + 1000.0,
        funding_known=px2.notna(), bar_available_at=g2 + pd.Timedelta(hours=1),
    )

    acc = _BoundExecutionReplayAccumulator(w1, 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec(), False)
    acc.consume(w1)
    # Given: float dust left on B (below QTY_EPS), e.g. from cumsum vs += accumulation
    acc.units_arr[1] = 1e-15
    # When: must not raise
    acc.consume(w2)
    # Then
    assert acc.ledger_start_ns is not None


def test_fills_in_span_matches_bruteforce_and_advances_scan_cursor() -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator
    grid = pd.date_range('2025-01-01', periods=4, freq='1h', tz='UTC')
    px = pd.DataFrame({'A': 100.0, 'B': 100.0}, index=grid)
    w = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=('A', 'B'), symbols=('A', 'B'),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=pd.DataFrame({'A': [0.0], 'B': [0.0]}, index=[grid[0]]),
        signal_available_at=pd.DatetimeIndex([grid[0]]), quote_volumes=px * 0.0 + 1.0,
        funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(hours=1),
    )
    acc = _BoundExecutionReplayAccumulator(w, 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec(), False)
    rng = np.random.default_rng(7)
    # 앞쪽 두 체결을 확실히 floor(700) 이하로 둬 커서 전진을 결정론적으로 검증한다
    acc.fill_bar_ns = [50, 60] + [int(x) for x in rng.integers(0, 100, size=58) * 10]
    acc.fill_gcol = [int(x) for x in rng.integers(0, 2, size=60)]
    acc.fill_qty = [float(x) for x in rng.normal(size=60)]
    grid_ns = np.arange(0, 1000, 10, dtype='int64')

    def brute(last_ns, target_ns, lo, gpos):
        floor = -1 if last_ns is None else int(last_ns)
        gl = gpos.tolist()
        out = []
        for fns, gcol, fqty in zip(acc.fill_bar_ns, acc.fill_gcol, acc.fill_qty, strict=True):
            if fns <= floor or fns > target_ns:
                continue
            rel = int(np.searchsorted(grid_ns, fns, side='left')) - lo
            if gcol in gl:
                out.append((max(rel, 0), gl.index(gcol), float(fqty)))
        return out

    spans = [(None, 150, 0), (150, 420, 16), (420, 700, 43), (700, 990, 71)]
    for gpos in (np.array([0, 1]), np.array([1])):
        acc._span_scan_from = 0
        for last_ns, target_ns, lo in spans:
            assert acc._fills_in_span(last_ns, target_ns, grid_ns, lo, gpos) == brute(last_ns, target_ns, lo, gpos)
    # Then: every fill at or before the last floor (700) is permanently skipped
    prefix = 0
    while prefix < len(acc.fill_bar_ns) and acc.fill_bar_ns[prefix] <= 700:
        prefix += 1
    assert acc._span_scan_from == prefix




def test_idle_tail_event_snapshots_keep_held_inventory() -> None:

    import numpy as np
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow

    def _window(grid, qv_values, mark_values, decisions, weights):
        px = pd.DataFrame({'A': mark_values}, index=grid)
        return ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1], columns=('A',), symbols=('A',),
            minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
            target_weights=pd.DataFrame({'A': weights}, index=pd.DatetimeIndex(decisions)),
            signal_available_at=pd.DatetimeIndex(decisions),
            quote_volumes=pd.DataFrame({'A': qv_values}, index=grid),
            funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(hours=1),
        )

    from src.mhs.execution import ExecutionSpec
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator
    grid = pd.date_range('2025-01-01', periods=40, freq='1h', tz='UTC')
    qv = np.where(np.arange(40) <= 3, 1000.0, 0.0)
    marks = np.where(np.arange(40) <= 3, 100.0, 80.0)
    w = _window(grid, qv, marks, [grid[0], grid[30]], [0.5, 0.0])
    acc = _BoundExecutionReplayAccumulator(w, 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec(), True)
    # When
    acc.consume(w)
    # Then: no inferred settlement exists; snapshots keep the held inventory
    assert acc.termination_counts.get('DELIST_SETTLEMENT', 0) == 0
    assert 'delist_settlement' not in acc.fill_reason
    ts, units = acc.units_after_events[-1]
    assert float(units[0]) == pytest.approx(5.0)
    nts, notional = acc.notional_after_events[-1]
    assert nts == ts
    assert float(notional[0]) == pytest.approx(5.0 * 100.0)


def test_idle_settlement_skipped_when_signal_bar_is_beyond_window() -> None:

    import numpy as np
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow

    def _window(grid, qv_values, mark_values, decisions, weights):
        px = pd.DataFrame({'A': mark_values}, index=grid)
        return ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1], columns=('A',), symbols=('A',),
            minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
            target_weights=pd.DataFrame({'A': weights}, index=pd.DatetimeIndex(decisions)),
            signal_available_at=pd.DatetimeIndex(decisions),
            quote_volumes=pd.DataFrame({'A': qv_values}, index=grid),
            funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(hours=1),
        )

    from src.mhs.execution import ExecutionSpec
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator
    grid = pd.date_range('2025-01-01', periods=40, freq='1h', tz='UTC')
    qv = np.where(np.arange(40) <= 3, 1000.0, 0.0)
    # Given: the idle decision sits on the last grid bar, so its signal bar (spos) == n_grid
    w = _window(grid, qv, np.full(40, 100.0), [grid[0], grid[39]], [0.5, 0.0])
    acc = _BoundExecutionReplayAccumulator(w, 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec(), False)
    # When
    acc.consume(w)
    # Then: no settlement booked, inventory still held
    assert acc.termination_counts.get('DELIST_SETTLEMENT', 0) == 0
    assert 'delist_settlement' not in acc.fill_reason
    assert abs(float(acc.units_arr[0])) > 0.0


def test_idle_settlement_skipped_when_settle_bar_mark_is_not_finite() -> None:

    import numpy as np
    import pandas as pd
    from src.mhs.execution import ExecutionReplayWindow

    def _window(grid, qv_values, mark_values, decisions, weights):
        px = pd.DataFrame({'A': mark_values}, index=grid)
        return ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1], columns=('A',), symbols=('A',),
            minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
            target_weights=pd.DataFrame({'A': weights}, index=pd.DatetimeIndex(decisions)),
            signal_available_at=pd.DatetimeIndex(decisions),
            quote_volumes=pd.DataFrame({'A': qv_values}, index=grid),
            funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(hours=1),
        )

    from src.mhs.execution import ExecutionSpec
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator
    grid = pd.date_range('2025-01-01', periods=40, freq='1h', tz='UTC')
    qv = np.where(np.arange(40) <= 3, 1000.0, 0.0)
    marks = np.full(40, 100.0)
    marks[31] = np.nan  # the would-be settlement bar has no usable mark
    import dataclasses
    w = _window(grid, qv, marks, [grid[0], grid[30]], [0.5, 0.0])
    # bar_funding must stay finite (fail-closed validation); only the mark is missing
    w = dataclasses.replace(w, bar_funding=w.bar_funding.fillna(0.0))
    acc = _BoundExecutionReplayAccumulator(w, 1000.0, 'OHLCV_IMMEDIATE_TAKER', ExecutionSpec(), False)
    # When
    acc.consume(w)
    # Then: never settles at an invented price
    assert acc.termination_counts.get('DELIST_SETTLEMENT', 0) == 0
    assert 'delist_settlement' not in acc.fill_reason
    assert abs(float(acc.units_arr[0])) > 0.0




def test_bounded_ledger_held_unknown_funding_stays_visible() -> None:
    """Held inventory over unknown nonzero funding invalidates without charging."""
    grid = pd.date_range("2025-01-01", periods=6, freq="3min", tz="UTC")
    px = pd.DataFrame({"A": 100.0}, index=grid)
    funding = pd.DataFrame({"A": 0.0001}, index=grid)
    known = pd.DataFrame({"A": [True, True, True, False, False, False]}, index=grid)
    window = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=("A",), symbols=("A",),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=funding,
        target_weights=pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([grid[0]])),
        signal_available_at=pd.DatetimeIndex([grid[0]]),
        quote_volumes=pd.DataFrame({"A": 1000.0}, index=grid),
        funding_known=known, bar_available_at=grid + pd.Timedelta(minutes=3),
    )
    acc = _BoundExecutionReplayAccumulator(window, 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False)
    acc.consume(window)
    result = acc.finalize()
    assert acc.first_held_funding is not None
    assert acc.first_held_funding[0] == "A"
    assert not result.ledger.primary_valid
    assert any(g.code == "MISSING_HELD_FUNDING" for g in result.ledger.data_gaps)
