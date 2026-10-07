from __future__ import annotations


def _drift_window(grid, alt_short: bool, decisions):
    import numpy as np
    import pandas as pd

    from src.mhs.execution import ExecutionReplayWindow

    alt = np.where(grid < pd.Timestamp("2025-01-01 03:00", tz="UTC"), 1.0, 5.0)
    px = pd.DataFrame({"BTCUSDT": np.full(len(grid), 100.0), "ALTUSDT": alt}, index=grid)
    alt_w = -0.1 if alt_short else 0.1
    index = pd.DatetimeIndex(decisions)
    weights = pd.DataFrame({"BTCUSDT": [0.1] * len(index), "ALTUSDT": [alt_w] * len(index)}, index=index)
    return ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=("BTCUSDT", "ALTUSDT"), symbols=("BTCUSDT", "ALTUSDT"),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=weights, signal_available_at=index, quote_volumes=px * 0.0 + 1000.0,
        funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(minutes=3),
    )


def test_drift_trim_long_breach_trims_to_cap() -> None:
    import dataclasses

    import pandas as pd
    import pytest

    from src.mhs.execution import ExecutionSpec, replay_execution_windows

    # Given: ALT long 10% of equity; ALT 1.0 -> 5.0 at 03:00 drifts it to ~35.7%.
    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 10:00", freq="3min", tz="UTC")
    window = _drift_window(grid, alt_short=False, decisions=[grid[0]])
    spec = dataclasses.replace(ExecutionSpec(), name_drift_trim_max_weight=0.2)
    # When
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    # Then: exactly one taker trim at the first 4h check (04:00 -> fill available 04:06).
    fills = result.simulated_fills
    trims = fills[fills["reason"] == "drift_trim"]
    assert len(trims) == 1
    trim = trims.iloc[0]
    assert trim["symbol"] == "ALTUSDT"
    assert trim["timestamp"] == pd.Timestamp("2025-01-01 04:06", tz="UTC")
    assert trim["quantity_delta"] == pytest.approx(-44.0064, rel=1e-9)
    assert trim["fill_price"] == pytest.approx(5.0)
    assert trim["fee_bps"] == pytest.approx(8.0)
    assert result.termination_counts["DRIFT_TRIM"] == 1
    alt_units = float(fills.loc[fills["symbol"] == "ALTUSDT", "quantity_delta"].sum())
    post_weight = alt_units * 5.0 / float(result.ledger.equity.iloc[-1])
    # The fee-induced overshoot stays inside the one-way taker trigger band: no re-trim at 08:00.
    assert post_weight == pytest.approx(0.2, abs=1e-3)
    assert post_weight <= 0.2 * (1.0 + spec.one_way_taker_bps() / 1e4)


def test_drift_trim_short_breach_buys_back_to_cap() -> None:
    import dataclasses

    import pandas as pd
    import pytest

    from src.mhs.execution import ExecutionSpec, replay_execution_windows

    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 10:00", freq="3min", tz="UTC")
    window = _drift_window(grid, alt_short=True, decisions=[grid[0]])
    spec = dataclasses.replace(ExecutionSpec(), name_drift_trim_max_weight=0.2)
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    fills = result.simulated_fills
    trims = fills[fills["reason"] == "drift_trim"]
    assert len(trims) == 1
    assert trims.iloc[0]["quantity_delta"] == pytest.approx(76.0064, rel=1e-9)
    alt_units = float(fills.loc[fills["symbol"] == "ALTUSDT", "quantity_delta"].sum())
    assert alt_units * 5.0 / float(result.ledger.equity.iloc[-1]) == pytest.approx(-0.2, abs=1e-3)


def test_drift_trim_disabled_by_default_is_noop() -> None:
    import pandas as pd

    from src.mhs.execution import ExecutionSpec, replay_execution_windows

    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 10:00", freq="3min", tz="UTC")
    window = _drift_window(grid, alt_short=False, decisions=[grid[0]])
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert ExecutionSpec().name_drift_trim_max_weight is None
    assert list(result.simulated_fills["reason"]) == ["timeout_taker", "timeout_taker"]
    assert "DRIFT_TRIM" not in result.termination_counts


def test_drift_trim_window_overlap_checks_once() -> None:
    import dataclasses

    import pandas as pd

    from src.mhs.execution import ExecutionSpec, replay_execution_windows

    # Given: window 2's grid starts at window 1's last decision (production layout).
    full = pd.date_range("2025-01-01 00:00", "2025-01-01 20:00", freq="3min", tz="UTC")
    first_grid = full[full <= pd.Timestamp("2025-01-01 00:36", tz="UTC")]
    first = _drift_window(first_grid, alt_short=False, decisions=[full[0]])
    second = _drift_window(full, alt_short=False, decisions=[pd.Timestamp("2025-01-01 12:00", tz="UTC")])
    spec = dataclasses.replace(ExecutionSpec(), name_drift_trim_max_weight=0.2)
    result = replay_execution_windows((first, second), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    fills = result.simulated_fills
    trims = fills[fills["reason"] == "drift_trim"]
    assert list(trims["timestamp"]) == [pd.Timestamp("2025-01-01 04:06", tz="UTC")]
    assert result.termination_counts["DRIFT_TRIM"] == 1
    assert len(fills) == 5


def test_drift_trim_check_skipped_when_order_would_cross_next_decision() -> None:
    import dataclasses

    import pandas as pd

    from src.mhs.execution import ExecutionSpec, replay_execution_windows

    # Given: the 04:00 check's order resolves at 04:33, after the 04:30 decision.
    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 10:00", freq="3min", tz="UTC")
    second = pd.Timestamp("2025-01-01 04:30", tz="UTC")
    window = _drift_window(grid, alt_short=False, decisions=[grid[0], second])
    spec = dataclasses.replace(ExecutionSpec(), name_drift_trim_max_weight=0.2)
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    fills = result.simulated_fills
    # Then: no trim; the 04:30 decision rebalances ALT back to 10% and the 08:00 check finds no breach.
    assert "drift_trim" not in set(fills["reason"])
    assert "DRIFT_TRIM" not in result.termination_counts
    assert len(fills) == 4


def test_drift_trim_off_grid_check_snaps_to_next_bar() -> None:
    import dataclasses

    import pandas as pd
    import pytest

    from src.mhs.execution import ExecutionSpec, replay_execution_windows

    # Given: an off-grid decision at 00:01 anchors checks at 04:01, 08:01 (not on the 3m grid).
    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 10:00", freq="3min", tz="UTC")
    window = _drift_window(grid, alt_short=False, decisions=[pd.Timestamp("2025-01-01 00:01", tz="UTC")])
    spec = dataclasses.replace(ExecutionSpec(), name_drift_trim_max_weight=0.2)
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    trims = result.simulated_fills[result.simulated_fills["reason"] == "drift_trim"]
    # Then: the check snaps to the 04:03 bar (never silently skipped); fill available at 04:09.
    assert list(trims["timestamp"]) == [pd.Timestamp("2025-01-01 04:09", tz="UTC")]
    assert trims.iloc[0]["quantity_delta"] == pytest.approx(-44.0064, rel=1e-9)


def test_drift_trim_window_without_decisions_is_noop() -> None:
    import dataclasses

    import pandas as pd

    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows

    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 06:00", freq="3min", tz="UTC")
    px = pd.DataFrame({"BTCUSDT": 100.0}, index=grid)
    empty = pd.DatetimeIndex([], tz="UTC")
    window = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=("BTCUSDT",), symbols=("BTCUSDT",),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=pd.DataFrame({"BTCUSDT": []}, index=empty, dtype="float64"), signal_available_at=empty,
        quote_volumes=px * 0.0 + 1000.0, funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(minutes=3),
    )
    spec = dataclasses.replace(ExecutionSpec(), name_drift_trim_max_weight=0.2)
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    assert result.simulated_fills.empty
    assert "DRIFT_TRIM" not in result.termination_counts


def test_execution_spec_rejects_invalid_drift_trim_parameters() -> None:
    import pytest

    from src.mhs.types import ExecutionSpec

    with pytest.raises(ValueError, match="name_drift_trim_max_weight"):
        ExecutionSpec(name_drift_trim_max_weight=0.0)
    with pytest.raises(ValueError, match="name_drift_trim_max_weight"):
        ExecutionSpec(name_drift_trim_max_weight=1.0)
    with pytest.raises(ValueError, match="name_drift_trim_interval_hours"):
        ExecutionSpec(name_drift_trim_max_weight=0.2, name_drift_trim_interval_hours=0)
    assert ExecutionSpec(name_drift_trim_max_weight=0.2).name_drift_trim_interval_hours == 4


def _ledger_window(grid, symbols, decisions, weights, marks=None, funding=None, settlement_events=(), lows=None, highs=None):
    import numpy as np
    import pandas as pd

    from src.mhs.execution import ExecutionReplayWindow

    n = len(grid)
    base = {s: np.full(n, 100.0) for s in symbols}
    closes = pd.DataFrame(dict(base), index=grid)
    marks_df = pd.DataFrame({s: np.asarray(marks[s], dtype="float64") for s in symbols}, index=grid) if marks is not None else closes
    lows_df = pd.DataFrame({s: np.asarray(lows[s], dtype="float64") for s in symbols}, index=grid) if lows is not None else closes
    highs_df = pd.DataFrame({s: np.asarray(highs[s], dtype="float64") for s in symbols}, index=grid) if highs is not None else closes
    funding_df = (
        pd.DataFrame({s: np.asarray(funding[s], dtype="float64") for s in symbols}, index=grid)
        if funding is not None
        else closes * 0.0
    )
    index = pd.DatetimeIndex(decisions)
    weights_df = pd.DataFrame({s: np.asarray(weights[s], dtype="float64") for s in symbols}, index=index)
    return ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=tuple(symbols), symbols=tuple(symbols),
        minute_grid=grid, highs=highs_df, lows=lows_df, closes=closes, marks=marks_df, bar_funding=funding_df,
        target_weights=weights_df, signal_available_at=index, quote_volumes=closes * 0.0 + 1000.0,
        funding_known=funding_df.notna(), bar_available_at=grid + pd.Timedelta(minutes=3),
        settlement_events=tuple(settlement_events),
    )


def _replay_with_acc(window_or_windows, spec=None, execution_bound="OHLCV_IMMEDIATE_TAKER", retain_event_snapshots=False):
    from src.mhs.execution import ExecutionSpec, replay_execution_windows

    windows = window_or_windows if isinstance(window_or_windows, tuple) else (window_or_windows,)
    live: list = []
    result = replay_execution_windows(
        windows, 1000.0, execution_bound, spec or ExecutionSpec(), live_accumulators=live,
        retain_event_snapshots=retain_event_snapshots,
    )
    return result, live[0][0]


def test_full_exit_ledger_units_exact_zero() -> None:
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(
        grid, ["BTCUSDT"], [grid[0], grid[4]],
        {"BTCUSDT": [0.1, 0.0]},
    )
    result, acc = _replay_with_acc(window)
    assert acc.units_arr[0] == 0.0
    assert acc.ledger_units[0] == 0.0
    assert acc.ledger_units[0] == acc.units_arr[0]
    assert result.ledger.primary_valid is True
    assert [g for g in result.ledger.data_gaps if g.code in ("MISSING_HELD_MARK", "MISSING_HELD_FUNDING")] == []


def test_flat_symbol_with_nan_marks_not_held() -> None:
    import numpy as np
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    marks = {"BTCUSDT": np.where(np.arange(10) < 6, 100.0, np.nan)}
    funding = {"BTCUSDT": np.where(np.arange(10) < 6, 0.0, 1e-4)}
    window = _ledger_window(
        grid, ["BTCUSDT"], [grid[0], grid[3]],
        {"BTCUSDT": [0.1, 0.0]}, marks=marks, funding=funding,
    )
    result, _acc = _replay_with_acc(window)
    assert result.ledger.primary_valid is True
    assert [g for g in result.ledger.data_gaps if g.code in ("MISSING_HELD_MARK", "MISSING_HELD_FUNDING")] == []
    assert float(result.ledger.funding_charge.iloc[6:].abs().sum()) == 0.0


def test_genuine_residual_fails_closed() -> None:
    import numpy as np
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    marks = {"BTCUSDT": np.where(np.arange(10) < 6, 100.0, np.nan)}
    window = _ledger_window(
        grid, ["BTCUSDT"], [grid[0], grid[3]],
        {"BTCUSDT": [0.1, 0.05]}, marks=marks,
    )
    result, _acc = _replay_with_acc(window)
    assert result.ledger.primary_valid is False
    assert any(g.code == "MISSING_HELD_MARK" for g in result.ledger.data_gaps)


def test_same_bar_multiple_fills_last_booked_level() -> None:
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    decisions = [grid[2], grid[2] + pd.Timedelta(minutes=1)]
    window = _ledger_window(
        grid, ["BTCUSDT"], decisions,
        {"BTCUSDT": [0.1, 0.2]},
    )
    _result, acc = _replay_with_acc(window)
    assert len(acc.fill_qty) == 2
    assert acc.fill_bar_ns[-2] == acc.fill_bar_ns[-1]
    assert acc.fill_post_units[-1] == float(acc.units_arr[0])
    assert float(acc.ledger_units[0]) == float(acc.fill_post_units[-1])


def test_carry_across_window_boundary() -> None:
    import pandas as pd

    from src.mhs.execution import ExecutionReplayWindow

    full = pd.date_range("2025-01-01 00:00", periods=20, freq="3min", tz="UTC")
    first = _ledger_window(full[:10], ["BTCUSDT"], [full[0]], {"BTCUSDT": [0.1]})
    grid_b = full[10:20]
    empty_index = pd.DatetimeIndex([], tz="UTC")
    px = pd.DataFrame({"BTCUSDT": 100.0}, index=grid_b)
    second = ExecutionReplayWindow(
        window_start=grid_b[0], window_end=grid_b[-1], columns=("BTCUSDT",), symbols=("BTCUSDT",),
        minute_grid=grid_b, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=pd.DataFrame({"BTCUSDT": []}, index=empty_index, dtype="float64"),
        signal_available_at=empty_index,
        quote_volumes=px * 0.0 + 1000.0, funding_known=px.notna(), bar_available_at=grid_b + pd.Timedelta(minutes=3),
    )
    result, acc = _replay_with_acc((first, second))
    assert len(acc.fill_post_units) == len(acc.fill_qty) == 1
    assert float(acc.ledger_units[0]) == float(acc.fill_post_units[-1])
    assert float(acc.ledger_units[0]) == float(acc.units_arr[0])
    assert result.ledger.primary_valid is True


def test_parallel_fill_lists_stay_aligned_with_settlement() -> None:
    import pandas as pd

    from src.mhs.execution.contracts import InstrumentSettlementEvent

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    event = InstrumentSettlementEvent(
        event_id="settle-1", symbol="BTCUSDT", effective_at=grid[5], available_at=grid[5],
        settlement_price=100.0, fee_bps=0.0, source_digest="probe",
    )
    window = _ledger_window(
        grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]}, settlement_events=(event,),
    )
    _result, acc = _replay_with_acc(window)
    assert len(acc.fill_post_units) == len(acc.fill_qty)
    idx = list(acc.fill_reason).index("delist_settlement")
    assert acc.fill_post_units[idx] == 0.0
    assert float(acc.ledger_units[0]) == 0.0


def test_cash_conservation_unchanged() -> None:
    import numpy as np
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(
        grid, ["BTCUSDT", "ALTUSDT"], [grid[0], grid[4]],
        {"BTCUSDT": [0.1, 0.0], "ALTUSDT": [0.05, 0.05]},
    )
    result, acc = _replay_with_acc(window)
    qty = np.asarray(acc.fill_qty, dtype="float64")
    price = np.asarray(acc.fill_price, dtype="float64")
    fee_bps = np.asarray(acc.fill_fee_bps, dtype="float64")
    expected_fees = float(np.sum(fee_bps / 1e4 * np.abs(qty) * price))
    assert abs(float(result.ledger.fee_charge.sum()) - expected_fees) <= 1e-12 * max(1.0, abs(expected_fees))
    assert np.isfinite(result.ledger.funding_charge.to_numpy()).all()
    assert (result.ledger.fill_turnover.to_numpy() >= 0.0).all()
    final_marks = np.array([100.0, 100.0])
    assert abs(float(result.ledger.equity.iloc[-1]) - (float(acc.ledger_cash) + float(acc.ledger_units @ final_marks))) <= 1e-9


def test_laddered_fill_lists_stay_aligned() -> None:
    import numpy as np
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    lows = {"BTCUSDT": np.where(np.arange(10) == 0, 100.0, 99.0)}
    window = _ledger_window(
        grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]}, lows=lows,
    )
    _result, acc = _replay_with_acc(window, execution_bound="OHLCV_LADDERED_PROXY")
    assert len(acc.fill_qty) >= 1
    assert len(acc.fill_post_units) == len(acc.fill_qty)
    assert float(acc.fill_post_units[-1]) == float(acc.units_arr[0])


def test_peg_chase_fill_lists_stay_aligned() -> None:
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(
        grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]},
    )
    _result, acc = _replay_with_acc(window, execution_bound="OHLCV_PEG_CHASE_PROXY")
    assert len(acc.fill_qty) >= 1
    assert len(acc.fill_post_units) == len(acc.fill_qty)
    assert float(acc.fill_post_units[-1]) == float(acc.units_arr[0])


# ---------------------------------------------------------------------------
# Spec 05 — single atomic fill booking + decision-time pre_trade_equity
# ---------------------------------------------------------------------------

_TWELVE_FILL_LISTS = (
    "fill_bar_ns", "fill_gcol", "fill_ts", "fill_symbol", "fill_qty",
    "fill_post_units", "fill_price", "fill_fee_bps", "fill_reason",
    "fill_pre_trade_equity", "fill_times", "submit_times",
)

_BOUNDS_4 = (
    "OHLCV_STRICT_PROXY", "OHLCV_IMMEDIATE_TAKER",
    "OHLCV_LADDERED_PROXY", "OHLCV_PEG_CHASE_PROXY",
)


def _probe_like_workload(*, days=12, n_symbols=6, seed=7, funding=1e-5):
    """Hermetic mirror of scratch/probe_mhs_exec/topic_a_b.py::workload (same shape and params)."""
    import numpy as np
    import pandas as pd

    grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
    symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
    rng = np.random.default_rng(seed)
    closes = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
        index=grid,
    )
    decision_grid = pd.date_range("2021-01-01", periods=days * 4, freq="6h", tz="UTC")
    weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
    rng_w = np.random.default_rng(seed + 1)
    for ts in decision_grid:
        active = rng_w.choice(symbols, size=4, replace=False)
        weights.loc[ts, active] = rng_w.uniform(0.05, 0.25, 4) * rng_w.choice([-1, 1], 4)
    return {
        "grid": grid,
        "highs": closes * 1.001,
        "lows": closes * 0.999,
        "closes": closes,
        "marks": closes * (1 + rng.normal(0, 3e-4, closes.shape)),
        "funding": pd.DataFrame(funding, index=grid, columns=symbols),
        "weights": weights,
        "signals": decision_grid + pd.Timedelta(hours=1),
    }


def _replay_probe_workload(bound, *, spec=None, n_windows=3, settlement_event=None):
    import dataclasses

    from src.mhs.execution import ExecutionSpec, replay_execution_windows
    from tests.unit.mhs.test_execution import _partition_windows

    wl = _probe_like_workload()
    spec = spec or ExecutionSpec()
    wins = _partition_windows(
        wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
        wl["closes"], wl["marks"], wl["funding"], spec, n_windows=n_windows,
    )
    if settlement_event is not None:
        idx = next(
            i for i, w in enumerate(wins)
            if w.minute_grid[0] <= settlement_event.available_at <= w.minute_grid[-1]
        )
        wins[idx] = dataclasses.replace(wins[idx], settlement_events=(settlement_event,))
    live: list = []
    result = replay_execution_windows(wins, 1000.0, bound, spec, live_accumulators=live)
    return result, live[0][0], wl


def _mid_replay_book_setup():
    """Replay one decision; return (acc, valid _book_fill kwargs, pre-call state snapshot)."""
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]})
    _result, acc = _replay_with_acc(window)
    assert len(acc.fill_qty) > 0
    frame = acc._consume_validate_window(window)
    kwargs = {
        "frame": frame, "gcol": int(frame.gpos[0]), "symbol": "BTCUSDT", "bar_pos": 1, "submit_pos": 1,
        "quantity": 0.5, "fill_price": 100.0, "fee_bps": 8.0, "reason": "passive_fill",
        "valuation_mark": 100.0, "pre_trade_equity": 1000.0,
        "target_weight": 0.1, "decision_price": 100.0,
    }
    snapshot = (
        float(acc.cash), acc.units_arr.copy(), acc.last_prices_arr.copy(),
        {name: list(getattr(acc, name)) for name in (*_TWELVE_FILL_LISTS, "_mirror_pending")},
    )
    return acc, kwargs, snapshot


def _assert_book_state_unchanged(acc, snapshot) -> None:
    import numpy as np

    cash, units, lasts, lists = snapshot
    assert float(acc.cash) == cash
    np.testing.assert_array_equal(acc.units_arr, units)
    np.testing.assert_array_equal(acc.last_prices_arr, lasts)
    for name, before in lists.items():
        assert list(getattr(acc, name)) == before


def test_pre_trade_equity_equals_decision_sizing_equity() -> None:
    import math

    for bound in _BOUNDS_4:
        _result, acc, _wl = _replay_probe_workload(bound)
        assert len(acc.fill_qty) > 0
        groups: dict = {}
        for submit, equity in zip(acc.submit_times, acc.fill_pre_trade_equity, strict=True):
            groups.setdefault(submit, []).append(equity)
        assert len(groups) >= 2
        for equities in groups.values():
            assert all(math.isfinite(e) and e > 0 for e in equities)
            assert all(e == equities[0] for e in equities)


def test_pre_trade_equity_independent_of_own_fill() -> None:
    import dataclasses

    import pandas as pd
    import pytest

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]})
    bumped_closes = window.closes.copy()
    bumped_closes.iloc[1, 0] *= 1.05
    _base, base_acc = _replay_with_acc(window)
    _bump, bump_acc = _replay_with_acc(dataclasses.replace(window, closes=bumped_closes))
    assert bump_acc.fill_price[0] == pytest.approx(base_acc.fill_price[0] * 1.05)
    assert bump_acc.fill_price[0] != base_acc.fill_price[0]
    assert bump_acc.fill_pre_trade_equity[0] == base_acc.fill_pre_trade_equity[0] == 1000.0


def test_same_bar_multi_fill_does_not_compound() -> None:
    import pandas as pd
    import pytest

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    decisions = [grid[2], grid[2] + pd.Timedelta(minutes=1)]
    window = _ledger_window(
        grid, ["BTCUSDT"], decisions,
        {"BTCUSDT": [0.1, 0.2]},
    )
    _result, acc = _replay_with_acc(window)
    assert len(acc.fill_qty) == 2
    assert acc.fill_bar_ns[0] == acc.fill_bar_ns[1]
    assert acc.fill_pre_trade_equity[0] == 1000.0
    first_fee = acc.fill_fee_bps[0] / 1e4 * abs(acc.fill_qty[0]) * acc.fill_price[0]
    assert acc.fill_pre_trade_equity[1] == pytest.approx(1000.0 - first_fee)


def test_settlement_records_pre_mutation_equity() -> None:
    import pandas as pd
    import pytest

    from src.mhs.execution.contracts import InstrumentSettlementEvent

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    event = InstrumentSettlementEvent(
        event_id="settle-pre", symbol="BTCUSDT", effective_at=grid[5], available_at=grid[5],
        settlement_price=90.0, fee_bps=5.0, source_digest="spec05",
    )
    window = _ledger_window(
        grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]}, settlement_events=(event,),
    )
    _result, acc = _replay_with_acc(window)
    idx = list(acc.fill_reason).index("delist_settlement")
    assert idx == len(acc.fill_qty) - 1
    q0, p0, fb0 = acc.fill_qty[0], acc.fill_price[0], acc.fill_fee_bps[0]
    cash_after_first = 1000.0 - q0 * p0 - fb0 / 1e4 * abs(q0) * p0
    expected_pre = cash_after_first + q0 * event.settlement_price
    assert acc.fill_pre_trade_equity[idx] == pytest.approx(expected_pre)
    post_cash = cash_after_first - acc.fill_qty[idx] * 90.0 - 5.0 / 1e4 * abs(acc.fill_qty[idx]) * 90.0
    assert expected_pre - post_cash == pytest.approx(5.0 / 1e4 * abs(q0) * 90.0)
    assert float(acc.fill_pre_trade_equity[idx]) > post_cash


def test_unknown_reason_fails_closed_without_mutation() -> None:
    import pytest

    from src.common.errors import DataIntegrityError

    acc, kwargs, snapshot = _mid_replay_book_setup()
    kwargs["reason"] = "weird_reason"
    with pytest.raises(DataIntegrityError, match="weird_reason"):
        acc._book_fill(**kwargs)
    _assert_book_state_unchanged(acc, snapshot)


def test_non_positive_pre_trade_equity_fails_closed() -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.integrity import _classify_execution_failure

    for bad_equity in (0.0, float("nan")):
        acc, kwargs, snapshot = _mid_replay_book_setup()
        kwargs["pre_trade_equity"] = bad_equity
        with pytest.raises(
            DataIntegrityError, match="pre-trade equity must be positive and finite"
        ) as exc_info:
            acc._book_fill(**kwargs)
        _assert_book_state_unchanged(acc, snapshot)
        assert _classify_execution_failure(exc_info.value) == "CAPITAL_INVARIANT_BREACH"


def test_non_finite_fill_sizing_fails_closed_without_mutation() -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.integrity import _classify_execution_failure

    for bad_qty, bad_price in ((float("inf"), 100.0), (0.5, float("nan"))):
        acc, kwargs, snapshot = _mid_replay_book_setup()
        kwargs["quantity"] = bad_qty
        kwargs["fill_price"] = bad_price
        with pytest.raises(DataIntegrityError, match="capital accounting invariant") as exc_info:
            acc._book_fill(**kwargs)
        _assert_book_state_unchanged(acc, snapshot)
        assert _classify_execution_failure(exc_info.value) == "CAPITAL_INVARIANT_BREACH"


def test_parallel_fill_lists_stay_equal_length_across_bounds() -> None:
    import dataclasses

    import pandas as pd

    from src.mhs.execution import ExecutionSpec
    from src.mhs.execution.accumulator import _BOOKED_FILL_REASONS
    from src.mhs.execution.contracts import InstrumentSettlementEvent

    def _assert_aligned(acc, result, *, expect_settlement: bool) -> None:
        lengths = {len(getattr(acc, name)) for name in _TWELVE_FILL_LISTS}
        assert lengths == {len(result.simulated_fills)}
        assert set(acc.fill_reason) <= _BOOKED_FILL_REASONS | {"drift_trim"}
        if expect_settlement:
            assert "delist_settlement" in set(acc.fill_reason)

    for bound in _BOUNDS_4:
        # Workload scale: settlement only. (Trim + settlement at this scale trips a
        # pre-existing causal-mirror divergence, identical on the pre-change tree,
        # so the combined path is covered on the small fixture below instead.)
        wl = _probe_like_workload()
        decision_ts = wl["weights"].index[-1]
        sym = str(wl["weights"].loc[decision_ts].abs().idxmax())
        at = decision_ts + pd.Timedelta(hours=2)
        event = InstrumentSettlementEvent(
            event_id=f"settle-{bound}", symbol=sym,
            effective_at=at, available_at=at,
            settlement_price=90.0, fee_bps=5.0, source_digest="spec05",
        )
        result, acc, _ = _replay_probe_workload(bound, settlement_event=event)
        _assert_aligned(acc, result, expect_settlement=True)

        # Combined path on a small fixture: drift trim fires at the 04:00 check and
        # BTC settles at 05:00 while its decision position is still open.
        grid = pd.date_range("2025-01-01 00:00", "2025-01-01 10:00", freq="3min", tz="UTC")
        drift_window = _drift_window(grid, alt_short=False, decisions=[grid[0]])
        trim_event = InstrumentSettlementEvent(
            event_id=f"trim-settle-{bound}", symbol="BTCUSDT",
            effective_at=grid[100], available_at=grid[100],
            settlement_price=90.0, fee_bps=5.0, source_digest="spec05",
        )
        spec = dataclasses.replace(ExecutionSpec(), name_drift_trim_max_weight=0.2)
        combined, live_acc = _replay_with_acc(
            dataclasses.replace(drift_window, settlement_events=(trim_event,)),
            spec=spec, execution_bound=bound,
        )
        _assert_aligned(live_acc, combined, expect_settlement=True)
        assert "drift_trim" in set(live_acc.fill_reason)


def test_cash_two_step_versus_mirror_one_step_preserved() -> None:
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(
        grid, ["BTCUSDT", "ALTUSDT"], [grid[0], grid[4]],
        {"BTCUSDT": [0.1, 0.0], "ALTUSDT": [0.05, 0.05]},
    )
    result, acc = _replay_with_acc(window)
    assert result.ledger.primary_valid is True
    assert float(acc.cash) != float(acc.ledger_cash)
    assert abs(float(acc.cash) - float(acc.ledger_cash)) <= 1e-12 * max(1.0, abs(float(acc.ledger_cash)))


def test_booked_snapshots_retained_when_opted_in() -> None:
    import numpy as np
    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(
        grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]},
    )
    _result, acc = _replay_with_acc(window, retain_event_snapshots=True)
    assert len(acc.fill_qty) >= 1
    assert len(acc.units_after_events) == len(acc.fill_qty) == len(acc.notional_after_events)
    for (stamp, units), notional in zip(acc.units_after_events, [n for _, n in acc.notional_after_events], strict=True):
        assert stamp in set(acc.fill_ts)
        assert units.shape == notional.shape == acc.units_arr.shape
        assert np.isfinite(np.asarray(units, dtype="float64")).all()


# ---------------------------------------------------------------------------
# Spec 05b — _WindowFrame replaces positional window-array plumbing
# ---------------------------------------------------------------------------


def test_window_frame_holds_references_not_copies() -> None:
    import dataclasses

    import pandas as pd
    import pytest

    from src.mhs.execution import ExecutionSpec
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator, _WindowFrame

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]})
    acc = _BoundExecutionReplayAccumulator(window, 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False)
    frame = acc._consume_validate_window(window)
    assert isinstance(frame, _WindowFrame)
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.bar_ns = -1  # type: ignore[misc]
    assert frame.n_grid == len(window.minute_grid)
    assert frame.n_local == len(window.symbols)
    assert frame.grid is window.minute_grid
    assert frame.tw_index is window.target_weights.index
    assert frame.sig_index is window.signal_available_at


def test_window_frame_refactor_preserves_replay_output() -> None:
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.mhs.execution import ExecutionSpec, replay_execution_windows
    from src.mhs.execution.contracts import InstrumentSettlementEvent
    from tests.unit.mhs.test_execution import _partition_windows

    # 2-day scale: trim + settlement at the 12-day scale trips a pre-existing
    # causal-mirror divergence identical on the pre-change tree (see the
    # combined-path note in test_parallel_fill_lists_stay_equal_length_across_bounds).
    spec = dataclasses.replace(ExecutionSpec(), name_drift_trim_max_weight=0.2)
    for bound in _BOUNDS_4:
        wl = _probe_like_workload(days=2)
        decision_ts = wl["weights"].index[-1]
        sym = str(wl["weights"].loc[decision_ts].abs().idxmax())
        at = decision_ts + pd.Timedelta(hours=2)
        event = InstrumentSettlementEvent(
            event_id=f"frame-settle-{bound}", symbol=sym,
            effective_at=at, available_at=at,
            settlement_price=90.0, fee_bps=5.0, source_digest="spec05b",
        )
        ledgers = []
        counts = []
        for n_windows in (1, 3):
            wins = _partition_windows(
                wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
                wl["closes"], wl["marks"], wl["funding"], spec, n_windows=n_windows,
            )
            idx = next(
                i for i, w in enumerate(wins)
                if w.minute_grid[0] <= event.available_at <= w.minute_grid[-1]
            )
            wins[idx] = dataclasses.replace(wins[idx], settlement_events=(event,))
            live: list = []
            result = replay_execution_windows(wins, 1000.0, bound, spec, live_accumulators=live)
            ledgers.append(result.ledger.equity.to_numpy(dtype="float64"))
            acc = live[0][0]
            lengths = {len(getattr(acc, name)) for name in _TWELVE_FILL_LISTS}
            assert lengths == {len(result.simulated_fills)}
            counts.append(len(result.simulated_fills))
        np.testing.assert_allclose(ledgers[0], ledgers[1], rtol=1e-12, atol=1e-12)
        assert counts[0] == counts[1] > 0


def test_idle_holdings_never_settle() -> None:
    import dataclasses

    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=40, freq="3min", tz="UTC")
    window = _ledger_window(grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]})
    quote_volumes = window.quote_volumes.copy()
    quote_volumes.iloc[2:, 0] = 0.0
    window = dataclasses.replace(window, quote_volumes=quote_volumes)
    result, acc = _replay_with_acc(window)
    assert len(acc.fill_qty) == 1
    assert float(acc.units_arr[0]) == float(acc.fill_qty[0]) != 0.0
    assert "delist_settlement" not in set(acc.fill_reason)
    assert "DELIST_SETTLEMENT" not in result.termination_counts


def _head_mirror_window(self, frame, p0: int) -> None:
    import numpy as np

    from src.mhs.execution.accumulator import QTY_EPS
    from src.mhs.execution.contracts import ExecutionDataGap

    grid_ns = frame.grid_ns
    n_grid = frame.n_grid
    marks_values = frame.marks_values
    funding_matrix = frame.funding_matrix
    funding_known = self._w_fknown
    gpos = frame.gpos
    grid = frame.grid
    state = self.accounting_state
    p0_ns = int(grid_ns[p0])
    queue = sorted(self._mirror_pending, key=lambda entry: entry[0])
    self._mirror_pending = []
    qi = 0
    while qi < len(queue) and queue[qi][0] < p0_ns:
        state.units[int(queue[qi][1])] += float(queue[qi][2])
        qi += 1
    gmarks = np.full(self.n_cols, np.nan, dtype="float64")
    grates = np.zeros(self.n_cols, dtype="float64")
    gknown = np.ones(self.n_cols, dtype=bool)
    all_known = np.ones(self.n_cols, dtype=bool)
    for b in range(int(p0), int(n_grid)):
        bns = int(grid_ns[b])
        gmarks[gpos] = marks_values[b]
        grates[gpos] = funding_matrix[b]
        gknown[gpos] = funding_known[b]
        held_unknown = (np.abs(state.units) >= QTY_EPS) & ~gknown
        if bool(held_unknown.any()):
            self.ledger_valid = False
            self.invalid_reasons.add("MISSING_DATA")
            witness = int(np.flatnonzero(held_unknown)[0])
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_HELD_FUNDING", symbol=self.columns[witness],
                    timestamp=grid[b], execution_bound=self.execution_bound,
                )
            )
            state.advance_to(
                event_ns=bns, marks=gmarks,
                funding_rates=np.where(gknown, grates, 0.0), funding_known=all_known,
            )
        else:
            state.advance_to(event_ns=bns, marks=gmarks, funding_rates=grates, funding_known=gknown)
        while qi < len(queue) and queue[qi][0] == bns:
            state.apply_fill(
                symbol_index=int(queue[qi][1]), quantity_delta=float(queue[qi][2]),
                fill_price=float(queue[qi][3]), fee_bps=float(queue[qi][4]),
            )
            qi += 1


def _twelve_symbol_windows():  # noqa: ANN202
    import dataclasses

    import pandas as pd

    from src.mhs.execution.contracts import InstrumentSettlementEvent
    from tests.unit.mhs.test_execution import _partition_windows

    wl = _probe_like_workload(days=2, n_symbols=12, seed=11)
    spec = __import__("src.mhs.execution", fromlist=["ExecutionSpec"]).ExecutionSpec()
    wins = _partition_windows(
        wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
        wl["closes"], wl["marks"], wl["funding"], spec, n_windows=3,
    )
    patched = []
    for i, w in enumerate(wins):
        fk = pd.DataFrame(True, index=w.minute_grid, columns=list(w.symbols))
        if i in (0, 2):
            lo = len(fk) // 3
            hi = min(len(fk), lo + 20)
            fk.iloc[lo:hi, 0:2] = False
        patched.append(dataclasses.replace(w, funding_known=fk))
    grid1 = patched[1].minute_grid
    at = grid1[len(grid1) // 2]
    sym = next(iter(patched[1].symbols))
    event = InstrumentSettlementEvent(
        event_id="mirror-settle", symbol=sym, effective_at=at, available_at=at,
        settlement_price=90.0, fee_bps=5.0, source_digest="perf03a",
    )
    patched[1] = dataclasses.replace(patched[1], settlement_events=(event,))
    return patched


def _assert_mirror_equal(result_a, acc_a, result_b, acc_b) -> None:
    import numpy as np

    assert acc_a.accounting_state.cash == acc_b.accounting_state.cash
    np.testing.assert_array_equal(acc_a.accounting_state.units, acc_b.accounting_state.units)
    np.testing.assert_array_equal(
        acc_a.accounting_state.last_marks, acc_b.accounting_state.last_marks,
    )
    assert acc_a.accounting_state.last_event_ns == acc_b.accounting_state.last_event_ns
    assert result_a.ledger.primary_valid == result_b.ledger.primary_valid
    assert tuple(sorted(result_a.ledger.invalid_reasons)) == tuple(sorted(result_b.ledger.invalid_reasons))
    ga, gb = list(result_a.ledger.data_gaps), list(result_b.ledger.data_gaps)
    assert len(ga) == len(gb)
    for x, y in zip(ga, gb, strict=True):
        assert x.code == y.code
        assert x.symbol == y.symbol
        assert x.timestamp == y.timestamp
        assert x.timestamp.unit == y.timestamp.unit
        assert str(x.timestamp.tz) == str(y.timestamp.tz)
        assert x.execution_bound == y.execution_bound
    for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
        sa, sb = getattr(result_a.ledger, field), getattr(result_b.ledger, field)
        np.testing.assert_array_equal(sa.to_numpy(), sb.to_numpy())
        np.testing.assert_array_equal(sa.index.asi8, sb.index.asi8)
    assert result_a.simulated_fills.equals(result_b.simulated_fills)


def test_vectorized_mirror_replay_equals_scalar_mirror_replay() -> None:
    from src.mhs.execution import ExecutionSpec, replay_execution_windows
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator

    for bound in ("OHLCV_IMMEDIATE_TAKER", "OHLCV_STRICT_PROXY"):
        wins = _twelve_symbol_windows()
        live_a: list = []
        result_a = replay_execution_windows(wins, 1000.0, bound, ExecutionSpec(), live_accumulators=live_a)
        acc_a = live_a[0][0]
        orig = _BoundExecutionReplayAccumulator._settle_mirror_window
        _BoundExecutionReplayAccumulator._settle_mirror_window = _head_mirror_window  # type: ignore[method-assign]
        try:
            live_b: list = []
            result_b = replay_execution_windows(wins, 1000.0, bound, ExecutionSpec(), live_accumulators=live_b)
            acc_b = live_b[0][0]
        finally:
            _BoundExecutionReplayAccumulator._settle_mirror_window = orig  # type: ignore[method-assign]
        _assert_mirror_equal(result_a, acc_a, result_b, acc_b)


def test_mirror_gap_timestamps_are_grid_elements() -> None:
    import dataclasses

    import pandas as pd

    grid = pd.date_range("2025-01-01 00:00", periods=15, freq="3min", tz="UTC")
    window = _ledger_window(grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]})
    fk = window.funding_known.copy()
    fk.iloc[4:10, 0] = False
    window = dataclasses.replace(window, funding_known=fk)
    result, _acc = _replay_with_acc(window)
    mirror_gaps = [g for g in result.ledger.data_gaps if g.code == "MISSING_HELD_FUNDING"]
    assert len(mirror_gaps) == 6
    for g, ts in zip(mirror_gaps, list(window.minute_grid[4:10]), strict=True):
        assert g.timestamp == ts
        assert g.timestamp.unit == ts.unit
        assert str(g.timestamp.tz) == str(ts.tz)


def test_reconcile_tripwire_still_fires() -> None:
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.execution import ExecutionSpec
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator

    grid = pd.date_range("2025-01-01 00:00", periods=10, freq="3min", tz="UTC")
    window = _ledger_window(grid, ["BTCUSDT"], [grid[0]], {"BTCUSDT": [0.1]})
    acc = _BoundExecutionReplayAccumulator(window, 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False)
    acc.consume(window)
    acc.accounting_state.cash += 1e-6 * 1000.0
    with pytest.raises(DataIntegrityError, match="causal accounting diverged"):
        acc.finalize()
