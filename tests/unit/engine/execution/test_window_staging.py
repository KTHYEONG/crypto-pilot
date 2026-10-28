"""Invariant guards for bound-invariant window staging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.engine.execution.contracts import ExecutionReplayWindow
from src.engine.execution.window_staging import WindowStaging, stage_window_arrays


def _window(
    grid, qv_values, mark_values, decisions, weights, *, explicit: bool = True, marks_none: bool = False,
):  # type: ignore[no-untyped-def]
    px = pd.DataFrame({"A": mark_values}, index=grid)
    qv = pd.DataFrame({"A": qv_values}, index=grid) if explicit else None
    marks = None if marks_none else px
    known = px.notna() if explicit else None
    avail = grid + pd.Timedelta(hours=1) if explicit else None
    return ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=("A",), symbols=("A",),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=marks, bar_funding=px * 0.0,
        target_weights=pd.DataFrame({"A": weights}, index=pd.DatetimeIndex(decisions)),
        signal_available_at=pd.DatetimeIndex(decisions),
        quote_volumes=qv, funding_known=known, bar_available_at=avail,
    )


def _grid(n: int = 40):  # type: ignore[no-untyped-def]
    return pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")


def test_staged_arrays_equal_private_per_bound_staging() -> None:
    from src.engine.execution import ExecutionSpec, replay_execution_windows
    from src.engine.execution.accumulator import _BoundExecutionReplayAccumulator

    grid = _grid()
    qv = np.full(40, 1000.0)
    marks = np.full(40, 100.0)
    w_explicit = _window(grid, qv, marks, [grid[0], grid[10]], [0.5, 0.0], explicit=True)
    w_legacy = _window(grid, qv, marks, [grid[0], grid[10]], [0.5, 0.0], explicit=False, marks_none=True)
    for w in (w_explicit, w_legacy):
        first = stage_window_arrays(w)
        second = stage_window_arrays(w)
        assert first is not second
        for name in (
            "grid_ns", "marks_values", "highs_values", "lows_values", "closes_values",
            "mark_valid", "funding_matrix", "last_close_idx", "quote_volumes",
            "last_liquid_idx", "funding_known", "avail_ns", "mark_avail",
            "decision_ns_all", "spos_all", "dpos_all", "on_grid_all", "target_values",
        ):
            a = getattr(first, name)
            b = getattr(second, name)
            assert a.dtype == b.dtype
            assert a.shape == b.shape
            assert np.array_equal(a, b)
        acc = _BoundExecutionReplayAccumulator(w, 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False)
        frame = acc._consume_validate_window(w)
        assert np.array_equal(frame.grid_ns, first.grid_ns)
        assert np.array_equal(frame.marks_values, first.marks_values)
        assert np.array_equal(frame.target_values, first.target_values)
        assert np.array_equal(acc._w_qv, first.quote_volumes)
        assert np.array_equal(acc._w_avail_ns, first.avail_ns)
        assert acc._w_avail_explicit == first.avail_explicit
        res = replay_execution_windows((w,), 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
        assert res is not None


def test_staged_arrays_are_read_only() -> None:
    grid = _grid()
    w = _window(grid, np.full(40, 1000.0), np.full(40, 100.0), [grid[0]], [0.5])
    staged = stage_window_arrays(w)
    for name in (
        "grid_ns", "marks_values", "highs_values", "lows_values", "closes_values",
        "mark_valid", "funding_matrix", "last_close_idx", "quote_volumes",
        "last_liquid_idx", "funding_known", "avail_ns", "mark_avail",
        "decision_ns_all", "spos_all", "dpos_all", "on_grid_all", "target_values",
    ):
        arr = getattr(staged, name)
        assert arr.flags.writeable is False
        with pytest.raises(ValueError, match=r".+"):
            arr[0] = arr[0]


def test_staging_failure_order_and_messages() -> None:
    grid = _grid()
    qv = np.full(40, 1000.0)
    marks = np.full(40, 100.0)
    thin = _window(grid[:1], qv[:1], marks[:1], [grid[0]], [0.5])
    with pytest.raises(DataIntegrityError, match="an execution window must span at least two grid bars"):
        stage_window_arrays(thin)
    import dataclasses

    w = _window(grid, qv, marks, [grid[0]], [0.5])
    shifted = grid + pd.Timedelta(hours=1)
    misaligned = dataclasses.replace(w, bar_funding=w.bar_funding.set_axis(shifted))
    with pytest.raises(DataIntegrityError, match="bar_funding must align exactly to the window minute grid"):
        stage_window_arrays(misaligned)
    bad_fund = w.bar_funding.copy()
    bad_fund.iloc[0, 0] = float("nan")
    bad_marks = w.marks.copy()
    bad_marks.iloc[1, 0] = -5.0
    both = dataclasses.replace(w, bar_funding=bad_fund, marks=bad_marks)
    with pytest.raises(DataIntegrityError, match="bar_funding must be finite"):
        stage_window_arrays(both)
    neg = dataclasses.replace(w, marks=bad_marks)
    with pytest.raises(DataIntegrityError, match="finite marks must be strictly positive"):
        stage_window_arrays(neg)


def test_staging_failures_are_not_memoized() -> None:
    grid = _grid()
    w = _window(grid, np.full(40, 1000.0), np.full(40, 100.0), [grid[0]], [0.5])
    bad_fund = w.bar_funding.copy()
    bad_fund.iloc[0, 0] = float("nan")
    import dataclasses

    bad = dataclasses.replace(w, bar_funding=bad_fund)
    staging = WindowStaging(bad)
    with pytest.raises(DataIntegrityError, match="bar_funding must be finite"):
        staging.arrays()
    with pytest.raises(DataIntegrityError, match="bar_funding must be finite"):
        staging.arrays()


def test_consume_rejects_foreign_staging_before_mutation() -> None:
    from src.engine.execution import ExecutionSpec
    from src.engine.execution.accumulator import _BoundExecutionReplayAccumulator

    grid = _grid()
    qv = np.full(40, 1000.0)
    marks = np.full(40, 100.0)
    wa = _window(grid, qv, marks, [grid[0]], [0.5])
    wb = _window(grid, qv, marks, [grid[5]], [0.25])
    acc = _BoundExecutionReplayAccumulator(wa, 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False)
    foreign = WindowStaging(wb)
    cash, units = acc.cash, acc.units_arr.copy()
    fills, chunks = len(acc.fill_ts), len(acc.equity_chunks)
    with pytest.raises(ValueError, match="staging was built for a different window"):
        acc.consume(wa, foreign)
    assert acc.cash == cash
    assert np.array_equal(acc.units_arr, units)
    assert len(acc.fill_ts) == fills
    assert len(acc.equity_chunks) == chunks


def _release_grid() -> pd.DatetimeIndex:
    return pd.date_range("2024-03-14 22:00", periods=40, freq="3min", tz="UTC")


def _release_window(grid, signals, *, avail_step: str = "3min"):  # type: ignore[no-untyped-def]
    """One BUY intent released at ``signals``; distinct trending closes, flat
    200.0 lows/highs so a strict resting order never touches."""
    closes = pd.DataFrame({"A": 100.0 + np.arange(len(grid), dtype="float64")}, index=grid)
    flat = pd.DataFrame({"A": np.full(len(grid), 200.0)}, index=grid)
    idx = pd.DatetimeIndex(signals)
    return ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=("A",), symbols=("A",),
        minute_grid=grid, highs=flat, lows=flat, closes=closes, marks=closes,
        bar_funding=closes * 0.0,
        target_weights=pd.DataFrame({"A": [0.5] * len(idx)}, index=idx),
        signal_available_at=idx,
        quote_volumes=flat * 0.0 + 1000.0, funding_known=flat.notna(),
        bar_available_at=grid + pd.Timedelta(avail_step),
    )


def test_submission_bar_opens_at_on_grid_release() -> None:
    grid = _release_grid()
    release = grid[20]
    staged = stage_window_arrays(_release_window(grid, [release]))
    assert int(staged.spos_all[0]) == 20


def test_off_grid_release_submits_on_next_label() -> None:
    grid = _release_grid()
    staged = stage_window_arrays(_release_window(grid, [grid[20] + pd.Timedelta(seconds=70)]))
    assert int(staged.spos_all[0]) == 21


def test_off_grid_decision_uses_last_completed_bar() -> None:
    from src.engine.execution import ExecutionSpec, replay_execution_windows

    grid = _release_grid()
    release = grid[20] + pd.Timedelta(seconds=70)
    result = replay_execution_windows(
        (_release_window(grid, [release]),), 10000.0, "OHLCV_IMMEDIATE_TAKER",
        ExecutionSpec(decision_anchor="submit_bar"),
    )
    assert float(result.simulated_fills.iloc[0]["quantity_delta"]) == pytest.approx(5000.0 / 119.0)


def test_submit_anchor_without_completed_bar_fails_closed() -> None:
    from src.engine.execution import ExecutionSpec, replay_execution_windows

    grid = _release_grid()
    result = replay_execution_windows(
        (_release_window(grid, [grid[0]]),), 10000.0, "OHLCV_IMMEDIATE_TAKER",
        ExecutionSpec(decision_anchor="submit_bar"),
    )
    assert result.simulated_fills.empty
    assert any(g.code == "MISSING_DECISION_MARK" for g in result.data_gaps)


def test_decision_price_is_release_closing_bar() -> None:
    from src.engine.execution import ExecutionSpec, replay_execution_windows

    grid = _release_grid()
    release = grid[20]
    result = replay_execution_windows(
        (_release_window(grid, [release]),), 10000.0, "OHLCV_IMMEDIATE_TAKER",
        ExecutionSpec(decision_anchor="submit_bar"),
    )
    assert result.ledger.primary_valid
    assert float(result.simulated_fills.iloc[0]["quantity_delta"]) == pytest.approx(5000.0 / 119.0)
    assert pd.Timestamp(result.simulated_fills.iloc[0]["timestamp"]) == release + pd.Timedelta(minutes=3)


def test_no_future_data_gap_at_on_grid_release() -> None:
    from src.engine.execution import ExecutionSpec, replay_execution_windows

    grid = _release_grid()
    result = replay_execution_windows(
        (_release_window(grid, [grid[20]]),), 10000.0, "OHLCV_IMMEDIATE_TAKER",
        ExecutionSpec(decision_anchor="submit_bar"),
    )
    assert result.ledger.primary_valid
    assert not any(g.code == "FUTURE_DATA_REFERENCE" for g in result.ledger.data_gaps)


def test_passive_timeout_bar_moves_with_submission() -> None:
    from src.engine.execution import ExecutionSpec, replay_execution_windows

    grid = _release_grid()
    release = grid[20]
    result = replay_execution_windows(
        (_release_window(grid, [release]),), 10000.0, "OHLCV_STRICT_PROXY",
        ExecutionSpec(decision_anchor="submit_bar"),
    )
    assert result.ledger.primary_valid
    fills = result.simulated_fills
    assert (fills["reason"] == "timeout_taker").all()
    assert float(fills.iloc[0]["fill_price"]) == pytest.approx(float(100.0 + 30.0))
