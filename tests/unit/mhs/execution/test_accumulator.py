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

