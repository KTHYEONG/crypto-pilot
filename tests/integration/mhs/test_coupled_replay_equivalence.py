"""P5 equivalence gate: replay_execution_windows_coupled vs the exact two-pass path.

SCENARIO_MHS_PERF_P5_01_COUPLED_SCALE_EXACT / SCENARIO_MHS_PERF_P5_02_INCOMPLETE_DAY_FALLBACK.

``replay_execution_windows_coupled`` (docs/specs/mhs_perf_refactor.md P5) folds
the reference (unscaled) and rescaled replay passes into one window stream,
recomputing the causal exposure scale from the growing reference-return prefix
at every window boundary instead of from the full series computed once. This
module proves that recomputation is exact for a real, already-production
prefix-deterministic scale function (`_pnl_vol_target_scale`, the
`median_relative` mode `is_streaming_scale_mode` accepts), and that the
coordinator fails closed -- never silently wrong -- when a window boundary
would require a not-yet-covered day.

A synthetic multi-window market is used deliberately (not the full 446-symbol
production panel): the property under test is the coordinator's incremental
recomputation being exact relative to computing the same causal formula once
on the complete series, which is fully exercised by a modest multi-window
synthetic workload per the repo's minimal-synthetic-data testing directive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mhs.scaling import _pnl_vol_target_scale, is_streaming_scale_mode
from src.mhs.execution import (
    DataIntegrityError,
    ExecutionReplayWindow,
    ExecutionSpec,
    _rescale_window_weights,
    replay_execution_window_batch,
    replay_execution_windows,
    replay_execution_windows_coupled,
)
from src.mhs.params import PNL_VOL_TARGET_BURN_IN_DAYS

pytestmark = pytest.mark.slow

_FREQ = "5min"
_N_SYMBOLS = 6
_N_DAYS = 130  # several 31-day windows, past the causal burn-in


def _build_workload() -> dict[str, object]:
    grid = pd.date_range("2021-01-01", periods=_N_DAYS * 24 * 12, freq=_FREQ, tz="UTC")
    symbols = [f"SYM{i:02d}USDT" for i in range(_N_SYMBOLS)]
    rng = np.random.default_rng(20260822)
    drift = rng.normal(0.0, 0.0002, _N_SYMBOLS)
    closes = pd.DataFrame(
        {
            s: 100.0 * np.exp(np.cumsum(drift[i] + rng.normal(0.0, 0.003, len(grid))))
            for i, s in enumerate(symbols)
        },
        index=grid,
    )
    decision_grid = pd.date_range("2021-01-01", periods=_N_DAYS * 4, freq="6h", tz="UTC")
    weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
    rng_w = np.random.default_rng(20260823)
    for ts in decision_grid:
        active = rng_w.choice(symbols, size=3, replace=False)
        weights.loc[ts, active] = rng_w.uniform(0.005, 0.02, 3)
    return {
        "grid": grid,
        "symbols": symbols,
        "highs": closes * 1.001,
        "lows": closes * 0.999,
        "closes": closes,
        "marks": closes,
        "funding": pd.DataFrame(1.0e-5, index=grid, columns=symbols),
        "weights": weights,
        "signals": decision_grid + pd.Timedelta(hours=1),
    }


def _partition_windows_31d(wl: dict[str, object]) -> list[ExecutionReplayWindow]:
    """31-day contiguous windows with strict-timeout overlap (production shape)."""
    grid: pd.DatetimeIndex = wl["grid"]  # type: ignore[assignment]
    weights: pd.DataFrame = wl["weights"]  # type: ignore[assignment]
    signals: pd.DatetimeIndex = wl["signals"]  # type: ignore[assignment]
    spec = ExecutionSpec()
    full_ns = np.asarray(grid, dtype="datetime64[ns]").astype("int64")
    timeout = spec.passive_timeout_minutes * 60_000_000_000
    sig_ns = np.asarray(signals, dtype="datetime64[ns]").astype("int64")
    spos = np.searchsorted(full_ns, sig_ns, side="right")
    resolve: list[pd.Timestamp | None] = [None] * len(weights)
    for i in range(len(weights)):
        if spos[i] >= len(full_ns):
            continue
        tns = full_ns[spos[i]] + timeout
        tpos = int(np.searchsorted(full_ns, tns, side="left"))
        if tpos < len(full_ns) and full_ns[tpos] == tns:
            resolve[i] = pd.Timestamp(tns, unit="ns", tz="UTC")
    decision_times = pd.DatetimeIndex(weights.index)
    max_window = pd.Timedelta(days=31)
    bounds: list[tuple[int, int]] = []
    i0 = 0
    while i0 < len(decision_times):
        i1 = i0 + 1
        while i1 < len(decision_times) and decision_times[i1] - decision_times[i0] <= max_window:
            i1 += 1
        bounds.append((i0, i1))
        i0 = i1
    out: list[ExecutionReplayWindow] = []
    for bi, (i0, i1) in enumerate(bounds):
        is_last = bi == len(bounds) - 1
        ws = weights.iloc[i0:i1]
        sg = signals[i0:i1]
        grid_start = grid[0] if bi == 0 else decision_times[i0 - 1]
        grid_end = (
            grid[-1] if is_last
            else max(
                (resolve[i] for i in range(i0, i1) if resolve[i] is not None),
                default=decision_times[i1 - 1] + pd.Timedelta(hours=2),
            )
        )
        grid_end = min(grid_end, grid[-1])
        wgrid = pd.date_range(grid_start, grid_end, freq=_FREQ, tz="UTC")
        out.append(
            ExecutionReplayWindow(
                window_start=grid_start, window_end=grid_end,
                columns=tuple(weights.columns), symbols=tuple(weights.columns),
                minute_grid=wgrid,
                highs=wl["highs"].loc[wgrid], lows=wl["lows"].loc[wgrid],  # type: ignore[union-attr]
                closes=wl["closes"].loc[wgrid], marks=wl["marks"].loc[wgrid],  # type: ignore[union-attr]
                bar_funding=wl["funding"].loc[wgrid],  # type: ignore[union-attr]
                target_weights=ws, signal_available_at=sg,
            )
        )
    return out


def _two_pass_reference(windows: list[ExecutionReplayWindow]) -> pd.Series:
    """Exact production two-pass formula: unscaled reference -> full-series scale."""
    reference = replay_execution_windows(windows, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    daily_returns = reference.ledger.equity.resample("1D").last().pct_change()
    return _pnl_vol_target_scale(daily_returns)


def _two_pass_scaled_ledger(windows: list[ExecutionReplayWindow], scale: pd.Series) -> pd.Series:
    rescaled = [_rescale_window_weights(w, scale) for w in windows]
    result = replay_execution_window_batch(
        rescaled, 1.0, [("OHLCV_IMMEDIATE_TAKER", ExecutionSpec())],
    )
    return result[0].ledger.equity


def test_streaming_mode_is_median_relative_only_for_this_probe() -> None:
    """Sanity: the scale function this test uses is one of the two modes the
    production `is_streaming_scale_mode` actually accepts."""
    from src.mhs.contracts import MhsDiagnosticRequest

    assert is_streaming_scale_mode(MhsDiagnosticRequest(pnl_vol_target_mode="median_relative"))


def test_coupled_scale_matches_two_pass_exactly() -> None:
    """SCENARIO_MHS_PERF_P5_01_COUPLED_SCALE_EXACT: the coupled coordinator's
    incrementally-recomputed scale and resulting ledger equal the exact
    two-pass result, index-for-index."""
    wl = _build_workload()
    windows = _partition_windows_31d(wl)
    assert len(windows) >= 3, "need multiple windows to exercise incremental recomputation"

    two_pass_scale = _two_pass_reference(windows)
    two_pass_equity = _two_pass_scaled_ledger(windows, two_pass_scale)

    reference_result, batch = replay_execution_windows_coupled(
        windows, 1.0,
        ("OHLCV_IMMEDIATE_TAKER", ExecutionSpec()),
        [("OHLCV_IMMEDIATE_TAKER", ExecutionSpec())],
        _pnl_vol_target_scale,
    )
    coupled_equity = batch.results[0].ledger.equity  # type: ignore[union-attr]

    assert batch.isolated_failures == ()
    np.testing.assert_array_equal(
        coupled_equity.index.to_numpy(), two_pass_equity.index.to_numpy(),
    )
    assert [repr(v) for v in coupled_equity.to_numpy()] == [
        repr(v) for v in two_pass_equity.to_numpy()
    ], "coupled ledger equity must repr()-match the two-pass ledger exactly"

    two_pass_reference_returns = (
        reference_result.ledger.equity.resample("1D").last().pct_change()
    )
    coupled_scale = _pnl_vol_target_scale(two_pass_reference_returns)
    burn_in_cut = two_pass_scale.index[
        two_pass_scale.index >= two_pass_scale.index[0] + pd.Timedelta(days=PNL_VOL_TARGET_BURN_IN_DAYS)
    ]
    assert len(burn_in_cut) > 0
    for ts in burn_in_cut:
        assert repr(float(coupled_scale.get(ts, float("nan")))) == repr(
            float(two_pass_scale.get(ts, float("nan")))
        ), f"scale diverges at {ts}"


def test_incomplete_day_raises_and_signals_two_pass_fallback() -> None:
    """SCENARIO_MHS_PERF_P5_02_INCOMPLETE_DAY_FALLBACK: a window whose grid_end
    falls mid-day, followed by a decision on the NEXT day, is rejected instead
    of silently emitting a scale from an incomplete prefix."""
    wl = _build_workload()
    windows = _partition_windows_31d(wl)
    first = windows[0]

    # Truncate the first window's grid_end to mid-day so its coverage never
    # reaches the midnight boundary the second window's first decision needs,
    # reproducing the documented hazard directly.
    short_end = first.window_start + pd.Timedelta(hours=6)
    short_grid = pd.date_range(first.window_start, short_end, freq=_FREQ, tz="UTC")
    truncated_first = first.__class__(
        window_start=first.window_start, window_end=short_end,
        columns=first.columns, symbols=first.symbols, minute_grid=short_grid,
        highs=first.highs.loc[short_grid], lows=first.lows.loc[short_grid],
        closes=first.closes.loc[short_grid],
        marks=first.marks.loc[short_grid] if first.marks is not None else None,
        bar_funding=first.bar_funding.loc[short_grid],
        target_weights=first.target_weights.iloc[:1],
        signal_available_at=first.signal_available_at[:1],
    )

    with pytest.raises(DataIntegrityError, match="reference prefix has a gap"):
        replay_execution_windows_coupled(
            [truncated_first, *windows[1:]], 1.0,
            ("OHLCV_IMMEDIATE_TAKER", ExecutionSpec()),
            [("OHLCV_IMMEDIATE_TAKER", ExecutionSpec())],
            _pnl_vol_target_scale,
        )
    # The documented caller contract (src/application/research/mhs/evaluation.py
    # _book_outcome): a DataIntegrityError here means "fall back to the exact
    # two-pass path", which this test's other case already proves is exact.
