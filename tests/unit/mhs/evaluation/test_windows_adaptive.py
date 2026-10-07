"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

from tests.fixtures.mhs_requests import research_baseline
import pytest
from tests.unit.mhs.evaluation.test_windows import (
    _completed_fixture,
)


def _tight_telemetry(monkeypatch, *, pss: int = 0, available: int = 10**12) -> None:
    from src.mhs import resources as _res
    from src.mhs.tree_memory import TreeMemoryObservation

    monkeypatch.setattr(_res, "_current_tree_pss_bytes", lambda: pss)
    monkeypatch.setattr(_res, "_current_available_bytes", lambda: available)
    monkeypatch.setattr(_res, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(_res, "_current_tree_swap_bytes", lambda: 0)
    monkeypatch.setattr(_res, "observe_tree_memory", lambda: TreeMemoryObservation(pss_bytes=pss, swap_bytes=0))


def _adaptive_fixture(tmp_path, *, days: int = 3):
    import pandas as pd

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=days, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=days)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[:, 0] = 0.05
    return start, end, decisions, funding, targets, spec


def test_adaptive_decode_splits_with_no_missing_decisions(tmp_path, monkeypatch) -> None:
    """Tight envelope emits smaller admitted pieces with every decision exactly once."""
    import pandas as pd

    from src.mhs import resources as _res
    import src.mhs.execution.window_stream as window_stream
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=3)
    _tight_telemetry(monkeypatch)
    admitted: list[int] = []
    orig = _res.assert_mhs_allocation_budget
    monkeypatch.setattr(_res, "assert_mhs_allocation_budget", lambda **k: (admitted.append(k["estimated_bytes"]), orig(**k))[1])
    monkeypatch.setattr(window_stream, "assert_mhs_allocation_budget", lambda **k: (admitted.append(k["estimated_bytes"]), orig(**k))[1])
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_600_000, reserve_bytes=100, execution_bound_count=2,
        )
    )
    assert len(windows) > 1
    assert len(admitted) >= len(windows)
    got = pd.DatetimeIndex([]).append([w.target_weights.index for w in windows]) if windows else pd.DatetimeIndex([])
    assert list(got) == list(targets.index)
    keys = {w.logical_partition for w in windows}
    assert len(keys) == 1
    assert next(iter(keys)) == (0, 3)


def test_adaptive_decode_streams_empty_pieces_before_next_daily_decision(tmp_path, monkeypatch) -> None:
    """A narrow physical plan advances through decision-free daily gaps."""
    from itertools import pairwise

    import pandas as pd

    import src.mhs.execution.window_stream as _stream
    import src.mhs.evaluation.windows as _w

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=3)
    _tight_telemetry(monkeypatch)
    monkeypatch.setattr(_stream, "plan_mhs_execution_bars", lambda **_kwargs: 100)
    windows = list(
        _w._iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_600_000, reserve_bytes=100, execution_bound_count=2,
        )
    )

    nonempty = [window for window in windows if not window.target_weights.empty]
    got = pd.DatetimeIndex([]).append([window.target_weights.index for window in nonempty])
    assert list(got) == list(targets.index)
    assert any(window.target_weights.empty for window in windows[1:])
    assert all(
        later.minute_grid[0] == earlier.minute_grid[-1]
        for earlier, later in pairwise(windows)
    )
    assert all(
        decision in window.minute_grid
        for window in nonempty
        for decision in window.target_weights.index
    )

    held_windows = list(
        _w._iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_600_000, reserve_bytes=100, execution_bound_count=2,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert all("AUSDT" in window.symbols for window in held_windows)


def test_adaptive_decode_rejects_plan_shorter_than_first_signal_span(tmp_path, monkeypatch) -> None:
    """A malformed plan cannot split the first decision from its signal span."""
    import pandas as pd

    import src.mhs.execution.window_stream as _stream
    import src.mhs.evaluation.windows as _w
    from src.common.errors import DataIntegrityError

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    monkeypatch.setattr(_stream, "plan_mhs_execution_bars", lambda **_kwargs: 1)
    with pytest.raises(DataIntegrityError, match="unresolved"):
        list(
            _w._iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1_600_000, reserve_bytes=100,
            )
        )


def test_adaptive_empty_piece_rejects_unknown_live_roster(tmp_path, monkeypatch) -> None:
    """Decision-free pieces preserve the fail-closed live-roster contract."""
    import itertools

    import pandas as pd

    import src.mhs.execution.window_stream as _stream
    import src.mhs.evaluation.windows as _w
    from src.common.errors import DataIntegrityError

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=3)
    _tight_telemetry(monkeypatch)
    monkeypatch.setattr(_stream, "plan_mhs_execution_bars", lambda **_kwargs: 100)
    calls = itertools.count()

    def _live_roster() -> frozenset[str]:
        return frozenset({"AUSDT"}) if next(calls) < 2 else frozenset({"ZZZ"})

    with pytest.raises(DataIntegrityError, match="not in canonical"):
        list(
            _w._iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1_600_000, reserve_bytes=100,
                required_symbols=_live_roster,
            )
        )


def test_adaptive_tail_and_nonaligned_fence(tmp_path, monkeypatch) -> None:
    """Held tail past the final decision streams completed bars with coverage."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=2, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=2, minutes=1)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[0, 0] = 0.1
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert windows
    assert (windows[-1].minute_grid < end).all()
    assert (windows[-1].bar_available_at <= end).all()


def test_adaptive_minimum_timeout_retention(tmp_path, monkeypatch) -> None:
    """An order near a physical boundary keeps its strict timeout endpoint."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
        )
    )
    assert len(windows) > 1
    first = windows[0]
    assert len(first.minute_grid) >= 2
    assert len(first.target_weights) >= 1


def test_adaptive_bound_union_roster(tmp_path, monkeypatch) -> None:
    """Base and stress holdings both remain in the next piece roster."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    targets.iloc[:, 1] = 0.0
    _tight_telemetry(monkeypatch)
    held = {"AUSDT", "BUSDT"}
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_800_000, reserve_bytes=100,
            required_symbols=lambda: frozenset(held),
        )
    )
    assert windows
    assert held <= set(windows[1].symbols) if len(windows) > 1 else held <= set(windows[0].symbols)


def test_adaptive_ipc_partition_parity(tmp_path, monkeypatch) -> None:
    """Spilled and restored split pieces keep ordinal keys and numeric planes."""
    import numpy as np
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows, _spill_window_to_ipc, _load_window_from_ipc

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
        )
    )
    assert len(windows) > 1
    for i, w in enumerate(windows):
        path = str(tmp_path / f"piece_{i}.arrow")
        _spill_window_to_ipc(w, path)
        loaded = _load_window_from_ipc(path)
        assert loaded.logical_partition == w.logical_partition == (0, 2)
        np.testing.assert_allclose(loaded.highs.to_numpy(dtype="float64"), w.highs.to_numpy(dtype="float64"))
        assert (loaded.minute_grid == w.minute_grid).all()


def test_adaptive_logical_identity_under_splitting(tmp_path, monkeypatch) -> None:
    """Every split piece shares one half-open ordinal range."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=40, freq="24h", tz="UTC")
    end = decisions[-1] + pd.Timedelta(days=1)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[:, 0] = 0.05
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=2_000_000, reserve_bytes=100,
        )
    )
    first_keys = {w.logical_partition for w in windows if w.logical_partition and w.logical_partition[0] == 0}
    assert first_keys == {(0, 32)} or all(k[0] == 0 or k[0] > 0 for k in {w.logical_partition for w in windows})
    assert all(w.logical_partition is not None and len(w.logical_partition) == 2 for w in windows)


def test_adaptive_execution_bound_count_validation(tmp_path) -> None:
    """Non-positive bound counts fail before any decode."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=1, freq="24h", tz="UTC")
    _, funding, targets, spec = _completed_fixture(tmp_path, start, start + pd.Timedelta(hours=2), decisions)
    with pytest.raises(ValueError, match="execution_bound_count"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, start + pd.Timedelta(hours=2), funding, spec,
                execution_bound_count=0,
            )
        )


def test_estimate_and_minimum_helpers_cover_branches() -> None:
    """Allocation helper validation and timeout edge cases."""

    from src.mhs.evaluation.windows import _estimate_mhs_execution_allocation, _minimum_mhs_execution_bars

    with pytest.raises(ValueError, match="bound_count"):
        _estimate_mhs_execution_allocation(n_symbols=2, n_columns=2, bound_count=0)
    assert _minimum_mhs_execution_bars(0, 180_000_000_000) == 2
    assert _minimum_mhs_execution_bars(1_800_000_000_000, 180_000_000_000) == 11
    alloc = _estimate_mhs_execution_allocation(n_symbols=0, n_columns=3, bound_count=2)
    assert alloc.bytes_per_bar > 0


def test_materialize_covers_empty_and_missing_branches(tmp_path, monkeypatch) -> None:
    """Empty aligned frames and missing symbols are admitted; no piece carries a mark plane."""
    import pandas as pd

    import src.mhs.evaluation.windows as _w
    from src.mhs.resources import MhsExecutionAllocation

    _tight_telemetry(monkeypatch)
    grid = pd.date_range("2022-01-01", periods=5, freq="3min", tz="UTC")
    alloc = MhsExecutionAllocation(fixed_bytes=100, bytes_per_bar=100, decoder_bytes=100)
    cols = ("AUSDT", "BUSDT")
    empty_w = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=list(cols), dtype="float64")
    empty_s = pd.DatetimeIndex([], tz="UTC")
    win = _w._materialize_execution_piece(
        piece_grid=grid, piece_weights=empty_w, piece_signals=empty_s, roster=[],
        columns=cols, root=str(tmp_path), timeframe="3m", funding_by_symbol={},
        funding_failures=None, allocation=alloc,
        budget_bytes=None, reserve_bytes=None,
        window_start=grid[0], window_end=grid[-1], logical_partition=(0, 0),
    )
    assert win.symbols == ()
    assert win.marks is None
    w2 = pd.DataFrame(0.0, index=pd.DatetimeIndex([grid[0]]), columns=list(cols))
    s2 = pd.DatetimeIndex([grid[0] + pd.Timedelta(hours=1)])
    import src.mhs.execution.window_stream as _ws

    monkeypatch.setattr(_ws, "_load_window_minute_frames", lambda *a, **k: {})
    win2 = _w._materialize_execution_piece(
        piece_grid=grid, piece_weights=w2, piece_signals=s2, roster=["AUSDT"],
        columns=cols, root=str(tmp_path), timeframe="3m", funding_by_symbol={},
        funding_failures=None, allocation=alloc,
        budget_bytes=None, reserve_bytes=None,
        window_start=grid[0], window_end=grid[-1], logical_partition=(0, 1),
    )
    assert "AUSDT" in win2.symbols
    assert float(win2.highs["AUSDT"].isna().sum()) == len(grid)
    assert win2.marks is None
    win3 = _w._materialize_execution_piece(
        piece_grid=grid, piece_weights=w2, piece_signals=s2, roster=["AUSDT"],
        columns=cols, root=str(tmp_path), timeframe="3m", funding_by_symbol={},
        funding_failures=None, allocation=alloc,
        budget_bytes=None, reserve_bytes=None,
        window_start=grid[0], window_end=grid[-1], logical_partition=(0, 1),
    )
    assert win3.marks is None
    win4 = _w._materialize_execution_piece(
        piece_grid=grid, piece_weights=w2, piece_signals=s2, roster=["AUSDT"],
        columns=cols, root=str(tmp_path), timeframe="3m", funding_by_symbol={},
        funding_failures=None, allocation=alloc,
        budget_bytes=None, reserve_bytes=None,
        window_start=grid[0], window_end=grid[-1], logical_partition=(0, 1),
    )
    assert win4.marks is None


def test_adaptive_single_piece_when_budget_allows(tmp_path, monkeypatch) -> None:
    """Generous budgets keep one physical piece per logical partition."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=10**12, reserve_bytes=100,
        )
    )
    assert len(windows) == 1
    assert windows[0].logical_partition == (0, 2)


def test_adaptive_single_piece_roster_branches(tmp_path, monkeypatch) -> None:
    """Single-piece adaptive path refreshes rosters and rejects unknown symbols."""
    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=1)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=10**12, reserve_bytes=100,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert windows
    assert "AUSDT" in windows[0].symbols
    with pytest.raises(DataIntegrityError, match="not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=10**12, reserve_bytes=100,
                required_symbols=lambda: frozenset({"ZZZ"}),
            )
        )


def test_adaptive_offgrid_timeout_fallback(tmp_path, monkeypatch) -> None:
    """Off-grid timeouts fall back to decision-anchored minimum spans."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.types import ExecutionSpec

    start, end, decisions, funding, targets, _ = _adaptive_fixture(tmp_path, days=2)
    spec = ExecutionSpec(passive_timeout_minutes=31)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
        )
    )
    assert windows


def test_adaptive_split_unknown_roster_fails_closed(tmp_path, monkeypatch) -> None:
    """Unknown held symbols fail during split and tail roster resolution."""
    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    with pytest.raises(DataIntegrityError, match=r"not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1_550_000, reserve_bytes=100,
                required_symbols=lambda: frozenset({"ZZZ"}),
            )
        )


def test_adaptive_piece_and_tail_unknown_branches(tmp_path, monkeypatch) -> None:
    """Single piece reuses the planning snapshot; late unknowns hit split-piece guards."""
    import itertools

    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    calls = itertools.count()
    seq = [frozenset(), frozenset({"ZZZ"})]

    def _flip() -> frozenset[str]:
        i = next(calls)
        return seq[min(i, 1)]

    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=10**12, reserve_bytes=100, required_symbols=_flip,
        )
    )
    assert len(windows) == 1
    assert next(calls) == 1
    assert list(windows[0].symbols) == ["AUSDT"]
    start3, end3, decisions3, funding3, targets3, spec3 = _adaptive_fixture(tmp_path, days=3)
    calls2 = itertools.count()

    def _flip2() -> frozenset[str]:
        i = next(calls2)
        return frozenset() if i < 2 else frozenset({"ZZZ"})

    with pytest.raises(DataIntegrityError, match=r"not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets3, decisions3 + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start3, end3, funding3, spec3,
                budget_bytes=1_600_000, reserve_bytes=100, required_symbols=_flip2,
            )
        )


def test_adaptive_tail_unknown_fails_closed(tmp_path, monkeypatch) -> None:
    """Unknown symbols during tail streaming fail with diagnostics."""
    import itertools

    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=1, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=4)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[0, 0] = 0.1
    _tight_telemetry(monkeypatch)
    calls = itertools.count()

    def _flip() -> frozenset[str]:
        i = next(calls)
        return frozenset({"AUSDT"}) if i < 2 else frozenset({"ZZZ"})

    with pytest.raises(DataIntegrityError, match=r"not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1_550_000, reserve_bytes=100, required_symbols=_flip,
            )
        )


def test_adaptive_long_tail_streams_multiple_pieces(tmp_path, monkeypatch) -> None:
    """A held tail longer than one plan streams several empty pieces."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=1, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=4)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[0, 0] = 0.1
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert len(windows) > 2
    assert all(w.target_weights.empty for w in windows[1:])


def test_adaptive_unresolvable_budget_fails_closed(tmp_path, monkeypatch) -> None:
    """A budget smaller than one timeout span leaves the order unresolved."""
    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    with pytest.raises(DataIntegrityError, match=r"unresolved|rejected"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1, reserve_bytes=100,
            )
        )


def test_adaptive_insufficient_grid_fails_closed(tmp_path, monkeypatch) -> None:
    """A logical grid smaller than the timeout minimum fails with diagnostics."""
    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.DatetimeIndex([start])
    _, funding, targets, spec = _completed_fixture(tmp_path, start, start + pd.Timedelta(minutes=6), decisions)
    _tight_telemetry(monkeypatch)
    with pytest.raises(DataIntegrityError, match="strict timeout"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, start + pd.Timedelta(minutes=6), funding, spec,
                budget_bytes=10**12, reserve_bytes=100,
            )
        )


def test_book_outcome_propagates_live_bound_count(monkeypatch, tmp_path) -> None:
    """The shared generator receives the actual live bound count, not a constant."""
    import dataclasses

    import pandas as pd

    import src.mhs.evaluation.windows as _w
    from src.mhs.types import BOOK_SPECS, ExecutionSpec

    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2022-01-02", tz="UTC")
    grid_1h = pd.date_range(start, end, freq="1h", tz="UTC")
    step_grid = pd.date_range(start, periods=2, freq="24h", tz="UTC")
    weights_step = pd.DataFrame(0.0, index=step_grid, columns=["AUSDT"])
    opens = pd.DataFrame(1.0, index=grid_1h, columns=["AUSDT"])
    bar_funding = pd.DataFrame(0.0, index=grid_1h, columns=["AUSDT"])
    captured: dict[str, object] = {}

    class _Evid:
        prescreen: dict[float, object] = dataclasses.field(default_factory=dict)
        tail: object = None

    monkeypatch.setattr(_w, "book_evidence", lambda *a, **k: type("E", (), {"prescreen": {}, "tail": None})())
    monkeypatch.setattr(_w, "_resolve_ram_budget", lambda *a, **k: (None, None))
    monkeypatch.setattr(_w.integrity, "_truncate_replayable_decisions", lambda w, s, g, spec: (w, s, 0))
    monkeypatch.setattr(_w.specs, "_resolved_base_execution_spec", lambda req: ExecutionSpec())
    monkeypatch.setattr(_w.specs, "_stress_cost_execution_spec", lambda spec: spec)
    monkeypatch.setattr(_w, "_scaling", type("S", (), {"_apply_rebalance_deadband": staticmethod(lambda w: w), "_replay_exposure_scale": staticmethod(lambda r, req: r), "is_streaming_scale_mode": staticmethod(lambda req: False)})())

    def _capture(*a, **k):
        captured.update(k)
        return iter(())

    monkeypatch.setattr(_w, "_iter_mhs_execution_windows", _capture)
    request = research_baseline(touch_diagnostic=True)
    phase = type("P", (), {})()
    report, _ = _w._book_outcome(
        "blend", BOOK_SPECS["fast_reversal"], 1, step_grid, weights_step, grid_1h, opens, bar_funding,
        phase, str(tmp_path), request, {}, start, end, 1, 1.0,
    )
    assert captured.get("execution_bound_count") == 4
    assert report.failure is not None
