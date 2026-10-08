"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/lab/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import pytest
from src.lab.mhs.backtest.inventory import (
    _execution_fence,
    _validate_replay_window,
)
from tests.lab.mhs.evaluation.test_windows import (
    _local_replay_fixtures,
)


def _p3_base(n: int = 2):  # type: ignore[no-untyped-def]
    path, windows = _local_replay_fixtures(n)
    fence = _execution_fence(path.target_weights)
    cols = list(path.target_weights.columns)
    return path, windows, fence, cols


def test_p3_validate_window_returns_next_cursor_on_valid_window() -> None:
    path, windows, fence, cols = _p3_base()
    w = windows[0]
    import dataclasses

    import pandas as pd

    mg = w.minute_grid.as_unit("us")
    frames = {k: getattr(w, k).copy() for k in ("highs", "lows", "closes", "marks", "bar_funding", "quote_volumes", "funding_known")}
    for f in frames.values():
        f.index = mg
    dec = pd.DatetimeIndex(w.target_weights.index.as_unit("ns"))
    tw = w.target_weights.copy()
    tw.index = dec
    sig = pd.DatetimeIndex(w.signal_available_at.as_unit("ns"))
    staged = dataclasses.replace(
        w, minute_grid=mg, target_weights=tw, signal_available_at=sig,
        bar_available_at=mg, **frames,
    )
    nxt = _validate_replay_window(
        staged, expected_columns=cols, expected_targets=path.target_weights,
        cursor=0, fence=fence,
    )
    assert nxt == 1


def test_p3_validate_window_signal_before_decision_rejected() -> None:
    import dataclasses

    import pandas as pd

    from src.common.errors import DataIntegrityError

    path, windows, fence, cols = _p3_base()
    w = windows[0]
    dec = w.target_weights.index
    sig = pd.DatetimeIndex(dec.asi8 - 1, tz="UTC")
    bad = dataclasses.replace(w, signal_available_at=sig)
    with pytest.raises(DataIntegrityError, match="signal_available_at must be no earlier than its decision label"):
        _validate_replay_window(bad, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)


def test_p3_validate_window_signal_equal_to_decision_accepted() -> None:
    import dataclasses

    path, windows, fence, cols = _p3_base()
    w = windows[0]
    same = dataclasses.replace(w, signal_available_at=w.target_weights.index)
    nxt = _validate_replay_window(same, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)
    assert nxt == 1


def test_p3_validate_window_unsorted_bar_availability_rejected() -> None:
    import dataclasses

    import pandas as pd

    from src.common.errors import DataIntegrityError

    path, windows, fence, cols = _p3_base()
    w = windows[0]
    arr = w.bar_available_at.as_unit("ns").asi8.copy()
    arr[5], arr[6] = arr[6], arr[5]
    swapped = dataclasses.replace(w, bar_available_at=pd.DatetimeIndex(arr, tz="UTC"))
    with pytest.raises(DataIntegrityError, match="bar_available_at must be increasing and aligned to minute_grid"):
        _validate_replay_window(swapped, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)
    dup = arr.copy()
    dup.sort()
    dup[6] = dup[5]
    dupe = dataclasses.replace(w, bar_available_at=pd.DatetimeIndex(dup, tz="UTC"))
    with pytest.raises(DataIntegrityError, match="bar_available_at must be increasing and aligned to minute_grid"):
        _validate_replay_window(dupe, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)


def test_p3_validate_window_availability_before_bar_label_rejected() -> None:
    import dataclasses

    import pandas as pd

    from src.common.errors import DataIntegrityError

    path, windows, fence, cols = _p3_base()
    w = windows[0]
    arr = w.bar_available_at.as_unit("ns").asi8.copy()
    arr[10] = arr[10] - 1_000_000_000
    early = dataclasses.replace(w, bar_available_at=pd.DatetimeIndex(arr, tz="UTC"))
    with pytest.raises(DataIntegrityError, match="bar_available_at must be no earlier than its bar label"):
        _validate_replay_window(early, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)


def test_p3_validate_window_fence_boundaries_exact() -> None:
    import dataclasses

    import pandas as pd

    from src.common.errors import DataIntegrityError

    path, windows, fence, cols = _p3_base()
    w = windows[0]
    mg = w.minute_grid
    sig = w.signal_available_at
    bar = w.bar_available_at
    mg2 = mg + (fence - mg[-1])
    frames2 = {k: getattr(w, k).copy() for k in ("highs", "lows", "closes", "marks", "bar_funding", "quote_volumes", "funding_known")}
    for f in frames2.values():
        f.index = mg2
    grid_hit = dataclasses.replace(w, minute_grid=mg2, bar_available_at=mg2, **frames2)
    with pytest.raises(DataIntegrityError, match="execution bar labels and signals must be before the fence"):
        _validate_replay_window(grid_hit, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)
    sig_ns = sig.as_unit("ns").asi8.copy()
    sig_ns[-1] = fence.value
    sig_hit = dataclasses.replace(w, signal_available_at=pd.DatetimeIndex(sig_ns, tz="UTC"))
    with pytest.raises(DataIntegrityError, match="execution bar labels and signals must be before the fence"):
        _validate_replay_window(sig_hit, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)
    bar_ns = bar.as_unit("ns").asi8.copy()
    bar_ns[-1] = fence.value
    bar_ok = dataclasses.replace(w, bar_available_at=pd.DatetimeIndex(bar_ns, tz="UTC"))
    assert _validate_replay_window(bar_ok, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence) == 1
    bar_ns[-1] = fence.value + 1
    bar_late = dataclasses.replace(w, bar_available_at=pd.DatetimeIndex(bar_ns, tz="UTC"))
    with pytest.raises(DataIntegrityError, match="completed bar availability must not be beyond the fence"):
        _validate_replay_window(bar_late, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)


def test_p3_validate_window_far_future_label_does_not_overflow() -> None:
    import dataclasses

    import pandas as pd

    from src.common.errors import DataIntegrityError

    path, windows, fence, cols = _p3_base()
    w = windows[0]
    mg = pd.date_range("2300-01-01", periods=len(w.minute_grid), freq="1min", tz="UTC").as_unit("us")
    frames = {k: getattr(w, k).copy() for k in ("highs", "lows", "closes", "marks", "bar_funding", "quote_volumes", "funding_known")}
    for f in frames.values():
        f.index = mg
    far = dataclasses.replace(w, minute_grid=mg, bar_available_at=mg, **frames)
    with pytest.raises(DataIntegrityError, match="execution bar labels and signals must be before the fence"):
        _validate_replay_window(far, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)


def test_p3_validate_window_check_order_preserved() -> None:
    import dataclasses

    import pandas as pd

    from src.common.errors import DataIntegrityError

    path, windows, fence, cols = _p3_base()
    w = windows[0]
    dec = w.target_weights.index
    sig = pd.DatetimeIndex(dec.asi8 - 1, tz="UTC")
    mg2 = w.minute_grid + (fence - w.minute_grid[-1])
    frames2 = {k: getattr(w, k).copy() for k in ("highs", "lows", "closes", "marks", "bar_funding", "quote_volumes", "funding_known")}
    for f in frames2.values():
        f.index = mg2
    both = dataclasses.replace(w, signal_available_at=sig, minute_grid=mg2, bar_available_at=mg2, **frames2)
    with pytest.raises(DataIntegrityError, match="signal_available_at must be no earlier than its decision label"):
        _validate_replay_window(both, expected_columns=cols, expected_targets=path.target_weights, cursor=0, fence=fence)
