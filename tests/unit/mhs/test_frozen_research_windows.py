from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution import ExecutionReplayWindow, InstrumentSettlementEvent
from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V1, FrozenMhsCandidate
from src.mhs.frozen_research_windows import validated_frozen_research_windows

_SETTLEMENT_BARS = 10


def _validated(
    candidate: FrozenMhsCandidate,
    windows: object,
    settlement_bars: int = _SETTLEMENT_BARS,
) -> object:
    return validated_frozen_research_windows(candidate, windows, settlement_bars=settlement_bars)  # type: ignore[arg-type]

_SYMBOLS = ("AAA", "BBB")
_DAY0 = pd.Timestamp("2021-06-01", tz="UTC")


def _candidate(
    n_days: int = 4, zero_tail: bool = False, symbols: tuple[str, ...] = _SYMBOLS
) -> FrozenMhsCandidate:
    labels = pd.DatetimeIndex([_DAY0 + pd.Timedelta(days=i + 1) for i in range(n_days)], tz="UTC")
    weights = pd.DataFrame(0.0, index=labels, columns=list(symbols), dtype="float64")
    if "AAA" in weights.columns:
        weights["AAA"] = 0.05
    if "BBB" in weights.columns:
        weights["BBB"] = -0.05
    if zero_tail:
        weights.loc[labels[2:], :] = 0.0
    avail = pd.DatetimeIndex([label - pd.Timedelta(hours=1) for label in labels], tz="UTC")
    return FrozenMhsCandidate(
        target_weights=weights, signal_available_at=avail, strategy=FROZEN_MHS_TOP20_V1
    )


def _window(
    grid: pd.DatetimeIndex,
    candidate: FrozenMhsCandidate,
    labels: list[pd.Timestamp],
    symbols: tuple[str, ...] | None = None,
    **overrides: object,
) -> ExecutionReplayWindow:
    canon = tuple(candidate.target_weights.columns)
    local = tuple(symbols) if symbols is not None else canon
    frames = {}
    for name in ("highs", "lows", "closes", "marks"):
        frames[name] = pd.DataFrame(100.0, index=grid, columns=list(local), dtype="float64")
    frames["highs"] = pd.DataFrame(101.0, index=grid, columns=list(local), dtype="float64")
    frames["lows"] = pd.DataFrame(99.0, index=grid, columns=list(local), dtype="float64")
    bar_funding = pd.DataFrame(0.0, index=grid, columns=list(local), dtype="float64")
    quote_volumes = pd.DataFrame(1000.0, index=grid, columns=list(local), dtype="float64")
    funding_known = pd.DataFrame(True, index=grid, columns=list(local))
    if labels:
        weights = candidate.target_weights.loc[labels, list(local)].copy()
    else:
        weights = candidate.target_weights.iloc[0:0].loc[:, list(local)].copy()
    avail = pd.DatetimeIndex([candidate.signal_available_at[candidate.target_weights.index.get_loc(l)] for l in labels], tz="UTC")
    params: dict[str, object] = {
        "window_start": grid[0], "window_end": grid[-1], "columns": canon, "symbols": local,
        "minute_grid": grid, "bar_funding": bar_funding,
        "target_weights": weights, "signal_available_at": avail,
        "quote_volumes": quote_volumes, "funding_known": funding_known,
        "bar_available_at": grid + pd.Timedelta(minutes=3),
    }
    params.update(frames)
    params.update(overrides)
    return ExecutionReplayWindow(**params)  # type: ignore[arg-type]


def _grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="3min", tz="UTC")


def _pair(candidate: FrozenMhsCandidate) -> tuple[ExecutionReplayWindow, ExecutionReplayWindow]:
    labels = list(candidate.target_weights.index)
    first = _window(_grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2)), candidate, labels[:2])
    second = _window(_grid(labels[2] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2)), candidate, labels[2:])
    return first, second


def test_close_fallback_accepted_unchanged() -> None:
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    window = _window(grid, candidate, labels[:2], marks=None)
    out = list(_validated(candidate, [window, _pair(candidate)[1]]))
    assert out[0] is window
    assert window.marks is None


def test_invalid_effective_fallback_mark_fails() -> None:
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    held_bar = int(grid.searchsorted(labels[0])) + 1
    bad_close = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    bad_close.iloc[held_bar, 0] = float("nan")
    with pytest.raises(DataIntegrityError, match=r"finite|fallback"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], marks=None, closes=bad_close)]))
    zero_close = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    zero_close.iloc[held_bar + 2, 1] = 0.0
    with pytest.raises(DataIntegrityError, match=r"finite|fallback|positive"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], marks=None, closes=zero_close)]))


def test_dynamic_local_roster_projects_canonical_targets() -> None:
    canon = tuple(f"S{i:03d}" for i in range(84))
    candidate = _candidate(n_days=4, symbols=canon)
    labels = list(candidate.target_weights.index)
    local = canon[:29]
    first = _window(_grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2)), candidate, labels[:2], symbols=local)
    second = _window(_grid(labels[2] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2)), candidate, labels[2:], symbols=local)
    out = list(_validated(candidate, [first, second]))
    assert len(out) == 2
    assert out[0] is first


def test_omitted_nonzero_target_rejected() -> None:
    canon = ("AAA", "BBB", *[f"S{i:03d}" for i in range(4)])
    candidate = _candidate(n_days=4, symbols=canon)
    labels = list(candidate.target_weights.index)
    pruned = tuple(s for s in canon if s != "BBB")
    assert "BBB" not in pruned
    narrow = _window(_grid(labels[0] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2)), candidate, labels, symbols=pruned)
    with pytest.raises(DataIntegrityError, match="nonzero"):
        list(_validated(candidate, [narrow]))


def test_reordered_local_roster_rejected() -> None:
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    canon = tuple(candidate.target_weights.columns)
    rev = tuple(reversed(canon))
    weights = candidate.target_weights.loc[labels[:2], list(rev)].copy()
    frames = {
        name: pd.DataFrame(100.0, index=grid, columns=list(rev), dtype="float64")
        for name in ("highs", "lows", "closes", "marks")
    }
    window = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=canon, symbols=rev,
        minute_grid=grid, highs=frames["highs"], lows=frames["lows"], closes=frames["closes"],
        marks=frames["marks"], bar_funding=pd.DataFrame(0.0, index=grid, columns=list(rev)),
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex([candidate.signal_available_at[0], candidate.signal_available_at[1]], tz="UTC"),
        quote_volumes=pd.DataFrame(1000.0, index=grid, columns=list(rev)),
        funding_known=pd.DataFrame(True, index=grid, columns=list(rev)),
        bar_available_at=grid + pd.Timedelta(minutes=3),
    )
    with pytest.raises(DataIntegrityError, match="canonical ordering"):
        list(_validated(candidate, [window]))


def test_release_to_entry_causality_exact() -> None:
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    equal_avail = pd.DatetimeIndex([labels[0], labels[1]], tz="UTC")
    with pytest.raises(DataIntegrityError, match="precede entry"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], signal_available_at=equal_avail)]))
    future_avail = pd.DatetimeIndex([labels[0] + pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=1)], tz="UTC")
    with pytest.raises(DataIntegrityError, match="precede entry"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], signal_available_at=future_avail)]))


def test_validated_windows_yield_exact_ordered_coverage() -> None:
    """Two disjoint windows stream once in order as the original objects."""
    candidate = _candidate()
    first, second = _pair(candidate)
    out = list(_validated(candidate, [first, second]))
    assert out[0] is first
    assert out[1] is second
    assert [list(w.target_weights.index) for w in out] == [list(candidate.target_weights.index[:2]), list(candidate.target_weights.index[2:])]


def test_validated_windows_reject_duplicate_decision() -> None:
    """A repeated target row fails instead of double-counting the backtest."""
    candidate = _candidate()
    first, second = _pair(candidate)
    labels = list(candidate.target_weights.index)
    duped = _window(_grid(labels[2] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2)), candidate, [labels[2], labels[2]])
    with pytest.raises(DataIntegrityError, match="duplicat"):
        list(_validated(candidate, [first, duped, second]))


def test_validated_windows_reject_omitted_decision() -> None:
    """A missing target row fails instead of shortening the backtest."""
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    wide = _window(_grid(labels[0] - pd.Timedelta(hours=1), labels[-1] + pd.Timedelta(hours=2)), candidate, labels[:2])
    with pytest.raises(DataIntegrityError, match="omitted or truncated"):
        list(_validated(candidate, [wide]))


def test_validated_windows_enforce_release_clock() -> None:
    """Shifted or same-time releases cannot replace the availability."""
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    shifted_avail = pd.DatetimeIndex([labels[0], labels[1]], tz="UTC")
    bad = _window(grid, candidate, labels[:2], signal_available_at=shifted_avail)
    with pytest.raises(DataIntegrityError, match="precede entry"):
        list(_validated(candidate, [bad]))
    short_avail = pd.DatetimeIndex([candidate.signal_available_at[0]], tz="UTC")
    bad_len = _window(grid, candidate, labels[:2], signal_available_at=short_avail)
    with pytest.raises(DataIntegrityError, match="share one length"):
        list(_validated(candidate, [bad_len]))


def test_validated_windows_require_knowledge() -> None:
    """Absent knowledge is no evidence."""
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    no_knowledge = _window(grid, candidate, labels[:2], funding_known=None)
    with pytest.raises(DataIntegrityError, match="explicit marks"):
        list(_validated(candidate, [no_knowledge]))


def test_validated_windows_reject_nonfinite_or_impossible_source() -> None:
    """Infinite prices, non-positive marks, and negative volumes fail closed."""
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    held_bar = int(grid.searchsorted(labels[0])) + 1
    bad_close = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    bad_close.iloc[held_bar, 0] = float("inf")
    with pytest.raises(DataIntegrityError, match="finite"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], closes=bad_close)]))
    bad_mark = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    bad_mark.iloc[held_bar + 2, 1] = 0.0
    with pytest.raises(DataIntegrityError, match="marks"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], marks=bad_mark)]))
    bad_qv = pd.DataFrame(1000.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    bad_qv.iloc[held_bar + 4, 0] = -2.0
    with pytest.raises(DataIntegrityError, match="volumes"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], quote_volumes=bad_qv)]))


def test_validated_windows_reject_misaligned_frames() -> None:
    """Frames off the grid or missing an active symbol cannot be replayed."""
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    short = pd.DataFrame(100.0, index=grid[:-5], columns=list(_SYMBOLS), dtype="float64")
    with pytest.raises(DataIntegrityError, match="align to the grid"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], closes=short)]))
    narrow = pd.DataFrame(100.0, index=grid, columns=["AAA"], dtype="float64")
    with pytest.raises(DataIntegrityError, match="align to the grid"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], closes=narrow)]))


def test_validated_windows_reject_bad_clocks() -> None:
    """Naive, non-UTC, unordered, or off-grid clocks fail before economics."""
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))

    def _bad(**overrides: object) -> ExecutionReplayWindow:
        return _window(grid, candidate, labels[:2], **overrides)

    naive_start = grid[0].tz_localize(None)
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(window_start=naive_start)]))
    eastern = grid[0].tz_convert(timezone(timedelta(hours=-5)))
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(window_start=eastern)]))
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(window_start=grid[-1], window_end=grid[0])]))
    single = grid[:1]
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(
            minute_grid=single, window_start=single[0], window_end=single[0] + pd.Timedelta(minutes=3),
            bar_available_at=single + pd.Timedelta(minutes=3),
        )]))
    dup = grid.append(grid[:1])
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(minute_grid=dup)]))
    coarse = pd.date_range(grid[0], grid[-1], freq="5min", tz="UTC")
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(minute_grid=coarse)]))
    shifted = grid + pd.Timedelta(hours=9)
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(minute_grid=shifted)]))
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(bar_available_at=None)]))
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(bar_available_at=grid[:-1] + pd.Timedelta(minutes=3))]))
    early_avail = grid - pd.Timedelta(minutes=3)
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(bar_available_at=early_avail)]))
    non_utc_grid = grid.tz_convert(timezone(timedelta(hours=-5)))
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(minute_grid=non_utc_grid)]))
    naive_grid = grid.tz_localize(None)
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(minute_grid=naive_grid)]))
    naive_avail = grid.tz_localize(None) + pd.Timedelta(minutes=3)
    with pytest.raises(DataIntegrityError, match="clocks"):
        list(_validated(candidate, [_bad(bar_available_at=naive_avail)]))


def test_validated_windows_reject_unknown_or_altered_targets() -> None:
    """Labels outside the candidate and altered values never enter the replay."""
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    outsider = pd.Timestamp("2021-07-01", tz="UTC")
    alien_weights = pd.DataFrame({"AAA": [0.05], "BBB": [-0.05]}, index=pd.DatetimeIndex([outsider], tz="UTC"))
    alien_avail = pd.DatetimeIndex([outsider - pd.Timedelta(hours=1)], tz="UTC")
    with pytest.raises(DataIntegrityError, match="unknown to or duplicated"):
        list(_validated(candidate, [_window(
            grid, candidate, labels[:1], target_weights=alien_weights, signal_available_at=alien_avail)]))
    doped = candidate.target_weights.loc[labels[:2]].copy()
    doped.iloc[0, 0] += 0.5
    with pytest.raises(DataIntegrityError, match="equal the frozen candidate"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], target_weights=doped)]))


def test_validated_windows_reject_unresolved_final_grid() -> None:
    """A final decision with no execution horizon after it is not completable."""
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    clipped = _grid(labels[0] - pd.Timedelta(hours=1), labels[-1])
    only = _window(clipped, candidate, labels, window_end=labels[-1])
    with pytest.raises(DataIntegrityError, match="execution horizon"):
        list(_validated(candidate, [only]))
    with pytest.raises(DataIntegrityError, match="execution horizon"):
        list(_validated(candidate, []))


def test_validated_windows_allow_zero_target_roster_exit() -> None:
    """A symbol may leave after its later target rows are zero."""
    candidate = _candidate(zero_tail=True)
    labels = list(candidate.target_weights.index)
    first = _window(_grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2)), candidate, labels[:2])
    narrow_grid = _grid(labels[2] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2))
    dropped = _window(narrow_grid, candidate, labels[2:], symbols=("BBB",))
    out = list(_validated(candidate, [first, dropped]))
    assert len(out) == 2


def test_validated_windows_allow_evidenced_settlement_exit() -> None:
    """An evidenced settlement lets a targeted symbol leave the roster."""
    candidate = _candidate(zero_tail=True)
    labels = list(candidate.target_weights.index)
    first = _window(_grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2)), candidate, labels[:2])
    event = InstrumentSettlementEvent(
        event_id="delist-aaa", symbol="AAA", effective_at=labels[2],
        available_at=labels[2] + pd.Timedelta(hours=1),
        settlement_price=95.0, fee_bps=5.0, source_digest="archive",
    )
    narrow_grid = _grid(labels[2] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2))
    settled = _window(narrow_grid, candidate, labels[2:], symbols=("BBB",), settlement_events=(event,))
    out = list(_validated(candidate, [first, settled]))
    assert len(out) == 2
    assert out[1] is settled


def test_validated_windows_reject_column_and_roster_mismatch() -> None:
    """Canonical columns and nonzero-target coverage are enforced per window."""
    candidate = _candidate()
    first, _ = _pair(candidate)
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    swapped = _window(grid, candidate, labels[:2], columns=("BBB", "AAA"))
    with pytest.raises(DataIntegrityError, match="canonical column order"):
        list(_validated(candidate, [swapped]))
    assert first.columns == _SYMBOLS
    thin = _window(grid, candidate, labels[:2], symbols=("AAA",))
    with pytest.raises(DataIntegrityError, match="nonzero"):
        list(_validated(candidate, [thin]))


def test_validated_windows_reject_out_of_order_stream() -> None:
    """Unsorted labels and backward windows never enter the replay."""
    candidate = _candidate()
    first, second = _pair(candidate)
    with pytest.raises(DataIntegrityError, match="chronological order"):
        list(_validated(candidate, [second, first]))
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    shuffled = _window(grid, candidate, [labels[1], labels[0]])
    with pytest.raises(DataIntegrityError, match="chronological order"):
        list(_validated(candidate, [shuffled]))


def test_validated_windows_stream_without_retaining_source() -> None:
    """Windows are consumed once; the first read never touches later frames."""
    from collections.abc import Iterator

    candidate = _candidate()
    first, second = _pair(candidate)
    reads: list[int] = []

    def _source() -> Iterator[ExecutionReplayWindow]:
        for pos, window in enumerate([first, second]):
            reads.append(pos)
            yield window

    stream = _validated(candidate, _source())
    head = next(stream)
    assert head is first
    assert reads == [0]
    assert np.isfinite(head.closes.to_numpy(dtype="float64")).all()


def test_validated_windows_reject_empty_candidate() -> None:
    """A candidate without target rows cannot anchor any window."""
    empty = FrozenMhsCandidate(
        target_weights=pd.DataFrame(columns=list(_SYMBOLS), dtype="float64"),
        signal_available_at=pd.DatetimeIndex([], tz="UTC"), strategy=FROZEN_MHS_TOP20_V1,
    )
    with pytest.raises(DataIntegrityError, match="at least one target row"):
        list(_validated(empty, []))


def test_local_roster_subset_and_order_branches() -> None:
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2))
    empty_weights = candidate.target_weights.loc[labels, []].copy()
    empty_window = _window(grid, candidate, labels, symbols=_SYMBOLS, target_weights=empty_weights)
    with pytest.raises(DataIntegrityError, match="non-empty"):
        list(_validated(candidate, [empty_window]))
    alien_weights = candidate.target_weights.loc[labels[:2]].copy()
    alien_weights.columns = ["AAA", "ZZZ"]
    alien = _window(grid, candidate, labels[:2], target_weights=alien_weights)
    with pytest.raises(DataIntegrityError, match="subset"):
        list(_validated(candidate, [alien]))
    ordered_cols = _window(grid, candidate, labels[:2])
    reordered_syms = __import__("dataclasses").replace(ordered_cols, symbols=("BBB", "AAA"))
    with pytest.raises(DataIntegrityError, match="canonical ordering"):
        list(_validated(candidate, [reordered_syms]))
    mismatched = __import__("dataclasses").replace(ordered_cols, symbols=("AAA", "BBB"),
        target_weights=candidate.target_weights.loc[labels[:2], ["AAA"]].copy())
    with pytest.raises(DataIntegrityError, match="same local roster"):
        list(_validated(candidate, [mismatched]))
    thin_target = candidate.target_weights.loc[labels[:2], ["AAA"]].copy()
    thin_both = _window(grid, candidate, labels[:2], symbols=("AAA",), target_weights=thin_target)
    with pytest.raises(DataIntegrityError, match="nonzero"):
        list(_validated(candidate, [thin_both]))
    narrow_marks = pd.DataFrame(100.0, index=grid, columns=["AAA"], dtype="float64")
    with pytest.raises(DataIntegrityError, match="align to the grid"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], marks=narrow_marks)]))


def _idle_candidate() -> FrozenMhsCandidate:
    base = _candidate()
    weights = base.target_weights.copy()
    weights.loc[list(weights.index)[:2]] = 0.0
    return FrozenMhsCandidate(
        target_weights=weights, signal_available_at=base.signal_available_at, strategy=base.strategy
    )


def test_unheld_gap_passes_through() -> None:
    candidate = _idle_candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    gappy = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    gappy.iloc[10:20, 0] = float("nan")
    first = _window(grid, candidate, labels[:2], closes=gappy)
    second = _window(_grid(labels[2] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2)), candidate, labels[2:])
    out = list(_validated(candidate, [first, second]))
    assert out[0] is first


def test_held_gap_fails_closed() -> None:
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    bad_close = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    bad_close.iloc[int(grid.searchsorted(labels[0])), 0] = float("nan")
    with pytest.raises(DataIntegrityError, match="finite"):
        list(_validated(candidate, [_window(grid, candidate, labels[:2], closes=bad_close)]))


def test_settlement_window_gap_fails_closed() -> None:
    candidate = _candidate(zero_tail=True)
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2))
    zero_bar = int(grid.searchsorted(labels[2]))
    bad_close = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    bad_close.iloc[zero_bar + 3, 0] = float("nan")
    with pytest.raises(DataIntegrityError, match="finite"):
        list(_validated(candidate, [_window(grid, candidate, labels, closes=bad_close)], settlement_bars=_SETTLEMENT_BARS))


def test_post_settlement_gap_passes() -> None:
    candidate = _candidate(zero_tail=True)
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2))
    zero_bar = int(grid.searchsorted(labels[2]))
    gappy = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    gappy.iloc[zero_bar + _SETTLEMENT_BARS + 5, 0] = float("nan")
    out = list(_validated(candidate, [_window(grid, candidate, labels)], settlement_bars=_SETTLEMENT_BARS))
    assert len(out) == 1


def test_carried_hold_across_pieces_fails() -> None:
    candidate = _candidate()
    labels = list(candidate.target_weights.index)
    first = _window(_grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2)), candidate, labels[:2])
    grid = _grid(labels[2] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2))
    bad_close = pd.DataFrame(100.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    bad_close.iloc[0, 0] = float("nan")
    second = _window(grid, candidate, labels[2:], closes=bad_close)
    with pytest.raises(DataIntegrityError, match="finite"):
        list(_validated(candidate, [first, second]))


def test_structure_checked_for_unheld_roster() -> None:
    base = _candidate()
    weights = base.target_weights.copy()
    weights.loc[:, :] = 0.0
    flat = FrozenMhsCandidate(
        target_weights=weights, signal_available_at=base.signal_available_at, strategy=base.strategy
    )
    labels = list(flat.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2))
    narrow = pd.DataFrame(100.0, index=grid, columns=["AAA"], dtype="float64")
    with pytest.raises(DataIntegrityError, match="align to the grid"):
        list(_validated(flat, [_window(grid, flat, labels[:2], closes=narrow)]))


def test_negative_volume_only_matters_when_held() -> None:
    base = _candidate()
    weights = base.target_weights.copy()
    weights["BBB"] = 0.0
    candidate = FrozenMhsCandidate(
        target_weights=weights, signal_available_at=base.signal_available_at, strategy=base.strategy
    )
    labels = list(candidate.target_weights.index)
    grid = _grid(labels[0] - pd.Timedelta(hours=1), labels[3] + pd.Timedelta(hours=2))
    idle_bad = pd.DataFrame(1000.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    idle_bad.iloc[40, 1] = -2.0
    out = list(_validated(candidate, [_window(grid, candidate, labels, quote_volumes=idle_bad)]))
    assert len(out) == 1
    held_bad = pd.DataFrame(1000.0, index=grid, columns=list(_SYMBOLS), dtype="float64")
    held_bad.iloc[40, 0] = -2.0
    with pytest.raises(DataIntegrityError, match="volumes"):
        list(_validated(candidate, [_window(grid, candidate, labels, quote_volumes=held_bad)]))


def test_healthy_sequence_passes_unchanged() -> None:
    candidate = _candidate()
    first, second = _pair(candidate)
    out = list(_validated(candidate, [first, second], settlement_bars=_SETTLEMENT_BARS))
    assert len(out) == 2
    assert out[0] is first
    assert out[1] is second
