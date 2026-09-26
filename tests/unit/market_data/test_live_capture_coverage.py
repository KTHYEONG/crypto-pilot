from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.market_data.streams.coverage import CoverageTracker, load_coverage


def test_quiet_connected_stretch_is_attested(tmp_path: Path) -> None:
    """A quiet connected stretch still attests its interval."""
    tracker = CoverageTracker("liquidations", tmp_path)
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:00:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:07:00Z"))
    tracker.flush()
    out = load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T11:00:00Z"))
    assert len(out) == 1
    assert out.iloc[0]["start"] == pd.Timestamp("2026-09-22T10:00:00Z")
    assert out.iloc[0]["end"] == pd.Timestamp("2026-09-22T10:07:00Z")


def test_error_closes_at_last_success(tmp_path: Path) -> None:
    """Transport errors close coverage at the last success, leaving a gap."""
    tracker = CoverageTracker("liquidations", tmp_path)
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:00:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:03:00Z"))
    tracker.mark_error(pd.Timestamp("2026-09-22T10:05:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:09:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:10:00Z"))
    tracker.flush()
    out = load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T11:00:00Z"))
    assert len(out) == 2
    assert out.iloc[0]["start"] == pd.Timestamp("2026-09-22T10:00:00Z")
    assert out.iloc[0]["end"] == pd.Timestamp("2026-09-22T10:03:00Z")
    assert out.iloc[1]["start"] == pd.Timestamp("2026-09-22T10:09:00Z")
    assert out.iloc[1]["end"] == pd.Timestamp("2026-09-22T10:10:00Z")


def test_incremental_flushes_merge_back(tmp_path: Path) -> None:
    """Consecutive flushes of one segment merge into a single interval."""
    tracker = CoverageTracker("liquidations", tmp_path)
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:00:00Z"))
    tracker.flush()
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:02:00Z"))
    tracker.flush()
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:04:00Z"))
    tracker.flush()
    out = load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T11:00:00Z"))
    assert len(out) == 1
    assert out.iloc[0]["start"] == pd.Timestamp("2026-09-22T10:00:00Z")
    assert out.iloc[0]["end"] == pd.Timestamp("2026-09-22T10:04:00Z")


def test_idle_flush_writes_nothing(tmp_path: Path) -> None:
    """Flushing without any success creates no files."""
    tracker = CoverageTracker("liquidations", tmp_path)
    assert tracker.flush() == []
    assert list(tmp_path.rglob("*")) == []


def test_midnight_split(tmp_path: Path) -> None:
    """Intervals spanning midnight split across both day files."""
    tracker = CoverageTracker("liquidations", tmp_path)
    tracker.mark_ok(pd.Timestamp("2026-09-22T23:58:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-23T00:02:00Z"))
    tracker.flush()
    day_d = tmp_path / "coverage" / "liquidations" / "20260922.jsonl"
    day_next = tmp_path / "coverage" / "liquidations" / "20260923.jsonl"
    assert day_d.exists()
    assert day_next.exists()
    out = load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T23:00:00Z"), end=pd.Timestamp("2026-09-23T01:00:00Z"))
    assert len(out) == 1
    assert out.iloc[0]["start"] == pd.Timestamp("2026-09-22T23:58:00Z")
    assert out.iloc[0]["end"] == pd.Timestamp("2026-09-23T00:02:00Z")


def test_naive_timestamp_rejected(tmp_path: Path) -> None:
    """Naive timestamps are rejected."""
    tracker = CoverageTracker("liquidations", tmp_path)
    with pytest.raises(ValueError, match="tz-aware"):
        tracker.mark_ok(pd.Timestamp("2026-09-22 10:00:00"))
    with pytest.raises(ValueError, match="tz-aware"):
        tracker.mark_error(pd.Timestamp("2026-09-22 10:00:00"))


def test_coverage_tracker_edge_cases(tmp_path: Path) -> None:
    """Decreasing stamps, error-only flushes, and closed flushes behave."""
    tracker = CoverageTracker("liquidations", tmp_path)
    tracker.mark_error(pd.Timestamp("2026-09-22T10:00:00Z"))
    assert tracker.flush() == []
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:05:00Z"))
    with pytest.raises(ValueError, match="non-decreasing"):
        tracker.mark_ok(pd.Timestamp("2026-09-22T10:04:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:06:00Z"))
    tracker.mark_error(pd.Timestamp("2026-09-22T10:07:00Z"))
    tracker.flush()
    out = load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:10:00Z"))
    assert len(out) == 1
    assert load_coverage(tmp_path, "missing", start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:10:00Z")).empty
    assert load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T10:10:00Z"), end=pd.Timestamp("2026-09-22T10:00:00Z")).empty


def test_load_coverage_skips_malformed_lines(tmp_path: Path) -> None:
    """Malformed and blank JSONL lines are ignored."""
    tracker = CoverageTracker("liquidations", tmp_path)
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:00:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-22T10:02:00Z"))
    tracker.flush()
    day_file = tmp_path / "coverage" / "liquidations" / "20260922.jsonl"
    with open(day_file, "a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("not json\n")
    out = load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T11:00:00Z"))
    assert len(out) == 1
    assert load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T12:00:00Z"), end=pd.Timestamp("2026-09-22T13:00:00Z")).empty
    (tmp_path / "coverage" / "liquidations" / "20260923.jsonl").mkdir()
    out2 = load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-24T00:00:00Z"))
    assert len(out2) == 1


def test_coverage_discard_unflushed_withdraws_pending_only(tmp_path: Path) -> None:
    """Discarding drops the open span; flushed history and later segments survive."""
    import pandas as pd

    from src.market_data.streams.coverage import CoverageTracker, load_coverage

    tracker = CoverageTracker("liquidations", tmp_path)
    t0 = pd.Timestamp("2026-09-22T10:00:00Z")
    t1 = t0 + pd.Timedelta(seconds=10)
    t2 = t1 + pd.Timedelta(seconds=10)
    tracker.mark_ok(t0)
    tracker.mark_ok(t1)
    tracker.flush()
    tracker.mark_ok(t2)
    tracker.discard_unflushed(t2)
    assert tracker.flush() == []
    t3 = t2 + pd.Timedelta(seconds=10)
    t4 = t3 + pd.Timedelta(seconds=10)
    tracker.mark_ok(t3)
    tracker.mark_ok(t4)
    tracker.flush()
    out = load_coverage(
        tmp_path, "liquidations",
        start=pd.Timestamp("2026-09-22T09:00:00Z"), end=pd.Timestamp("2026-09-22T12:00:00Z"),
    )
    assert len(out) == 2
    assert out.iloc[0]["start"] == t0
    assert out.iloc[0]["end"] == t1
    assert out.iloc[1]["start"] == t3
    assert out.iloc[1]["end"] == t4


def test_coverage_discard_unflushed_rejects_naive_timestamp(tmp_path: Path) -> None:
    """Naive discard instants fail closed."""
    import pandas as pd
    import pytest

    from src.market_data.streams.coverage import CoverageTracker

    tracker = CoverageTracker("liquidations", tmp_path)
    with pytest.raises(ValueError, match="tz-aware"):
        tracker.discard_unflushed(pd.Timestamp("2026-09-22 10:00:00"))


def test_snapshot_state_requires_flushed_segments(tmp_path) -> None:
    """snapshot_state raises while closed segments are unflushed."""
    tracker = CoverageTracker("liquidations", tmp_path)
    tracker.mark_ok(pd.Timestamp("2026-09-26T10:00:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-26T10:01:00Z"))
    tracker.flush()
    tracker.mark_ok(pd.Timestamp("2026-09-26T10:02:00Z"))
    tracker.mark_error(pd.Timestamp("2026-09-26T10:03:00Z"))
    with pytest.raises(ValueError, match="unflushed closed"):
        tracker.snapshot_state()
    tracker.flush()
    state = tracker.snapshot_state()
    assert state.open_last is None
    assert state.cursor is None


def test_restore_continues_open_segment(tmp_path) -> None:
    """A restored tracker extends from the cursor; a fresh state opens anew."""
    tracker = CoverageTracker("liquidations", tmp_path)
    tracker.mark_ok(pd.Timestamp("2026-09-26T10:00:00Z"))
    tracker.mark_ok(pd.Timestamp("2026-09-26T10:01:00Z"))
    tracker.flush()
    state = tracker.snapshot_state()
    assert state.open_last == pd.Timestamp("2026-09-26T10:01:00Z")
    restored = CoverageTracker.restore("liquidations", tmp_path, state)
    restored.mark_ok(pd.Timestamp("2026-09-26T10:02:00Z"))
    restored.flush()
    merged = load_coverage(tmp_path, "liquidations", start=pd.Timestamp("2026-09-26T10:00:00Z"),
                           end=pd.Timestamp("2026-09-26T11:00:00Z"))
    assert len(merged) == 1
    assert merged.iloc[0]["start"] == pd.Timestamp("2026-09-26T10:00:00Z")
    assert merged.iloc[0]["end"] == pd.Timestamp("2026-09-26T10:02:00Z")
