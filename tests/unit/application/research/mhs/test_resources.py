"""Tests for the MHS RAM-budget guard primitives."""

from __future__ import annotations

import pytest

from src.application.research.mhs import resources
from src.common.errors import DataIntegrityError


def test_resolve_ram_budget_guard_disabled_returns_none_pair() -> None:
    """ram_guard=False yields the legacy unlimited (None, None) semantics."""
    assert resources._resolve_ram_budget(max_rss_bytes=123, ram_guard=False) == (None, None)


def test_resolve_ram_budget_explicit_max_rss_overrides_auto_budget(monkeypatch) -> None:
    """An explicit max_rss_bytes wins over the auto total*fraction budget."""
    class _Mem:
        total = 100_000_000_000

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())

    budget, reserve = resources._resolve_ram_budget(max_rss_bytes=42, ram_guard=True)

    assert budget == 42
    assert reserve is not None
    assert reserve > 0


def test_resolve_ram_budget_auto_budget_from_total(monkeypatch) -> None:
    """With no explicit cap, the budget is total * RAM_BUDGET_FRACTION."""
    class _Mem:
        total = 1_000_000_000

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())

    budget, reserve = resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True)

    assert budget == int(1_000_000_000 * resources.RAM_BUDGET_FRACTION)
    assert reserve == max(
        int(1_000_000_000 * resources.RAM_RESERVE_FRACTION),
        resources.RAM_RESERVE_FLOOR_BYTES,
    )


def test_resolve_ram_budget_psutil_failure_disables_guard(monkeypatch) -> None:
    """A psutil failure is an observational failure: guard disables, never raises."""
    def _raise() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _raise())

    assert resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True) == (None, None)


def test_resolve_ram_budget_nonpositive_total_disables_guard(monkeypatch) -> None:
    """A non-positive reported total disables the guard rather than dividing by it."""
    class _Mem:
        total = 0

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())

    assert resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True) == (None, None)


def test_assert_stage_rss_budget_noop_when_both_none() -> None:
    """Both budget and reserve None makes the guard a no-op."""
    resources._assert_stage_rss_budget("stage", None, None)


def test_assert_stage_rss_budget_raises_when_rss_exceeds_budget(monkeypatch) -> None:
    """A current RSS above the budget raises DataIntegrityError naming the stage."""
    monkeypatch.setattr(resources, "_current_rss_bytes", lambda: 1000)

    with pytest.raises(DataIntegrityError, match="my_stage"):
        resources._assert_stage_rss_budget("my_stage", budget_bytes=500, reserve_bytes=None)


def test_assert_stage_rss_budget_passes_when_rss_within_budget(monkeypatch) -> None:
    """A current RSS at or below the budget does not raise."""
    monkeypatch.setattr(resources, "_current_rss_bytes", lambda: 100)

    resources._assert_stage_rss_budget("my_stage", budget_bytes=500, reserve_bytes=None)


def test_assert_stage_rss_budget_raises_when_reserve_breached(monkeypatch) -> None:
    """Available system memory below the reserve raises DataIntegrityError."""
    class _Mem:
        available = 10

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())

    with pytest.raises(DataIntegrityError, match="reserve"):
        resources._assert_stage_rss_budget("my_stage", budget_bytes=None, reserve_bytes=1000)


def test_assert_stage_rss_budget_swallows_psutil_failure_on_reserve_probe(monkeypatch) -> None:
    """A psutil failure while probing the reserve is swallowed (observational)."""
    def _raise() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _raise())

    resources._assert_stage_rss_budget("my_stage", budget_bytes=None, reserve_bytes=1000)


def test_assert_execution_rss_budget_noop_when_both_none() -> None:
    """Neither budget nor reserve set makes the guard a no-op."""
    resources._assert_execution_rss_budget("stage", None, completed_windows=3)


def test_assert_execution_rss_budget_raises_with_window_context(monkeypatch) -> None:
    """An exceeded budget raises with stage, observed rss, budget, and window count."""
    monkeypatch.setattr(resources, "_current_rss_bytes", lambda: 2000)

    with pytest.raises(DataIntegrityError, match="completed_windows=7"):
        resources._assert_execution_rss_budget(
            "exec_stage", budget=1000, completed_windows=7,
        )


def test_assert_execution_rss_budget_reserve_breach_maps_to_rss_budget_message(monkeypatch) -> None:
    """A system-reserve breach uses the stable 'rss budget'-prefixed message."""
    class _Mem:
        available = 5

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(resources, "_current_rss_bytes", lambda: 0)

    with pytest.raises(DataIntegrityError, match="execution RSS budget"):
        resources._assert_execution_rss_budget(
            "exec_stage", budget=None, completed_windows=1, reserve_bytes=100,
        )


def test_stage_recorder_records_ordered_measurements_and_tracks_peak() -> None:
    """Records accumulate in call order and the peak RSS never decreases."""
    recorder = resources._StageRecorder(log_run=False)

    recorder.record("stage_a", n_symbols=3)
    recorder.record("stage_b", n_symbols=5)

    assert [r.stage for r in recorder.records] == ["stage_a", "stage_b"]
    assert recorder.records[0].n_symbols == 3
    assert recorder.records[1].peak_rss_bytes >= recorder.records[0].peak_rss_bytes


def test_stage_recorder_absorb_merges_records_and_folds_peak() -> None:
    """absorb() appends frozen records and folds their peak into the recorder's own."""
    recorder = resources._StageRecorder(log_run=False)
    recorder.record("parent_stage")

    other = resources._StageRecorder(log_run=False)
    other.record("child_stage")
    child_records = other.records

    recorder.absorb(child_records)

    assert recorder.records[-1].stage == "child_stage"
    assert recorder._peak_rss >= max(r.peak_rss_bytes or 0 for r in child_records)


def test_stage_recorder_absorb_empty_is_noop() -> None:
    """absorb(()) does not append anything or reset timing state unexpectedly."""
    recorder = resources._StageRecorder(log_run=False)
    recorder.record("only_stage")

    recorder.absorb(())

    assert len(recorder.records) == 1


def test_peak_rss_bytes_empty_measurements_returns_none() -> None:
    """No measurements yields None rather than raising on an empty max()."""
    assert resources._peak_rss_bytes(()) is None


def test_peak_rss_bytes_returns_max_rss_across_measurements() -> None:
    """The reported peak is the maximum rss_bytes across all measurements."""
    recorder = resources._StageRecorder(log_run=False)
    recorder.record("a")
    recorder.record("b")

    peak = resources._peak_rss_bytes(recorder.records)

    assert peak == max(r.rss_bytes for r in recorder.records)
