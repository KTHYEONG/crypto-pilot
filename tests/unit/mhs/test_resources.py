"""Tests for the MHS RAM-budget guard primitives."""

from __future__ import annotations

import dataclasses
import multiprocessing
import time

import psutil
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


def _child_hold_private_bytes(ready_queue, release_queue, nbytes: int) -> None:
    """Fork child: allocate ``nbytes`` private pages and hold until released."""
    buf = bytearray(nbytes)
    for i in range(0, nbytes, 4096):
        buf[i] = 1  # touch every page so the allocation becomes resident
    ready_queue.put("held")
    release_queue.get(timeout=60)


class TestProcessTreeMemoryStats:
    """SCENARIO_MHS_PERF_P0_03_TREE_MEMORY_METRIC."""

    def test_has_no_sum_of_rss_field(self) -> None:
        """Sum-of-RSS is deliberately absent: it double-counts COW-shared pages."""
        names = {f.name for f in dataclasses.fields(resources.ProcessTreeMemoryStats)}
        assert names == {
            "tree_pss_peak_bytes",
            "tree_uss_peak_bytes",
            "min_system_available_bytes",
            "max_concurrent_procs",
            "samples_taken",
        }
        assert not any("rss" in name for name in names)

    def test_fork_child_private_allocation_exceeds_parent_rss(self) -> None:
        """A child's private 200 MB shows up in tree PSS but not parent RSS."""
        ctx = multiprocessing.get_context("fork")
        ready = ctx.Queue()
        release = ctx.Queue()
        # Pre-fork baselines. The >= 150 MB margin is asserted against the
        # parent's own PSS: RSS double-counts COW-shared pages (the very
        # defect this metric exists to fix), so PSS attribution is the stable
        # reference -- the child contributes its ~200 MB private set on top.
        parent_info_before = psutil.Process().memory_full_info()

        sampler = resources._TreeMemorySampler(interval_seconds=0.05)
        sampler.start()
        try:
            proc = ctx.Process(
                target=_child_hold_private_bytes,
                args=(ready, release, 200 * 2**20),
            )
            proc.start()
            assert ready.get(timeout=60) == "held"
            time.sleep(0.6)  # guarantee several samples catch the allocation
        finally:
            release.put("done")
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
            stats = sampler.stop()

        assert stats.samples_taken >= 1
        assert (
            stats.tree_pss_peak_bytes
            - int(getattr(parent_info_before, "pss", parent_info_before.rss))
            >= 150 * 2**20
        )

    def test_min_system_available_below_total(self) -> None:
        """The available floor can never exceed total memory."""
        ctx = multiprocessing.get_context("fork")
        sampler = resources._TreeMemorySampler(interval_seconds=0.05)
        sampler.start()
        try:
            time.sleep(0.2)
        finally:
            stats = sampler.stop()
        assert stats.min_system_available_bytes < psutil.virtual_memory().total
        assert stats.max_concurrent_procs >= 1

    def test_psutil_failure_never_raises_into_the_run(self, monkeypatch) -> None:
        """An injected psutil failure yields samples_taken >= 0 with no raise."""
        def _boom(*_args, **_kwargs):
            raise RuntimeError("injected psutil failure")

        monkeypatch.setattr(resources.psutil, "virtual_memory", _boom)
        monkeypatch.setattr(resources.psutil, "Process", _boom)

        sampler = resources._TreeMemorySampler(interval_seconds=0.05)
        sampler.start()
        time.sleep(0.15)
        stats = sampler.stop()
        assert stats.samples_taken >= 0


def test_worker_plan_observer_records_stage_and_budget() -> None:
    """record_worker_plan stores granted workers plus the per-worker budget."""
    recorder = resources._StageRecorder(log_run=False)
    observer = resources._worker_plan_observer(recorder, "books", 3 * 2**30)
    assert observer is not None
    observer("ram_guard", 3, 2, (10 * 2**30), (1 * 2**30))

    plan = recorder.worker_plan
    assert plan["books"] == 2
    assert plan["books_per_worker_bytes"] == 3 * 2**30


def test_worker_plan_observer_none_recorder_is_none() -> None:
    """No recorder means no observer: existing call sites stay untouched."""
    assert resources._worker_plan_observer(None, "books") is None
