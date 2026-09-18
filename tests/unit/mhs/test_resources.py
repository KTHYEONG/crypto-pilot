"""Tests for the MHS RAM-budget guard primitives."""

from __future__ import annotations

import dataclasses
import io
import multiprocessing
import time

import psutil
import pytest

from src.mhs import resources
from src.common.errors import DataIntegrityError


def _isolate_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "_read_cgroup_limit_bytes", lambda: None)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)


def _admit_all(monkeypatch: pytest.MonkeyPatch, *, available: int = 10**12) -> None:
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: available)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)


def test_resolve_ram_budget_guard_disabled_returns_none_pair() -> None:
    """ram_guard=False yields the legacy unlimited (None, None) semantics."""
    assert resources._resolve_ram_budget(max_rss_bytes=123, ram_guard=False) == (None, None)


def test_resolve_ram_budget_explicit_max_rss_overrides_auto_budget(monkeypatch) -> None:
    """An explicit max_rss_bytes wins over the auto total*fraction budget."""
    class _Mem:
        total = 100_000_000_000

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())
    _isolate_cgroup(monkeypatch)

    budget, reserve = resources._resolve_ram_budget(max_rss_bytes=42, ram_guard=True)

    assert budget == 42
    assert reserve is not None
    assert reserve > 0


def test_resolve_ram_budget_explicit_budget_capped_by_adoption_ceiling(monkeypatch) -> None:
    """An explicit budget above the tree ceiling is capped at 2.5GiB."""
    class _Mem:
        total = 100_000_000_000

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())
    _isolate_cgroup(monkeypatch)

    budget, _ = resources._resolve_ram_budget(max_rss_bytes=10 * 2**30, ram_guard=True)

    assert budget == resources.MHS_TREE_PSS_BUDGET_BYTES


def test_resolve_ram_budget_invalid_explicit_budget_rejected(monkeypatch) -> None:
    """A non-positive explicit budget raises ValueError."""
    _isolate_cgroup(monkeypatch)

    with pytest.raises(ValueError, match="max_rss_bytes"):
        resources._resolve_ram_budget(max_rss_bytes=0, ram_guard=True)
    with pytest.raises(ValueError, match="max_rss_bytes"):
        resources._resolve_ram_budget(max_rss_bytes=-8, ram_guard=True)


def test_resolve_ram_budget_auto_budget_from_total(monkeypatch) -> None:
    """With no explicit cap, the budget is total * RAM_BUDGET_FRACTION."""
    class _Mem:
        total = 1_000_000_000

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())
    _isolate_cgroup(monkeypatch)

    budget, reserve = resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True)

    assert budget == int(1_000_000_000 * resources.RAM_BUDGET_FRACTION)
    assert reserve == max(
        int(1_000_000_000 * resources.RAM_RESERVE_FRACTION),
        resources.RAM_RESERVE_FLOOR_BYTES,
        resources.MHS_AVAILABLE_FLOOR_BYTES,
    )
    assert reserve >= resources.MHS_AVAILABLE_FLOOR_BYTES


def test_resolve_ram_budget_telemetry_failure_fails_closed(monkeypatch) -> None:
    """A psutil failure with the guard on fails closed, never bypasses."""
    def _raise() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _raise())

    with pytest.raises(DataIntegrityError, match="telemetry unavailable"):
        resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True)


def test_resolve_ram_budget_nonpositive_total_fails_closed(monkeypatch) -> None:
    """A non-positive reported total fails closed rather than disabling the guard."""
    class _Mem:
        total = 0

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())

    with pytest.raises(DataIntegrityError, match="telemetry unavailable"):
        resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True)


def test_resolve_ram_budget_impossible_work_fails_closed(monkeypatch) -> None:
    """A trivially small effective total leaves no usable budget."""
    class _Mem:
        total = 1

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())
    _isolate_cgroup(monkeypatch)

    with pytest.raises(DataIntegrityError, match="minimum safe work"):
        resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True)


def test_resolve_ram_budget_smaller_cgroup_limit_governs(monkeypatch) -> None:
    """A readable cgroup limit below the host total governs the budget."""
    class _Mem:
        total = 2 * 2**30

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(resources, "_read_cgroup_limit_bytes", lambda: 1 * 2**30)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)

    budget, _ = resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True)

    assert budget == int(1 * 2**30 * resources.RAM_BUDGET_FRACTION)


def test_resolve_ram_budget_cgroup_remaining_caps_budget(monkeypatch) -> None:
    """Remaining cgroup capacity below the computed budget caps admission."""
    class _Mem:
        total = 8 * 2**30

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(resources, "_read_cgroup_limit_bytes", lambda: None)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: 100)

    budget, _ = resources._resolve_ram_budget(max_rss_bytes=None, ram_guard=True)

    assert budget == 100


def _fake_open(files: dict[str, str]):  # type: ignore[no-untyped-def]
    def _open(path: object, *args: object, **kwargs: object) -> io.StringIO:
        key = str(path)
        if key in files:
            return io.StringIO(files[key])
        raise OSError(f"missing {key}")

    return _open


def test_read_cgroup_limit_numeric_v2(monkeypatch) -> None:
    """A numeric cgroup v2 limit is read."""
    monkeypatch.setattr("builtins.open", _fake_open({"/sys/fs/cgroup/memory.max": "4294967296\n"}))

    assert resources._read_cgroup_limit_bytes() == 4294967296


def test_read_cgroup_limit_unlimited_v2_returns_none(monkeypatch) -> None:
    """A 'max' cgroup v2 limit means no readable cap."""
    monkeypatch.setattr("builtins.open", _fake_open({"/sys/fs/cgroup/memory.max": "max\n"}))

    assert resources._read_cgroup_limit_bytes() is None


def test_read_cgroup_limit_falls_back_to_v1(monkeypatch) -> None:
    """A missing v2 hierarchy falls back to the v1 limit file."""
    monkeypatch.setattr(
        "builtins.open",
        _fake_open({"/sys/fs/cgroup/memory/memory.limit_in_bytes": "1073741824\n"}),
    )

    assert resources._read_cgroup_limit_bytes() == 1073741824


def test_read_cgroup_limit_invalid_and_missing_returns_none(monkeypatch) -> None:
    """Unparseable or absent limits are treated as unreadable, not zero."""
    monkeypatch.setattr("builtins.open", _fake_open({"/sys/fs/cgroup/memory.max": "not-a-number\n"}))

    assert resources._read_cgroup_limit_bytes() is None

    monkeypatch.setattr("builtins.open", _fake_open({}))

    assert resources._read_cgroup_limit_bytes() is None


def test_read_cgroup_remaining_v2_reports_headroom(monkeypatch) -> None:
    """Remaining v2 capacity is limit minus current usage."""
    monkeypatch.setattr(
        "builtins.open",
        _fake_open({"/sys/fs/cgroup/memory.max": "1000\n", "/sys/fs/cgroup/memory.current": "400\n"}),
    )

    assert resources._read_cgroup_remaining_bytes() == 600


def test_read_cgroup_remaining_exhausted_reports_zero(monkeypatch) -> None:
    """Usage at or above the limit reports zero, never negative."""
    monkeypatch.setattr(
        "builtins.open",
        _fake_open({"/sys/fs/cgroup/memory.max": "1000\n", "/sys/fs/cgroup/memory.current": "1500\n"}),
    )

    assert resources._read_cgroup_remaining_bytes() == 0


def test_read_cgroup_remaining_unlimited_or_missing_returns_none(monkeypatch) -> None:
    """An unlimited or absent hierarchy yields no remaining-capacity cap."""
    monkeypatch.setattr(
        "builtins.open",
        _fake_open({"/sys/fs/cgroup/memory.max": "max\n", "/sys/fs/cgroup/memory.current": "10\n"}),
    )

    assert resources._read_cgroup_remaining_bytes() is None

    monkeypatch.setattr("builtins.open", _fake_open({}))

    assert resources._read_cgroup_remaining_bytes() is None


def test_read_cgroup_remaining_falls_back_to_v1(monkeypatch) -> None:
    """A missing v2 hierarchy falls back to the v1 usage/limit files."""
    monkeypatch.setattr(
        "builtins.open",
        _fake_open({
            "/sys/fs/cgroup/memory/memory.usage_in_bytes": "300\n",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": "1000\n",
        }),
    )

    assert resources._read_cgroup_remaining_bytes() == 700


def test_allocation_budget_bypass_when_both_none() -> None:
    """No budget and no reserve keeps the intentional legacy bypass."""
    resources.assert_mhs_allocation_budget(estimated_bytes=10**18, budget_bytes=None, reserve_bytes=None)


def test_allocation_budget_available_probe_failure_fails_closed(monkeypatch) -> None:
    """A psutil failure while reading available memory fails closed."""
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)

    def _boom() -> object:
        raise RuntimeError("psutil unavailable")

    monkeypatch.setattr(resources.psutil, "virtual_memory", _boom)

    with pytest.raises(DataIntegrityError, match="cannot read available"):
        resources.assert_mhs_allocation_budget(estimated_bytes=8, budget_bytes=None, reserve_bytes=1000)


def test_allocation_budget_tree_enumeration_failure_fails_closed(monkeypatch) -> None:
    """A psutil failure while enumerating the tree fails closed."""
    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("no process")

    monkeypatch.setattr(resources.psutil, "Process", _boom)

    with pytest.raises(DataIntegrityError, match="cannot enumerate"):
        resources.assert_mhs_allocation_budget(estimated_bytes=8, budget_bytes=1000, reserve_bytes=None)


def test_tree_swap_unmeasurable_returns_none(monkeypatch) -> None:
    """An unmeasurable tree yields null swap, never a fabricated zero."""
    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("no process")

    monkeypatch.setattr(resources.psutil, "Process", _boom)

    assert resources._current_tree_swap_bytes() is None


def test_tree_helpers_skip_unreadable_process(monkeypatch) -> None:
    """A generically unreadable process is skipped, not fatal."""
    class _BadProc:
        def memory_full_info(self) -> object:
            raise RuntimeError("unreadable")

        def children(self, recursive: bool = True) -> list[object]:
            return [self]

    monkeypatch.setattr(resources.psutil, "Process", lambda *args: _BadProc())

    with pytest.raises(DataIntegrityError, match="cannot read process memory"):
        resources._current_tree_pss_bytes()
    assert resources._current_tree_swap_bytes() is None


def test_allocation_budget_rejects_estimate_exceeding_budget(monkeypatch) -> None:
    """An estimate that would exceed budget raises before any allocation begins."""
    _admit_all(monkeypatch)
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 900)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)

    with pytest.raises(DataIntegrityError, match="no decoder or plane allocation begins"):
        resources.assert_mhs_allocation_budget(estimated_bytes=200, budget_bytes=1000, reserve_bytes=None)


def test_allocation_budget_admits_estimate_within_budget(monkeypatch) -> None:
    """An estimate fully inside budget and reserve is admitted."""
    _admit_all(monkeypatch)

    resources.assert_mhs_allocation_budget(estimated_bytes=100, budget_bytes=1000, reserve_bytes=128)


def test_allocation_budget_rejects_low_available(monkeypatch) -> None:
    """Small tree usage with available below the floor is rejected."""
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 8)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 16)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)

    with pytest.raises(DataIntegrityError, match="reserve breached"):
        resources.assert_mhs_allocation_budget(estimated_bytes=8, budget_bytes=10**12, reserve_bytes=2 * 2**30)


def test_allocation_budget_rejects_estimate_consuming_reserve(monkeypatch) -> None:
    """An estimate that would push available below the floor is rejected."""
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 1000)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)

    with pytest.raises(DataIntegrityError, match="reserve breached"):
        resources.assert_mhs_allocation_budget(estimated_bytes=900, budget_bytes=10**12, reserve_bytes=500)


def test_allocation_budget_telemetry_failure_fails_closed(monkeypatch) -> None:
    """Unmeasurable tree or available memory fails closed."""
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 10**12)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)

    def _boom() -> int:
        raise DataIntegrityError("cannot measure")

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", _boom)

    with pytest.raises(DataIntegrityError, match="cannot measure"):
        resources.assert_mhs_allocation_budget(estimated_bytes=8, budget_bytes=1000, reserve_bytes=None)

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)

    def _boom_available() -> int:
        raise DataIntegrityError("cannot read available")

    monkeypatch.setattr(resources, "_current_available_bytes", _boom_available)

    with pytest.raises(DataIntegrityError, match="cannot read available"):
        resources.assert_mhs_allocation_budget(estimated_bytes=8, budget_bytes=None, reserve_bytes=1000)


def test_allocation_budget_rejects_invalid_inputs(monkeypatch) -> None:
    """Negative estimates and non-positive limits are rejected deterministically."""
    _admit_all(monkeypatch)

    with pytest.raises(DataIntegrityError, match="estimate is invalid"):
        resources.assert_mhs_allocation_budget(estimated_bytes=-1, budget_bytes=100, reserve_bytes=None)
    with pytest.raises(DataIntegrityError, match="budget is invalid"):
        resources.assert_mhs_allocation_budget(estimated_bytes=1, budget_bytes=0, reserve_bytes=None)
    with pytest.raises(DataIntegrityError, match="reserve is invalid"):
        resources.assert_mhs_allocation_budget(estimated_bytes=1, budget_bytes=None, reserve_bytes=0)


def test_allocation_budget_aborts_on_swap_growth(monkeypatch) -> None:
    """Observed per-process swap growth aborts replay safely with diagnostics."""
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 10**12)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 4096)

    with pytest.raises(DataIntegrityError, match="swap growth"):
        resources.assert_mhs_allocation_budget(estimated_bytes=8, budget_bytes=10**12, reserve_bytes=128)


def test_tree_pss_uses_shared_page_accounting(monkeypatch) -> None:
    """Tree memory sums PSS, never substituting summed RSS for shared pages."""
    class _Info:
        def __init__(self, pss: int, rss: int) -> None:
            self.pss = pss
            self.rss = rss
            self.uss = pss
            self.swap = 0

    class _Proc:
        def __init__(self, info: _Info, children: list[_Proc]) -> None:
            self._info = info
            self._children = children

        def memory_full_info(self) -> _Info:
            return self._info

        def children(self, recursive: bool = True) -> list[_Proc]:
            return self._children

    parent = _Proc(_Info(pss=100, rss=1000), [])
    child = _Proc(_Info(pss=50, rss=900), [])
    parent._children.append(child)
    monkeypatch.setattr(resources.psutil, "Process", lambda *args: parent)
    monkeypatch.setattr(resources.os, "getpid", lambda: 1)

    assert resources._current_tree_pss_bytes() == 150


def test_tree_pss_unmeasurable_tree_fails_closed(monkeypatch) -> None:
    """A tree with no readable process fails admission instead of reporting zero."""
    class _Proc:
        def memory_full_info(self):  # type: ignore[no-untyped-def]
            raise psutil.NoSuchProcess(pid=1)

        def children(self, recursive: bool = True) -> list[object]:
            return [self]

    monkeypatch.setattr(resources.psutil, "Process", lambda *args: _Proc())

    with pytest.raises(DataIntegrityError, match="no process-tree PSS"):
        resources._current_tree_pss_bytes()
    assert resources._current_tree_swap_bytes() is None


def test_available_negative_reading_fails_closed(monkeypatch) -> None:
    """A negative available reading is treated as missing telemetry."""
    class _Mem:
        available = -1

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())

    with pytest.raises(DataIntegrityError, match="telemetry unavailable"):
        resources._current_available_bytes()


def test_worker_plan_bounded_before_fork(monkeypatch) -> None:
    """Parallel private demand beyond budget bounds workers before fork."""
    from src.mhs.parallel import plan_worker_count
    import src.mhs.parallel as parallel

    class _Mem:
        available = 3 * 2**30

    monkeypatch.setattr(parallel.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(parallel.psutil, "cpu_count", lambda: 8)
    monkeypatch.setattr(parallel, "_system_reserve_bytes", lambda: 2 * 2**30)

    granted = plan_worker_count(8, 1 * 2**30, True)

    assert granted == 1


def test_wide_window_splits_without_changing_strategy_inputs(monkeypatch) -> None:
    """An over-budget wide window is split into admittable IO pieces only."""
    from src.mhs.types import ExecutionSpec

    _admit_all(monkeypatch)
    spec = ExecutionSpec()
    clock_before = spec.passive_timeout_minutes
    wide_estimate = 1500
    budget = 1000

    with pytest.raises(DataIntegrityError, match="budget exceeded"):
        resources.assert_mhs_allocation_budget(
            estimated_bytes=wide_estimate, budget_bytes=budget, reserve_bytes=None
        )
    for piece in (wide_estimate // 2, wide_estimate - wide_estimate // 2):
        resources.assert_mhs_allocation_budget(estimated_bytes=piece, budget_bytes=budget, reserve_bytes=None)

    assert spec.passive_timeout_minutes == clock_before


def test_resource_scope_reports_times_and_peaks_on_failure() -> None:
    """A controlled failure still leaves elapsed times and sampled peaks available."""
    sampler = resources._TreeMemorySampler(interval_seconds=0.01)
    sampler.start()
    try:
        sampler._sample_once()
        raise RuntimeError("controlled replay failure")
    except RuntimeError:
        stats = sampler.stop()

    assert stats.samples_taken >= 1
    assert stats.wall_seconds >= 0.0
    assert stats.cpu_seconds >= 0.0
    assert stats.tree_pss_peak_bytes >= 0
    assert stats.parent_rss_peak_bytes is not None


def test_sampler_stop_without_start_reports_null_peaks() -> None:
    """A sampler that never started reports null optional peaks, not zeros."""
    stats = resources._TreeMemorySampler(interval_seconds=0.01).stop()

    assert stats.samples_taken == 0
    assert stats.wall_seconds == 0.0
    assert stats.parent_rss_peak_bytes is None
    assert stats.child_rss_peak_bytes is None
    assert stats.process_swap_growth_bytes is None


def test_sampler_stop_survives_cpu_probe_failure(monkeypatch) -> None:
    """A failing CPU probe at stop still yields wall time without raising."""
    sampler = resources._TreeMemorySampler(interval_seconds=0.01)
    sampler.start()

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("cpu unavailable")

    monkeypatch.setattr(resources.psutil, "Process", _boom)
    stats = sampler.stop()

    assert stats.wall_seconds >= 0.0
    assert stats.cpu_seconds == 0.0


def test_assert_stage_rss_budget_noop_when_both_none() -> None:
    """Both budget and reserve None makes the guard a no-op."""
    resources._assert_stage_rss_budget("stage", None, None)


def test_assert_stage_rss_budget_raises_when_rss_exceeds_budget(monkeypatch) -> None:
    """A current tree PSS above the budget raises DataIntegrityError naming the stage."""
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 1000)

    with pytest.raises(DataIntegrityError, match="my_stage"):
        resources._assert_stage_rss_budget("my_stage", budget_bytes=500, reserve_bytes=None)


def test_assert_stage_rss_budget_passes_when_rss_within_budget(monkeypatch) -> None:
    """A current tree PSS at or below the budget does not raise."""
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 100)

    resources._assert_stage_rss_budget("my_stage", budget_bytes=500, reserve_bytes=None)


def test_assert_stage_rss_budget_raises_when_reserve_breached(monkeypatch) -> None:
    """Available system memory below the reserve raises DataIntegrityError."""
    class _Mem:
        available = 10

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())

    with pytest.raises(DataIntegrityError, match="reserve"):
        resources._assert_stage_rss_budget("my_stage", budget_bytes=None, reserve_bytes=1000)


def test_assert_stage_rss_budget_swallows_psutil_failure_on_reserve_probe(monkeypatch) -> None:
    """A psutil failure while probing required headroom fails closed."""
    def _raise() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _raise())

    with pytest.raises(DataIntegrityError, match="telemetry unavailable"):
        resources._assert_stage_rss_budget("my_stage", budget_bytes=None, reserve_bytes=1000)


def test_assert_execution_rss_budget_noop_when_both_none() -> None:
    """Neither budget nor reserve set makes the guard a no-op."""
    resources._assert_execution_rss_budget("stage", None, completed_windows=3)


def test_assert_execution_rss_budget_raises_with_window_context(monkeypatch) -> None:
    """An exceeded budget raises with stage, observed pss, budget, and window count."""
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 2000)

    with pytest.raises(DataIntegrityError, match="completed_windows=7"):
        resources._assert_execution_rss_budget(
            "exec_stage", budget=1000, completed_windows=7,
        )


def test_assert_execution_rss_budget_reserve_breach_maps_to_rss_budget_message(monkeypatch) -> None:
    """A system-reserve breach uses the stable 'rss budget'-prefixed message."""
    class _Mem:
        available = 5

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)

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


def _child_hold_private_bytes(ready_queue, release_queue, nbytes: int) -> None:  # type: ignore[no-untyped-def]
    """Fork child: allocate ``nbytes`` private pages and hold until released."""
    buf = bytearray(nbytes)
    for i in range(0, nbytes, 4096):
        buf[i] = 1  # touch every page so the allocation becomes resident
    ready_queue.put("held")
    release_queue.get(timeout=60)


class TestProcessTreeMemoryStats:
    """Process-tree memory telemetry uses PSS and carries sampled run evidence."""

    def test_has_no_sum_of_rss_field(self) -> None:
        """Sum-of-RSS is deliberately absent: it double-counts COW-shared pages."""
        names = {f.name for f in dataclasses.fields(resources.ProcessTreeMemoryStats)}
        assert names == {
            "tree_pss_peak_bytes",
            "tree_uss_peak_bytes",
            "min_system_available_bytes",
            "max_concurrent_procs",
            "samples_taken",
            "parent_rss_peak_bytes",
            "child_rss_peak_bytes",
            "wall_seconds",
            "cpu_seconds",
            "process_swap_growth_bytes",
            "preparation_tree_pss_peak_bytes",
            "replay_tree_pss_peak_bytes",
        }
        assert not any("sum" in name for name in names)

    def test_optional_observations_default_to_null(self) -> None:
        """Missing optional observations are null, not zero."""
        stats = resources.ProcessTreeMemoryStats(
            tree_pss_peak_bytes=1,
            tree_uss_peak_bytes=2,
            min_system_available_bytes=3,
            max_concurrent_procs=1,
            samples_taken=0,
        )

        assert stats.parent_rss_peak_bytes is None
        assert stats.child_rss_peak_bytes is None
        assert stats.process_swap_growth_bytes is None
        assert stats.wall_seconds == 0.0
        assert stats.cpu_seconds == 0.0

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
        def _boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
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


def _stage_budget(total: int = 1000, replay: int = 400, floor: int = 100) -> object:
    return resources.MhsMemoryBudget(
        total_tree_pss_bytes=total, replay_tree_pss_bytes=replay, min_available_bytes=floor,
    )


def _admit_setup(monkeypatch, *, pss: int = 100, available: int = 10**12, swap: int | None = 0) -> None:
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: pss)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: available)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: swap)


def test_memory_budget_defaults_and_validation() -> None:
    """Defaults are 2.5/1.5/2 GiB with replay bounded by total."""
    budget = resources.MhsMemoryBudget()
    assert budget.total_tree_pss_bytes == resources.MHS_TREE_PSS_BUDGET_BYTES
    assert budget.replay_tree_pss_bytes == resources.MHS_REPLAY_BUDGET_BYTES
    assert budget.min_available_bytes == resources.MHS_AVAILABLE_FLOOR_BYTES
    assert budget.total_tree_pss_bytes == int(2.5 * 2**30)
    assert budget.replay_tree_pss_bytes == int(1.5 * 2**30)
    assert budget.min_available_bytes == 2 * 2**30
    with pytest.raises(ValueError, match="positive integer"):
        resources.MhsMemoryBudget(total_tree_pss_bytes=0)
    with pytest.raises(ValueError, match="positive integer"):
        resources.MhsMemoryBudget(min_available_bytes=-5)
    with pytest.raises(ValueError, match="cannot exceed"):
        resources.MhsMemoryBudget(total_tree_pss_bytes=100, replay_tree_pss_bytes=200)
    with pytest.raises(ValueError, match="positive integer"):
        resources.MhsMemoryBudget(total_tree_pss_bytes=True)  # type: ignore[arg-type]
    err = resources.MhsResourceAdmissionError(stage="s", error_code="MEMORY_BUDGET", message="m")
    assert err.stage == "s"
    assert err.error_code == "MEMORY_BUDGET"
    assert isinstance(err, DataIntegrityError)


def test_stage_admission_applies_separate_limits(monkeypatch) -> None:
    """Same residency passes total but fails replay; phase peaks stay distinct."""
    _admit_setup(monkeypatch, pss=500, available=10**12)
    budget = _stage_budget(total=1000, replay=400, floor=100)
    resources.assert_mhs_stage_allocation(
        stage="setup", estimated_bytes=100, budget=budget, replay=False, initial_swap_bytes=0,
    )
    with pytest.raises(resources.MhsResourceAdmissionError) as exc:
        resources.assert_mhs_stage_allocation(
            stage="replay", estimated_bytes=100, budget=budget, replay=True, initial_swap_bytes=0,
        )
    assert exc.value.stage == "replay"
    assert exc.value.error_code == "MEMORY_BUDGET"
    sampler = resources._TreeMemorySampler(interval_seconds=0.01)
    sampler.set_stage("preparation")
    sampler._sample_once()
    sampler.set_stage("replay")
    sampler._sample_once()
    stats = sampler.stop()
    assert stats.preparation_tree_pss_peak_bytes is not None
    assert stats.replay_tree_pss_peak_bytes is not None
    assert stats.tree_pss_peak_bytes >= 0
    fresh = resources._TreeMemorySampler(interval_seconds=0.01).stop()
    assert fresh.preparation_tree_pss_peak_bytes is None
    assert fresh.replay_tree_pss_peak_bytes is None
    with pytest.raises(ValueError, match="unsupported"):
        sampler.set_stage("invalid")  # type: ignore[arg-type]
    sampler.set_stage("replay")


def test_stage_admission_rejects_before_decoder_runs(monkeypatch) -> None:
    """An inadmissible estimate fails before any decoder constructor runs."""
    _admit_setup(monkeypatch, pss=900, available=10**12)
    budget = _stage_budget(total=1000, replay=1000, floor=100)
    called: list[str] = []

    def _decoder() -> None:
        called.append("decode")

    with pytest.raises(DataIntegrityError, match="tree_pss=900"):
        resources.assert_mhs_stage_allocation(
            stage="process_prepare_panel", estimated_bytes=200, budget=budget,
            replay=False, initial_swap_bytes=0,
        )
    assert called == []


def test_stage_admission_rejects_unreadable_live_child(monkeypatch) -> None:
    """An inaccessible live child fails closed with the named stage."""
    class _Info:
        pss = 10
        uss = 10
        swap = 0
        rss = 10

    class _Good:
        def memory_full_info(self) -> object:
            return _Info()

        def children(self, recursive: bool = True) -> list[object]:
            return [_Bad()]

    class _Bad:
        def memory_full_info(self) -> object:
            raise resources.psutil.AccessDenied(pid=9)

    monkeypatch.setattr(resources.psutil, "Process", lambda *args: _Good())
    with pytest.raises(DataIntegrityError, match="unreadable live process"):
        resources._current_tree_pss_bytes()
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 10**12)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)
    with pytest.raises(resources.MhsResourceAdmissionError) as exc:
        resources.assert_mhs_stage_allocation(
            stage="named", estimated_bytes=10**12, budget=_stage_budget(),
            replay=False, initial_swap_bytes=0,
        )
    assert exc.value.stage == "named"


def test_stage_admission_rejects_tighter_headroom(monkeypatch) -> None:
    """Whichever of host or cgroup headroom is tighter governs rejection."""
    budget = _stage_budget(total=10**12, replay=10**12, floor=1000)
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 500)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)
    with pytest.raises(resources.MhsResourceAdmissionError) as exc:
        resources.assert_mhs_stage_allocation(
            stage="host", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=0,
        )
    assert exc.value.error_code == "MEMORY_RESERVE"
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 10**12)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: 100)
    with pytest.raises(resources.MhsResourceAdmissionError) as exc2:
        resources.assert_mhs_stage_allocation(
            stage="cgroup", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=0,
        )
    assert exc2.value.error_code == "MEMORY_RESERVE"
    assert exc2.value.stage == "cgroup"


def test_stage_admission_swap_baseline_behaviour(monkeypatch) -> None:
    """Only measured swap growth triggers the swap error; unknown stays null."""
    budget = _stage_budget(total=10**12, replay=10**12, floor=100)
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 10**12)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 100)
    resources.assert_mhs_stage_allocation(
        stage="s", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=100,
    )
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 200)
    with pytest.raises(resources.MhsResourceAdmissionError) as exc:
        resources.assert_mhs_stage_allocation(
            stage="s", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=100,
        )
    assert exc.value.error_code == "SWAP_GROWTH"
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: None)
    resources.assert_mhs_stage_allocation(
        stage="s", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=100,
    )
    resources.assert_mhs_stage_allocation(
        stage="s", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=None,
    )


def test_stage_admission_telemetry_and_parameter_errors(monkeypatch) -> None:
    """Required telemetry failures and malformed inputs raise explicitly."""
    budget = _stage_budget()

    def _boom() -> int:
        raise DataIntegrityError("cannot enumerate")

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", _boom)
    with pytest.raises(resources.MhsResourceAdmissionError) as exc:
        resources.assert_mhs_stage_allocation(
            stage="s", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=0,
        )
    assert exc.value.error_code == "RESOURCE_TELEMETRY"
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)

    def _boom_host() -> int:
        raise DataIntegrityError("cannot read available")

    monkeypatch.setattr(resources, "_current_available_bytes", _boom_host)
    with pytest.raises(resources.MhsResourceAdmissionError) as exc2:
        resources.assert_mhs_stage_allocation(
            stage="s", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=0,
        )
    assert exc2.value.error_code == "RESOURCE_TELEMETRY"
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: 10**12)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)
    with pytest.raises(ValueError, match="stage"):
        resources.assert_mhs_stage_allocation(
            stage="", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=0,
        )
    with pytest.raises(ValueError, match="estimated_bytes"):
        resources.assert_mhs_stage_allocation(
            stage="s", estimated_bytes=-1, budget=budget, replay=False, initial_swap_bytes=0,
        )
    with pytest.raises(ValueError, match="replay"):
        resources.assert_mhs_stage_allocation(
            stage="s", estimated_bytes=0, budget=budget, replay="yes", initial_swap_bytes=0,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="initial_swap_bytes"):
        resources.assert_mhs_stage_allocation(
            stage="s", estimated_bytes=0, budget=budget, replay=False, initial_swap_bytes=-1,
        )
    with pytest.raises(ValueError, match="budget"):
        resources.assert_mhs_stage_allocation(
            stage="s", estimated_bytes=0, budget="bad", replay=False, initial_swap_bytes=0,  # type: ignore[arg-type]
        )


def test_tree_pss_missing_observation_fails_closed(monkeypatch) -> None:
    """A live process without PSS and vanishing children never fabricate zero."""
    class _NoPss:
        uss = 1
        swap = 0
        rss = 1

    class _Proc:
        def memory_full_info(self) -> object:
            return _NoPss()

        def children(self, recursive: bool = True) -> list[object]:
            return []

    monkeypatch.setattr(resources.psutil, "Process", lambda *args: _Proc())
    with pytest.raises(DataIntegrityError, match="PSS observation missing"):
        resources._current_tree_pss_bytes()

    class _Vanish:
        def children(self, recursive: bool = True) -> list[object]:
            raise resources.psutil.NoSuchProcess(pid=1)

    class _Me:
        def children(self, recursive: bool = True) -> list[object]:
            return []

        def memory_full_info(self) -> object:
            from types import SimpleNamespace

            return SimpleNamespace(pss=5, uss=5, swap=0, rss=5)

    monkeypatch.setattr(resources.psutil, "Process", lambda *args: _Me())
    assert resources._current_tree_pss_bytes() == 5
    with pytest.raises(ValueError, match="budget"):
        resources._resolve_memory_budget("bad")  # type: ignore[arg-type]
    assert resources._resolve_memory_budget(None).total_tree_pss_bytes == resources.MHS_TREE_PSS_BUDGET_BYTES

    class _BadPss:
        pss = "bad"
        uss = 1
        swap = 0
        rss = 1

    class _BadPssProc:
        def memory_full_info(self) -> object:
            return _BadPss()

        def children(self, recursive: bool = True) -> list[object]:
            return []

    monkeypatch.setattr(resources.psutil, "Process", lambda *args: _BadPssProc())
    with pytest.raises(DataIntegrityError, match="invalid PSS"):
        resources._current_tree_pss_bytes()


def test_sampler_stage_boundary_failures_stay_observational(monkeypatch) -> None:
    """Boundary observation failures never raise into the run."""
    sampler = resources._TreeMemorySampler(interval_seconds=0.01)

    def _boom(*args, **kwargs) -> object:
        raise RuntimeError("no tree")

    monkeypatch.setattr(resources.psutil, "Process", _boom)
    sampler.set_stage("preparation")
    sampler.set_stage("replay")
    stats = sampler.stop()
    assert stats.preparation_tree_pss_peak_bytes is None
    assert stats.replay_tree_pss_peak_bytes is None

    class _BadMem:
        def memory_full_info(self) -> object:
            raise RuntimeError("unreadable")

    class _BadTree:
        def children(self, recursive: bool = True) -> list[object]:
            return [self]

        def memory_full_info(self) -> object:
            raise RuntimeError("unreadable")

    monkeypatch.setattr(resources.psutil, "Process", lambda *args: _BadTree())
    sampler2 = resources._TreeMemorySampler(interval_seconds=0.01)
    sampler2.set_stage("preparation")
    assert sampler2.stop().preparation_tree_pss_peak_bytes is None
    assert _BadMem is not None


def _plan_allocation(fixed: int = 1000, per_bar: int = 100, decoder: int = 500) -> object:
    return resources.MhsExecutionAllocation(fixed_bytes=fixed, bytes_per_bar=per_bar, decoder_bytes=decoder)


def _plan_setup(monkeypatch, *, pss: int = 0, available: int = 10**12) -> None:
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: pss)
    monkeypatch.setattr(resources, "_current_available_bytes", lambda: available)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 0)


def test_plan_returns_largest_legal_grid(monkeypatch) -> None:
    """Largest admissible grid within both budgets never exceeds the request."""
    _plan_setup(monkeypatch, pss=0, available=10**12)
    alloc = _plan_allocation(fixed=1000, per_bar=100, decoder=500)
    got = resources.plan_mhs_execution_bars(requested_bars=20, minimum_bars=2, allocation=alloc, budget_bytes=3000, reserve_bytes=100)
    assert got == 15
    assert got <= 20


def test_plan_minimum_failure_names_bytes(monkeypatch) -> None:
    """Insufficient space for the minimum timeout grid fails before any decode."""
    _plan_setup(monkeypatch, pss=900, available=10**12)
    alloc = _plan_allocation(fixed=1000, per_bar=100, decoder=500)
    with pytest.raises(DataIntegrityError, match="require"):
        resources.plan_mhs_execution_bars(requested_bars=10, minimum_bars=5, allocation=alloc, budget_bytes=1000, reserve_bytes=None)


def test_plan_decoder_dominance_constrains(monkeypatch) -> None:
    """Large decoder transients constrain the plan rather than being ignored."""
    _plan_setup(monkeypatch, pss=0, available=10**12)
    small = resources.plan_mhs_execution_bars(requested_bars=100, minimum_bars=2, allocation=_plan_allocation(100, 10, 5000), budget_bytes=6000, reserve_bytes=None)
    large = resources.plan_mhs_execution_bars(requested_bars=100, minimum_bars=2, allocation=_plan_allocation(100, 10, 100), budget_bytes=6000, reserve_bytes=None)
    assert small < large


def test_plan_invalid_and_unknown_fails_closed(monkeypatch) -> None:
    """Invalid components or unavailable telemetry fail without zero grids."""
    _plan_setup(monkeypatch)
    with pytest.raises(ValueError, match="bytes_per_bar"):
        resources.MhsExecutionAllocation(fixed_bytes=0, bytes_per_bar=0, decoder_bytes=0)
    with pytest.raises(ValueError, match="non-negative"):
        resources.MhsExecutionAllocation(fixed_bytes=-1, bytes_per_bar=10, decoder_bytes=0)
    with pytest.raises(ValueError, match="requested_bars"):
        resources.plan_mhs_execution_bars(requested_bars=1, minimum_bars=2, allocation=_plan_allocation(), budget_bytes=100, reserve_bytes=None)
    with pytest.raises(ValueError, match="minimum_bars"):
        resources.plan_mhs_execution_bars(requested_bars=5, minimum_bars=1, allocation=_plan_allocation(), budget_bytes=100, reserve_bytes=None)
    with pytest.raises(ValueError, match="allocation"):
        resources.plan_mhs_execution_bars(requested_bars=5, minimum_bars=2, allocation="bad", budget_bytes=100, reserve_bytes=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="budget_bytes"):
        resources.plan_mhs_execution_bars(requested_bars=5, minimum_bars=2, allocation=_plan_allocation(), budget_bytes=0, reserve_bytes=None)
    with pytest.raises(ValueError, match="requested_bars"):
        resources.plan_mhs_execution_bars(requested_bars=0, minimum_bars=2, allocation=_plan_allocation(), budget_bytes=100, reserve_bytes=None)


def test_plan_reserve_breach_skips_large_grids(monkeypatch) -> None:
    """Post-allocation reserve skips large grids before returning a smaller one."""
    _plan_setup(monkeypatch, pss=0, available=5000)
    alloc = _plan_allocation(fixed=100, per_bar=10, decoder=100)
    got = resources.plan_mhs_execution_bars(requested_bars=100, minimum_bars=2, allocation=alloc, budget_bytes=None, reserve_bytes=4000)
    assert got < 100
    assert got >= 2

    def _boom() -> int:
        raise DataIntegrityError("cannot enumerate")

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", _boom)
    with pytest.raises(DataIntegrityError, match="cannot enumerate"):
        resources.plan_mhs_execution_bars(requested_bars=5, minimum_bars=2, allocation=_plan_allocation(), budget_bytes=100, reserve_bytes=None)
    assert resources.plan_mhs_execution_bars(requested_bars=7, minimum_bars=2, allocation=_plan_allocation(), budget_bytes=None, reserve_bytes=None) == 7
