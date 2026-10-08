"""Contract tests for the MHS fork-COW parallel primitives.

- ``SCENARIO_MHS_REFACTOR_01``: ``fork_shared_payload``/``resolve_fork_shared``
  resolve the same payload in a fork child by token, and an unregistered token
  fails closed with ``DataIntegrityError``.
- ``SCENARIO_MHS_REFACTOR_02``: ``plan_worker_count`` clamps to RAM, respects
  the CPU bound, honors ``ram_guard=False``, and rejects non-positive
  ``per_worker_bytes``.
- ``SCENARIO_MHS_REFACTOR_03``: ``assert_fork_admission`` raises a
  ``fork admission``-prefixed ``DataIntegrityError`` when projected demand
  breaches the reserve, and is a no-op otherwise.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import threading
import weakref
from collections.abc import Iterator
from types import SimpleNamespace

import psutil
import pytest

from src.common.errors import DataIntegrityError
from src.core.parallel import (
    FORK_CONTEXT,
    assert_fork_admission,
    collect_window_garbage,
    fork_shared_payload,
    frozen_gc_heap,
    plan_worker_count,
    resolve_fork_shared,
)

_GB = 2**30


@pytest.fixture
def _gc_freeze_guard() -> Iterator[None]:
    """Pin the process-global GC freeze state: 0 on entry, restored on exit."""
    assert gc.get_freeze_count() == 0
    try:
        yield
    finally:
        gc.unfreeze()


def _fake_virtual_memory(total_gb: float, available_gb: float) -> SimpleNamespace:
    return SimpleNamespace(
        total=int(total_gb * _GB),
        available=int(available_gb * _GB),
    )


def test_scenario_01_fork_child_resolves_shared_payload() -> None:
    """SCENARIO_MHS_REFACTOR_01: a fork child resolves the payload by token."""
    payload = {"grid": [1, 2, 3], "label": "panels"}

    def _child_entry(token: str, queue) -> None:
        resolved = resolve_fork_shared(token)
        queue.put((resolved["label"], list(resolved["grid"])))

    ctx = mp.get_context("fork")
    with fork_shared_payload(payload) as token:
        queue = ctx.Queue()
        proc = ctx.Process(target=_child_entry, args=(token, queue))
        proc.start()
        proc.join(60)
        assert proc.exitcode == 0
        label, grid = queue.get(timeout=30)
    assert label == "panels"
    assert grid == [1, 2, 3]


def test_scenario_01_token_removed_after_context_exit() -> None:
    """SCENARIO_MHS_REFACTOR_01: after exit the token fails closed."""
    with fork_shared_payload({"a": 1}) as token:
        assert resolve_fork_shared(token)["a"] == 1
    with pytest.raises(DataIntegrityError, match="unregistered fork-shared"):
        resolve_fork_shared(token)


def test_scenario_01_fork_context_pinned() -> None:
    """SCENARIO_MHS_REFACTOR_01: the pinned fork context is a fork context."""
    assert FORK_CONTEXT.get_start_method() == "fork"


def test_scenario_02_clamps_to_one_when_ram_is_tight(monkeypatch) -> None:
    """SCENARIO_MHS_REFACTOR_02: projected demand below reserve clamps to 1."""
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _fake_virtual_memory(100.0, 5.5),
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda: 8)
    # reserve = max(5% of 100GB, 256MiB) = 5GB; (5.5-5)GB < 6GB per worker.
    assert plan_worker_count(3, int(6.0 * _GB), True) == 1


def test_scenario_02_respects_cpu_bound_when_ram_ample(monkeypatch) -> None:
    """SCENARIO_MHS_REFACTOR_02: ample RAM returns min(requested, cpu_count)."""
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _fake_virtual_memory(100.0, 100.0),
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda: 8)
    assert plan_worker_count(3, _GB, True) == 3


def test_scenario_02_ram_guard_off_returns_cpu_bound(monkeypatch) -> None:
    """SCENARIO_MHS_REFACTOR_02: ram_guard=False ignores available memory."""
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _fake_virtual_memory(100.0, 0.1),
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda: 8)
    assert plan_worker_count(3, _GB, False) == 3


def test_scenario_02_rejects_non_positive_per_worker(monkeypatch) -> None:
    """SCENARIO_MHS_REFACTOR_02: per_worker_bytes <= 0 raises ValueError."""
    monkeypatch.setattr(psutil, "cpu_count", lambda: 8)
    with pytest.raises(ValueError, match="per_worker_bytes"):
        plan_worker_count(3, 0, False)
    with pytest.raises(ValueError, match="per_worker_bytes"):
        plan_worker_count(3, -1, True)


def test_worker_plan_uses_the_same_reserve_as_fork_admission(monkeypatch) -> None:
    """A RAM-clamped plan remains admissible under its caller's reserve."""
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _fake_virtual_memory(16.58, 10.6),
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda: 8)

    reserve = int(2.0 * _GB)
    workers = plan_worker_count(
        3, int(3.0 * _GB), True, reserve_bytes=reserve,
    )

    assert workers == 2
    assert_fork_admission("books", workers, int(3.0 * _GB), reserve)


def test_scenario_03_admission_raises_when_reserve_breached(monkeypatch) -> None:
    """SCENARIO_MHS_REFACTOR_03: projected demand breaching reserve fails."""
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _fake_virtual_memory(100.0, 5.5),
    )
    with pytest.raises(DataIntegrityError, match=r"^fork admission"):
        assert_fork_admission("books", 3, int(2.0 * _GB), int(6.0 * _GB))


def test_scenario_03_admission_noop_with_headroom(monkeypatch) -> None:
    """SCENARIO_MHS_REFACTOR_03: sufficient headroom is a no-op."""
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _fake_virtual_memory(100.0, 50.0),
    )
    result = assert_fork_admission("books", 3, _GB, int(6.0 * _GB))
    assert result is None


def test_scenario_03_admission_noop_when_reserve_none(monkeypatch) -> None:
    """SCENARIO_MHS_REFACTOR_03: reserve_bytes=None is a no-op."""
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _fake_virtual_memory(100.0, 0.0),
    )
    result = assert_fork_admission("books", 3, _GB, None)
    assert result is None


class TestWorkerPlanObserver:
    """SCENARIO_MHS_PERF_P0_04_WORKER_PLAN_RECORDED."""

    def test_observer_invoked_once_with_granted_and_ram_state(self, monkeypatch) -> None:
        """obs fires exactly once with the granted count and available/reserve."""
        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: _fake_virtual_memory(19.53, 12.4),
        )
        monkeypatch.setattr(psutil, "cpu_count", lambda: 8)
        calls: list[tuple] = []

        granted = plan_worker_count(
            3, int(3.0 * _GB), True,
            observer=lambda *args: calls.append(args),
        )
        direct = plan_worker_count(3, int(3.0 * _GB), True)

        assert granted == direct == 3
        assert len(calls) == 1
        stage, requested, obs_granted, available, reserve = calls[0]
        assert requested == 3
        assert obs_granted == granted
        assert available == int(12.4 * _GB)
        assert reserve == max(int(19.53 * _GB * 0.05), 256 * 2**20)

    def test_observer_not_invoked_when_ram_guard_off(self, monkeypatch) -> None:
        """ram_guard=False short-circuits before the observer would fire."""
        monkeypatch.setattr(psutil, "cpu_count", lambda: 8)
        calls: list[tuple] = []
        assert plan_worker_count(
            3, _GB, False, observer=lambda *a: calls.append(a),
        ) == 3
        assert calls == []

    def test_observer_failure_is_observational(self, monkeypatch) -> None:
        """A raising observer never changes the planner result."""
        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: _fake_virtual_memory(19.53, 12.4),
        )
        monkeypatch.setattr(psutil, "cpu_count", lambda: 8)

        def _boom(*_a):
            raise RuntimeError("observer boom")

        assert plan_worker_count(3, int(3.0 * _GB), True, observer=_boom) == 3

    @pytest.mark.parametrize(
        ("available_gb", "expected"),
        [(6.0, 1), (12.4, 3)],
        ids=["collapsed_to_one_worker", "full_three_workers"],
    )
    def test_planner_collapse_reproduces_baseline_divergence(
        self, monkeypatch, available_gb: float, expected: int,
    ) -> None:
        """6.0 GiB available grants 1 worker (737 s outlier); 12.4 grants 3."""
        monkeypatch.setattr(
            psutil,
            "virtual_memory",
            lambda: _fake_virtual_memory(19.53, available_gb),
        )
        monkeypatch.setattr(psutil, "cpu_count", lambda: 8)
        assert plan_worker_count(3, int(3.0 * _GB), True) == expected


class TestFrozenGcHeap:
    """SCENARIO_PERF_03B: window-stream GC scoped heap freeze."""

    pytestmark = pytest.mark.usefixtures("_gc_freeze_guard")

    def test_freezes_and_releases(self) -> None:
        """Entering freezes the setup heap; exit releases it."""
        with frozen_gc_heap():
            assert gc.get_freeze_count() > 0
        assert gc.get_freeze_count() == 0

    def test_releases_on_exception(self) -> None:
        """A fatal body error propagates and still releases the freeze."""
        with pytest.raises(DataIntegrityError, match="fatal window"), frozen_gc_heap():
            raise DataIntegrityError("fatal window")
        assert gc.get_freeze_count() == 0

    def test_overlapping_scopes_share_one_freeze(self) -> None:
        """A exits while B is inside: still frozen; 0 only after B exits."""
        entered = threading.Event()
        release = threading.Event()
        done = threading.Event()
        state: dict[str, object] = {}

        def _other() -> None:
            with frozen_gc_heap():
                entered.set()
                assert release.wait(60)
                state["still_frozen"] = gc.get_freeze_count() > 0
            state["after"] = gc.get_freeze_count()
            done.set()

        worker = threading.Thread(target=_other)
        worker.start()
        try:
            with frozen_gc_heap():
                assert entered.wait(60)
            release.set()
            assert done.wait(60)
            assert state["still_frozen"] is True
            assert state["after"] == 0
            assert gc.get_freeze_count() == 0
        finally:
            release.set()
            worker.join(60)

    def test_collect_freezes_survivors_only_inside_scope(self) -> None:
        """Survivors freeze inside a scope; outside the count stays put."""
        with frozen_gc_heap():
            base = gc.get_freeze_count()
            survivor = {"window": list(range(100))}
            collect_window_garbage()
            assert gc.get_freeze_count() >= base + 1
            del survivor
        before = gc.get_freeze_count()
        transient = {"window": list(range(100))}
        collect_window_garbage()
        assert gc.get_freeze_count() == before
        del transient

    def test_collect_reclaims_consumed_window_cycle(self) -> None:
        """A dropped self-referencing window cycle is reclaimed with count ≥ 1."""
        class _Node:
            def __init__(self) -> None:
                self.peer: _Node | None = None

        with frozen_gc_heap():
            node = _Node()
            node.peer = node
            ref = weakref.ref(node)
            del node
            found = collect_window_garbage()
            assert ref() is None
            assert found >= 1

    def test_fork_child_never_unfreezes_inherited_heap(self) -> None:
        """A child forked under a scope keeps the inherited heap frozen on exit."""
        def _child_entry(queue) -> None:
            import gc as _gc

            from src.core.parallel import frozen_gc_heap as _scope

            with _scope():
                pass
            queue.put(_gc.get_freeze_count())

        with frozen_gc_heap():
            parent_count = gc.get_freeze_count()
            assert parent_count > 0
            queue = FORK_CONTEXT.Queue()
            proc = FORK_CONTEXT.Process(target=_child_entry, args=(queue,))
            proc.start()
            proc.join(60)
            assert proc.exitcode == 0
            child_after = queue.get(timeout=30)
        assert child_after >= parent_count
        assert gc.get_freeze_count() == 0

    def test_fork_child_without_inherited_heap_unfreezes(self) -> None:
        """A child forked outside any scope exits its own scope at 0."""
        def _child_entry(queue) -> None:
            import gc as _gc

            from src.core.parallel import frozen_gc_heap as _scope

            with _scope():
                assert _gc.get_freeze_count() > 0
            queue.put(_gc.get_freeze_count())

        queue = FORK_CONTEXT.Queue()
        proc = FORK_CONTEXT.Process(target=_child_entry, args=(queue,))
        proc.start()
        proc.join(60)
        assert proc.exitcode == 0
        assert queue.get(timeout=30) == 0

    def test_fork_child_gets_fresh_scope_lock(self) -> None:
        """A child forked while a helper holds the scope lock still scopes freely."""
        import src.core.parallel as _parallel

        held = threading.Event()
        release = threading.Event()

        def _holder() -> None:
            with _parallel._SCOPE_LOCK:
                held.set()
                assert release.wait(60)

        helper = threading.Thread(target=_holder)
        helper.start()
        try:
            assert held.wait(60)

            def _child_entry(queue) -> None:
                import gc as _gc

                from src.core.parallel import frozen_gc_heap as _scope

                with _scope():
                    pass
                queue.put(_gc.get_freeze_count())

            queue = FORK_CONTEXT.Queue()
            proc = FORK_CONTEXT.Process(target=_child_entry, args=(queue,))
            proc.start()
            proc.join(60)
            assert proc.exitcode == 0
            assert queue.get(timeout=30) == 0
        finally:
            release.set()
            helper.join(60)
