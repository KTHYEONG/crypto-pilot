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

import multiprocessing as mp
from types import SimpleNamespace

import psutil
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.parallel import (
    MHS_FORK_CONTEXT,
    assert_fork_admission,
    fork_shared_payload,
    plan_worker_count,
    resolve_fork_shared,
)

_GB = 2**30


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
    assert MHS_FORK_CONTEXT.get_start_method() == "fork"


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
