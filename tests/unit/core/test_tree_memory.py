"""Invariant scenarios for single-sweep process-tree memory observation."""

from __future__ import annotations

import pytest

from src.common.errors import DataIntegrityError
from src.core import resources
from src.core.tree_memory import TreeMemoryObservation, observe_tree_memory


class _Info:
    def __init__(self, pss, swap=0, with_swap: bool = True) -> None:
        if pss is not None:
            self.pss = pss
        if with_swap:
            self.swap = swap


class _Proc:
    def __init__(self, info_or_exc, calls: dict[int, int], index: int) -> None:
        self._payload = info_or_exc
        self._calls = calls
        self._index = index

    def memory_full_info(self):
        self._calls[self._index] = self._calls.get(self._index, 0) + 1
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Root:
    def __init__(self, procs: list[_Proc]) -> None:
        self._procs = procs

    def children(self, recursive: bool = True):
        return self._procs[1:]

    def memory_full_info(self):
        return self._procs[0].memory_full_info()


def _install(monkeypatch, payloads: list) -> dict[int, int]:
    import psutil

    calls: dict[int, int] = {}
    procs = [_Proc(payload, calls, i) for i, payload in enumerate(payloads)]
    root = _Root(procs)
    monkeypatch.setattr(psutil, "Process", lambda *args, **kwargs: root)
    return calls


def test_tree_memory_sums_pss_and_swap_in_one_read_per_process(monkeypatch) -> None:
    """Four processes contribute summed PSS and swap with one read each."""
    calls = _install(
        monkeypatch,
        [_Info(100, 0), _Info(200, 10), _Info(300, 0), _Info(400, 5)],
    )
    obs = observe_tree_memory()
    assert (obs.pss_bytes, obs.swap_bytes) == (1000, 15)
    assert calls == {0: 1, 1: 1, 2: 1, 3: 1}


def test_tree_memory_matches_legacy_swap_helper_and_pinned_pss(monkeypatch) -> None:
    """Combined swap equals the legacy helper and PSS matches pinned sums."""
    import psutil

    cases: list[tuple[list, int, int | None]] = [
        ([_Info(100, 0), _Info(200, 10)], 300, 10),
        ([_Info(50, 7)], 50, 7),
    ]
    for payloads, pinned_pss, pinned_swap in cases:
        calls: dict[int, int] = {}
        procs = [_Proc(payload, calls, i) for i, payload in enumerate(payloads)]
        root = _Root(procs)
        monkeypatch.setattr(psutil, "Process", lambda *a, _r=root, **k: _r)
        obs = observe_tree_memory()
        assert obs.pss_bytes == pinned_pss
        assert obs.swap_bytes == resources._current_tree_swap_bytes() == pinned_swap


def test_tree_memory_skips_vanished_processes(monkeypatch) -> None:
    """A NoSuchProcess child contributes to neither sum."""
    import psutil

    calls = _install(monkeypatch, [_Info(100, 4), psutil.NoSuchProcess(pid=9)])
    obs = observe_tree_memory()
    assert (obs.pss_bytes, obs.swap_bytes) == (100, 4)


def test_tree_memory_rejects_unreadable_live_process(monkeypatch) -> None:
    """AccessDenied and generic read failures fail closed with distinct causes."""
    import psutil

    _install(monkeypatch, [_Info(10, 0), psutil.AccessDenied(pid=9)])
    with pytest.raises(DataIntegrityError, match="unreadable live process"):
        observe_tree_memory()
    _install(monkeypatch, [_Info(10, 0), RuntimeError("unreadable")])
    with pytest.raises(DataIntegrityError, match="cannot read process memory"):
        observe_tree_memory()


def test_tree_memory_missing_pss_fails_closed(monkeypatch) -> None:
    """An info object without PSS fails closed instead of fabricating zero."""
    _install(monkeypatch, [_Info(None, 0)])
    with pytest.raises(DataIntegrityError, match="process-tree PSS observation missing"):
        observe_tree_memory()


def test_tree_memory_swap_missing_counts_as_zero_and_observed(monkeypatch) -> None:
    """Infos without a swap field contribute zero yet mark swap observed."""
    _install(monkeypatch, [_Info(100, with_swap=False), _Info(200, with_swap=False)])
    obs = observe_tree_memory()
    assert isinstance(obs, TreeMemoryObservation)
    assert obs.pss_bytes == 300
    assert obs.swap_bytes == 0


def test_tree_memory_invalid_swap_skips_only_that_contribution(monkeypatch) -> None:
    """A non-numeric swap field drops only its contribution without raising."""
    _install(monkeypatch, [_Info(100, 5), _Info(200, "bad"), _Info(300, 7)])
    obs = observe_tree_memory()
    assert obs.pss_bytes == 600
    assert obs.swap_bytes == 12


def test_tree_memory_enumeration_failure_fails_closed(monkeypatch) -> None:
    """A failing tree enumeration fails closed with its cause."""
    import psutil

    def _boom(*args, **kwargs):
        raise RuntimeError("no tree")

    monkeypatch.setattr(psutil, "Process", _boom)
    with pytest.raises(DataIntegrityError, match="cannot enumerate process tree"):
        observe_tree_memory()


def test_tree_memory_invalid_pss_fails_closed(monkeypatch) -> None:
    """A non-numeric PSS value fails closed instead of coercing silently."""
    _install(monkeypatch, [_Info("bad", 0)])
    with pytest.raises(DataIntegrityError, match="invalid PSS observation"):
        observe_tree_memory()


def test_tree_memory_all_vanished_fails_closed(monkeypatch) -> None:
    """A tree with no readable process never fabricates a zero observation."""
    import psutil

    _install(monkeypatch, [psutil.NoSuchProcess(pid=1), psutil.NoSuchProcess(pid=2)])
    with pytest.raises(DataIntegrityError, match="no process-tree PSS observation"):
        observe_tree_memory()


def test_tree_memory_unreadable_swap_skips_only_that_contribution(monkeypatch) -> None:
    """A swap field that raises outside int() coercion drops only its share."""

    class _BadInt:
        def __int__(self) -> int:
            raise RuntimeError("unreadable swap")

    _install(monkeypatch, [_Info(100, 5), _Info(200, _BadInt())])
    obs = observe_tree_memory()
    assert obs.pss_bytes == 300
    assert obs.swap_bytes == 5
