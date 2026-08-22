"""Fork-COW shared payload and RAM-aware worker planning for MHS parallel paths.

``ProcessPoolExecutor`` (fork) pickles every ``submit`` argument into the call
queue even in a fork context, so large read-only panels travel at full copy
cost.  ``fork_shared_payload`` instead registers a payload in a module-global
registry that fork children inherit copy-on-write; only a short token crosses
the ``submit`` boundary, and the payload is resolved child-side with
``resolve_fork_shared``.  ``plan_worker_count`` derives the worker count from
system RAM rather than a hardcoded constant, and ``assert_fork_admission`` is a
pre-fork barrier that fails closed before the projected fork demand can breach
the system reserve (closing the guard blind spot between staged RSS checks).
"""

from __future__ import annotations

import gc
import multiprocessing
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from multiprocessing.context import BaseContext
from typing import Any

import psutil

from src.common.errors import DataIntegrityError
from src.mhs.types import RAM_RESERVE_FLOOR_BYTES, RAM_RESERVE_FRACTION

__all__ = [
    "FORK_CONTEXT",
    "assert_fork_admission",
    "fork_shared_payload",
    "plan_worker_count",
    "resolve_fork_shared",
]

#: The explicit fork context pinned as ``mp_context`` for every MHS
#: ``ProcessPoolExecutor``.  ``spawn`` is unusable here: it re-imports the
#: module in each worker and drops the caller's monkeypatched
#: ``funding_path``/``_mark_price_path``, and pickling still copies the panels.
FORK_CONTEXT: BaseContext = multiprocessing.get_context("fork")

#: Module-global read-only payload registry inherited copy-on-write by fork
#: children.  Keys are uuid4 hex tokens; values are arbitrary mappings.
_FORK_SHARED: dict[str, Mapping[str, Any]] = {}


def _system_reserve_bytes() -> int:
    """The system RAM reserve floor: ``max(5% of total, 256 MiB)``."""
    total = int(psutil.virtual_memory().total)
    return max(int(total * RAM_RESERVE_FRACTION), RAM_RESERVE_FLOOR_BYTES)


@contextmanager
def fork_shared_payload(payload: Mapping[str, Any]) -> Iterator[str]:
    """Register ``payload`` in the fork-shared registry and yield its token.

    The payload stays registered while the context is active, so fork workers
    created inside the block inherit it copy-on-write and resolve it by token
    with ``resolve_fork_shared`` at zero pickle cost.  On exit the token is
    removed and ``gc.collect()`` releases the parent reference.
    """
    token = uuid.uuid4().hex
    _FORK_SHARED[token] = payload
    try:
        yield token
    finally:
        _FORK_SHARED.pop(token, None)
        gc.collect()


def resolve_fork_shared(token: str) -> Mapping[str, Any]:
    """Resolve a fork-shared payload token inside a child process.

    An unregistered token raises ``DataIntegrityError`` so a pool accidentally
    started with a non-fork start method fails closed instead of silently
    recomputing.
    """
    try:
        return _FORK_SHARED[token]
    except KeyError:
        raise DataIntegrityError(
            f"unregistered fork-shared payload token: {token}"
        ) from None


def plan_worker_count(
    requested: int,
    per_worker_bytes: int,
    ram_guard: bool,
    *,
    observer: Callable[[str, int, int, int, int], None] | None = None,
) -> int:
    """Clamp the requested worker count to what the available RAM supports.

    ``min(requested, cpu_count)`` is further clamped by
    ``max(1, (available - reserve) // per_worker_bytes)`` when ``ram_guard``
    is True; with ``ram_guard=False`` the CPU bound is returned unchanged.
    ``per_worker_bytes <= 0`` raises ``ValueError``.  A psutil observational
    failure disables only the RAM clamp, never the CPU bound.

    ``observer`` (default ``None`` keeps every existing call byte-identical)
    is invoked once per RAM-clamped decision with
    ``(stage, requested, granted, available_bytes, reserve_bytes)`` so the
    decision is recorded in the report rather than silently swallowed; an
    observer failure is itself observational and never changes the result.
    """
    if per_worker_bytes <= 0:
        raise ValueError("per_worker_bytes must be positive")
    capped = min(requested, psutil.cpu_count() or 1)
    if not ram_guard:
        return capped
    try:
        available = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - observational
        return capped
    reserve = _system_reserve_bytes()
    by_ram = (available - reserve) // per_worker_bytes
    granted = max(1, min(capped, by_ram))
    if observer is not None:
        with suppress(Exception):
            observer("ram_guard", requested, granted, available, reserve)
    return granted


def assert_fork_admission(
    stage: str,
    workers: int,
    per_worker_bytes: int,
    reserve_bytes: int | None,
) -> None:
    """Fail-closed pre-fork barrier: projected demand must keep headroom.

    ``workers * per_worker_bytes`` projected against
    ``psutil.virtual_memory().available`` must leave at least ``reserve_bytes``
    free; otherwise ``DataIntegrityError`` whose message begins with
    ``fork admission`` is raised BEFORE the pool forks (the OS OOM killer
    otherwise terminates the whole WSL VM).  ``reserve_bytes=None`` is a
    no-op.  psutil failures are observational (no-op).
    """
    if reserve_bytes is None:
        return
    try:
        available = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - observational
        return
    projected = workers * per_worker_bytes
    if available - projected < reserve_bytes:
        raise DataIntegrityError(
            f"fork admission {stage}: projected demand {projected} bytes leaves "
            f"{available - projected} bytes available below the {reserve_bytes}-byte "
            "system reserve"
        )
