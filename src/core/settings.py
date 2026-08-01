"""Environment-overridable hardware settings.

Every execution-budget value lives here so automatic parallelism is never
introduced without an explicit, auditable configuration. ``HARDWARE_MAX_WORKERS``
defaults to the detected CPU count and an environment override may only lower
the cap; a guessed fixed worker count is never used.
"""

from __future__ import annotations

import os

_HARDWARE_MAX_WORKERS_ENV = "HARDWARE_MAX_WORKERS"


def _default_hardware_max_workers() -> int:
    detected = os.cpu_count() or 1
    raw = os.environ.get(_HARDWARE_MAX_WORKERS_ENV)
    if raw is None:
        return detected
    try:
        cap = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_HARDWARE_MAX_WORKERS_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if cap < 1:
        raise ValueError(
            f"{_HARDWARE_MAX_WORKERS_ENV} must be >= 1, got {cap}"
        )
    # The environment override may only reduce the effective process-pool cap.
    return min(detected, cap)


HARDWARE_MAX_WORKERS: int = _default_hardware_max_workers()

assert HARDWARE_MAX_WORKERS >= 1


def effective_worker_count(
    distinct_symbol_count: int,
    *,
    requested: int | None = None,
    hardware_cap: int = HARDWARE_MAX_WORKERS,
) -> int:
    """Bounded execution budget for the symbol-level admission workers.

    The effective count is ``min(hardware_cap, requested, distinct_symbol_count)``
    and is never an admission criterion: it is telemetry that must not
    participate in an admission or proposal fingerprint. ``requested=1`` forces
    the identical sequential code path.
    """
    if distinct_symbol_count < 1:
        raise ValueError("distinct_symbol_count must be >= 1")
    if requested is not None and requested < 1:
        raise ValueError(f"requested must be >= 1, got {requested}")
    if hardware_cap < 1:
        raise ValueError(f"hardware_cap must be >= 1, got {hardware_cap}")
    requested_value = requested if requested is not None else hardware_cap
    return min(hardware_cap, requested_value, distinct_symbol_count)
