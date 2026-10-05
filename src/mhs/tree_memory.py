"""Single-sweep process-tree memory observation for fail-closed admission.

Each ``memory_full_info`` call parses one process's smaps and briefly holds its
memory-map lock; admission that needs both resident PSS and swap reads them from
the same per-process observation instead of walking the tree twice.
"""

from __future__ import annotations

import dataclasses
import os

import psutil

from src.common.errors import DataIntegrityError


@dataclasses.dataclass(frozen=True, slots=True)
class TreeMemoryObservation:
    """Process-tree memory at one sweep.

    Attributes:
        pss_bytes: Sum of PSS over every readable live process of the tree.
        swap_bytes: Sum of swap over processes whose swap field was readable;
            None when no process contributed (unknown, never a fabricated zero).
    """

    pss_bytes: int
    swap_bytes: int | None


def observe_tree_memory() -> TreeMemoryObservation:
    """Observe this process tree's PSS (mandatory) and swap (optional) in one sweep.

    Returns:
        Observation whose ``pss_bytes`` equals ``_current_tree_pss_bytes()`` and
        whose ``swap_bytes`` equals ``_current_tree_swap_bytes()`` for the same
        tree state.
    Raises:
        DataIntegrityError: ``"physical-memory telemetry unavailable: ..."`` when
            the tree cannot be enumerated, a live process is unreadable or
            access-denied, a PSS value is missing or invalid, or no process
            yields PSS.
    """
    try:
        me = psutil.Process(os.getpid())
        procs = [me, *me.children(recursive=True)]
    except Exception as exc:  # noqa: BLE001
        raise DataIntegrityError(
            f"physical-memory telemetry unavailable: cannot enumerate process tree: {exc}"
        ) from exc
    pss_total = 0
    swap_total = 0
    observed = 0
    swap_observed = False
    for proc in procs:
        try:
            info = proc.memory_full_info()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise DataIntegrityError(
                f"physical-memory telemetry unavailable: unreadable live process: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise DataIntegrityError(
                f"physical-memory telemetry unavailable: cannot read process memory: {exc}"
            ) from exc
        pss = getattr(info, "pss", None)
        if pss is None:
            raise DataIntegrityError(
                "physical-memory telemetry unavailable: process-tree PSS observation missing"
            )
        try:
            pss_total += int(pss)
        except (TypeError, ValueError) as exc:
            raise DataIntegrityError(
                f"physical-memory telemetry unavailable: invalid PSS observation: {exc}"
            ) from exc
        observed += 1
        try:
            swap_total += int(getattr(info, "swap", 0))
            swap_observed = True
        except (TypeError, ValueError):
            continue
        except Exception:  # noqa: BLE001, S112 - optional observation never raises
            continue
    if observed == 0:
        raise DataIntegrityError(
            "physical-memory telemetry unavailable: no process-tree PSS observation"
        )
    return TreeMemoryObservation(
        pss_bytes=pss_total, swap_bytes=swap_total if swap_observed else None
    )
