"""Fixed parameter snapshot for research provenance.

Carries the decision-constant snapshot bound into preregistered procedures and
run-history records. Sealed deployment params, bootstrap envelopes, digests and
runtime gates were retired with the legacy live stack.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

PARAMS_SNAPSHOT_KEYS: tuple[str, ...] = (
    "SIGNAL_PANEL_WINDOW_DAYS",
    "SIGNAL_REPLAY_WARMUP_DAYS",
    "SIGNAL_RETURN_TAIL_DAYS",
    "SIGNAL_OVERLAP_TOLERANCE",
    "FOLD_PANEL_WARMUP_HOURS",
    "COMMITTEE_PURGE_HOURS",
    "COMMITTEE_OOS_START",
    "PNL_VOL_TARGET_SCALE_FLOOR",
    "COMMITTEE_KELLY_WINDOW_DAYS",
    "COMMITTEE_KELLY_FRACTION",
    "COMMITTEE_KELLY_LCB_Z",
    "COMMITTEE_ADMISSION_PROCEDURE",
    "INSTRUMENT_LIFECYCLE_PROCEDURE",
    "INSTRUMENT_SETTLEMENT_REGISTRY_DIGEST",
    "VENUE_HALT_REGISTRY_DIGEST",
)


#: Snapshot keys owned by ``src.lab.mhs.params`` after the part-4 split; every
#: other plain key is owned by ``src.core.params``.
_LAB_SNAPSHOT_KEYS: frozenset[str] = frozenset({
    "SIGNAL_REPLAY_WARMUP_DAYS",
    "SIGNAL_RETURN_TAIL_DAYS",
    "SIGNAL_OVERLAP_TOLERANCE",
    "FOLD_PANEL_WARMUP_HOURS",
    "PNL_VOL_TARGET_SCALE_FLOOR",
    "COMMITTEE_KELLY_WINDOW_DAYS",
    "COMMITTEE_KELLY_FRACTION",
    "COMMITTEE_KELLY_LCB_Z",
    "COMMITTEE_ADMISSION_PROCEDURE",
    "INSTRUMENT_LIFECYCLE_PROCEDURE",
})


def capture_params_snapshot() -> dict[str, Any]:
    from src.core import params as core_params
    from src.lab.mhs import params as lab_params

    snap: dict[str, Any] = {}
    for key in PARAMS_SNAPSHOT_KEYS:
        if key == "INSTRUMENT_SETTLEMENT_REGISTRY_DIGEST":
            from src.core.instrument_settlements import load_instrument_settlement_registry

            snap[key] = load_instrument_settlement_registry().digest
            continue
        if key == "VENUE_HALT_REGISTRY_DIGEST":
            from src.core.venue_halts import load_venue_halt_registry

            snap[key] = load_venue_halt_registry().digest
            continue
        val = getattr(lab_params if key in _LAB_SNAPSHOT_KEYS else core_params, key)
        if key == "COMMITTEE_OOS_START":
            snap[key] = pd.Timestamp(val).isoformat()
        else:
            snap[key] = val
    return snap
