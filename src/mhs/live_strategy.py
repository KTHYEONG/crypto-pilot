# ruff: noqa
"""Frozen parameter snapshot for research provenance.

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
)


def capture_params_snapshot() -> dict[str, Any]:
    from src.mhs import params as mhs_params

    snap: dict[str, Any] = {}
    for key in PARAMS_SNAPSHOT_KEYS:
        val = getattr(mhs_params, key)
        if key == "COMMITTEE_OOS_START":
            snap[key] = pd.Timestamp(val).isoformat()
        else:
            snap[key] = val
    return snap
