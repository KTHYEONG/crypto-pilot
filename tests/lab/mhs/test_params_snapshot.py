"""Params snapshot unchanged by the part-4 split (spec 38 part 4).

The snapshot captured at the pre-move commit is recorded literally below;
``capture_params_snapshot()`` must equal it, and the trial identity of a
fixture run record must be unchanged, so every historical run keeps its key.
"""

from __future__ import annotations

# Byte-identical capture from the pre-move commit (whatever orders the dict,
# equality is order-insensitive; values must match exactly).
EXPECTED_SNAPSHOT: dict[str, object] = {
    "COMMITTEE_ADMISSION_PROCEDURE": "boundary_frozen_warmup_excluded_v1",
    "COMMITTEE_KELLY_FRACTION": 0.5,
    "COMMITTEE_KELLY_LCB_Z": 0.0,
    "COMMITTEE_KELLY_WINDOW_DAYS": 42,
    "COMMITTEE_OOS_START": "2023-01-01T00:00:00+00:00",
    "COMMITTEE_PURGE_HOURS": 720,
    "FOLD_PANEL_WARMUP_HOURS": 912,
    "INSTRUMENT_LIFECYCLE_PROCEDURE": "pit_registry_settlement_halts_causal_exclusions_exit_deferral",
    "INSTRUMENT_SETTLEMENT_REGISTRY_DIGEST": "sha256:9832902381e85e9ebf6d693f2e7571ff2ebc5c8556ccf4240393859b71aac8b9",
    "PNL_VOL_TARGET_SCALE_FLOOR": 0.2,
    "SIGNAL_OVERLAP_TOLERANCE": 1e-09,
    "SIGNAL_PANEL_WINDOW_DAYS": 400,
    "SIGNAL_REPLAY_WARMUP_DAYS": 30,
    "SIGNAL_RETURN_TAIL_DAYS": 400,
    "VENUE_HALT_REGISTRY_DIGEST": "sha256:67b420f149af3f192788192cdba81d262c5744093f1281a8bde6527c9867409c",
}

EXPECTED_TRIAL_IDENTITY_KEY = '{"committee_member_set": "flow_momentum", "data_policy": "legacy", "params_snapshot": {"COMMITTEE_ADMISSION_PROCEDURE": "boundary_frozen_warmup_excluded_v1", "COMMITTEE_KELLY_FRACTION": 0.5, "COMMITTEE_KELLY_LCB_Z": 0.0, "COMMITTEE_KELLY_WINDOW_DAYS": 42, "COMMITTEE_OOS_START": "2023-01-01T00:00:00+00:00", "COMMITTEE_PURGE_HOURS": 720, "FOLD_PANEL_WARMUP_HOURS": 912, "INSTRUMENT_LIFECYCLE_PROCEDURE": "pit_registry_settlement_halts_causal_exclusions_exit_deferral", "INSTRUMENT_SETTLEMENT_REGISTRY_DIGEST": "sha256:9832902381e85e9ebf6d693f2e7571ff2ebc5c8556ccf4240393859b71aac8b9", "PNL_VOL_TARGET_SCALE_FLOOR": 0.2, "SIGNAL_OVERLAP_TOLERANCE": 1e-09, "SIGNAL_PANEL_WINDOW_DAYS": 400, "SIGNAL_REPLAY_WARMUP_DAYS": 30, "SIGNAL_RETURN_TAIL_DAYS": 400, "VENUE_HALT_REGISTRY_DIGEST": "sha256:67b420f149af3f192788192cdba81d262c5744093f1281a8bde6527c9867409c"}}'


def test_params_snapshot_unchanged_by_split() -> None:
    from src.lab.mhs.live_strategy import PARAMS_SNAPSHOT_KEYS, capture_params_snapshot

    assert set(PARAMS_SNAPSHOT_KEYS) == set(EXPECTED_SNAPSHOT)
    assert capture_params_snapshot() == EXPECTED_SNAPSHOT


def test_trial_identity_key_unchanged() -> None:
    from src.lab.mhs.live_strategy import capture_params_snapshot
    from src.lab.mhs.run_history import trial_identity_key

    record = {
        "run_id": "snapshot-pinned",
        "status": "COMPLETE",
        "flags": {"committee_member_set": "flow_momentum"},
        "start": "2021-01-01T00:00:00+00:00",
        "resolved_end": "2025-12-31T23:59:59+00:00",
        "blend": {"primary_naive_sharpe": 2.0},
        "research_go": {"reason_codes": [], "data_integrity_reason_codes": []},
        "params_snapshot": capture_params_snapshot(),
    }
    assert trial_identity_key(record) == EXPECTED_TRIAL_IDENTITY_KEY
