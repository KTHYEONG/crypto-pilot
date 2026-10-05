"""Frozen trial-identity baseline (MHS-1 Stage 0, I-ID-STABLE / I-DEFAULT-DECOUPLED).

The identity key of a recorded configuration is canonicalized against
``TRIAL_IDENTITY_BASELINE``, never against live request defaults, so a later
default change cannot re-key recorded trials.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from dataclasses import fields
from typing import Any

import pytest

import src.mhs.contracts as contracts_module
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.params import SEARCH_TRIALS_ATTEMPTED
from src.mhs.run_history import (
    RESEARCH_NEUTRAL_FLAGS,
    TRIAL_IDENTITY_BASELINE,
    _identity_dump,
    _sparse_identity_key,
    append_run_history_record,
    derive_trials_attempted,
    trial_identity_key,
)

_BASELINE_V1: dict[str, Any] = {
    "start": None,
    "end": None,
    "partition": "dev",
    "data_root": None,
    "execution_timeframe": "3m",
    "execution_universe_size": 30,
    "max_rss_bytes": None,
    "log_run": True,
    "touch_diagnostic": False,
    "ladder_diagnostic": False,
    "peg_chase_diagnostic": False,
    "liquidity_cost_model": "flat",
    "passive_timeout_minutes": 30,
    "discovery_gate": False,
    "discovery_gate_adjusted_net_t": False,
    "discovery_gate_regime_scaled_net_t": False,
    "fold_safe_horizon_selection": False,
    "crash_regime_tilt_alpha": None,
    "slow_book_mode": "single_horizon",
    "fast_book_mode": "single_horizon",
    "rebalance_filter": "per_symbol_deadband",
    "beta_neutralize": False,
    "ensemble_signal": "raw",
    "trend_efficiency_overlay": False,
    "pnl_vol_target": True,
    "pnl_vol_target_mode": "median_relative",
    "trend_sleeve": False,
    "trend_sleeve_gross": 0.0,
    "multi_feature_book": False,
    "committee_book": False,
    "committee_kelly_sizing": False,
    "committee_growth_diagnostic": False,
    "committee_capital": False,
    "committee_member_set": "risk_premia",
    "committee_tranche_smoothing": False,
    "committee_regime_adaptive_tranche": False,
    "committee_tranche_count": 3,
    "committee_target_gross": "<builtins.object>",
    "committee_evidence_weighting": False,
    "funding_carry_sleeve": False,
    "funding_carry_weight": 0.0,
    "execution_coverage_gate": False,
    "exposure_scale_two_sided": False,
    "exposure_drawdown_brake": False,
    "name_drift_trim": False,
    "ram_guard": True,
    "growth_envelope": "conservative",
    "committee_member_attribution": False,
    "final_oos_2026h1": False,
    "forward_registration_digest": None,
    "data_policy": "zombie_mask_v1",
    "input_manifest_path": None,
    "forward_execution_quality_dir": None,
    "forward_strategy_digest": None,
}


def _pinned_records() -> list[dict[str, Any]]:

    run_config_flags = json.loads(json.dumps(dataclasses.asdict(MhsDiagnosticRequest())))
    return [
        {"flags": {}, "params_snapshot": {"K": 1}},
        {
            "flags": {
                "committee_capital": True,
                "committee_target_gross": 0.92,
                "execution_universe_size": 60,
                "pnl_vol_target_mode": "growth_budget",
            },
            "params_snapshot": {"K": 1},
        },
        {
            "flags": {
                "log_run": False,
                "committee_target_gross": None,
                "data_policy": "zombie_mask_v1",
            },
            "params_snapshot": {"K": 1},
        },
        {"flags": {"unknown_alpha": 3}},
        {"flags": run_config_flags, "params_snapshot": {"K": 1}},
    ]


_PINNED_KEYS = (
    '{"data_policy": "legacy", "params_snapshot": {"K": 1}}',
    '{"committee_capital": true, "committee_target_gross": 0.92, "data_policy": "legacy", "execution_universe_size": 60, "params_snapshot": {"K": 1}, "pnl_vol_target_mode": "growth_budget"}',
    '{"params_snapshot": {"K": 1}}',
    '{"data_policy": "legacy", "params_snapshot": {"__legacy_params_snapshot__": "missing"}, "unknown_alpha": 3}',
    '{"committee_capital": true, "committee_evidence_weighting": true, "committee_kelly_sizing": true, "committee_member_set": "flow_momentum", "committee_regime_adaptive_tranche": true, "committee_target_gross": 0.92, "execution_universe_size": 60, "exposure_scale_two_sided": true, "funding_carry_sleeve": true, "funding_carry_weight": 0.3, "growth_envelope": "growth_extreme_budgeted", "params_snapshot": {"K": 1}, "pnl_vol_target_mode": "growth_budget"}',
)


def _unified_request_cls():  # type: ignore[no-untyped-def]

    cfg = MhsDiagnosticRequest()
    spec = []
    for f in fields(MhsDiagnosticRequest):
        v = getattr(cfg, f.name)
        v = v.value if hasattr(v, "value") else v
        spec.append((f.name, object, dataclasses.field(default=v, metadata=dict(f.metadata))))
    return dataclasses.make_dataclass("MhsDiagnosticRequest", spec, frozen=True)


def _bool_inverted_request_cls():  # type: ignore[no-untyped-def]
    spec = []
    for f in fields(MhsDiagnosticRequest):
        v = f.default
        if isinstance(v, bool):
            v = not v
        spec.append((f.name, object, dataclasses.field(default=v, metadata=dict(f.metadata))))
    return dataclasses.make_dataclass("MhsDiagnosticRequest", spec, frozen=True)


def _trial_record(run_id: str, flags: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "COMPLETE",
        "flags": flags,
        "start": "2021-01-01T00:00:00+00:00",
        "resolved_end": "2025-12-31T23:59:59+00:00",
        "blend": {"primary_naive_sharpe": 2.0},
        "research_go": {"reason_codes": [], "data_integrity_reason_codes": []},
    }


def test_pinned_keys_are_stable() -> None:
    records = _pinned_records()
    assert len(records) == len(_PINNED_KEYS) == 5
    for record, pinned in zip(records, _PINNED_KEYS, strict=True):
        assert trial_identity_key(record) == pinned


def test_default_decoupled_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    records = _pinned_records()
    baseline_items = list(TRIAL_IDENTITY_BASELINE.items())
    rng = np.random.default_rng(11)
    random_flags: list[dict[str, Any]] = []
    for _ in range(20):
        chosen = rng.choice(len(baseline_items), size=int(rng.integers(0, len(baseline_items) + 1)), replace=False)
        flags: dict[str, Any] = {}
        for idx in chosen:
            name, base = baseline_items[int(idx)]
            if rng.random() < 0.5:
                flags[name] = base
            elif isinstance(base, bool):
                flags[name] = not base
            elif isinstance(base, int) and not isinstance(base, bool):
                flags[name] = base + 30
            elif isinstance(base, float):
                flags[name] = base + 0.5
            elif base is None:
                flags[name] = "alt"
            else:
                flags[name] = f"{base}_alt"
        random_flags.append(flags)
    random_records = [{"flags": f, "params_snapshot": {"K": 1}} for f in random_flags]
    all_records = [*records, *random_records]
    expected = [trial_identity_key(r) for r in all_records]
    expected_sparse = [_sparse_identity_key(k) for k in expected if k is not None]

    for clone in (_unified_request_cls(), _bool_inverted_request_cls()):
        monkeypatch.setattr(contracts_module, "MhsDiagnosticRequest", clone)
        for record, want, want_sparse in zip(all_records, expected, expected_sparse, strict=True):
            got = trial_identity_key(record)
            assert got == want
            assert got is not None
            assert want is not None
            assert _sparse_identity_key(got) == want_sparse


def test_baseline_covers_the_request_schema() -> None:
    request_names = {f.name for f in fields(MhsDiagnosticRequest)}
    assert request_names - RESEARCH_NEUTRAL_FLAGS <= set(TRIAL_IDENTITY_BASELINE), (
        "new MhsDiagnosticRequest field must be appended to TRIAL_IDENTITY_BASELINE "
        "with its introduction default"
    )


def test_baseline_is_append_only() -> None:
    assert _BASELINE_V1.items() <= TRIAL_IDENTITY_BASELINE.items()
    assert list(TRIAL_IDENTITY_BASELINE)[:54] == list(_BASELINE_V1)


def test_baseline_is_immutable() -> None:
    with pytest.raises(TypeError):
        TRIAL_IDENTITY_BASELINE["start"] = "x"  # type: ignore[index]


def test_sparse_rekey_is_idempotent() -> None:
    dense_key = _identity_dump({**_BASELINE_V1, "committee_capital": True, "params_snapshot": {"K": 1}})
    once = _sparse_identity_key(dense_key)
    twice = _sparse_identity_key(once)
    expected = trial_identity_key(
        {"flags": {"committee_capital": True, "data_policy": "zombie_mask_v1"}, "params_snapshot": {"K": 1}}
    )
    assert once == twice == expected


def test_stored_identity_drift_fails_closed(tmp_path) -> None:
    from src.common.errors import DataIntegrityError

    history_dir = tmp_path / "history"
    append_run_history_record(_trial_record("r0", {"committee_capital": True}), history_dir)
    registry = history_dir / "registry.sqlite3"
    conn = sqlite3.connect(str(registry))
    try:
        conn.execute("UPDATE history_records SET identity_key = ?", ('{"tampered": 1}',))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError, match="trial identity drift"):
        derive_trials_attempted(history_dir)


def test_non_sparse_ledger_key_fails_closed(tmp_path) -> None:
    from src.common.errors import DataIntegrityError

    history_dir = tmp_path / "history"
    append_run_history_record(_trial_record("r0", {"committee_capital": True}), history_dir)
    registry = history_dir / "registry.sqlite3"
    dense_key = _identity_dump({"beta_neutralize": False, "params_snapshot": {"K": 1}})
    assert '"beta_neutralize": false' in dense_key
    conn = sqlite3.connect(str(registry))
    try:
        conn.execute(
            "INSERT OR IGNORE INTO trials (namespace, identity_key, first_seen, provenance_json)"
            " VALUES (?, ?, ?, ?)",
            ("mhs_legacy_horizon", dense_key, "2026-09-02T00:00:00+00:00", "{}"),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError, match="baseline form"):
        derive_trials_attempted(history_dir)


def test_no_denominator_inflation_under_default_change(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    history_dir = tmp_path / "history"
    sparse_sets = ({}, {"committee_capital": True}, {"execution_universe_size": 60})
    for index, flags in enumerate(sparse_sets):
        append_run_history_record(_trial_record(f"r{index}", dict(flags)), history_dir)
    before, _ = derive_trials_attempted(history_dir)
    assert before == SEARCH_TRIALS_ATTEMPTED + 3

    monkeypatch.setattr(contracts_module, "MhsDiagnosticRequest", _unified_request_cls())
    after_patch, _ = derive_trials_attempted(history_dir)
    assert after_patch == SEARCH_TRIALS_ATTEMPTED + 3

    for index, flags in enumerate(sparse_sets):
        append_run_history_record(_trial_record(f"re{index}", dict(flags)), history_dir)
    after_reappend, _ = derive_trials_attempted(history_dir)
    assert after_reappend == SEARCH_TRIALS_ATTEMPTED + 3
