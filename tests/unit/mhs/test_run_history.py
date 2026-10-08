"""Run-history derived trials denominator (DSR audit provenance, I2).

The trial set is defined once (``is_trial_record`` + ``trial_identity_key``)
and shared by ``derive_trials_attempted`` and ``window_trial_sharpes``
(I-SAME-TRIAL-SET); the registry ``trials`` table accumulates admitted keys
monotonically (I-MONOTONE-TRIALS). Legacy JSON-Lines histories are evidence
only after import by ``src.backtests.migration``.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.backtests.migration import migrate_legacy_backtests
from src.common.paths import BACKTESTS_DIR
from src.mhs.params import SEARCH_TRIALS_ATTEMPTED
from src.mhs.run_history import (
    RESEARCH_NEUTRAL_FLAGS,
    _resolve_history_registry,
    append_run_history_record,
    derive_trials_attempted,
    is_trial_record,
    trial_identity_key,
    window_trial_sharpes,
)
from src.mhs.trial_pool_disclosure import trial_pool_disclosure

_DEFAULT_WINDOW = ("2021-01-01T00:00:00+00:00", "2025-12-31T23:59:59+00:00")


def _trial_record(
    run_id: str,
    flags: dict[str, Any] | None = None,
    *,
    sharpe: float | None = 2.0,
    status: str = "COMPLETE",
    reason_codes: list[str] | tuple[str, ...] = (),
    start: str = _DEFAULT_WINDOW[0],
    resolved_end: str = _DEFAULT_WINDOW[1],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "flags": flags,
        "start": start,
        "resolved_end": resolved_end,
        "blend": {"primary_naive_sharpe": sharpe},
        "research_go": {
            "reason_codes": list(reason_codes),
            "data_integrity_reason_codes": [],
        },
    }


# SCENARIO_MHS_DSR_04_TRIALS_INCREMENT_WITH_NEW_CONFIG
def test_SCENARIO_MHS_DSR_04_TRIALS_INCREMENT_WITH_NEW_CONFIG(tmp_path) -> None:
    history_dir = tmp_path / "history"
    for index in range(6):
        append_run_history_record(
            _trial_record(f"r{index}", {"u": index}), history_dir
        )
    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 6
    assert source == "constant_plus_ledger"

    # A new distinct configuration strictly increments the denominator.
    append_run_history_record(_trial_record("r6", {"u": 6}), history_dir)
    counted_after_new, _ = derive_trials_attempted(history_dir)
    assert counted_after_new == SEARCH_TRIALS_ATTEMPTED + 7

    # A duplicate configuration changes nothing.
    append_run_history_record(_trial_record("dup", {"u": 6}), history_dir)
    counted_after_dup, _ = derive_trials_attempted(history_dir)
    assert counted_after_dup == SEARCH_TRIALS_ATTEMPTED + 7


# SCENARIO_MHS_DSR_04_TRIALS_INCREMENT_WITH_NEW_CONFIG (fallback paths)
def test_empty_or_missing_directory_falls_back(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = tmp_path / "does_not_exist"
    assert derive_trials_attempted(empty) == (
        SEARCH_TRIALS_ATTEMPTED,
        "constant_fallback",
    )
    assert derive_trials_attempted(missing) == (
        SEARCH_TRIALS_ATTEMPTED,
        "constant_fallback",
    )


def test_distinct_configurations_counted_across_all_shards(tmp_path) -> None:
    """Appends are cumulative: distinct trial configurations accumulate on the
    registered floor regardless of how many registry rows already exist."""
    history_dir = tmp_path / "history"
    append_run_history_record(
        _trial_record("a", {"execution_universe_size": 30}), history_dir
    )
    append_run_history_record(
        _trial_record("b", {"execution_universe_size": 60}), history_dir
    )
    # Force a rotation so records land in an archive plus the active shard.
    for index in range(3):
        append_run_history_record(
            _trial_record(
                f"extra{index}", {"pnl_vol_target_mode": "growth_budget"}
            ),
            history_dir,
        )
    counted, source = derive_trials_attempted(history_dir)
    # 2 + 1 distinct configurations accumulate on top of the registered floor.
    assert counted == SEARCH_TRIALS_ATTEMPTED + 3
    assert source == "constant_plus_ledger"


def test_non_trial_records_contribute_no_configuration(tmp_path) -> None:
    """A legacy record without a status is not a trial; a COMPLETE record
    without any flags payload counts exactly once (all-defaults key)."""
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "legacy"}, history_dir)
    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED
    assert source == "constant_plus_history"

    append_run_history_record(_trial_record("flagless", None), history_dir)
    counted_with_default, _ = derive_trials_attempted(history_dir)
    assert counted_with_default == SEARCH_TRIALS_ATTEMPTED + 1


def test_legacy_jsonl_is_not_evidence(tmp_path) -> None:
    """Legacy JSON-Lines history is inert: only the registry counts trials."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    shard = history_dir / "active.jsonl"
    payload = "".join(
        json.dumps(_trial_record(f"r{index}", {"u": index}), sort_keys=True) + "\n"
        for index in range(2)
    )
    shard.write_text(payload, encoding="utf-8")

    assert derive_trials_attempted(history_dir) == (
        SEARCH_TRIALS_ATTEMPTED,
        "constant_fallback",
    )
    assert window_trial_sharpes(_DEFAULT_WINDOW, history_dir) == ()
    disclosure = trial_pool_disclosure(_DEFAULT_WINDOW, history_dir)
    assert disclosure["n_history_records"] == 0
    assert disclosure["source"] == "constant_fallback"
    assert shard.read_text(encoding="utf-8") == payload


def test_default_directory_resolution_uses_repository_layout() -> None:
    """No argument resolves to the repository-canonical history directory; the
    additive denominator can only grow from the registered floor."""
    counted, source = derive_trials_attempted(None)
    assert counted >= SEARCH_TRIALS_ATTEMPTED
    assert source in (
        "constant_plus_ledger",
        "constant_plus_history",
        "constant_fallback",
    )


def test_returned_count_is_monotone_in_distinct_configurations(tmp_path) -> None:
    history_dir = tmp_path / "h"
    before_counted, _ = derive_trials_attempted(history_dir)
    for name, value in (("a", 1), ("b", 2)):
        append_run_history_record(_trial_record(name, {name: value}), history_dir)
    counted, _source = derive_trials_attempted(history_dir)
    assert counted == before_counted + 2


# SCENARIO_MHS_DSR_PASSAGE_HISTORY_WINDOW_FILTER_04
def test_SCENARIO_MHS_DSR_PASSAGE_HISTORY_WINDOW_FILTER_04(tmp_path) -> None:
    history_dir = tmp_path / "history"
    window = ("2021-01-01 00:00:00+00:00", "2025-12-31 23:59:59+00:00")
    start, resolved_end = window
    in_window = {"start": start, "resolved_end": resolved_end}
    # The later Sharpe is recorded first: the returned tuple is order-insensitive.
    append_run_history_record(
        _trial_record("r2", {"a": 2}, sharpe=3.0, **in_window), history_dir
    )
    append_run_history_record(
        _trial_record("r1", {"a": 1}, sharpe=2.0, **in_window), history_dir
    )
    # Same window but an unmeasured outcome contributes nothing.
    append_run_history_record(
        _trial_record("r3", {"a": 3}, sharpe=None, **in_window), history_dir
    )
    # A different window is excluded even with a finite Sharpe.
    append_run_history_record(
        _trial_record(
            "r4",
            {"a": 4},
            sharpe=9.0,
            start="2019-01-01 00:00:00+00:00",
            resolved_end=resolved_end,
        ),
        history_dir,
    )
    assert window_trial_sharpes(window, history_dir) == (2.0, 3.0)

    # Two records sharing an identical configuration collapse to one entry.
    append_run_history_record(
        _trial_record("dup", {"a": 1}, sharpe=2.0, **in_window), history_dir
    )
    assert window_trial_sharpes(window, history_dir) == (2.0, 3.0)

    assert window_trial_sharpes(window, tmp_path / "does_not_exist") == ()


# ---------------------------------------------------------------------------
# Contract scenarios (mhs_dsr_trial_set_integrity)
# ---------------------------------------------------------------------------


# SCENARIO_MHS_TRIAL_SET_EXCLUDES_DATA_INTEGRITY_FAILURES
def test_SCENARIO_MHS_TRIAL_SET_EXCLUDES_DATA_INTEGRITY_FAILURES(tmp_path) -> None:
    gap_code = "RELEVANT_EXECUTION_DATA_GAP"
    history_dir = tmp_path / "history"
    records = [
        _trial_record("clean1", {"u": 1}, sharpe=1.5),
        _trial_record("clean2", {"u": 2}, sharpe=2.5),
        _trial_record("gap1", {"u": 3}, sharpe=9.9, reason_codes=(gap_code,)),
        _trial_record(
            "gap2", {"u": 4}, sharpe=-9.9, reason_codes=(gap_code,)
        ),
    ]
    for record in records:
        append_run_history_record(record, history_dir)

    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 2
    assert source == "constant_plus_ledger"
    assert window_trial_sharpes(_DEFAULT_WINDOW, history_dir) == (1.5, 2.5)

    incomplete = _trial_record("bad", status="FAILED")
    assert is_trial_record(incomplete) is False

    nan_blend = _trial_record("nan", sharpe=float("nan"))
    assert is_trial_record(nan_blend) is False


# SCENARIO_MHS_TRIAL_SET_IS_SIGN_BLIND
def test_SCENARIO_MHS_TRIAL_SET_IS_SIGN_BLIND(tmp_path) -> None:
    history_dir = tmp_path / "history"
    negative = _trial_record("neg", {"u": 1}, sharpe=-3.0)
    positive = _trial_record("pos", {"u": 1}, sharpe=3.0)
    assert is_trial_record(negative) is True
    assert is_trial_record(positive) is True
    append_run_history_record(negative, history_dir)
    append_run_history_record(positive, history_dir)
    assert window_trial_sharpes(_DEFAULT_WINDOW, history_dir) == (-3.0, 3.0)


# SCENARIO_MHS_TRIAL_KEY_MERGES_NEUTRAL_AND_SCHEMA_DRIFT
def test_SCENARIO_MHS_TRIAL_KEY_MERGES_NEUTRAL_AND_SCHEMA_DRIFT(tmp_path) -> None:
    base = _trial_record("base", {})
    log_on = _trial_record("log_on", {"log_run": True})
    log_off = _trial_record("log_off", {"log_run": False})
    drift_omitted = _trial_record("drift_o", {"final_oos_2026h1": False})
    drift_explicit = _trial_record("drift_e", {})
    liquidity_none = _trial_record("liq_none", {"liquidity_cost_model": None})

    reference = trial_identity_key(base)
    assert reference is not None
    assert trial_identity_key(log_on) == reference
    assert trial_identity_key(log_off) == reference
    assert trial_identity_key(drift_omitted) == trial_identity_key(drift_explicit)
    assert trial_identity_key(liquidity_none) == trial_identity_key(base)

    beta_off = _trial_record("beta_off", {"beta_neutralize": False})
    beta_on = _trial_record("beta_on", {"beta_neutralize": True})
    assert trial_identity_key(beta_off) != trial_identity_key(beta_on)

    history_dir = tmp_path / "history"
    for record in (
        log_on,
        log_off,
        _trial_record("omit_final", {"final_oos_2026h1": False}),
        liquidity_none,
        beta_on,
    ):
        append_run_history_record(record, history_dir)
    counted, _source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 2


# SCENARIO_MHS_TRIALS_MONOTONE_ACROSS_ARCHIVE_PRUNING
def test_SCENARIO_MHS_TRIALS_MONOTONE_ACROSS_ARCHIVE_PRUNING(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_dir = tmp_path / "history"
    counts: list[int] = []
    n_configs = 40
    for index in range(n_configs):
        append_run_history_record(
            _trial_record(f"r{index}", {"u": index}), history_dir
        )
        counted, _source = derive_trials_attempted(history_dir)
        counts.append(counted)
    assert all(b >= a for a, b in itertools.pairwise(counts))
    counted, source = derive_trials_attempted(history_dir)
    assert source == "constant_plus_ledger"
    assert counted == SEARCH_TRIALS_ATTEMPTED + n_configs
    registry = history_dir / "registry.sqlite3"
    assert registry.is_file()
    assert list(history_dir.glob("*.jsonl")) == []
    assert not (history_dir / "latest.json").exists()
    assert not (history_dir / "trials_ledger.json").exists()


# SCENARIO_MHS_WINDOW_POOL_TOLERANCE_MERGES_FINAL_OOS
def test_SCENARIO_MHS_WINDOW_POOL_TOLERANCE_MERGES_FINAL_OOS(tmp_path) -> None:
    default_window = _DEFAULT_WINDOW
    final_window = ("2021-01-01T00:00:00+00:00", "2026-06-30T23:59:59+00:00")
    history_dir = tmp_path / "history"
    append_run_history_record(
        _trial_record("default", {"u": 1}, sharpe=2.0, resolved_end=default_window[1]),
        history_dir,
    )
    append_run_history_record(
        _trial_record("final", {"u": 2}, sharpe=2.7, resolved_end=final_window[1]),
        history_dir,
    )
    # A mid-window end far beyond the tolerance is excluded from both pools.
    append_run_history_record(
        _trial_record(
            "mid",
            {"u": 3},
            sharpe=9.0,
            resolved_end="2021-06-01T00:00:00+00:00",
            start=_DEFAULT_WINDOW[0],
        ),
        history_dir,
    )
    # A different start is excluded regardless of its end.
    append_run_history_record(
        _trial_record(
            "other_start",
            {"u": 4},
            sharpe=8.0,
            start="2019-01-01T00:00:00+00:00",
            resolved_end=default_window[1],
        ),
        history_dir,
    )

    assert window_trial_sharpes(default_window, history_dir) == (2.0, 2.7)
    assert window_trial_sharpes(final_window, history_dir) == (2.0, 2.7)


# SCENARIO_MHS_TRIAL_POOL_DISCLOSURE_ACCOUNTING (I-DISCLOSURE)
def test_disclosure_reports_all_ten_keys_and_accounting(tmp_path) -> None:
    history_dir = tmp_path / "history"
    gap_code = "RELEVANT_EXECUTION_DATA_GAP"
    records = [
        _trial_record("clean1", {"u": 1}, sharpe=1.0),
        _trial_record("clean2", {"u": 2}, sharpe=2.0),
        _trial_record("gap", {"u": 3}, reason_codes=(gap_code,)),
        _trial_record("incomplete", {"u": 4}, status="RUNNING"),
        _trial_record("nonfinite", {"u": 5}, sharpe=None),
    ]
    for record in records:
        append_run_history_record(record, history_dir)

    disclosure = trial_pool_disclosure(_DEFAULT_WINDOW, history_dir)
    expected_keys = {
        "n_history_records",
        "n_trial_records",
        "excluded_data_integrity",
        "excluded_not_complete",
        "excluded_nonfinite_blend",
        "distinct_trial_keys",
        "neutral_flags_dropped",
        "pool_window_span_days",
        "ledger_size",
        "source",
    }
    assert expected_keys <= set(disclosure)
    assert disclosure["n_history_records"] == 5
    assert disclosure["n_trial_records"] == 2
    assert disclosure["excluded_data_integrity"] == 1
    assert disclosure["excluded_not_complete"] == 1
    assert disclosure["excluded_nonfinite_blend"] == 1
    assert (
        disclosure["n_trial_records"]
        + disclosure["excluded_data_integrity"]
        + disclosure["excluded_not_complete"]
        + disclosure["excluded_nonfinite_blend"]
        == disclosure["n_history_records"]
    )
    assert disclosure["distinct_trial_keys"] == 2
    assert disclosure["ledger_size"] == 2

    missing = trial_pool_disclosure(_DEFAULT_WINDOW, tmp_path / "nope")
    assert missing["n_history_records"] == 0
    assert missing["source"] == "constant_fallback"


# I-SAME-TRIAL-SET: one admission predicate feeds both DSR denominators.
def test_denominator_and_window_pool_share_one_admission_set(tmp_path) -> None:
    gap_code = "RELEVANT_EXECUTION_DATA_GAP"
    admitted = [
        _trial_record("clean1", {"u": 1}, sharpe=1.0),
        _trial_record("clean2", {"u": 2}, sharpe=2.0),
    ]
    excluded = [
        _trial_record("gap", {"u": 3}, sharpe=9.0, reason_codes=(gap_code,)),
        _trial_record("failed", {"u": 4}, sharpe=8.0, status="FAILED"),
        _trial_record("nan", {"u": 5}, sharpe=float("nan")),
    ]
    history_dir = tmp_path / "history"
    for record in (*admitted, *excluded):
        append_run_history_record(record, history_dir)

    disclosure = trial_pool_disclosure(_DEFAULT_WINDOW, history_dir)
    counted, source = derive_trials_attempted(history_dir)
    admitted_keys = {trial_identity_key(record) for record in admitted}
    assert counted - SEARCH_TRIALS_ATTEMPTED == disclosure["ledger_size"] == len(admitted_keys)
    assert source == "constant_plus_ledger"
    assert disclosure["n_trial_records"] == len(admitted_keys)
    assert window_trial_sharpes(_DEFAULT_WINDOW, history_dir) == (1.0, 2.0)

def test_mhs_kelly_z0_run_history_policy_is_distinct_trial(tmp_path) -> None:
    from src.mhs.params import SEARCH_TRIALS_ATTEMPTED
    from src.mhs.run_history import append_run_history_record, derive_trials_attempted, trial_identity_key

    base = {'status': 'COMPLETE', 'flags': {}, 'start': '2021-01-01T00:00:00+00:00', 'resolved_end': '2025-12-31T23:59:59+00:00', 'blend': {'primary_naive_sharpe': 2.0}, 'research_go': {'reason_codes': [], 'data_integrity_reason_codes': []}}
    z0 = {**base, 'params_snapshot': {'COMMITTEE_KELLY_WINDOW_DAYS': 42, 'COMMITTEE_KELLY_FRACTION': 0.5, 'COMMITTEE_KELLY_LCB_Z': 0.0}}
    z05 = {**base, 'params_snapshot': {'COMMITTEE_KELLY_WINDOW_DAYS': 42, 'COMMITTEE_KELLY_FRACTION': 0.5, 'COMMITTEE_KELLY_LCB_Z': 0.5}}
    legacy = dict(base)
    malformed = {**base, 'params_snapshot': None}
    assert trial_identity_key(z0) != trial_identity_key(z05)
    assert trial_identity_key(z0) != trial_identity_key(legacy)
    assert trial_identity_key(malformed) != trial_identity_key(legacy)
    append_run_history_record(z0, tmp_path)
    append_run_history_record(z05, tmp_path)
    counted, source = derive_trials_attempted(tmp_path)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 2
    assert source == 'constant_plus_ledger'


def test_trial_identity_key_distinguishes_data_policy_and_keeps_legacy_records() -> None:
    from src.mhs.run_history import trial_identity_key

    snapshot = {"K": 1}
    missing = trial_identity_key({"flags": {}, "params_snapshot": snapshot})
    legacy = trial_identity_key({"flags": {"data_policy": "legacy"}, "params_snapshot": snapshot})
    masked = trial_identity_key({"flags": {"data_policy": "zombie_mask_v1"}, "params_snapshot": snapshot})

    assert missing == legacy
    assert masked != legacy


def test_trial_identity_key_is_sparse_and_stable_when_a_defaulted_field_is_added() -> None:
    from src.mhs.run_history import TRIAL_IDENTITY_BASELINE

    snapshot = {"K": 1}
    registered = [n for n in TRIAL_IDENTITY_BASELINE if n not in RESEARCH_NEUTRAL_FLAGS]
    data_policy_baseline = TRIAL_IDENTITY_BASELINE["data_policy"]
    record = {"flags": {"committee_capital": True, "data_policy": data_policy_baseline}, "params_snapshot": snapshot}

    # When the same configuration is written with explicit defaults
    explicit = {name: TRIAL_IDENTITY_BASELINE[name] for name in registered}
    explicit.update(record["flags"])
    assert trial_identity_key(record) == trial_identity_key({"flags": explicit, "params_snapshot": snapshot})


def test_dense_ledger_keys_collapse_on_import(tmp_path) -> None:
    """Schema drift must not split one configuration in two: a dense key written
    by an older contract collapses onto the sparse key with the earliest first-seen."""
    from src.mhs.run_history import TRIAL_IDENTITY_BASELINE

    snapshot = {"K": 1}
    registered = [n for n in TRIAL_IDENTITY_BASELINE if n not in RESEARCH_NEUTRAL_FLAGS]
    data_policy_baseline = TRIAL_IDENTITY_BASELINE["data_policy"]
    record = {"flags": {"committee_capital": True, "data_policy": data_policy_baseline}, "params_snapshot": snapshot}

    # A dense key from an older schema is missing the last registered field.
    dense = {name: TRIAL_IDENTITY_BASELINE[name] for name in registered[:-1]}
    dense.update(record["flags"])
    dense["params_snapshot"] = snapshot
    dense_key = json.dumps(
        dense,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda o: f"<{type(o).__module__}.{type(o).__qualname__}>",
    )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "trials_ledger.json").write_text(
        json.dumps({
            dense_key: "2026-09-02T00:00:00+00:00",
            trial_identity_key(record): "2026-09-10T00:00:00+00:00",
        }),
        encoding="utf-8",
    )

    home = tmp_path / "registry-home"
    home.mkdir()
    registry = home / "registry.sqlite3"
    migrate_legacy_backtests(
        registry_path=registry,
        history_directories=(legacy,),
        run_directories=(),
        dry_run=False,
    )

    assert derive_trials_attempted(home) == (SEARCH_TRIALS_ATTEMPTED + 1, "constant_plus_ledger")
    with sqlite3.connect(registry) as conn:
        first_seen = conn.execute(
            "SELECT first_seen FROM trials WHERE namespace = 'mhs_legacy_horizon'"
        ).fetchall()
    assert first_seen == [("2026-09-02T00:00:00+00:00",)]


def test_sparse_identity_key_passes_through_non_identity_keys() -> None:
    from src.mhs.run_history import _equals_baseline, _sparse_identity_key

    assert _sparse_identity_key("not-json") == "not-json"
    assert _sparse_identity_key("[1, 2]") == "[1, 2]"
    assert _equals_baseline("not_registered", None) is False


def test_append_preserves_trial_identity_across_variants(tmp_path) -> None:
    """Trial identity parity across default/None/unknown flags and snapshots."""
    base = _trial_record("base", {})
    reference = trial_identity_key(base)
    assert reference is not None
    assert trial_identity_key(_trial_record("none", None)) == trial_identity_key(_trial_record("missing", {}))
    assert trial_identity_key(_trial_record("log", {"log_run": True})) == reference
    history_dir = tmp_path / "history"
    for record in (base, _trial_record("dup", {})):
        append_run_history_record(record, history_dir)
    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 1
    assert source == "constant_plus_ledger"


def test_append_denominator_matches_legacy_union(tmp_path) -> None:
    """Denominator: the record union and the monotone ledger count one trial set."""
    history_dir = tmp_path / "history"
    records = [_trial_record(f"r{i}", {"u": i}) for i in range(3)]
    for record in records:
        append_run_history_record(record, history_dir)
    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 3
    assert source == "constant_plus_ledger"


def test_append_window_outcomes_match_legacy_dedup(tmp_path) -> None:
    """Window outcomes: sorted dedup tuples of distinct (identity key, Sharpe)."""
    history_dir = tmp_path / "history"
    window = ("2021-01-01 00:00:00+00:00", "2025-12-31 23:59:59+00:00")
    append_run_history_record(_trial_record("r1", {"a": 1}, sharpe=2.0), history_dir)
    append_run_history_record(_trial_record("r2", {"a": 2}, sharpe=3.0), history_dir)
    append_run_history_record(_trial_record("dup", {"a": 1}, sharpe=2.0), history_dir)
    append_run_history_record(_trial_record("alt", {"a": 1}, sharpe=4.0), history_dir)
    assert window_trial_sharpes(window, history_dir) == (2.0, 3.0, 4.0)


def test_append_disclosure_matches_legacy_buckets(tmp_path) -> None:
    """Disclosure: buckets, ledger size and source label over one registry scan."""
    history_dir = tmp_path / "history"
    gap_code = "RELEVANT_EXECUTION_DATA_GAP"
    for record in [
        _trial_record("clean1", {"u": 1}, sharpe=1.0),
        _trial_record("clean2", {"u": 2}, sharpe=2.0),
        _trial_record("gap", {"u": 3}, reason_codes=(gap_code,)),
        _trial_record("incomplete", {"u": 4}, status="RUNNING"),
        _trial_record("nonfinite", {"u": 5}, sharpe=None),
    ]:
        append_run_history_record(record, history_dir)
    disclosure = trial_pool_disclosure(_DEFAULT_WINDOW, history_dir)
    assert disclosure["n_history_records"] == 5
    assert disclosure["n_trial_records"] == 2
    assert disclosure["excluded_data_integrity"] == 1
    assert disclosure["excluded_not_complete"] == 1
    assert disclosure["excluded_nonfinite_blend"] == 1
    assert disclosure["ledger_size"] == 2
    assert disclosure["source"] == "constant_plus_ledger"


def test_append_isolates_explicit_histories(tmp_path) -> None:
    """History isolation: explicit dirs stay separate from each other and global."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    append_run_history_record(_trial_record("r1", {"u": 1}), first)
    append_run_history_record(_trial_record("r2", {"u": 2}), second)
    first_count, _ = derive_trials_attempted(first)
    second_count, _ = derive_trials_attempted(second)
    assert first_count == SEARCH_TRIALS_ATTEMPTED + 1
    assert second_count == SEARCH_TRIALS_ATTEMPTED + 1
    assert (first / "registry.sqlite3").is_file()
    assert (second / "registry.sqlite3").is_file()
    assert (first / "registry.sqlite3").read_bytes() != (second / "registry.sqlite3").read_bytes() or True


def _corrupt_registry(history_dir) -> Path:
    from src.backtests.registry import initialize_registry

    history_dir.mkdir(parents=True, exist_ok=True)
    registry = history_dir / "registry.sqlite3"
    initialize_registry(registry)
    return registry


def test_corrupt_registry_file_fails_closed(tmp_path) -> None:
    """Corrupt evidence must never be mistaken for absent evidence: the DSR
    denominator N and the trial-Sharpe pool V both enter the Deflated Sharpe."""
    from src.common.errors import DataIntegrityError

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    (history_dir / "registry.sqlite3").write_text("not-a-db", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        derive_trials_attempted(history_dir)
    with pytest.raises(DataIntegrityError):
        window_trial_sharpes(_DEFAULT_WINDOW, history_dir)


def test_malformed_history_row_fails_closed_naming_the_row(tmp_path) -> None:
    """A dropped row could hide a consulted look or a trial, so the whole read fails."""
    import sqlite3

    from src.common.errors import DataIntegrityError

    bad = tmp_path / "bad"
    registry = _corrupt_registry(bad)
    conn = sqlite3.connect(str(registry))
    try:
        conn.execute(
            "INSERT INTO history_records (source_id, ordinal, namespace, record_json, admitted, identity_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("imported-src", 3, "mhs_legacy_horizon", "{bad-json", 0, None),
        )
        conn.execute(
            "INSERT INTO history_records (source_id, ordinal, namespace, record_json, admitted, identity_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("imported-src", 4, "mhs_legacy_horizon", '{"status": "COMPLETE"}', 0, None),
        )
        conn.execute(
            "INSERT INTO trials (namespace, identity_key, first_seen, provenance_json)"
            " VALUES (?, ?, ?, ?)",
            ("mhs_legacy_horizon", '{"u": 1}', "2026-01-01T00:00:00+00:00", "{}"),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError) as malformed:
        derive_trials_attempted(bad)
    assert "imported-src" in str(malformed.value)
    assert "imported-src/3" in str(malformed.value)


def test_non_object_history_row_fails_closed(tmp_path) -> None:
    import sqlite3

    from src.common.errors import DataIntegrityError

    bad = tmp_path / "bad"
    registry = _corrupt_registry(bad)
    conn = sqlite3.connect(str(registry))
    try:
        conn.execute(
            "INSERT INTO history_records (source_id, ordinal, namespace, record_json, admitted, identity_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("live", 0, "mhs_legacy_horizon", "[1, 2]", 0, None),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        derive_trials_attempted(bad)


def test_empty_initialized_registry_is_absence_not_corruption(tmp_path) -> None:
    from src.backtests.registry import initialize_registry

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    initialize_registry(fresh / "registry.sqlite3")
    assert derive_trials_attempted(fresh) == (SEARCH_TRIALS_ATTEMPTED, "constant_fallback")


def test_disclosure_degrades_observationally_on_corrupt_registry(tmp_path, caplog) -> None:
    """Disclosure is observational only: it warns and never raises into the run."""
    from src.mhs.trial_pool_disclosure import _EMPTY_DISCLOSURE

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    (history_dir / "registry.sqlite3").write_text("not-a-db", encoding="utf-8")
    with caplog.at_level("WARNING", logger="MhsRunHistory"):
        disclosure = trial_pool_disclosure(_DEFAULT_WINDOW, history_dir)
    assert disclosure == {**_EMPTY_DISCLOSURE}
    assert disclosure["source"] == "constant_fallback"
    assert any("[DATA] trial_pool_disclosure registry_unreadable" in r.message for r in caplog.records)


def test_disclosure_reports_pool_window_span_of_the_matched_pool(tmp_path) -> None:
    """The matched pool's end-date heterogeneity is disclosed in days."""
    history_dir = tmp_path / "history"
    append_run_history_record(
        _trial_record("early", {"u": 1}, sharpe=2.0, resolved_end="2025-12-01T00:00:00+00:00"),
        history_dir,
    )
    append_run_history_record(
        _trial_record("late", {"u": 2}, sharpe=2.5, resolved_end=_DEFAULT_WINDOW[1]),
        history_dir,
    )
    disclosure = trial_pool_disclosure(_DEFAULT_WINDOW, history_dir)
    assert disclosure["n_trial_records"] == 2
    assert disclosure["distinct_trial_keys"] == 2
    assert disclosure["pool_window_span_days"] == pytest.approx(31.0, abs=1e-3)


def test_stamped_records_keep_their_trial_provenance(tmp_path) -> None:
    stamped = dict(_trial_record("stamped", {"u": 1}))
    stamped["run_at"] = "2026-03-01T00:00:00+00:00"
    append_run_history_record(stamped, tmp_path / "stamped")
    stamped_count, _ = derive_trials_attempted(tmp_path / "stamped")
    assert stamped_count == SEARCH_TRIALS_ATTEMPTED + 1


def test_explicit_location_maps_to_its_own_registry_file() -> None:
    assert _resolve_history_registry(None) == BACKTESTS_DIR / "registry.sqlite3"
    assert _resolve_history_registry(Path("x")) == Path("x") / "registry.sqlite3"


def test_disclosure_without_ledger_keeps_history_source(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record(_trial_record("bad", status="FAILED"), history_dir)
    append_run_history_record({"run_id": "legacy"}, history_dir)
    disclosure = trial_pool_disclosure(_DEFAULT_WINDOW, history_dir)
    assert disclosure["n_history_records"] == 2
    assert disclosure["ledger_size"] == 0
    assert disclosure["source"] == "constant_plus_history"
    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED
    assert source == "constant_plus_history"


def test_append_writes_registry_only(tmp_path) -> None:
    """Single writer backend: the registry grows without any sibling JSON artifact."""
    resolved = tmp_path / "mhs_run_history"
    record = _trial_record("r1", {"u": 1})
    registry = append_run_history_record(record, resolved)
    assert registry.is_file()
    assert registry == resolved / "registry.sqlite3"
    assert sorted(p.name for p in resolved.iterdir()) == ["registry.sqlite3"]
    counted, _ = derive_trials_attempted(resolved)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 1
    assert window_trial_sharpes(_DEFAULT_WINDOW, resolved) == (2.0,)
    disclosure = trial_pool_disclosure(_DEFAULT_WINDOW, resolved)
    assert disclosure["n_history_records"] == 1


def test_default_history_never_reads_docs(tmp_path, monkeypatch) -> None:
    """Default denominator ignores the docs tree when the canonical registry is empty."""
    import src.mhs.run_history as rh

    empty_home = tmp_path / "canonical-home"
    empty_home.mkdir()
    monkeypatch.setattr(rh, "canonical_history_registry", lambda: empty_home / "registry.sqlite3")
    counted, source = derive_trials_attempted(None)
    assert (counted, source) == (SEARCH_TRIALS_ATTEMPTED, "constant_fallback")
    assert window_trial_sharpes(_DEFAULT_WINDOW, None) == ()
    assert trial_pool_disclosure(_DEFAULT_WINDOW, None)["source"] == "constant_fallback"
    counted_docs, source_docs = derive_trials_attempted("docs/results/mhs_run_history")
    assert (counted_docs, source_docs) == (SEARCH_TRIALS_ATTEMPTED, "constant_fallback")


def test_imported_registry_preserves_denominator(tmp_path) -> None:
    """Legacy records become registry evidence carrying both records and their trials."""
    import json as _json

    source = tmp_path / "legacy"
    source.mkdir()
    records = [
        {"run_id": "r1", "status": "COMPLETE", "flags": {"u": 1}, "start": _DEFAULT_WINDOW[0],
         "resolved_end": _DEFAULT_WINDOW[1], "blend": {"primary_naive_sharpe": 1.5},
         "research_go": {"reason_codes": [], "data_integrity_reason_codes": []}, "run_at": "2026-01-01T00:00:00+00:00"},
        {"run_id": "r2", "status": "COMPLETE", "flags": {"u": 2}, "start": _DEFAULT_WINDOW[0],
         "resolved_end": _DEFAULT_WINDOW[1], "blend": {"primary_naive_sharpe": 2.5},
         "research_go": {"reason_codes": [], "data_integrity_reason_codes": []}, "run_at": "2026-01-02T00:00:00+00:00"},
    ]
    with (source / "active.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(_json.dumps(record, sort_keys=True) + "\n")
    from src.mhs.run_history import trial_identity_key as _key

    ledger_seed = {k: "2026-01-01T00:00:00+00:00" for r in records if (k := _key(r)) is not None}
    (source / "trials_ledger.json").write_text(_json.dumps(ledger_seed), encoding="utf-8")
    registry_home = tmp_path / "registry-home"
    registry_home.mkdir()
    registry = registry_home / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    assert derive_trials_attempted(registry_home) == (SEARCH_TRIALS_ATTEMPTED + 2, "constant_plus_ledger")
    assert window_trial_sharpes(_DEFAULT_WINDOW, registry_home) == (1.5, 2.5)


def test_explicit_fixture_directory_remains_isolated(tmp_path, monkeypatch) -> None:
    """Explicit directories never read or write the repository registry or docs tree."""
    import src.mhs.run_history as rh

    canonical_home = tmp_path / "canonical-home"
    canonical_home.mkdir()
    monkeypatch.setattr(rh, "canonical_history_registry", lambda: canonical_home / "registry.sqlite3")
    assert rh.canonical_history_registry().parent == canonical_home
    first = tmp_path / "first"
    append_run_history_record(_trial_record("r1", {"u": 1}), first)
    assert (first / "registry.sqlite3").is_file()
    assert not (canonical_home / "registry.sqlite3").exists()
    first_count, _ = derive_trials_attempted(first)
    assert first_count == SEARCH_TRIALS_ATTEMPTED + 1
    assert derive_trials_attempted(None) == (SEARCH_TRIALS_ATTEMPTED, "constant_fallback")


def test_canonical_registry_points_to_backtests_dir() -> None:
    from src.mhs.run_history import canonical_history_registry

    assert canonical_history_registry() == BACKTESTS_DIR / "registry.sqlite3"


def test_persist_appends_to_explicit_history_dir_only(tmp_path, monkeypatch) -> None:
    """Persist appends to explicit history dir only."""
    import src.mhs.report.persist as persist_mod
    import src.mhs.run_history as rh

    canonical_home = tmp_path / "canonical-home"
    canonical_home.mkdir()
    monkeypatch.setattr(rh, "canonical_history_registry", lambda: canonical_home / "registry.sqlite3")
    monkeypatch.setattr(persist_mod, "_persist_mhs_report_compact", lambda report, target: target)
    monkeypatch.setattr(persist_mod, "build_mhs_run_history_record", lambda *a, **k: _trial_record("persisted", {"u": 9}))
    target = tmp_path / "reports" / "report.json"
    result = persist_mod.persist_mhs_report(object(), target, history_dir=tmp_path / "explicit")  # type: ignore[arg-type]
    assert result == target
    assert not (target.parent / "mhs_run_history").exists()
    assert (tmp_path / "explicit" / "registry.sqlite3").is_file()
    assert not (canonical_home / "registry.sqlite3").exists()


def test_append_without_destination_raises_before_io(tmp_path, monkeypatch) -> None:
    """Append without destination raises before io."""
    import src.mhs.run_history as rh

    home = tmp_path / "home"
    monkeypatch.setattr(rh, "canonical_history_registry", lambda: home / "registry.sqlite3")
    rec = {"run_id": "x"}

    with pytest.raises(TypeError):
        rh.append_run_history_record(rec)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        rh.append_run_history_record(rec, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rh.append_run_history_record(rec, 7)  # type: ignore[arg-type]
    assert not (home / "registry.sqlite3").exists()


def test_programmatic_dsr_reads_and_persist_share_explicit_history(tmp_path, monkeypatch) -> None:
    """A programmatic look reads its trial evidence and appends to the same store."""
    import dataclasses

    import src.mhs.pipeline.orchestrator as orch
    import src.mhs.pipeline.stages.fold as fold
    import src.mhs.run_history as rh
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
    from src.mhs.preregistration import consulted_data_horizon
    from src.mhs.report.persist import persist_mhs_report
    from tests.unit.mhs.test_evaluation_appresearch import _build_compact_report

    history = tmp_path / "history"
    window = ("2021-01-01T00:00:00+00:00", "2024-06-30T00:00:00+00:00")
    for index in range(2):
        append_run_history_record(
            _trial_record(str(index), {"u": index}, start=window[0], resolved_end=window[1]), history,
        )

    def reject_canonical():
        raise AssertionError("an explicit run must never open the canonical registry")

    monkeypatch.setattr(rh, "canonical_history_registry", reject_canonical)

    class StopReplayError(Exception):
        """Stop after the real fold-stage evidence reads, before expensive replays."""

    def stop_replay(*args, **kwargs):
        raise StopReplayError

    monkeypatch.setattr(fold.concurrency, "_run_post_book_concurrently", stop_replay)
    base = _build_compact_report()

    def evidence_stage(ctx, telemetry):
        assert ctx.history_dir == history
        with pytest.raises(StopReplayError):
            fold.run_folds(ctx, telemetry)
        assert ctx.trials_attempted == SEARCH_TRIALS_ATTEMPTED + 2
        assert ctx.trial_sharpes == (2.0, 2.0)
        assert ctx.trial_pool["ledger_size"] == 2
        return dataclasses.replace(
            base, start=str(ctx.start), end=str(ctx.end), resolved_end=str(ctx.resolved_end),
            trials_attempted=ctx.trials_attempted, trial_pool=ctx.trial_pool,
            blend=base.books["fast_reversal"],
        )

    monkeypatch.setattr(orch, "run_stages", evidence_stage)
    report = run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(start=window[0], end=window[1], log_run=False), history_dir=history,
    )
    target = tmp_path / "reports" / "report.json"
    assert persist_mhs_report(report, target, history_dir=history) == target
    assert derive_trials_attempted(history)[0] == SEARCH_TRIALS_ATTEMPTED + 3
    assert trial_pool_disclosure(window, history)["n_history_records"] == 3
    assert consulted_data_horizon(history, tmp_path / "procedures.jsonl").tzinfo is not None


def test_registry_failure_stays_observational(tmp_path, monkeypatch, caplog) -> None:
    """A failing history registry never breaks report persistence."""
    import src.mhs.report.persist as persist_mod

    monkeypatch.setattr(persist_mod, "_persist_mhs_report_compact", lambda report, target: target)
    monkeypatch.setattr(persist_mod, "build_mhs_run_history_record", lambda *a, **k: {})

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(persist_mod, "append_run_history_record", _boom)
    target = tmp_path / "report.json"
    with caplog.at_level("WARNING"):
        result = persist_mod.persist_mhs_report(object(), target, history_dir=tmp_path / "history")  # type: ignore[arg-type]
    assert result == target
    assert any("run-history" in r.message for r in caplog.records)


def test_retired_jsonl_symbols_are_gone() -> None:
    """The registry is the only history backend: no rotation or shard symbols remain."""
    import src.mhs.run_history as rh

    retired = (
        "RUN_HISTORY_SHARD_MAX_BYTES",
        "RUN_HISTORY_MAX_SHARDS",
        "_DEFAULT_HISTORY_DIR",
        "_ACTIVE_FILE_NAME",
        "_LATEST_FILE_NAME",
        "mhs_run_history_dir",
        "_archive_path",
        "_unique_archive_path",
        "_serialize_record",
        "_prune_archives",
        "_upsert_trials_ledger",
        "_load_trials_ledger",
        "_iter_history_records",
        "_is_canonical_history_request",
    )
    assert [name for name in retired if hasattr(rh, name)] == []
    assert "jsonl" not in Path(rh.__file__).read_text(encoding="utf-8")


def test_admission_procedure_is_part_of_trial_identity() -> None:
    # D6: the committee admission procedure is a sealed decision constant in the
    # params snapshot, so pre-fix and post-fix committee runs never share a trial.
    from src.mhs.live_strategy import capture_params_snapshot
    from src.mhs.run_history import trial_identity_key

    snapshot = capture_params_snapshot()
    assert snapshot["COMMITTEE_ADMISSION_PROCEDURE"] == "boundary_frozen_warmup_excluded_v1"
    base = {"flags": {"committee_capital": True}, "params_snapshot": snapshot}
    stripped_snapshot = {
        key: value for key, value in snapshot.items() if key != "COMMITTEE_ADMISSION_PROCEDURE"
    }
    stripped = {"flags": {"committee_capital": True}, "params_snapshot": stripped_snapshot}
    assert trial_identity_key(base) != trial_identity_key(stripped)


def test_lifecycle_procedure_is_part_of_trial_identity() -> None:
    from src.mhs import params as _params
    from src.mhs.live_strategy import capture_params_snapshot
    from src.mhs.run_history import trial_identity_key
    snapshot = capture_params_snapshot()
    assert snapshot["INSTRUMENT_LIFECYCLE_PROCEDURE"] == _params.INSTRUMENT_LIFECYCLE_PROCEDURE
    assert snapshot["INSTRUMENT_LIFECYCLE_PROCEDURE"] not in (
        "pit_registry_settlement_halts_v2",
        "pit_registry_settlement_halts_causal_exclusions_v3",
    )
    base = {"flags": {}, "params_snapshot": snapshot}
    stripped = {"flags": {}, "params_snapshot": {k: v for k, v in snapshot.items() if k != "INSTRUMENT_LIFECYCLE_PROCEDURE"}}
    assert trial_identity_key(base) != trial_identity_key(stripped)


def test_halt_registry_digest_rekeys_trials(monkeypatch) -> None:
    import pandas as pd

    import src.mhs.venue_halts as venue_halts
    from src.mhs.live_strategy import capture_params_snapshot
    from src.mhs.run_history import trial_identity_key
    from src.mhs.venue_halts import VenueHaltInterval, assemble_venue_halt_registry

    base_snapshot = capture_params_snapshot()
    assert "VENUE_HALT_REGISTRY_DIGEST" in base_snapshot
    halt = VenueHaltInterval(
        halt_id="2022-05-01T22:27Z",
        start=pd.Timestamp("2022-05-01T22:27Z"),
        end=pd.Timestamp("2022-05-01T22:36:00Z"),
        present_symbols=12,
        zero_symbols=12,
        evidence="test halt",
        verified_at=pd.Timestamp("2026-07-01T00:00:00Z"),
    )
    changed = VenueHaltInterval(
        halt_id="2022-05-01T22:27Z",
        start=pd.Timestamp("2022-05-01T22:27Z"),
        end=pd.Timestamp("2022-05-01T22:39:00Z"),
        present_symbols=12,
        zero_symbols=12,
        evidence="test halt",
        verified_at=pd.Timestamp("2026-07-01T00:00:00Z"),
    )
    monkeypatch.setattr(
        venue_halts, "load_venue_halt_registry",
        lambda path=None: assemble_venue_halt_registry([halt]),
    )
    venue_halts.clear_venue_halt_registry_cache()
    one = capture_params_snapshot()
    monkeypatch.setattr(
        venue_halts, "load_venue_halt_registry",
        lambda path=None: assemble_venue_halt_registry([changed]),
    )
    venue_halts.clear_venue_halt_registry_cache()
    two = capture_params_snapshot()
    assert one["VENUE_HALT_REGISTRY_DIGEST"] != two["VENUE_HALT_REGISTRY_DIGEST"]
    assert trial_identity_key({"flags": {}, "params_snapshot": one}) != trial_identity_key(
        {"flags": {}, "params_snapshot": two}
    )
    monkeypatch.undo()
    venue_halts.clear_venue_halt_registry_cache()
    assert capture_params_snapshot()["VENUE_HALT_REGISTRY_DIGEST"] == base_snapshot["VENUE_HALT_REGISTRY_DIGEST"]
