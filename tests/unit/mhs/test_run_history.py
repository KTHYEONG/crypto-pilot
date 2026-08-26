"""Run-history derived trials denominator (DSR audit provenance, I2).

The trial set is defined once (``is_trial_record`` + ``trial_identity_key``)
and shared by ``derive_trials_attempted`` and ``window_trial_sharpes``
(I-SAME-TRIAL-SET); the monotone ``trials_ledger.json`` survives archive
pruning (I-MONOTONE-TRIALS).
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import pytest

import src.mhs.run_history as run_history_mod
from src.mhs.params import SEARCH_TRIALS_ATTEMPTED
from src.mhs.run_history import (
    append_run_history_record,
    derive_trials_attempted,
    is_trial_record,
    trial_identity_key,
    trial_pool_disclosure,
    window_trial_sharpes,
)

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
    """Archives, the active shard, and the ledger are one logical ledger:
    distinct trial configurations accumulate across every JSONL shard."""
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


def test_malformed_history_falls_back_with_explicit_provenance(tmp_path) -> None:
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    (history_dir / "active.jsonl").write_text("{not-json}\n", encoding="utf-8")
    assert derive_trials_attempted(history_dir) == (
        SEARCH_TRIALS_ATTEMPTED,
        "constant_fallback",
    )


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
    monkeypatch.setattr(run_history_mod, "RUN_HISTORY_SHARD_MAX_BYTES", 200)
    history_dir = tmp_path / "history"

    counts: list[int] = []
    n_configs = 40
    for index in range(n_configs):
        append_run_history_record(
            _trial_record(f"r{index}", {"u": index}), history_dir
        )
        counted, _source = derive_trials_attempted(history_dir)
        counts.append(counted)
    # Strictly monotone non-decreasing across every intermediate append.
    assert all(b >= a for a, b in itertools.pairwise(counts))
    # Pruning actually deleted archives (rotation happened many times).
    archives = sorted(history_dir.glob("mhs_run_history_*.jsonl"))
    assert len(archives) <= run_history_mod.RUN_HISTORY_MAX_SHARDS
    # The surviving shards alone cannot hold every configuration anymore.
    assert len(archives) < n_configs
    # Yet the denominator still accounts for every configuration ever appended.
    counted, source = derive_trials_attempted(history_dir)
    assert source == "constant_plus_ledger"
    # The union covers every configuration even though shards were deleted.
    assert counted == SEARCH_TRIALS_ATTEMPTED + n_configs
    ledger_path = history_dir / "trials_ledger.json"
    assert ledger_path.exists()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger) >= n_configs

    # A corrupt ledger degrades to the shard-only scan without raising.
    ledger_path.write_text("{corrupt", encoding="utf-8")
    degraded_count, degraded_source = derive_trials_attempted(history_dir)
    assert degraded_source == "constant_plus_history"
    assert SEARCH_TRIALS_ATTEMPTED <= degraded_count <= counted

    # A deleted ledger behaves identically.
    ledger_path.unlink()
    deleted_count, deleted_source = derive_trials_attempted(history_dir)
    assert deleted_source == "constant_plus_history"
    assert deleted_count == degraded_count


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
