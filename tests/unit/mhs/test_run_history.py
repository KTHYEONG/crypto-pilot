"""Run-history derived trials denominator (DSR audit provenance, I2)."""

from __future__ import annotations

from pathlib import Path

from src.mhs.params import SEARCH_TRIALS_ATTEMPTED
from src.mhs.run_history import append_run_history_record, derive_trials_attempted


def _write_records(history_dir: Path, flags_list: list[dict[str, object]]) -> None:
    for index, flags in enumerate(flags_list):
        append_run_history_record({"run_id": f"r{index}", "flags": flags}, history_dir)


# SCENARIO_MHS_DSR_04_TRIALS_INCREMENT_WITH_NEW_CONFIG
def test_SCENARIO_MHS_DSR_04_TRIALS_INCREMENT_WITH_NEW_CONFIG(tmp_path) -> None:
    history_dir = tmp_path / "history"
    _write_records(history_dir, [{"u": i} for i in range(6)])
    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 6
    assert source == "constant_plus_history"

    # A new distinct configuration strictly increments the denominator.
    append_run_history_record({"run_id": "r6", "flags": {"u": 6}}, history_dir)
    counted_after_new, _ = derive_trials_attempted(history_dir)
    assert counted_after_new == SEARCH_TRIALS_ATTEMPTED + 7

    # A duplicate configuration changes nothing.
    append_run_history_record({"run_id": "dup", "flags": {"u": 6}}, history_dir)
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
    """Archives and the active shard are one logical ledger: distinct flag
    configurations are counted across every JSONL shard, duplicates collapse."""
    history_dir = tmp_path / "history"
    _write_records(
        history_dir,
        [{"execution_universe_size": 30}, {"execution_universe_size": 60}],
    )
    # Force a rotation so records land in an archive plus the active shard.
    for index in range(3):
        append_run_history_record(
            {"run_id": f"extra{index}", "flags": {"pnl_vol_target_mode": "growth_budget"}},
            history_dir,
        )
    counted, source = derive_trials_attempted(history_dir)
    # 2 + 1 distinct configurations accumulate on top of the registered floor.
    assert counted == SEARCH_TRIALS_ATTEMPTED + 3
    assert source == "constant_plus_history"


def test_missing_flags_payload_counts_as_one_configuration(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "legacy"}, history_dir)
    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED + 1
    assert source == "constant_plus_history"


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
    assert source in ("constant_plus_history", "constant_fallback")


def test_returned_count_is_monotone_in_distinct_configurations(tmp_path) -> None:
    history_dir = tmp_path / "h"
    before_counted, _ = derive_trials_attempted(history_dir)
    for flags in ({"a": 1}, {"b": 2}):
        append_run_history_record({"flags": flags}, history_dir)
    counted, _source = derive_trials_attempted(history_dir)
    assert counted == before_counted + 2
