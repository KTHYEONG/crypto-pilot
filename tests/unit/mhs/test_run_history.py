"""Run-history derived trials denominator (DSR audit provenance, I2)."""

from __future__ import annotations

from pathlib import Path

from src.mhs.params import SEARCH_TRIALS_ATTEMPTED
from src.mhs.run_history import append_run_history_record, derive_trials_attempted


def _write_records(history_dir: Path, flags_list: list[dict[str, object]]) -> None:
    for index, flags in enumerate(flags_list):
        append_run_history_record({"run_id": f"r{index}", "flags": flags}, history_dir)


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
    # 3 observed configurations <= registered floor: the constant binds and
    # the denominator can never drop below it.
    assert counted == SEARCH_TRIALS_ATTEMPTED
    assert source == "constant"


def test_missing_flags_payload_counts_as_one_configuration(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "legacy"}, history_dir)
    counted, source = derive_trials_attempted(history_dir)
    assert counted == SEARCH_TRIALS_ATTEMPTED
    assert source == "constant"


def test_malformed_history_falls_back_with_explicit_provenance(tmp_path) -> None:
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    (history_dir / "active.jsonl").write_text("{not-json}\n", encoding="utf-8")
    assert derive_trials_attempted(history_dir) == (
        SEARCH_TRIALS_ATTEMPTED,
        "constant_fallback",
    )


def test_default_directory_resolution_uses_repository_layout() -> None:
    """No argument resolves to the repository-canonical history directory; on a
    bare checkout it is absent, so the conservative fallback must engage."""
    counted, source = derive_trials_attempted(None)
    assert counted == SEARCH_TRIALS_ATTEMPTED
    assert source in ("constant", "constant_fallback", "history")
    assert counted >= SEARCH_TRIALS_ATTEMPTED


def test_returned_count_is_monotone_in_the_registered_floor(tmp_path) -> None:
    for flags in ({"a": 1}, {"b": 2}):
        append_run_history_record({"flags": flags}, tmp_path / "h")
    counted, _source = derive_trials_attempted(tmp_path / "h")
    assert counted == max(2, SEARCH_TRIALS_ATTEMPTED)
