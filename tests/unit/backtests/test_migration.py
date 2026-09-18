"""Invariant guards for explicit legacy migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from src.backtests.migration import (
    _derive_run_id,
    _is_hex32,
    _map_run_status,
    _normalize_gap,
    migrate_legacy_backtests,
    verify_legacy_history_migration,
)
from src.common.errors import DataIntegrityError
_WINDOW = ("2021-01-01T00:00:00+00:00", "2025-12-31T23:59:59+00:00")


def _trial_record(run_id: str, flags: dict[str, Any] | None = None, *, sharpe: float | None = 2.0) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "COMPLETE",
        "flags": flags or {},
        "start": _WINDOW[0],
        "resolved_end": _WINDOW[1],
        "blend": {"primary_naive_sharpe": sharpe},
        "research_go": {"reason_codes": [], "data_integrity_reason_codes": []},
        "run_at": "2026-01-01T00:00:00+00:00",
    }


def _write_history(source: Path, records: list[dict[str, Any]], ledger_extra: dict[str, str] | None = None) -> None:
    source.mkdir(parents=True, exist_ok=True)
    with (source / "active.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    ledger: dict[str, str] = {}
    if ledger_extra is not None:
        ledger.update(ledger_extra)
    (source / "trials_ledger.json").write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")


def _write_run(
    source: Path,
    *,
    status: Any = "resource_rejected",
    with_primary: bool = False,
    valid: bool | None = None,
    gaps: Any = None,
) -> None:
    source.mkdir(parents=True, exist_ok=True)
    run_doc = {
        "command": ["python", "-m", "worker"],
        "cpu_seconds": None,
        "wall_seconds": 5.0,
        "gnu_max_individual_rss_bytes": None,
        "sampled_tree_pss_peak_bytes": 10,
        "sampled_tree_uss_peak_bytes": None,
        "process_swap_growth_bytes": None,
        "signal_number": None,
        "exit_code": None,
        "start": "2021-01-01T00:00:00+00:00",
        "end": "2026-06-30T23:59:59+00:00",
        "data_root": None,
        "status": status,
        "termination_reason": "swap growth",
    }
    (source / "run.json").write_text(json.dumps(run_doc, sort_keys=True), encoding="utf-8")
    if with_primary:
        terminal = {"primary_valid": valid, "terminal_certified": valid, "data_gaps": gaps if gaps is not None else []}
        primary = {"base": {"terminal": terminal}, "stress": {"terminal": dict(terminal)}}
        (source / "primary.json").write_text(json.dumps(primary, sort_keys=True), encoding="utf-8")


def _counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db))
    try:
        return {
            "history": int(conn.execute("SELECT COUNT(*) FROM history_records").fetchone()[0]),
            "trials": int(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]),
            "runs": int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]),
            "migrations": int(conn.execute("SELECT COUNT(*) FROM migrations").fetchone()[0]),
        }
    finally:
        conn.close()


def test_migrate_preview_leaves_files_untouched(tmp_path: Path) -> None:
    """Preview immutability: dry run creates or mutates no file."""
    history = tmp_path / "history"
    _write_history(history, [_trial_record("r1", {"u": 1})])
    run = tmp_path / "run-abc"
    _write_run(run)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (*history.iterdir(), *run.iterdir())}
    registry = tmp_path / "registry.sqlite3"
    report = migrate_legacy_backtests(
        registry_path=registry, history_directories=(history,), run_directories=(run,), dry_run=True
    )
    assert report["imported"] == 2
    assert report["skipped"] == 0
    assert not registry.exists()
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (*history.iterdir(), *run.iterdir())}
    assert before == after


def test_migrate_retry_deduplicates_identity(tmp_path: Path) -> None:
    """Retry identity: re-running skips without duplicating metadata/trials/records."""
    history = tmp_path / "history"
    _write_history(history, [_trial_record("r1", {"u": 1}), _trial_record("r2", {"u": 1}, sharpe=2.0)])
    run = tmp_path / "run-valid"
    _write_run(run, status="completed", with_primary=True, valid=True)
    registry = tmp_path / "registry.sqlite3"
    first = migrate_legacy_backtests(
        registry_path=registry, history_directories=(history,), run_directories=(run,), dry_run=False
    )
    assert first["imported"] == 2
    counts = _counts(registry)
    second = migrate_legacy_backtests(
        registry_path=registry, history_directories=(history,), run_directories=(run,), dry_run=False
    )
    assert second["skipped"] == 2
    assert second["imported"] == 0
    assert _counts(registry) == counts


def test_migrate_preserves_ledger_only_trials(tmp_path: Path) -> None:
    """Ledger-only keys survive with first-seen provenance."""
    import json as _json

    from src.mhs.run_history import trial_identity_key

    history = tmp_path / "history"
    record = _trial_record("r1", {"u": 1})
    key = trial_identity_key(record)
    assert key is not None
    dense = dict(_json.loads(key))
    dense["log_run"] = True
    dense_key = _json.dumps(dense, sort_keys=True, ensure_ascii=False)
    _write_history(
        history,
        [record],
        ledger_extra={key: "2026-02-01T00:00:00+00:00", dense_key: "2026-01-01T00:00:00+00:00", "orphan-key": "2026-01-01T00:00:00+00:00"},
    )
    registry = tmp_path / "registry.sqlite3"
    report = migrate_legacy_backtests(
        registry_path=registry, history_directories=(history,), run_directories=(), dry_run=False
    )
    assert report["imported"] == 1
    conn = sqlite3.connect(str(registry))
    try:
        rows = {r[0]: r[1] for r in conn.execute("SELECT identity_key, first_seen FROM trials").fetchall()}
    finally:
        conn.close()
    assert rows[key] == "2026-01-01T00:00:00+00:00"
    assert rows["orphan-key"] == "2026-01-01T00:00:00+00:00"


def test_migrate_rejects_corrupt_ledger_without_shrinking_trials(tmp_path: Path) -> None:
    """Corrupt ledgers are rejected without empty replacement or trial loss."""
    good = tmp_path / "good"
    _write_history(good, [_trial_record("r1", {"u": 1})])
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(good,), run_directories=(), dry_run=False)
    before = _counts(registry)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "active.jsonl").write_text(json.dumps(_trial_record("r9", {"u": 9})) + "\n", encoding="utf-8")
    (broken / "trials_ledger.json").write_text("{corrupt", encoding="utf-8")
    report = migrate_legacy_backtests(
        registry_path=registry, history_directories=(broken,), run_directories=(), dry_run=False
    )
    assert report["rejected"] == 1
    assert _counts(registry) == before
    non_dict = tmp_path / "non-dict"
    non_dict.mkdir()
    (non_dict / "active.jsonl").write_text(json.dumps(_trial_record("r9", {"u": 9})) + "\n", encoding="utf-8")
    (non_dict / "trials_ledger.json").write_text("[1, 2]", encoding="utf-8")
    second = migrate_legacy_backtests(
        registry_path=registry, history_directories=(non_dict,), run_directories=(), dry_run=False
    )
    assert second["rejected"] == 1
    bad_shard = tmp_path / "bad-shard"
    bad_shard.mkdir()
    (bad_shard / "active.jsonl").write_text("{not-json}\n", encoding="utf-8")
    third = migrate_legacy_backtests(
        registry_path=registry, history_directories=(bad_shard,), run_directories=(), dry_run=False
    )
    assert third["rejected"] == 1
    assert _counts(registry) == before


def test_migrate_preserves_resource_rejected_runs(tmp_path: Path) -> None:
    """Primary-less runs keep failure class, cause and null observations."""
    run = tmp_path / "run-rejected"
    _write_run(run, status="resource_rejected")
    registry = tmp_path / "registry.sqlite3"
    report = migrate_legacy_backtests(registry_path=registry, history_directories=(), run_directories=(run,), dry_run=False)
    assert report["imported"] == 1
    assert report["protected"] == 1
    conn = sqlite3.connect(str(registry))
    try:
        row = conn.execute("SELECT status, primary_valid, outcome_json FROM finalizations").fetchone()
    finally:
        conn.close()
    assert row[0] == "resource_rejected"
    assert row[1] is None
    outcome = json.loads(str(row[2]))
    assert outcome["raw_status"] == "resource_rejected"
    assert outcome["cpu_seconds"] is None
    assert outcome["gnu_max_individual_rss_bytes"] is None
    assert outcome["wall_seconds"] == 5.0


def test_migrate_preserves_unresolved_primary_evidence(tmp_path: Path) -> None:
    """Invalid runs keep lossless gap evidence with unknown intents and protected originals."""
    run = tmp_path / "run-invalid"
    gaps = [
        {"code": "MISSING_HELD_FUNDING", "symbol": "ANCUSDT", "timestamp": "2022-05-13T00:03:00+00:00"},
        {
            "code": "X",
            "symbol": "Y",
            "timestamp": "2022-05-14T00:03:00+00:00",
            "decision_time": "2022-05-14T00:00:00+00:00",
            "signal_time": "2022-05-14T00:01:00+00:00",
            "execution_bound": "close",
        },
    ]
    _write_run(run, status="COMPLETE", with_primary=True, valid=False, gaps=gaps)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
    registry = tmp_path / "registry.sqlite3"
    report = migrate_legacy_backtests(registry_path=registry, history_directories=(), run_directories=(run,), dry_run=False)
    assert report["imported"] == 1
    assert report["protected"] == 1
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
    assert before == after
    conn = sqlite3.connect(str(registry))
    try:
        row = conn.execute("SELECT status, primary_valid, outcome_json FROM finalizations").fetchone()
    finally:
        conn.close()
    assert row[0] == "completed"
    assert row[1] == 0
    outcome = json.loads(str(row[2]))
    assert outcome["raw_status"] == "COMPLETE"
    assert outcome["gap_count"] == 4
    assert outcome["gaps"][0]["execution_bound"] == "unknown"
    assert outcome["primary_sha256"] is not None


def test_migrate_rejects_unsafe_source_selection(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite3"
    with pytest.raises(ValueError, match=".+"):
        migrate_legacy_backtests(
            registry_path=cast(Path, "not-a-path"),
            history_directories=(),
            run_directories=(),
            dry_run=True,
        )
    with pytest.raises(ValueError, match=".+"):
        migrate_legacy_backtests(
            registry_path=registry, history_directories=(tmp_path / "missing",), run_directories=(), dry_run=True
        )


def test_migrate_helper_branches_and_ops_wiring(tmp_path: Path) -> None:
    assert _is_hex32("ab" * 16) is True
    assert _is_hex32("short") is False
    assert _is_hex32("z" * 32) is False
    assert _derive_run_id("ab" * 16) == ("ab" * 16).lower()
    assert _derive_run_id("run-abc") == uuid.uuid5(uuid.NAMESPACE_URL, "run-abc").hex
    assert _map_run_status("COMPLETE") == "completed"
    assert _map_run_status("timed_out") == "timed_out"
    assert _map_run_status(None) == "failed"
    assert _map_run_status("weird") == "failed"
    assert _normalize_gap(None)["code"] == "unknown"
    full = {
        "code": "C",
        "symbol": "S",
        "timestamp": "T",
        "decision_time": "D",
        "signal_time": "G",
        "execution_bound": "B",
    }
    assert _normalize_gap(full) == full
    history = tmp_path / "history"
    _write_history(history, [_trial_record("r1", {"u": 1})])
    (history / "extra.txt").write_text("x", encoding="utf-8")
    with (history / "active.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("[1, 2]\n")
    no_ledger = tmp_path / "no-ledger-history"
    no_ledger.mkdir()
    (no_ledger / "active.jsonl").write_text(json.dumps(_trial_record("nl", {"u": 5})) + "\n", encoding="utf-8")
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(history,), run_directories=(), dry_run=False)
    migrate_legacy_backtests(registry_path=registry, history_directories=(no_ledger,), run_directories=(), dry_run=False)
    preview_history = tmp_path / "preview-history"
    _write_history(preview_history, [_trial_record("r2", {"u": 2})])
    preview_run = tmp_path / "preview-run"
    _write_run(preview_run, status="failed", with_primary=True, valid=None, gaps="not-a-list")
    (preview_run / "failure.json").write_text(json.dumps({"error": "boom"}), encoding="utf-8")
    before = _counts(registry)
    preview = migrate_legacy_backtests(
        registry_path=registry,
        history_directories=(preview_history,),
        run_directories=(preview_run,),
        dry_run=True,
    )
    assert preview["imported"] == 2
    assert _counts(registry) == before
    bad_run = tmp_path / "bad-run"
    bad_run.mkdir()
    (bad_run / "run.json").write_text("[1, 2]", encoding="utf-8")
    rejected = migrate_legacy_backtests(
        registry_path=registry, history_directories=(), run_directories=(bad_run,), dry_run=False
    )
    assert rejected["rejected"] == 1
    bad_primary = tmp_path / "bad-primary"
    bad_primary.mkdir()
    (bad_primary / "run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (bad_primary / "primary.json").write_text("[1]", encoding="utf-8")
    rejected_primary = migrate_legacy_backtests(
        registry_path=registry, history_directories=(), run_directories=(bad_primary,), dry_run=False
    )
    assert rejected_primary["rejected"] == 1
    odd_terminal = tmp_path / "odd-terminal"  # noqa: E501
    odd_terminal.mkdir()
    (odd_terminal / "run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (odd_terminal / "primary.json").write_text(
        json.dumps({"base": {"terminal": "bad"}, "stress": {"terminal": "bad"}}), encoding="utf-8"
    )
    odd_report = migrate_legacy_backtests(
        registry_path=registry, history_directories=(), run_directories=(odd_terminal,), dry_run=False
    )
    assert odd_report["imported"] == 1
    parser = argparse.ArgumentParser()
    from src.cli.commands.ops import add_ops_commands

    add_ops_commands(parser)
    args = parser.parse_args(
        ["backtests-migrate", "--registry-path", str(registry), "--history-directory", str(preview_history), "--apply"]
    )
    args.handler(args)
    assert registry.is_file()


def _write_sharded_history(source: Path, shards: list[list[dict[str, Any]]], ledger: dict[str, str]) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for index, records in enumerate(shards):
        with (source / f"shard-{index}.jsonl").open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    (source / "trials_ledger.json").write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")


def test_verify_imported_history_matches_counts_and_provenance(tmp_path: Path) -> None:
    """Imported history verifies exactly with matching counts and earliest timestamps."""
    from src.mhs.run_history import trial_identity_key

    first = _trial_record("r1", {"u": 1})
    first["run_at"] = "2026-01-01T00:00:00+00:00"
    second = _trial_record("r2", {"u": 2})
    second["run_at"] = "2026-01-02T00:00:00+00:00"
    key = trial_identity_key(first)
    assert key is not None
    source = tmp_path / "history"
    _write_sharded_history(source, [[first], [second]], {key: "2026-01-01T00:00:00+00:00"})
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    report = verify_legacy_history_migration(registry_path=registry, source=source)
    assert report["verified"] is True
    assert report["source_records"] == 2
    assert report["registry_records"] == 2
    assert report["distinct_trial_identities"] == 2


def test_verify_changed_source_fails_closed_until_reimport(tmp_path: Path) -> None:
    """Changed source fails closed until a new explicit import completes."""
    record = _trial_record("r1", {"u": 1})
    source = tmp_path / "history"
    _write_sharded_history(source, [[record]], {})
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    verify_legacy_history_migration(registry_path=registry, source=source)
    with (source / "shard-0.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_trial_record("r2", {"u": 2})) + "\n")
    with pytest.raises(DataIntegrityError):
        verify_legacy_history_migration(registry_path=registry, source=source)
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    report = verify_legacy_history_migration(registry_path=registry, source=source)
    assert report["source_records"] == 2


def test_verify_corrupt_source_raises_without_mutation(tmp_path: Path) -> None:
    """Corrupt source cannot be retired and leaves registry rows unchanged."""
    source = tmp_path / "history"
    _write_sharded_history(source, [[_trial_record("r1", {"u": 1})]], {})
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    before = _counts(registry)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "shard-0.jsonl").write_text("{not-json}\n", encoding="utf-8")
    (broken / "trials_ledger.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        verify_legacy_history_migration(registry_path=registry, source=broken)
    bad_ledger = tmp_path / "bad-ledger"
    bad_ledger.mkdir()
    (bad_ledger / "shard-0.jsonl").write_text(json.dumps(_trial_record("r1", {"u": 1})) + "\n", encoding="utf-8")
    (bad_ledger / "trials_ledger.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        verify_legacy_history_migration(registry_path=registry, source=bad_ledger)
    non_dict = tmp_path / "non-dict"
    non_dict.mkdir()
    (non_dict / "shard-0.jsonl").write_text(json.dumps(_trial_record("r1", {"u": 1})) + "\n", encoding="utf-8")
    (non_dict / "trials_ledger.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        verify_legacy_history_migration(registry_path=registry, source=non_dict)
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    (unreadable / "x.jsonl").mkdir()
    (unreadable / "trials_ledger.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        verify_legacy_history_migration(registry_path=registry, source=unreadable)
    with pytest.raises(DataIntegrityError):
        verify_legacy_history_migration(registry_path=registry, source=tmp_path / "missing")
    with pytest.raises(DataIntegrityError):
        verify_legacy_history_migration(registry_path=tmp_path / "no-registry.sqlite3", source=source)
    assert _counts(registry) == before


def test_verify_ignores_run_bundles_and_cli_wiring(tmp_path: Path) -> None:
    """Run bundles are excluded from history verification, including CLI wiring."""
    source = tmp_path / "history"
    _write_sharded_history(source, [[_trial_record("r1", {"u": 1})]], {})
    run = tmp_path / "run-abc"
    _write_run(run)
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    report = verify_legacy_history_migration(registry_path=registry, source=source)
    assert report["verified"] is True
    assert _counts(registry)["runs"] == 0
    assert _counts(registry)["migrations"] == 1
    parser = argparse.ArgumentParser()
    from src.cli.commands.ops import add_ops_commands

    add_ops_commands(parser)
    args = parser.parse_args(
        ["backtests-verify-history-migration", "--registry-path", str(registry), "--history-directory", str(source)]
    )
    args.handler(args)
    with pytest.raises(SystemExit):
        bad = parser.parse_args(
            ["backtests-verify-history-migration", "--registry-path", str(registry), "--history-directory", str(tmp_path / "missing")]
        )
        bad.handler(bad)
    assert _counts(registry)["runs"] == 0


def test_verify_duplicate_trials_keep_earliest_identity(tmp_path: Path) -> None:
    """Duplicate trials remain monotone with one identity and earliest provenance."""
    import json as _json

    from src.mhs.run_history import trial_identity_key

    record = _trial_record("r1", {"u": 1})
    record["run_at"] = "2026-01-01T00:00:00+00:00"
    key = trial_identity_key(record)
    assert key is not None
    dense = dict(_json.loads(key))
    dense["log_run"] = True
    dense_key = _json.dumps(dense, sort_keys=True, ensure_ascii=False)
    source = tmp_path / "history"
    _write_sharded_history(source, [[record, dict(record)]], {key: "2026-01-01T00:00:00+00:00", dense_key: "2026-02-01T00:00:00+00:00"})
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    report = verify_legacy_history_migration(registry_path=registry, source=source)
    assert report["source_records"] == 2
    assert report["distinct_trial_identities"] == 1
    conn = sqlite3.connect(str(registry))
    try:
        rows = {r[0]: r[1] for r in conn.execute("SELECT identity_key, first_seen FROM trials").fetchall()}
    finally:
        conn.close()
    assert rows[key] == "2026-01-01T00:00:00+00:00"


def test_verify_detects_registry_drift(tmp_path: Path) -> None:
    """Tampered registry rows fail verification without mutating the registry."""
    source = tmp_path / "history"
    _write_sharded_history(source, [[_trial_record("r1", {"u": 1}), _trial_record("r2", {"u": 2})]], {})
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    verify_legacy_history_migration(registry_path=registry, source=source)
    conn = sqlite3.connect(str(registry))
    try:
        conn.execute("DELETE FROM history_records WHERE ordinal = 0")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError, match="count"):
        verify_legacy_history_migration(registry_path=registry, source=source)
    conn = sqlite3.connect(str(registry))
    try:
        row = conn.execute("SELECT source_id FROM migrations").fetchone()
        conn.execute("DELETE FROM history_records")
        conn.execute(
            "INSERT INTO history_records (source_id, ordinal, namespace, record_json, admitted, identity_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (row[0], 0, "mhs_legacy_horizon", json.dumps(_trial_record("r1", {"u": 1})), 1, "missing-key"),
        )
        conn.execute(
            "INSERT INTO history_records (source_id, ordinal, namespace, record_json, admitted, identity_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (row[0], 1, "mhs_legacy_horizon", json.dumps(_trial_record("r2", {"u": 2})), 1, "missing-key-2"),
        )
        conn.execute("DELETE FROM trials")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError, match="identity|provenance"):
        verify_legacy_history_migration(registry_path=registry, source=source)
    empty_db = tmp_path / "empty.sqlite3"
    empty_conn = sqlite3.connect(str(empty_db))
    try:
        empty_conn.execute("CREATE TABLE t (a TEXT)")
        empty_conn.commit()
    finally:
        empty_conn.close()
    with pytest.raises(DataIntegrityError):
        verify_legacy_history_migration(registry_path=empty_db, source=source)
