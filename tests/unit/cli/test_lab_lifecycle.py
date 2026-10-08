"""Lifecycle invariants for the canonical lab process-backtest command."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import pytest

import src.cli.commands.backtest as backtest_mod
from src.backtests.contracts import RetentionPolicy
from src.cli.commands.lab import (
    _resolve_process_destinations,
    _resolve_process_fingerprint,
    _resolve_process_retention_policy,
    run_process_backtest,
)
from src.cli.main import build_root_parser
from src.core.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING


def _parse(argv: list[str]) -> argparse.Namespace:
    return build_root_parser(argv).parse_args(argv)


def test_process_defaults_and_explicit_budget_reach_supervision(tmp_path, monkeypatch) -> None:
    from src.core.resources import MhsMemoryBudget

    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    seen = _install_supervisor(monkeypatch)
    args = _parse(["lab", "process-backtest", "--force"])
    assert args.handler is run_process_backtest
    run_process_backtest(args)
    assert (seen["start"], seen["end"]) == (DISCOVERY_START, PROCESS_EVALUATION_CEILING)
    assert seen["memory_budget"] == MhsMemoryBudget()
    assert seen["targets_output"] is None
    assert seen["timeout_seconds"] is None
    assert seen["poll_seconds"] == 0.25
    run_process_backtest(_parse([
        "lab", "process-backtest", "--force",
        "--total-tree-pss-bytes", str(4 * 2**30),
        "--replay-tree-pss-bytes", str(3 * 2**30),
        "--min-available-bytes", str(2 * 2**30),
    ]))
    assert seen["memory_budget"] == MhsMemoryBudget(
        total_tree_pss_bytes=4 * 2**30, replay_tree_pss_bytes=3 * 2**30,
        min_available_bytes=2 * 2**30,
    )


@pytest.mark.parametrize("offset", ["", "T00:00:00+00:00", "T09:00:00+09:00"])
def test_process_cli_dates_normalize_to_utc(tmp_path, monkeypatch, offset) -> None:
    import pandas as pd

    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    seen = _install_supervisor(monkeypatch)
    run_process_backtest(_parse([
        "lab", "process-backtest", "--force",
        "--start", "2022-01-01" + offset, "--end", "2022-01-04" + offset,
    ]))
    assert seen["start"] == pd.Timestamp("2022-01-01", tz="UTC")
    assert seen["end"] == pd.Timestamp("2022-01-04", tz="UTC")


@pytest.mark.parametrize("controls", [
    ["--start", "not-a-date"],
    ["--timeout-seconds", "nan"],
    ["--replay-tree-pss-bytes", str(2**60)],
])
def test_process_invalid_controls_never_launch(tmp_path, monkeypatch, controls) -> None:
    import subprocess

    def forbidden_launch(*args, **kwargs):
        raise AssertionError("invalid controls must fail before launch")

    monkeypatch.setattr(subprocess, "Popen", forbidden_launch)
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit):
        run_process_backtest(_parse(["lab", "process-backtest", "--output", str(output), *controls]))
    assert not output.exists()


@pytest.mark.parametrize("status", ["failed", "timed_out", "signaled", "resource_rejected", "interrupted"])
def test_process_non_success_status_never_reports_success(tmp_path, monkeypatch, status, capsys) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    _install_supervisor(monkeypatch, status=status)
    with pytest.raises(SystemExit) as exc:
        run_process_backtest(_parse(["lab", "process-backtest", "--force"]))
    assert exc.value.code == 1
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "index.jsonl").exists()


def test_process_completed_headline_retains_financial_values(tmp_path, monkeypatch) -> None:
    import json
    import types

    import src.lab.mhs.app.supervisor as supervisor

    def completed(**kwargs):
        kwargs["result_output"].write_text(
            json.dumps({"financial": {"base": {"cagr": 0.42, "max_drawdown": -0.1}}}), encoding="utf-8",
        )
        return types.SimpleNamespace(status="completed")

    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    monkeypatch.setattr(supervisor, "run_mhs_process_backtest", completed)
    run_process_backtest(_parse(["lab", "process-backtest", "--force"]))
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert (row["kind"], row["base_cagr"], row["base_max_drawdown"]) == ("mhs", 0.42, -0.1)
    assert row["run_dir"].startswith("runs/")


def _install_supervisor(monkeypatch, status: str = "completed") -> dict:
    import src.lab.mhs.app.supervisor as supmod
    from src.lab.mhs.app.supervisor import MhsSupervisedRun

    seen: dict = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        result = kwargs["result_output"]
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text('{"financial": {"base": {}}}', encoding="utf-8")
        return MhsSupervisedRun(
            status=status, command=("worker",), exit_code=0, signal_number=None,
            start="s", end="e", data_root=None, result_output_path="o",
            log_path="l", domain_artifact_written=True,
            wall_seconds=1.0, cpu_seconds=None,
            gnu_max_individual_rss_bytes=None, sampled_tree_pss_peak_bytes=None,
            sampled_tree_uss_peak_bytes=None, min_available_bytes=None,
            process_swap_growth_bytes=None, samples_taken=0, sample_interval_seconds=0.25,
            cpu_scope="cpu", memory_scope="memory", termination_reason=None,
            memory_budget=kwargs.get("memory_budget"), run_id=kwargs.get("run_id", ""),
        )

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _fake)
    return seen


def _seed_finalized_run(
    registry: Path, fingerprint: str, *, status: str = "completed", delete_artifacts: bool = False,
) -> tuple[str, Path]:
    import hashlib as _hashlib

    from src.backtests.contracts import ArtifactReference, RunFinalization, RunRegistration
    from src.backtests.registry import finalize_run, initialize_registry, register_run

    initialize_registry(registry)
    run_id = uuid.uuid4().hex
    run_dir = registry.parent / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result_path.write_bytes(b'{"ok": true, "run": "%s"}' % run_id.encode())
    digest = _hashlib.sha256(result_path.read_bytes()).hexdigest()
    register_run(
        registry,
        RunRegistration(
            run_id=run_id, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": fingerprint, "window": "3m"},
            managed_directory=run_dir,
        ),
    )
    finalize_run(
        registry,
        RunFinalization(
            run_id=run_id, status=status, finalized_at="2026-01-02T00:00:00+00:00",  # type: ignore[arg-type]
            primary_valid=True, terminal_certified=True, outcome={"ok": True},
        ),
        (
            ArtifactReference(
                run_id=run_id, role="result", path=result_path, sha256=digest,
                byte_count=result_path.stat().st_size, managed=True, evidence_id=None,
            ),
        ),
    )
    if delete_artifacts:
        import shutil as _shutil

        _shutil.rmtree(run_dir)
    return run_id, result_path


def test_resolve_destinations_creates_single_result_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    args = _parse(["lab", "process-backtest"])
    result, targets = _resolve_process_destinations(args)
    assert result.parent.parent == tmp_path / "runs"
    assert result.name == "result.json"
    uuid.UUID(hex=result.parent.name)
    assert targets is None
    assert not (result.parent / "primary.json").exists()
    assert not (result.parent / "failure.json").exists()
    assert not (result.parent / "run.json").exists()


def test_equivalent_finalized_request_reuses_without_launch(tmp_path, monkeypatch, caplog, capsys) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    registry = tmp_path / "registry.sqlite3"
    probe = _parse(["lab", "process-backtest"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(probe)
    fingerprint = _resolve_process_fingerprint(probe, start, end, budget)
    existing, result_path = _seed_finalized_run(registry, fingerprint)
    runs_root = tmp_path / "runs"
    before = set(runs_root.iterdir()) if runs_root.is_dir() else set()

    def _boom(**kwargs):
        raise AssertionError("supervisor must not launch on reuse")

    import src.lab.mhs.app.supervisor as supmod

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _boom)
    with caplog.at_level("INFO"):
        run_process_backtest(_parse(["lab", "process-backtest"]))
    captured = capsys.readouterr()
    assert captured.out == str(result_path) + "\n"
    assert any(existing in r.getMessage() and str(result_path) in r.getMessage() for r in caplog.records)
    after = set(runs_root.iterdir()) if runs_root.is_dir() else set()
    assert after == before
    assert not (tmp_path / "index.jsonl").exists()


def test_failed_equivalent_run_falls_through_to_fresh_execution(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    registry = tmp_path / "registry.sqlite3"
    probe = _parse(["lab", "process-backtest"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(probe)
    fingerprint = _resolve_process_fingerprint(probe, start, end, budget)
    existing, _ = _seed_finalized_run(registry, fingerprint, status="failed")
    seen = _install_supervisor(monkeypatch)
    with caplog.at_level("WARNING"):
        run_process_backtest(_parse(["lab", "process-backtest"]))
    assert seen["run_id"] != existing
    assert any(existing in r.getMessage() and "reuse_rejected" in r.getMessage() for r in caplog.records)


def test_leaked_run_with_deleted_directory_falls_through_to_fresh_execution(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    registry = tmp_path / "registry.sqlite3"
    probe = _parse(["lab", "process-backtest"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(probe)
    fingerprint = _resolve_process_fingerprint(probe, start, end, budget)
    existing, _ = _seed_finalized_run(registry, fingerprint, delete_artifacts=True)
    seen = _install_supervisor(monkeypatch)
    with caplog.at_level("WARNING"):
        run_process_backtest(_parse(["lab", "process-backtest"]))
    assert seen["run_id"] != existing
    assert any(
        existing in r.getMessage() and "artifact_missing" in r.getMessage() for r in caplog.records
    )


def test_corrupt_registry_aborts_before_any_destination(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    registry = tmp_path / "registry.sqlite3"
    registry.write_bytes(b"not a sqlite database")
    runs_root = tmp_path / "runs"
    before = set(runs_root.iterdir()) if runs_root.is_dir() else set()

    def _boom(**kwargs):
        raise AssertionError("supervisor must not launch on corrupt registry")

    import src.lab.mhs.app.supervisor as supmod

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _boom)
    with pytest.raises(SystemExit) as excinfo:
        run_process_backtest(_parse(["lab", "process-backtest"]))
    assert str(excinfo.value.code).startswith("registry integrity failure")
    after = set(runs_root.iterdir()) if runs_root.is_dir() else set()
    assert after == before


def test_explicit_destination_skips_reuse(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    registry = tmp_path / "registry.sqlite3"
    probe = _parse(["lab", "process-backtest"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(probe)
    fingerprint = _resolve_process_fingerprint(probe, start, end, budget)
    _seed_finalized_run(registry, fingerprint)
    seen = _install_supervisor(monkeypatch)
    with caplog.at_level("INFO"):
        run_process_backtest(_parse(["lab", "process-backtest", "--output", str(tmp_path / "fresh.json")]))
    assert seen["result_output"] == tmp_path / "fresh.json"
    assert any("reuse_skipped reason=explicit_destination" in r.getMessage() for r in caplog.records)
    seen_targets = _install_supervisor(monkeypatch)
    with caplog.at_level("INFO"):
        run_process_backtest(
            _parse(["lab", "process-backtest", "--targets-output", str(tmp_path / "fresh.parquet")])
        )
    assert seen_targets["targets_output"] == tmp_path / "fresh.parquet"


def test_force_bypasses_reuse_with_fresh_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    registry = tmp_path / "registry.sqlite3"
    probe = _parse(["lab", "process-backtest"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(probe)
    fingerprint = _resolve_process_fingerprint(probe, start, end, budget)
    existing, _ = _seed_finalized_run(registry, fingerprint)
    seen = _install_supervisor(monkeypatch)
    run_process_backtest(_parse(["lab", "process-backtest", "--force"]))
    assert seen["run_id"] != existing
    uuid.UUID(hex=seen["run_id"])


def test_completed_fresh_run_prints_result_path(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    seen = _install_supervisor(monkeypatch)
    run_process_backtest(_parse(["lab", "process-backtest"]))
    captured = capsys.readouterr()
    assert captured.out == str(seen["result_output"]) + "\n"


def test_index_failure_does_not_print_success_path(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    _install_supervisor(monkeypatch)

    def _fail_index(**kwargs):
        raise OSError("index persistence failed")

    monkeypatch.setattr("src.backtests.catalog.append_backtest_index", _fail_index)
    with pytest.raises(OSError, match="index persistence failed"):
        run_process_backtest(_parse(["lab", "process-backtest"]))
    assert capsys.readouterr().out == ""


def test_fingerprint_excludes_output_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    first = _parse(["lab", "process-backtest", "--output", str(tmp_path / "a.json")])
    second = _parse(["lab", "process-backtest", "--output", str(tmp_path / "b.json")])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(first)
    assert _resolve_process_fingerprint(first, start, end, budget) == _resolve_process_fingerprint(second, start, end, budget)


def test_fingerprint_includes_financial_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    base = _parse(["lab", "process-backtest"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(base)
    reference = _resolve_process_fingerprint(base, start, end, budget)
    altered_interval = _parse(["lab", "process-backtest", "--start", "2022-01-01", "--end", "2022-01-04"])
    new_start = backtest_mod._utc_timestamp("2022-01-01", "start", DISCOVERY_START)
    new_end = backtest_mod._utc_timestamp("2022-01-04", "end", PROCESS_EVALUATION_CEILING)
    assert _resolve_process_fingerprint(altered_interval, new_start, new_end, budget) != reference
    altered_data = _parse(["lab", "process-backtest", "--data-root", str(tmp_path)])
    assert _resolve_process_fingerprint(altered_data, start, end, budget) != reference
    altered_control = _parse(["lab", "process-backtest", "--rebalance-tracking-error-threshold", "0.2"])
    assert _resolve_process_fingerprint(altered_control, start, end, budget) != reference
    from src.core.resources import MhsMemoryBudget

    other_budget = MhsMemoryBudget(
        total_tree_pss_bytes=budget.total_tree_pss_bytes + 1,
        replay_tree_pss_bytes=budget.replay_tree_pss_bytes,
        min_available_bytes=budget.min_available_bytes,
    )
    assert _resolve_process_fingerprint(base, start, end, other_budget) != reference


def test_deprecated_output_flags_fail_early() -> None:
    with pytest.raises(SystemExit):
        _parse(["lab", "process-backtest", "--failure-output", "x.json"])
    with pytest.raises(SystemExit):
        _parse(["lab", "process-backtest", "--run-output", "x.json"])


def test_explicit_registry_and_retention_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    custom = tmp_path / "custom" / "registry.sqlite3"
    seen = _install_supervisor(monkeypatch)
    run_process_backtest(_parse(["lab", "process-backtest", "--registry-path", str(custom)]))
    assert seen["registry_path"] == custom
    assert seen["retention_policy"] == RetentionPolicy(max_detail_bytes=None, max_detail_runs=1)
    assert _resolve_process_retention_policy(_parse(["lab", "process-backtest"])) == RetentionPolicy(max_detail_bytes=None, max_detail_runs=1)
    budgeted = _parse(["lab", "process-backtest", "--max-detail-bytes", "100", "--max-detail-runs", "2"])
    policy = _resolve_process_retention_policy(budgeted)
    assert policy is not None
    assert policy.max_detail_bytes == 100
    assert policy.max_detail_runs == 2


def test_destinations_reject_invalid_targets_and_occupied_paths(tmp_path, monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_process_destinations(_parse(["lab", "process-backtest", "--targets-output", str(tmp_path / "bad.txt")]))
    occupied_targets = tmp_path / "taken.parquet"
    occupied_targets.write_bytes(b"x")
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_process_destinations(_parse(["lab", "process-backtest", "--targets-output", str(occupied_targets)]))
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_process_destinations(_parse(["lab", "process-backtest", "--output", str(tmp_path / "bad.csv")]))
    occupied = tmp_path / "taken.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_process_destinations(_parse(["lab", "process-backtest", "--output", str(occupied)]))

    def _no_launch(*args, **kwargs):
        raise AssertionError("worker must not launch")

    monkeypatch.setattr(subprocess, "Popen", _no_launch)
    with pytest.raises(SystemExit, match=r".+"):
        run_process_backtest(_parse(["lab", "process-backtest", "--output", str(occupied)]))
    bad_budget = _parse(["lab", "process-backtest", "--max-detail-bytes", "0"])
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_process_retention_policy(bad_budget)


def test_interrupted_supervised_run_exits_nonzero_without_index_append(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path)
    _install_supervisor(monkeypatch, status="interrupted")
    with pytest.raises(SystemExit) as excinfo:
        run_process_backtest(_parse(["lab", "process-backtest"]))
    assert excinfo.value.code == 1
    assert not (tmp_path / "index.jsonl").exists()


def test_lab_verify_history_migration_wiring(tmp_path) -> None:
    import argparse as _argparse
    import json as _json

    from src.lab.mhs.app.backtests_migration import migrate_legacy_backtests
    from src.cli.commands.lab import add_lab_commands

    source = tmp_path / "history"
    source.mkdir()
    record = {
        "run_id": "r1", "status": "COMPLETE", "flags": {}, "start": "2021-01-01T00:00:00+00:00",
        "resolved_end": "2025-12-31T23:59:59+00:00", "blend": {"primary_naive_sharpe": 2.0},
        "research_go": {"reason_codes": [], "data_integrity_reason_codes": []},
        "run_at": "2026-01-01T00:00:00+00:00",
    }
    with (source / "active.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(_json.dumps(record) + "\n")
    (source / "trials_ledger.json").write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry.sqlite3"
    migrate_legacy_backtests(registry_path=registry, history_directories=(source,), run_directories=(), dry_run=False)
    parser = _argparse.ArgumentParser()
    add_lab_commands(parser)
    args = parser.parse_args(
        ["backtests-verify-history-migration", "--registry-path", str(registry), "--history-directory", str(source)]
    )
    args.handler(args)
    bad = parser.parse_args(
        ["backtests-verify-history-migration", "--registry-path", str(registry), "--history-directory", str(tmp_path / "missing")]
    )
    with pytest.raises(SystemExit):
        bad.handler(bad)


def test_retention_default_limits_finalized_runs() -> None:
    """Unflagged retention policy keeps only the single latest finalized bundle, with no byte budget."""
    policy = _resolve_process_retention_policy(argparse.Namespace(max_detail_bytes=None, max_detail_runs=None))
    assert policy is not None
    assert policy.max_detail_runs == 1
    assert policy.max_detail_bytes is None


def test_retention_explicit_flags_override_default() -> None:
    """Explicit retention flags win over the new five-run default."""
    policy = _resolve_process_retention_policy(argparse.Namespace(max_detail_bytes=None, max_detail_runs=2))
    assert policy is not None
    assert policy.max_detail_runs == 2
