"""Lifecycle invariants for the canonical backtest mhs command."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import pytest

import src.cli.commands.backtest as backtest_mod
from src.cli.commands.backtest import _resolve_destinations, _resolve_fingerprint, _resolve_retention_policy, run_mhs_backtest
from src.cli.main import build_root_parser
from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING


def _parse(argv: list[str]) -> argparse.Namespace:
    return build_root_parser().parse_args(argv)


def _install_supervisor(monkeypatch, status: str = "completed") -> dict:
    import src.application.mhs_supervisor as supmod
    from src.application.mhs_supervisor import MhsSupervisedRun

    seen: dict = {}

    def _fake(**kwargs):
        seen.update(kwargs)
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


def _seed_finalized_run(registry: Path, fingerprint: str) -> str:
    from src.backtests.contracts import RunFinalization, RunRegistration
    from src.backtests.registry import finalize_run, initialize_registry, register_run

    initialize_registry(registry)
    run_id = uuid.uuid4().hex
    register_run(
        registry,
        RunRegistration(
            run_id=run_id, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": fingerprint, "window": "3m"},
            managed_directory=None,
        ),
    )
    finalize_run(
        registry,
        RunFinalization(
            run_id=run_id, status="completed", finalized_at="2026-01-02T00:00:00+00:00",
            primary_valid=True, terminal_certified=True, outcome={"ok": True},
        ),
        (),
    )
    return run_id


def test_resolve_destinations_creates_single_result_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    args = _parse(["backtest", "mhs"])
    result, targets = _resolve_destinations(args)
    assert result.parent.parent == tmp_path / "runs"
    assert result.name == "result.json"
    uuid.UUID(hex=result.parent.name)
    assert targets is None
    assert not (result.parent / "primary.json").exists()
    assert not (result.parent / "failure.json").exists()
    assert not (result.parent / "run.json").exists()


def test_equivalent_finalized_request_reuses_without_launch(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    registry = tmp_path / "registry.sqlite3"
    probe = _parse(["backtest", "mhs"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(probe)
    fingerprint = _resolve_fingerprint(probe, start, end, budget)
    existing = _seed_finalized_run(registry, fingerprint)

    def _boom(**kwargs):
        raise AssertionError("supervisor must not launch on reuse")

    import src.application.mhs_supervisor as supmod

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _boom)
    with caplog.at_level("INFO"):
        run_mhs_backtest(_parse(["backtest", "mhs"]))
    assert any(existing in r.getMessage() for r in caplog.records)


def test_force_bypasses_reuse_with_fresh_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    registry = tmp_path / "registry.sqlite3"
    probe = _parse(["backtest", "mhs"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(probe)
    fingerprint = _resolve_fingerprint(probe, start, end, budget)
    existing = _seed_finalized_run(registry, fingerprint)
    seen = _install_supervisor(monkeypatch)
    run_mhs_backtest(_parse(["backtest", "mhs", "--force"]))
    assert seen["run_id"] != existing
    uuid.UUID(hex=seen["run_id"])


def test_fingerprint_excludes_output_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    first = _parse(["backtest", "mhs", "--output", str(tmp_path / "a.json")])
    second = _parse(["backtest", "mhs", "--output", str(tmp_path / "b.json")])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(first)
    assert _resolve_fingerprint(first, start, end, budget) == _resolve_fingerprint(second, start, end, budget)


def test_fingerprint_includes_financial_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    base = _parse(["backtest", "mhs"])
    start = backtest_mod._utc_timestamp(None, "start", DISCOVERY_START)
    end = backtest_mod._utc_timestamp(None, "end", PROCESS_EVALUATION_CEILING)
    budget = backtest_mod._resolve_budget(base)
    reference = _resolve_fingerprint(base, start, end, budget)
    altered_interval = _parse(["backtest", "mhs", "--start", "2022-01-01", "--end", "2022-01-04"])
    new_start = backtest_mod._utc_timestamp("2022-01-01", "start", DISCOVERY_START)
    new_end = backtest_mod._utc_timestamp("2022-01-04", "end", PROCESS_EVALUATION_CEILING)
    assert _resolve_fingerprint(altered_interval, new_start, new_end, budget) != reference
    altered_data = _parse(["backtest", "mhs", "--data-root", str(tmp_path)])
    assert _resolve_fingerprint(altered_data, start, end, budget) != reference
    altered_control = _parse(["backtest", "mhs", "--rebalance-tracking-error-threshold", "0.2"])
    assert _resolve_fingerprint(altered_control, start, end, budget) != reference
    from src.mhs.resources import MhsMemoryBudget

    other_budget = MhsMemoryBudget(
        total_tree_pss_bytes=budget.total_tree_pss_bytes + 1,
        replay_tree_pss_bytes=budget.replay_tree_pss_bytes,
        min_available_bytes=budget.min_available_bytes,
    )
    assert _resolve_fingerprint(base, start, end, other_budget) != reference


def test_deprecated_output_flags_fail_early() -> None:
    with pytest.raises(SystemExit):
        _parse(["backtest", "mhs", "--failure-output", "x.json"])
    with pytest.raises(SystemExit):
        _parse(["backtest", "mhs", "--run-output", "x.json"])


def test_explicit_registry_and_retention_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    custom = tmp_path / "custom" / "registry.sqlite3"
    seen = _install_supervisor(monkeypatch)
    run_mhs_backtest(_parse(["backtest", "mhs", "--registry-path", str(custom)]))
    assert seen["registry_path"] == custom
    assert seen["retention_policy"] is None
    assert _resolve_retention_policy(_parse(["backtest", "mhs"])) is None
    budgeted = _parse(["backtest", "mhs", "--max-detail-bytes", "100", "--max-detail-runs", "2"])
    policy = _resolve_retention_policy(budgeted)
    assert policy is not None
    assert policy.max_detail_bytes == 100
    assert policy.max_detail_runs == 2


def test_destinations_reject_invalid_targets_and_occupied_paths(tmp_path, monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_destinations(_parse(["backtest", "mhs", "--targets-output", str(tmp_path / "bad.txt")]))
    occupied_targets = tmp_path / "taken.parquet"
    occupied_targets.write_bytes(b"x")
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_destinations(_parse(["backtest", "mhs", "--targets-output", str(occupied_targets)]))
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_destinations(_parse(["backtest", "mhs", "--output", str(tmp_path / "bad.csv")]))
    occupied = tmp_path / "taken.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_destinations(_parse(["backtest", "mhs", "--output", str(occupied)]))

    def _no_launch(*args, **kwargs):
        raise AssertionError("worker must not launch")

    monkeypatch.setattr(subprocess, "Popen", _no_launch)
    with pytest.raises(SystemExit, match=r".+"):
        run_mhs_backtest(_parse(["backtest", "mhs", "--output", str(occupied)]))
    bad_budget = _parse(["backtest", "mhs", "--max-detail-bytes", "0"])
    with pytest.raises(SystemExit, match=r".+"):
        _resolve_retention_policy(bad_budget)


def test_ops_verify_history_migration_wiring(tmp_path) -> None:
    import argparse as _argparse
    import json as _json

    from src.backtests.migration import migrate_legacy_backtests
    from src.cli.commands.ops import add_ops_commands

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
    add_ops_commands(parser)
    args = parser.parse_args(
        ["backtests-verify-history-migration", "--registry-path", str(registry), "--history-directory", str(source)]
    )
    args.handler(args)
    bad = parser.parse_args(
        ["backtests-verify-history-migration", "--registry-path", str(registry), "--history-directory", str(tmp_path / "missing")]
    )
    with pytest.raises(SystemExit):
        bad.handler(bad)
