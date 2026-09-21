"""Invariant scenarios for the canonical backtest mhs command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

import src.cli.commands.backtest as backtest_mod
from src.cli.commands.backtest import add_backtest_commands, run_mhs_backtest
from src.cli.main import build_root_parser
from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING
from src.mhs.resources import MhsMemoryBudget


def _parse(argv: list[str]) -> argparse.Namespace:
    return build_root_parser().parse_args(argv)


def _install_supervisor(monkeypatch, status: str = "completed") -> dict:
    import src.application.mhs_supervisor as supmod

    seen: dict = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(status=status)

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _fake)
    return seen


def test_backtest_mhs_canonical_default(tmp_path, monkeypatch) -> None:
    """Canonical default: no overrides select existing dates, budget and 3m supervision."""
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    args = _parse(["backtest", "mhs"])
    assert args.handler is run_mhs_backtest
    seen = _install_supervisor(monkeypatch)
    assert run_mhs_backtest(args) is None
    assert seen["start"] == DISCOVERY_START
    assert seen["end"] == PROCESS_EVALUATION_CEILING
    assert seen["memory_budget"] == MhsMemoryBudget()
    assert seen["targets_output"] is None
    assert seen["timeout_seconds"] is None
    assert seen["poll_seconds"] == 0.25


def test_backtest_mhs_evidence_destinations(tmp_path, monkeypatch) -> None:
    """Evidence destinations: omitted output yields unique run dirs with one result envelope."""

    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    first = _install_supervisor(monkeypatch)
    run_mhs_backtest(_parse(["backtest", "mhs"]))
    second = _install_supervisor(monkeypatch)
    run_mhs_backtest(_parse(["backtest", "mhs"]))
    assert first["result_output"] != second["result_output"]
    for seen in (first, second):
        assert seen["result_output"].parent.parent == tmp_path / "runs"
        assert seen["result_output"].name == "result.json"
        assert tmp_path in seen["result_output"].parents
    out = tmp_path / "custom.json"
    explicit = _install_supervisor(monkeypatch)
    run_mhs_backtest(
        _parse(
            [
                "backtest", "mhs", "--output", str(out),
                "--targets-output", str(tmp_path / "custom.targets.parquet"),
            ]
        )
    )
    assert explicit["result_output"] == out
    assert explicit["targets_output"] == tmp_path / "custom.targets.parquet"


def test_backtest_mhs_aware_dates(tmp_path, monkeypatch) -> None:
    """Aware dates: date-only UTC and timezone-aware ISO inputs normalize to identical instants."""
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    plain = _install_supervisor(monkeypatch)
    run_mhs_backtest(_parse(["backtest", "mhs", "--start", "2022-01-01", "--end", "2022-01-04"]))
    assert plain["start"] == pd.Timestamp("2022-01-01", tz="UTC")
    aware = _install_supervisor(monkeypatch)
    run_mhs_backtest(
        _parse(
            [
                "backtest", "mhs",
                "--start", "2022-01-01T00:00:00+00:00",
                "--end", "2022-01-04T00:00:00+00:00",
            ]
        )
    )
    assert aware["start"] == plain["start"]
    assert aware["end"] == plain["end"]
    shifted = _install_supervisor(monkeypatch)
    run_mhs_backtest(
        _parse(
            [
                "backtest", "mhs",
                "--start", "2022-01-01T09:00:00+09:00",
                "--end", "2022-01-04T09:00:00+09:00",
            ]
        )
    )
    assert shifted["start"] == plain["start"]
    assert shifted["end"] == plain["end"]


def test_backtest_mhs_explicit_budget(tmp_path, monkeypatch) -> None:
    """Explicit budget: 4/3/2 GiB controls reach supervision without an added CLI cap."""
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    seen = _install_supervisor(monkeypatch)
    run_mhs_backtest(
        _parse(
            [
                "backtest", "mhs",
                "--total-tree-pss-bytes", str(4 * 2**30),
                "--replay-tree-pss-bytes", str(3 * 2**30),
                "--min-available-bytes", str(2 * 2**30),
            ]
        )
    )
    assert seen["memory_budget"] == MhsMemoryBudget(
        total_tree_pss_bytes=4 * 2**30,
        replay_tree_pss_bytes=3 * 2**30,
        min_available_bytes=2 * 2**30,
    )


def _no_launch(*args, **kwargs):
    raise AssertionError("worker must not launch")


def test_backtest_mhs_rejects_invalid_controls(tmp_path, monkeypatch) -> None:
    """Invalid controls: bad timeframe, budget, timing, dates or occupied evidence fail before launch."""
    monkeypatch.setattr(subprocess, "Popen", _no_launch)
    out = tmp_path / "custom.json"
    args = _parse(["backtest", "mhs", "--output", str(out)])
    args.execution_timeframe = "1m"
    with pytest.raises(SystemExit, match=r".+"):
        run_mhs_backtest(args)
    with pytest.raises(SystemExit, match=r".+"):
        run_mhs_backtest(
            _parse(
                [
                    "backtest", "mhs", "--output", str(out),
                    "--replay-tree-pss-bytes", str(MhsMemoryBudget().total_tree_pss_bytes + 1),
                ]
            )
        )
    with pytest.raises(SystemExit, match=r".+"):
        run_mhs_backtest(_parse(["backtest", "mhs", "--output", str(out), "--start", "not-a-date"]))
    with pytest.raises(SystemExit, match=r".+"):
        run_mhs_backtest(
            _parse(["backtest", "mhs", "--output", str(out), "--timeout-seconds", str(float("nan"))])
        )
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match=r".+"):
        run_mhs_backtest(_parse(["backtest", "mhs", "--output", str(out)]))
    assert json.loads(out.read_text(encoding="utf-8")) == {}


def test_backtest_mhs_nonsuccess_exit(tmp_path, monkeypatch) -> None:
    """Non-success exit: every non-completed supervisor status exits nonzero after outcome recording."""
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)
    out = tmp_path / "custom.json"
    for status in ("failed", "timed_out", "signaled", "resource_rejected", "interrupted"):
        _install_supervisor(monkeypatch, status=status)
        with pytest.raises(SystemExit) as excinfo:
            run_mhs_backtest(_parse(["backtest", "mhs", "--output", str(out)]))
        assert excinfo.value.code == 1


def test_backtest_mhs_registers_leaf_handler() -> None:
    """Command wiring registers the mhs leaf with its supervised handler and 3m default."""
    sub = argparse.ArgumentParser().add_subparsers()
    add_backtest_commands(sub.add_parser("backtest"))
    leaf = sub.choices["backtest"]
    assert leaf is not None
    mhs = leaf.parse_args(["mhs"])
    assert mhs.handler is run_mhs_backtest
    assert mhs.execution_timeframe == "3m"
    assert mhs.total_tree_pss_bytes is None
    assert mhs.poll_seconds == 0.25


def test_backtest_mhs_source_owned_execution() -> None:
    """Source-owned execution: the worker launch references the application worker, never tools or the CLI."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "backtest", "mhs", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "--total-tree-pss-bytes" in proc.stdout
    import src.application.mhs_supervisor as supmod
    import src.application.mhs_worker as workermod

    supervisor_source = Path(supmod.__file__).read_text(encoding="utf-8")
    assert '"-m", "src.application.mhs_worker"' in supervisor_source
    assert "src.cli.main" not in supervisor_source
    assert "tools." not in supervisor_source
    worker_source = Path(workermod.__file__).read_text(encoding="utf-8")
    assert "src.cli" not in worker_source
    assert "tools." not in worker_source


def test_backtest_mhs_command_discovery(monkeypatch) -> None:
    """Independent command discovery: loading help never runs evaluation or needs retired tools."""
    import src.application.mhs_supervisor as supmod

    def _boom(*args, **kwargs):
        raise AssertionError("evaluation must not run during discovery")

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _boom)
    parser = build_root_parser()
    group = next(action for action in parser._actions if action.dest == "group")
    assert "backtest" in group.choices
    assert "backtest" in (build_root_parser.__doc__ or "")
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["backtest", "mhs", "--help"])
    assert excinfo.value.code == 0


def _frozen_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "backtest", "mhs-frozen",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--output", str(tmp_path / "frozen.json"), *extra,
    ]


def _install_frozen(monkeypatch: pytest.MonkeyPatch) -> dict:
    import src.mhs.frozen_research_run as run_mod

    seen: dict = {}

    def _fake_run(request: object) -> object:
        seen["request"] = request
        return types.SimpleNamespace(request=request)

    def _fake_persist(run: object, output: Path) -> Path:
        seen["output"] = output
        Path(output).write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(run_mod, "run_frozen_mhs_backtest", _fake_run)
    import src.mhs.frozen_research_report as report_mod

    monkeypatch.setattr(report_mod, "persist_frozen_mhs_backtest", _fake_persist)
    return seen


def test_backtest_mhs_frozen_distinct_from_legacy() -> None:
    """Frozen registration uses its own handler and legacy mhs semantics stay unchanged."""
    from src.cli.commands.backtest import run_frozen_mhs_backtest_command

    frozen = _parse(_frozen_argv(Path("frozen.json")))
    assert frozen.handler is run_frozen_mhs_backtest_command
    assert frozen.handler is not run_mhs_backtest
    legacy = _parse(["backtest", "mhs"])
    assert legacy.handler is run_mhs_backtest


def test_backtest_mhs_frozen_breadth_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitted breadth selects primary Top-20; breadth 40 names an explicit control variant."""
    seen = _install_frozen(monkeypatch)
    backtest_mod.run_frozen_mhs_backtest_command(_parse(_frozen_argv(tmp_path)))
    assert seen["request"].strategy.strategy_id == "frozen_mhs_top20_v1"
    assert seen["request"].strategy.breadth == 20
    assert seen["output"] == tmp_path / "frozen.json"
    out40 = tmp_path / "frozen40.json"
    backtest_mod.run_frozen_mhs_backtest_command(_parse([
        "backtest", "mhs-frozen",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--breadth", "40", "--output", str(out40),
    ]))
    assert seen["request"].strategy.breadth == 40
    assert "40" in seen["request"].strategy.strategy_id
    assert seen["request"].strategy.strategy_id != "frozen_mhs_top20_v1"


def test_backtest_mhs_frozen_required_dates_and_fresh_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing source start, invalid order, or an occupied output exits before runner invocation."""
    seen = _install_frozen(monkeypatch)
    with pytest.raises(SystemExit, match=r"source-start"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "a.json")])
        )
    with pytest.raises(SystemExit, match=r"source-start < start"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2025-03-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "b.json")])
        )
    occupied = tmp_path / "occupied.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"fresh"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(occupied)])
        )
    assert "request" not in seen


def test_backtest_mhs_frozen_variant_breadth_and_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Arbitrary breadth names a control variant; bad controls and replay failures exit nonzero."""
    seen = _install_frozen(monkeypatch)
    out12 = tmp_path / "frozen12.json"
    backtest_mod.run_frozen_mhs_backtest_command(_parse([
        "backtest", "mhs-frozen",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--breadth", "12", "--output", str(out12),
    ]))
    assert seen["request"].strategy.breadth == 12
    assert "12" in seen["request"].strategy.strategy_id
    with pytest.raises(SystemExit, match=r"breadth"):
        backtest_mod.run_frozen_mhs_backtest_command(_parse([
            "backtest", "mhs-frozen",
            "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
            "--breadth", "0", "--output", str(tmp_path / "zero.json"),
        ]))
    with pytest.raises(SystemExit, match=r"start is required"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "c.json")])
        )
    with pytest.raises(SystemExit, match=r"end is required"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--output", str(tmp_path / "d.json")])
        )
    auto = tmp_path / "auto"
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", auto)
    backtest_mod.run_frozen_mhs_backtest_command(
        _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    resolved = seen["output"]
    assert resolved.name == "result.json"
    assert resolved.parent.parent == auto
    assert resolved.parent.is_dir()
    assert (resolved.parent / "manifest.json").is_file()
    with pytest.raises(SystemExit, match=r"JSON path"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "e.txt")])
        )
    with pytest.raises(SystemExit, match=r"invalid frozen backtest request"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01T00:00:00+00:00", "--end", "2025-01-01T12:00:00+00:00", "--output", str(tmp_path / "f.json")])
        )
    import src.mhs.frozen_research_run as run_mod

    monkeypatch.setattr(run_mod, "run_frozen_mhs_backtest", lambda request: (_ for _ in ()).throw(ValueError("boom")))
    failed = tmp_path / "failed.json"
    with pytest.raises(SystemExit, match=r"frozen backtest failed"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse([*_frozen_argv(tmp_path)[:8], "--output", str(failed)])
        )
    assert not failed.exists()


def test_frozen_omitted_output_resolves_to_fresh_run_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitted output resolves into a fresh UUID run directory that already exists."""
    import re

    frozen_root = tmp_path / "frozen"
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    seen = _install_frozen(monkeypatch)
    backtest_mod.run_frozen_mhs_backtest_command(
        _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    resolved = seen["output"]
    assert resolved.name == "result.json"
    assert resolved.parent.parent == frozen_root
    assert re.fullmatch(r"[0-9a-f]{32}", resolved.parent.name) is not None
    assert resolved.parent.is_dir()


def test_frozen_manifest_matches_request_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifest beside the result records the request identity in JSON primitives."""
    frozen_root = tmp_path / "frozen"
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    seen = _install_frozen(monkeypatch)
    backtest_mod.run_frozen_mhs_backtest_command(
        _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    request = seen["request"]
    manifest = json.loads((seen["output"].parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_start"] == request.source_start.isoformat()
    assert manifest["evaluation_start"] == request.evaluation_start.isoformat()
    assert manifest["evaluation_end"] == request.evaluation_end.isoformat()
    assert manifest["breadth"] == 20
    assert manifest["strategy_id"] == request.strategy.strategy_id
    for key in ("source_start", "evaluation_start", "evaluation_end"):
        assert manifest[key].endswith("+00:00")


def test_frozen_explicit_output_rejects_invalid_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit frozen output keeps the legacy suffix and freshness guards."""
    _install_frozen(monkeypatch)
    with pytest.raises(SystemExit, match=r"JSON path"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "bad.txt")])
        )
    occupied = tmp_path / "occupied.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"fresh"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(occupied)])
        )


def test_retention_default_limits_finalized_runs() -> None:
    """Unflagged retention policy keeps five finalized bundles with no byte budget."""
    policy = backtest_mod._resolve_retention_policy(argparse.Namespace(max_detail_bytes=None, max_detail_runs=None))
    assert policy is not None
    assert policy.max_detail_runs == 5
    assert policy.max_detail_bytes is None


def test_retention_explicit_flags_override_default() -> None:
    """Explicit retention flags win over the new five-run default."""
    policy = backtest_mod._resolve_retention_policy(argparse.Namespace(max_detail_bytes=None, max_detail_runs=2))
    assert policy is not None
    assert policy.max_detail_runs == 2
