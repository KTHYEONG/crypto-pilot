"""Invariant scenarios for the canonical backtest mhs command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
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
    assert seen["request"].strategy.strategy_id == "frozen_mhs_top20_v2"
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
    assert seen["request"].strategy.strategy_id != "frozen_mhs_top20_v2"


def test_backtest_mhs_frozen_growth_variant_selects_registered_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--variant growth selects the registered clip + exposure policy at Top-20 breadth."""
    from src.mhs.params import FROZEN_GROWTH_EXPOSURE_MULTIPLIER, FROZEN_GROWTH_NAME_CLIP

    seen = _install_frozen(monkeypatch)
    backtest_mod.run_frozen_mhs_backtest_command(_parse([*_frozen_argv(tmp_path)[:8], "--variant", "growth", "--output", str(tmp_path / "growth.json")]))
    assert seen["request"].strategy.strategy_id == "frozen_mhs_top20_growth_v2"
    assert seen["request"].strategy.exposure_multiplier == FROZEN_GROWTH_EXPOSURE_MULTIPLIER
    assert seen["request"].strategy.name_clip == FROZEN_GROWTH_NAME_CLIP
    with pytest.raises(SystemExit):
        backtest_mod._frozen_strategy(20, "turbo")


def test_backtest_mhs_frozen_growth_rejects_non20_breadth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Growth with a non-20 breadth exits before any workload launch."""
    seen = _install_frozen(monkeypatch)
    with pytest.raises(SystemExit):
        backtest_mod.run_frozen_mhs_backtest_command(_parse([*_frozen_argv(tmp_path)[:8], "--variant", "growth", "--breadth", "40", "--output", str(tmp_path / "g40.json")]))
    assert "request" not in seen


def test_backtest_mhs_frozen_growth_run_directory_labelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A growth run without --output resolves into a top20_growth_ directory."""
    import argparse

    frozen_root = tmp_path / "frozen"
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    output = backtest_mod._resolve_frozen_destination(
        argparse.Namespace(output=None, variant="growth"),
        start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"), breadth=20, variant="growth",
    )
    assert "top20_growth_" in output.parent.name
    assert output.name == "result.json"


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
    assert re.fullmatch(r"20250101_20250201_top20_\d{8}T\d{6}Z", resolved.parent.name) is not None
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
    """Unflagged retention policy keeps only the single latest finalized bundle, with no byte budget."""
    policy = backtest_mod._resolve_retention_policy(argparse.Namespace(max_detail_bytes=None, max_detail_runs=None))
    assert policy is not None
    assert policy.max_detail_runs == 1
    assert policy.max_detail_bytes is None


def test_retention_explicit_flags_override_default() -> None:
    """Explicit retention flags win over the new five-run default."""
    policy = backtest_mod._resolve_retention_policy(argparse.Namespace(max_detail_bytes=None, max_detail_runs=2))
    assert policy is not None
    assert policy.max_detail_runs == 2


def test_frozen_index_and_prune_keep_only_recent_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each run appends one index line forever; only the newest `keep` run directories survive on disk."""
    frozen_root = tmp_path / "backtests" / "frozen" / "runs"
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    monkeypatch.setattr(backtest_mod, "DEFAULT_DETAIL_RETENTION_MAX_RUNS", 2)
    seen = _install_frozen(monkeypatch)
    for day in ("01", "02", "03"):
        backtest_mod.run_frozen_mhs_backtest_command(
            _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", f"2025-01-{day}", "--end", "2025-02-01"])
        )
    index_lines = (tmp_path / "backtests" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 3
    for line in index_lines:
        row = json.loads(line)
        assert row["kind"] == "mhs_frozen"
        assert row["strategy_id"] == "frozen_mhs_top20_v2"
    remaining = sorted(d for d in frozen_root.iterdir() if d.is_dir())
    assert len(remaining) == 2
    assert seen["output"].parent.exists()


def test_frozen_destination_dedupes_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run-name collision (same dates/breadth/second) appends a numeric suffix instead of clobbering."""
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir(parents=True)
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-02-01", tz="UTC")
    frozen_now = pd.Timestamp("2026-06-01T00:00:00Z")
    name = backtest_mod._frozen_run_name(start, end, 20, frozen_now)
    (frozen_root / name).mkdir()
    orig_now = pd.Timestamp.now
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, tz=None: frozen_now))
    try:
        output = backtest_mod._resolve_frozen_destination(
            argparse.Namespace(output=None), start=start, end=end, breadth=20,
        )
    finally:
        pd.Timestamp.now = orig_now
    assert output.parent.name == f"{name}-2"


def test_prune_frozen_runs_noop_when_directory_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pruning before any frozen run has ever been created is a safe no-op."""
    frozen_root = tmp_path / "never_created"
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    backtest_mod._prune_frozen_runs(keep=5)
    assert not frozen_root.exists()


def test_backtest_mhs_appends_headline_to_shared_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed canonical run appends one headline row to the shared cross-pipeline index."""
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path)

    def _fake(**kwargs):
        Path(kwargs["result_output"]).write_text(
            json.dumps({"financial": {"base": {"cagr": 0.42, "max_drawdown": -0.1}}}), encoding="utf-8",
        )
        return types.SimpleNamespace(status="completed")

    import src.application.mhs_supervisor as supmod

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _fake)
    run_mhs_backtest(_parse(["backtest", "mhs", "--start", "2022-01-01", "--end", "2022-01-04"]))
    lines = (tmp_path / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["kind"] == "mhs"
    assert row["base_cagr"] == 0.42
    assert row["base_max_drawdown"] == -0.1
    assert row["run_dir"].startswith("runs" + "/")


def test_append_backtest_index_falls_back_to_absolute_path_outside_root(tmp_path: Path) -> None:
    """A run directory outside the index root records its absolute path instead of raising."""
    index_path = tmp_path / "inside" / "index.jsonl"
    index_path.parent.mkdir(parents=True)
    outside_run_dir = tmp_path / "elsewhere" / "run1"
    outside_run_dir.mkdir(parents=True)
    now = pd.Timestamp("2026-01-01", tz="UTC")
    backtest_mod._append_backtest_index(
        index_path=index_path, kind="mhs", run_dir=outside_run_dir, created_at=now,
        evaluation_start=now, evaluation_end=now, strategy_id="s", base_cagr=None, base_max_drawdown=None,
    )
    row = json.loads(index_path.read_text(encoding="utf-8").strip())
    assert row["run_dir"] == str(outside_run_dir)


def test_frozen_specs_use_submit_anchor() -> None:
    """Both frozen cost cases cross from the last pre-submission mark at 6 and 18 bps."""
    base, stress = backtest_mod._frozen_specs()
    assert base.decision_anchor == "submit_bar"
    assert stress.decision_anchor == "submit_bar"
    assert base.one_way_taker_bps() == 6.0
    assert stress.one_way_taker_bps() == 18.0


def test_frozen_execution_flag_maps_to_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--execution maker selects the strict-proxy bound and labels the run directory."""
    frozen_root = tmp_path / "frozen"
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    seen = _install_frozen(monkeypatch)
    backtest_mod.run_frozen_mhs_backtest_command(
        _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--execution", "maker"])
    )
    assert seen["request"].execution_bound == "OHLCV_STRICT_PROXY"
    assert "_maker_" in seen["output"].parent.name


def test_frozen_default_execution_is_taker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No flag keeps the immediate-taker bound and today's run directory format."""
    import re

    frozen_root = tmp_path / "frozen"
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    seen = _install_frozen(monkeypatch)
    backtest_mod.run_frozen_mhs_backtest_command(
        _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    assert seen["request"].execution_bound == "OHLCV_IMMEDIATE_TAKER"
    assert re.fullmatch(r"20250101_20250201_top20_\d{8}T\d{6}Z", seen["output"].parent.name) is not None


def test_frozen_execution_rejects_unknown_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An execution value outside taker/maker exits before any workload launch."""
    seen = _install_frozen(monkeypatch)
    args = _parse(_frozen_argv(tmp_path))
    args.execution = "peg"
    with pytest.raises(SystemExit, match=r"execution"):
        backtest_mod.run_frozen_mhs_backtest_command(args)
    assert "request" not in seen


def test_frozen_index_records_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen index row carries execution while a canonical mhs row has no such key."""
    frozen_root = tmp_path / "backtests" / "frozen" / "runs"
    monkeypatch.setattr(backtest_mod, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", frozen_root)
    _install_frozen(monkeypatch)
    backtest_mod.run_frozen_mhs_backtest_command(
        _parse(["backtest", "mhs-frozen", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    frozen_row = json.loads((tmp_path / "backtests" / "index.jsonl").read_text(encoding="utf-8").strip())
    assert frozen_row["execution"] == "taker"

    def _fake(**kwargs):
        Path(kwargs["result_output"]).write_text(
            json.dumps({"financial": {"base": {"cagr": 0.1, "max_drawdown": -0.05}}}), encoding="utf-8",
        )
        return types.SimpleNamespace(status="completed")

    import src.application.mhs_supervisor as supmod

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _fake)
    run_mhs_backtest(_parse(["backtest", "mhs", "--start", "2022-01-01", "--end", "2022-01-04"]))
    rows = (tmp_path / "backtests" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    canonical_row = json.loads(rows[-1])
    assert canonical_row["kind"] == "mhs"
    assert "execution" not in canonical_row


def _fake_run_dir(tmp_path: Path, multiplier: float = 2.5) -> Path:
    run_dir = tmp_path / "frozen_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_id": "frozen_mhs_top20_growth_v2",
        "breadth": 20,
        "exposure_multiplier": multiplier,
        "execution_bound": "OHLCV_IMMEDIATE_TAKER",
        "source_start": "2024-01-01T00:00:00+00:00",
        "evaluation_start": "2025-01-01T00:00:00+00:00",
        "evaluation_end": "2025-02-01T00:00:00+00:00",
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    idx = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    pd.DataFrame(
        {"base_return": np.full(len(idx), 0.001), "max_name_weight": np.full(len(idx), 0.05)},
        index=idx,
    ).to_parquet(run_dir / "daily.parquet")
    return run_dir


def _install_exposure(monkeypatch: pytest.MonkeyPatch) -> dict:
    import src.mhs.frozen_research_universe as universe_mod
    import src.mhs.growth_exposure as growth_mod
    import src.mhs.panel as panel_mod
    import src.mhs.resources as resources_mod

    seen: dict = {}

    def _fake_panel(root, interval, columns, start, end, partition="all", selection_mode="causal_history", allocation_admission=None):
        seen["selection_mode"] = selection_mode
        if allocation_admission is not None:
            allocation_admission(1024)
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return {name: pd.DataFrame(100.0, index=idx, columns=["AAA", "BBB"], dtype="float64") for name in columns}

    def _fake_roster(daily_close, daily_quote_volume, census, *, breadth, blocked_decisions=None):
        seen["breadth"] = breadth
        seen["blocked_decisions"] = blocked_decisions
        return pd.DataFrame(False, index=daily_close.index, columns=list(census), dtype=bool)

    def _fake_solve(unit_returns, *, max_name_weight, gaps, mean_haircut, grid, plateau_tolerance, n_paths, horizon_years, mean_block_days, seed):
        seen["unit_returns"] = unit_returns
        seen["max_name_weight"] = max_name_weight
        seen["gaps"] = gaps
        seen["solver_params"] = {
            "mean_haircut": mean_haircut, "grid": grid, "plateau_tolerance": plateau_tolerance,
            "n_paths": n_paths, "horizon_years": horizon_years, "mean_block_days": mean_block_days, "seed": seed,
        }
        grid_tuple = tuple(grid)
        return types.SimpleNamespace(
            grid=grid_tuple, growth=(0.1,) * len(grid_tuple), ruin_probability=(0.0,) * len(grid_tuple),
            argmax=grid_tuple[-1], chosen=grid_tuple[0], gap_events_per_year=1.5, gap_sample_size=3,
        )

    monkeypatch.setattr(panel_mod, "load_base_panel", _fake_panel)
    monkeypatch.setattr(universe_mod, "build_frozen_pit_roster", _fake_roster)
    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", _fake_solve)
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda: frozenset())
    monkeypatch.setattr(resources_mod, "resolve_mhs_memory_budget", lambda budget: budget)
    monkeypatch.setattr(resources_mod, "assert_mhs_stage_allocation", lambda **kwargs: None)
    return seen


def test_frozen_exposure_unlevers_by_run_multiplier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The solver receives base returns and mean name weight divided by the run multiplier."""
    from src.mhs.params import (
        COMMITTEE_GROWTH_HORIZON_YEARS,
        COMMITTEE_GROWTH_N_PATHS,
        FROZEN_EXPOSURE_MEAN_HAIRCUT,
        NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
    )

    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure(monkeypatch)
    backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))
    np.testing.assert_allclose(seen["unit_returns"].to_numpy(), 0.001 / 2.5)
    assert seen["max_name_weight"] == pytest.approx(0.05 / 2.5)
    assert seen["solver_params"]["mean_haircut"] == FROZEN_EXPOSURE_MEAN_HAIRCUT
    assert seen["solver_params"]["n_paths"] == COMMITTEE_GROWTH_N_PATHS
    assert seen["solver_params"]["horizon_years"] == COMMITTEE_GROWTH_HORIZON_YEARS
    assert seen["solver_params"]["mean_block_days"] == NULL_BOOTSTRAP_MEAN_BLOCK_DAYS
    exposure = json.loads((run_dir / "exposure.json").read_text(encoding="utf-8"))
    assert exposure["chosen"] == exposure["grid"][0]
    assert exposure["execution_bound"] == "OHLCV_IMMEDIATE_TAKER"
    assert exposure["exposure_multiplier"] == 2.5


def test_frozen_exposure_gap_roster_ignores_trading_exclusions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap roster is built with no trading-exclusion filter over the evaluation window."""
    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure(monkeypatch)
    backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))
    assert seen["blocked_decisions"] is None
    assert seen["breadth"] == 20
    assert seen["selection_mode"] == "causal_history"


def test_frozen_exposure_gap_population_restricted_to_delisted_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only symbols with a registered DELISTED exclusion feed the gap sampler; the rest never do."""
    import src.mhs.growth_exposure as growth_mod

    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure(monkeypatch)
    monkeypatch.setattr(growth_mod, "structurally_excluded_symbols", lambda: frozenset({"AAA"}))
    real_sample = growth_mod.roster_gap_sample
    captured: dict = {}

    def _spy(daily_close, roster, *, threshold):
        captured["columns"] = list(daily_close.columns)
        return real_sample(daily_close, roster, threshold=threshold)

    monkeypatch.setattr(growth_mod, "roster_gap_sample", _spy)
    backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))
    assert captured["columns"] == ["AAA"]
    exposure = json.loads((run_dir / "exposure.json").read_text(encoding="utf-8"))
    assert exposure["gap_symbols"] == ["AAA"]


def test_frozen_exposure_empty_exclusion_registry_skips_sampler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No registered exclusion yields a zero-event gap sample without calling the sampler on zero columns."""
    import src.mhs.growth_exposure as growth_mod

    run_dir = _fake_run_dir(tmp_path)
    seen = _install_exposure(monkeypatch)

    def _boom(*args, **kwargs):
        raise AssertionError("roster_gap_sample must not be called for an empty exclusion set")

    monkeypatch.setattr(growth_mod, "roster_gap_sample", _boom)
    backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))
    assert seen["gaps"].events_per_year == 0.0
    assert seen["gaps"].magnitudes.size == 0
    exposure = json.loads((run_dir / "exposure.json").read_text(encoding="utf-8"))
    assert exposure["gap_symbols"] == []


def test_frozen_exposure_artifact_fresh_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing exposure.json exits before any panel load."""
    import src.mhs.panel as panel_mod

    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "exposure.json").write_text("{}", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("panel must not load")

    monkeypatch.setattr(panel_mod, "load_base_panel", _boom)
    with pytest.raises(SystemExit, match=r"fresh"):
        backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))


def test_frozen_exposure_rejects_missing_run_dir(tmp_path: Path) -> None:
    """A missing run-dir flag or a non-directory path exits before any workload."""
    with pytest.raises(SystemExit, match=r"run-dir is required"):
        backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure"]))
    with pytest.raises(SystemExit, match=r"existing frozen run directory"):
        backtest_mod.run_frozen_exposure_command(
            _parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(tmp_path / "missing")])
        )


def test_frozen_exposure_rejects_invalid_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run directory without result.json exits instead of solving garbage."""
    _install_exposure(monkeypatch)
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match=r"invalid frozen run artifacts"):
        backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))


def test_frozen_exposure_solver_rejection_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A solver rejection exits instead of persisting an exposure artifact."""
    import src.mhs.growth_exposure as growth_mod

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure(monkeypatch)

    def _reject(*args, **kwargs):
        raise ValueError("no admissible rung")

    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", _reject)
    with pytest.raises(SystemExit, match=r"frozen exposure failed"):
        backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))
    assert not (run_dir / "exposure.json").exists()


def _account_argv(*extra: str) -> list[str]:
    return [
        "backtest", "mhs-frozen-account",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        *extra,
    ]


def _install_account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, recon_fail: bool = False, stub_venue: bool = True) -> dict:
    import src.market_data.binance.venue_rules as venue_mod
    import src.mhs.account_ledger as ledger_mod
    import src.mhs.account_sources as sources_mod
    import src.mhs.frozen_research_run as run_mod
    from src.common.errors import DataIntegrityError
    from src.mhs.account_ledger import AccountLedgerResult
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2, FrozenMhsCandidate
    from src.mhs.frozen_research_run import FrozenSourceContext
    from src.market_data.binance.venue_rules import VenueRuleSnapshot

    seen: dict = {}
    dates = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    weights = pd.DataFrame({"AAA": [0.05, -0.05, 0.02]}, index=dates, dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex(dates - pd.Timedelta(hours=1)),
        strategy=FROZEN_MHS_TOP20_V2,
    )
    frame = pd.DataFrame({"AAA": 100.0}, index=dates, dtype="float64")
    context = FrozenSourceContext(
        census=("AAA",), funding_by_symbol={}, funding_failures={}, root="root",
        budget=MhsMemoryBudget(), daily_close=frame, daily_quote_volume=frame,
    )

    def _fake_build(request: object) -> tuple:
        seen["request"] = request
        return candidate, context

    def _fake_assemble(cand: object, ctx: object) -> tuple:
        seen["candidate"] = cand
        bars = pd.date_range("2025-01-01", periods=4, freq="3min", tz="UTC")
        plane = pd.DataFrame({"AAA": 100.0}, index=bars, dtype="float64")
        marks = ledger_mod.AccountMarkPanels(close=plane, high=plane, low=plane)
        unit = pd.DataFrame({"AAA": [0.05, -0.05, 0.02]}, index=dates, dtype="float64")
        flat = pd.DataFrame({"AAA": [0.0, 0.0, 0.0]}, index=dates, dtype="float64")
        rich = pd.DataFrame({"AAA": [1e9, 1e9, 1e9]}, index=dates, dtype="float64")
        calm = pd.DataFrame({"AAA": [0.02, 0.02, 0.02]}, index=dates, dtype="float64")
        return unit, marks, flat, rich, calm

    def _fake_replay(unit: object, marks: object, funding: object, adv: object, sigma: object, rules: object, policy: object, *, capital: float, taker_fee_bps: float, apply_order_filters: bool = True) -> AccountLedgerResult:
        seen.setdefault("replays", []).append({"capital": capital, "policy": policy, "filters": apply_order_filters, "fee": taker_fee_bps})
        if recon_fail and capital == 1e5:
            raise DataIntegrityError("recon boom")
        equity = pd.Series([capital, capital * 1.1, capital * 1.05], index=dates)
        exposure = pd.Series([1.0, 2.0, 1.5], index=dates)
        return AccountLedgerResult(
            capital=capital, daily_equity=equity, daily_exposure=exposure, liquidated_at=None,
            skipped_orders=3, untraded_fraction=0.01, initial_margin_breaches=0,
            fee_paid=1.0, impact_paid=2.0, funding_paid=0.5,
            fallback_ladder_symbols=(), missing_filter_symbols=(),
        )

    snapshot = VenueRuleSnapshot(captured_at=pd.Timestamp("2026-01-01", tz="UTC"), symbols={})
    monkeypatch.setattr(run_mod, "build_frozen_request_candidate", _fake_build)
    monkeypatch.setattr(sources_mod, "assemble_account_inputs", _fake_assemble)
    monkeypatch.setattr(ledger_mod, "replay_account", _fake_replay)
    if stub_venue:
        monkeypatch.setattr(venue_mod, "latest_venue_rule_snapshot", lambda root: tmp_path / "venue.json")
        monkeypatch.setattr(venue_mod, "load_venue_rule_snapshot", lambda path: snapshot)
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", tmp_path / "x" / "runs")
    return seen


def _run_dirs(tmp_path: Path) -> list[Path]:
    root = tmp_path / "x" / "runs"
    return sorted(root.iterdir()) if root.is_dir() else []


def test_account_command_defaults_to_growth_at_retail_capital(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No policy/capital flags select growth at the declared minimum retail start."""
    from src.mhs.params import (
        ACCOUNT_DEFAULT_CAPITAL_USDT,
        ACCOUNT_EXPOSURE_MAX,
        ACCOUNT_EXPOSURE_STEP,
        ACCOUNT_IMPACT_Y,
        ACCOUNT_INITIAL_MARGIN_CAP,
        ACCOUNT_MARGIN_RESERVE,
        ACCOUNT_MEAN_HAIRCUT,
        ACCOUNT_SHOCK_PER_UNIT,
        ACCOUNT_TAKER_FEE_BPS,
        ACCOUNT_UNIT_DAILY_MEAN,
        ACCOUNT_UNIT_DAILY_SIGMA,
    )

    seen = _install_account(monkeypatch, tmp_path)
    backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    policy = seen["replays"][0]["policy"]
    assert policy.kind == "growth"
    assert seen["replays"][0]["capital"] == ACCOUNT_DEFAULT_CAPITAL_USDT
    assert policy.mean_haircut == ACCOUNT_MEAN_HAIRCUT
    assert (policy.exposure_max, policy.exposure_step) == (ACCOUNT_EXPOSURE_MAX, ACCOUNT_EXPOSURE_STEP)
    assert (policy.unit_daily_mean, policy.unit_daily_sigma) == (ACCOUNT_UNIT_DAILY_MEAN, ACCOUNT_UNIT_DAILY_SIGMA)
    assert (policy.shock_per_unit, policy.margin_reserve, policy.initial_margin_cap) == (
        ACCOUNT_SHOCK_PER_UNIT, ACCOUNT_MARGIN_RESERVE, ACCOUNT_INITIAL_MARGIN_CAP,
    )
    assert policy.impact_y == ACCOUNT_IMPACT_Y
    assert seen["replays"][0]["fee"] == ACCOUNT_TAKER_FEE_BPS
    assert seen["replays"][0]["filters"] is True


def test_account_candidate_is_unlevered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The account book is unlevered clip-0.05; exposure is chosen by the policy, never pre-multiplied."""
    from src.mhs.params import FROZEN_GROWTH_NAME_CLIP

    seen = _install_account(monkeypatch, tmp_path)
    backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert seen["request"].strategy.exposure_multiplier == 1.0
    assert seen["request"].strategy.name_clip == FROZEN_GROWTH_NAME_CLIP


def test_account_fixed_policy_requires_exposure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--policy fixed without --fixed-exposure exits before any load."""
    seen = _install_account(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match=r"fixed-exposure"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv("--policy", "fixed")))
    assert "request" not in seen
    backtest_mod.run_frozen_account_command(_parse(_account_argv("--policy", "fixed", "--fixed-exposure", "2.5")))
    assert seen["replays"][0]["policy"].kind == "fixed"
    assert seen["replays"][0]["policy"].exposure_max == 2.5


def test_account_missing_venue_snapshot_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty venue-rules dir exits naming the collection command."""
    seen = _install_account(monkeypatch, tmp_path, stub_venue=False)
    empty = tmp_path / "venue_empty"
    empty.mkdir()
    monkeypatch.setattr(backtest_mod, "VENUE_RULES_DIR", empty)
    with pytest.raises(SystemExit, match=r"data collect venue-rules"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert "request" not in seen


def test_account_artifacts_and_disclosures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """account.json carries mandated disclosures and the daily parquet exists."""
    seen = _install_account(monkeypatch, tmp_path)
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps({"kind": "mhs", "strategy_id": "x", "base_cagr": 0.1}) + "\n"
        + json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": "runs/old", "evaluation_start": "2025-01-01", "evaluation_end": "2025-02-01", "base_cagr": 1.365, "base_max_drawdown": 0.29}) + "\n",
        encoding="utf-8",
    )
    with caplog.at_level("INFO", logger="MhsBacktestCli"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert "[EVAL] mhs-frozen-account" in caplog.text
    (run_dir,) = _run_dirs(tmp_path)
    assert run_dir.name.startswith("20250101_20250201_top20_account_growth_2100_")
    payload = json.loads((run_dir / "account.json").read_text(encoding="utf-8"))
    assert payload["in_sample_moments"] is True
    assert payload["venue_rules_applied_retroactively"] is True
    assert payload["capital"] == 2100.0
    assert payload["policy"]["kind"] == "growth"
    assert payload["cagr"] == pytest.approx((2205.0 / 2100.0) ** (365.0 / 3.0) - 1.0)
    assert payload["mdd"] == pytest.approx(2205.0 / 2310.0 - 1.0)
    assert payload["liquidated_at"] is None
    assert (payload["mean_exposure"], payload["min_exposure"], payload["last_exposure"]) == (1.5, 1.0, 1.5)
    recon = payload["reconciliation"]
    assert recon["reference_canonical"]["base_cagr"] == 1.365
    assert recon["cagr_gap"] == pytest.approx(recon["cagr"] - 1.365)
    daily = pd.read_parquet(run_dir / "account_daily.parquet")
    assert list(daily.columns) == ["equity", "exposure"]
    assert len(daily) == 3
    rows = index.read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["kind"] == "mhs_frozen_account"

    index.unlink()
    backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    _, second = _run_dirs(tmp_path)
    again = json.loads((second / "account.json").read_text(encoding="utf-8"))
    assert again["reconciliation"]["reference_canonical"] is None
    assert again["reconciliation"]["cagr_gap"] is None


def test_account_reconciliation_failure_still_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing reconciliation is disclosed, never fatal."""
    _install_account(monkeypatch, tmp_path, recon_fail=True)
    backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    (run_dir,) = _run_dirs(tmp_path)
    payload = json.loads((run_dir / "account.json").read_text(encoding="utf-8"))
    assert payload["reconciliation"]["status"] == "failed"
    assert (run_dir / "account_daily.parquet").exists()


def _write_held_parquet(root: Path, symbol: str, start: pd.Timestamp, bars: int, close: float = 100.0) -> pd.DatetimeIndex:
    grid = pd.date_range(start, periods=bars, freq="3min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": (grid.view("int64") // 10**6).astype("int64"),
        "open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
    })
    target = root / "3m"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / f"{symbol}.parquet")
    return grid


def _assemble_fixture(tmp_path: Path) -> tuple:
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2, FrozenMhsCandidate
    from src.mhs.frozen_research_run import FrozenSourceContext
    from src.mhs.resources import MhsMemoryBudget

    entries = pd.date_range("2021-04-01", periods=2, freq="D", tz="UTC")
    weights = pd.DataFrame({"AAA": [0.05, -0.05], "BBB": [0.0, 0.0]}, index=entries, dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex(entries - pd.Timedelta(hours=1)),
        strategy=FROZEN_MHS_TOP20_V2,
    )
    grid = _write_held_parquet(tmp_path, "AAA", entries[0], 960)
    days = pd.date_range("2021-01-01", periods=92, freq="D", tz="UTC")
    qv = pd.DataFrame({"AAA": 5_000_000.0, "BBB": 1_000_000.0}, index=days, dtype="float64")
    drift = pd.DataFrame(
        {"AAA": 100.0 + 0.01 * pd.Series(range(92), index=days), "BBB": 50.0},
        index=days, dtype="float64",
    )
    funding_ts = pd.DatetimeIndex([entries[0] + pd.Timedelta(hours=8, minutes=1), entries[1] + pd.Timedelta(hours=8)])
    funding = pd.Series([0.0001, 0.0002], index=funding_ts, dtype="float64")
    context = FrozenSourceContext(
        census=("AAA", "BBB"), funding_by_symbol={"AAA": funding}, funding_failures={},
        root=str(tmp_path), budget=MhsMemoryBudget(), daily_close=drift, daily_quote_volume=qv,
    )
    return candidate, context, entries, grid


def test_assemble_account_inputs_held_symbols_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only held symbols are read; funding is cumulated on 3m bars and sampled at each entry."""
    import src.mhs.account_sources as sources_mod

    candidate, context, entries, grid = _assemble_fixture(tmp_path)
    seen: dict = {}
    real_assert = sources_mod.assert_mhs_stage_allocation

    def _spy(*, stage: str, **kwargs: object) -> None:
        seen["stage"] = stage
        real_assert(stage=stage, **kwargs)

    monkeypatch.setattr(sources_mod, "assert_mhs_stage_allocation", _spy)
    unit, marks, funding_cum, adv, daily_sigma = sources_mod.assemble_account_inputs(candidate, context)

    assert list(unit.columns) == ["AAA"]
    assert seen["stage"] == "account_marks"
    assert marks.close.dtypes.iloc[0] == np.dtype("float32")
    assert marks.close.index[0] == entries[0]
    assert marks.close.index[-1] == entries[-1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=3)
    # 펀딩 누적은 진입 시각에서 표본화되어 replay_account의 일간 인덱스와 정렬돼야 한다.
    assert funding_cum.index.equals(unit.index)
    assert list(funding_cum.columns) == list(unit.columns)
    assert funding_cum.loc[entries[0], "AAA"] == pytest.approx(0.0)
    assert funding_cum.loc[entries[1], "AAA"] == pytest.approx(0.0001)
    assert adv.index.equals(unit.index)
    assert daily_sigma.index.equals(unit.index)
    assert adv.loc[entries[1], "AAA"] == pytest.approx(5_000_000.0)
    assert bool(np.isfinite(daily_sigma.loc[entries[1], "AAA"]))


def test_assemble_account_inputs_missing_held_source_fails(tmp_path: Path) -> None:
    """A held symbol without 3m source fails closed."""
    import src.mhs.account_sources as sources_mod

    from src.common.errors import DataIntegrityError

    candidate, context, _, _ = _assemble_fixture(tmp_path)
    (tmp_path / "3m" / "AAA.parquet").unlink()
    with pytest.raises(DataIntegrityError, match=r"AAA"):
        sources_mod.assemble_account_inputs(candidate, context)


def test_account_rejects_invalid_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid dates, policy, and controls exit before any load."""
    import argparse

    seen = _install_account(monkeypatch, tmp_path)
    run = backtest_mod.run_frozen_account_command
    with pytest.raises(SystemExit, match=r"source-start is required"):
        run(_parse(["backtest", "mhs-frozen-account", "--start", "2025-01-01", "--end", "2025-02-01"]))
    with pytest.raises(SystemExit, match=r"start is required"):
        run(_parse(["backtest", "mhs-frozen-account", "--source-start", "2024-01-01", "--end", "2025-02-01"]))
    with pytest.raises(SystemExit, match=r"end is required"):
        run(_parse(["backtest", "mhs-frozen-account", "--source-start", "2024-01-01", "--start", "2025-01-01"]))
    with pytest.raises(SystemExit, match=r"source-start < start < end"):
        run(_parse(["backtest", "mhs-frozen-account", "--source-start", "2025-03-01", "--start", "2025-01-01", "--end", "2025-02-01"]))
    base = {"source_start": "2024-01-01", "start": "2025-01-01", "end": "2025-02-01"}
    with pytest.raises(SystemExit, match=r"policy must be"):
        run(argparse.Namespace(**base, policy="turbo", fixed_exposure=None))
    with pytest.raises(SystemExit, match=r"invalid account controls"):
        run(argparse.Namespace(**base, policy="fixed", fixed_exposure="abc"))
    with pytest.raises(SystemExit, match=r"positive finite exposure"):
        run(_parse(_account_argv("--policy", "fixed", "--fixed-exposure", "0")))
    with pytest.raises(SystemExit, match=r"positive finite capital"):
        run(_parse(_account_argv("--capital", "0")))
    assert "request" not in seen


def test_account_run_directory_suffix_on_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated timestamp resolves to a fresh suffixed run directory."""
    frozen = pd.Timestamp("2025-03-03 12:00:00", tz="UTC")
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", tmp_path)
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: frozen))
    first = backtest_mod._resolve_account_destination(
        start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"),
        policy="growth", capital=2100.0,
    )
    second = backtest_mod._resolve_account_destination(
        start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"),
        policy="growth", capital=2100.0,
    )
    assert first.name.endswith("Z")
    assert second.name == f"{first.name}-2"


def test_assemble_account_inputs_malformed_source_fails(tmp_path: Path) -> None:
    """A 3m archive without OHLC columns fails closed."""
    import src.mhs.account_sources as sources_mod

    from src.common.errors import DataIntegrityError

    candidate, context, _, _ = _assemble_fixture(tmp_path)
    grid = pd.date_range("2021-04-01", periods=8, freq="3min", tz="UTC")
    pd.DataFrame({"timestamp": (grid.view("int64") // 10**6).astype("int64"), "open": 1.0}).to_parquet(
        tmp_path / "3m" / "AAA.parquet"
    )
    with pytest.raises(DataIntegrityError, match=r"malformed"):
        sources_mod.assemble_account_inputs(candidate, context)


def test_account_source_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A data-integrity failure during source assembly exits instead of replaying garbage."""
    import src.mhs.frozen_research_run as run_mod

    from src.common.errors import DataIntegrityError

    seen = _install_account(monkeypatch, tmp_path)

    def _boom(request: object) -> tuple:
        raise DataIntegrityError("source boom")

    monkeypatch.setattr(run_mod, "build_frozen_request_candidate", _boom)
    with pytest.raises(SystemExit, match=r"frozen account failed"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert "replays" not in seen


def test_account_replay_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay failure exits without persisting an account artifact."""
    import src.mhs.account_ledger as ledger_mod

    from src.common.errors import DataIntegrityError

    _install_account(monkeypatch, tmp_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("replay boom")

    monkeypatch.setattr(ledger_mod, "replay_account", _boom)
    with pytest.raises(SystemExit, match=r"frozen account failed"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert _run_dirs(tmp_path) == []
