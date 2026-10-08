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
from src.core.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING
from src.core.resources import MhsMemoryBudget
from tests.unit.application.test_mhs_frozen_account import (
    _fake_run_dir,
    _install_exposure_fakes as _install_exposure,
    _write_same_book_reference,
)


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
    """Source-owned execution: help stays available and worker avoids CLI/tools imports."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "backtest", "mhs", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "--total-tree-pss-bytes" in proc.stdout
    import src.application.mhs_supervisor as supmod
    import src.application.mhs_worker as workermod

    supervisor_source = Path(supmod.__file__).read_text(encoding="utf-8")
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
    import src.engine.strategy_backtest as run_mod

    seen: dict = {}

    def _fake_run(request: object) -> object:
        seen["request"] = request
        return types.SimpleNamespace(request=request)

    def _fake_persist(run: object, output: Path) -> Path:
        seen["output"] = output
        Path(output).write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(run_mod, "run_frozen_mhs_backtest", _fake_run)
    import src.engine.backtest_persist as report_mod

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
    from src.core.params import FROZEN_GROWTH_EXPOSURE_MULTIPLIER, FROZEN_GROWTH_NAME_CLIP

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
    import src.engine.strategy_backtest as run_mod

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


def test_frozen_exposure_artifact_fresh_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing exposure.json exits before any panel load."""
    import src.core.panel as panel_mod

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
    import src.evaluation.exposure as growth_mod

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


def _install_account(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *,
    unit_fail: bool = False, unit_liquidated: bool = False, stub_venue: bool = True,
    unit_intraday: float = 0.05, account_intraday: float = 0.05,
) -> dict:
    from tests.unit.application.test_mhs_frozen_account import _install_frozen_account_fakes

    seen = _install_frozen_account_fakes(
        monkeypatch, tmp_path, unit_fail=unit_fail, unit_liquidated=unit_liquidated,
        stub_venue=stub_venue, unit_intraday=unit_intraday, account_intraday=account_intraday,
    )
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", tmp_path / "x" / "runs")
    return seen


def _run_dirs(tmp_path: Path) -> list[Path]:
    root = tmp_path / "x" / "runs"
    return sorted(root.iterdir()) if root.is_dir() else []


def test_account_command_defaults_to_growth_at_retail_capital(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No policy/capital flags select growth at the declared minimum retail start."""
    from src.core.params import (
        ACCOUNT_DEFAULT_CAPITAL_USDT,
        ACCOUNT_EXPOSURE_MAX,
        ACCOUNT_EXPOSURE_STEP,
        ACCOUNT_IMPACT_Y,
        ACCOUNT_INITIAL_MARGIN_CAP,
        ACCOUNT_MARGIN_RESERVE,
        ACCOUNT_MEAN_HAIRCUT,
        ACCOUNT_MIN_MOMENT_DAYS,
        ACCOUNT_PRIOR_DAYS,
        ACCOUNT_SHOCK_PER_UNIT,
        ACCOUNT_TAKER_FEE_BPS,
        ACCOUNT_UNIT_REFERENCE_CAPITAL,
    )

    seen = _install_account(monkeypatch, tmp_path)
    backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert len(seen["replays"]) == 2
    unit_replay, main = seen["replays"]
    unit_policy = unit_replay["policy"]
    assert unit_policy.kind == "fixed"
    assert unit_policy.exposure_max == 1.0
    assert unit_policy.impact_y == 0.0
    assert unit_replay["capital"] == ACCOUNT_UNIT_REFERENCE_CAPITAL
    assert unit_replay["filters"] is False
    assert unit_replay["unit_equity"] is None
    policy = main["policy"]
    assert policy.kind == "growth"
    assert main["capital"] == ACCOUNT_DEFAULT_CAPITAL_USDT
    assert policy.mean_haircut == ACCOUNT_MEAN_HAIRCUT
    assert (policy.exposure_max, policy.exposure_step) == (ACCOUNT_EXPOSURE_MAX, ACCOUNT_EXPOSURE_STEP)
    assert (policy.prior_days, policy.min_moment_days) == (ACCOUNT_PRIOR_DAYS, ACCOUNT_MIN_MOMENT_DAYS)
    assert not hasattr(policy, "unit_daily_mean")
    assert (policy.shock_per_unit, policy.margin_reserve, policy.initial_margin_cap) == (
        ACCOUNT_SHOCK_PER_UNIT, ACCOUNT_MARGIN_RESERVE, ACCOUNT_INITIAL_MARGIN_CAP,
    )
    assert policy.impact_y == ACCOUNT_IMPACT_Y
    assert main["fee"] == ACCOUNT_TAKER_FEE_BPS
    assert main["filters"] is True
    assert main["unit_equity"] is not None
    pd.testing.assert_series_equal(main["unit_equity"], seen["equities"][0])


def test_account_fixed_policy_requires_exposure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--policy fixed without --fixed-exposure exits before any load."""
    seen = _install_account(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match=r"fixed-exposure"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv("--policy", "fixed")))
    assert "request" not in seen
    backtest_mod.run_frozen_account_command(_parse(_account_argv("--policy", "fixed", "--fixed-exposure", "2.5")))
    assert seen["replays"][1]["policy"].kind == "fixed"
    assert seen["replays"][1]["policy"].exposure_max == 2.5


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
    from src.core.params import (
        ACCOUNT_MIN_MOMENT_DAYS,
        ACCOUNT_PRIOR_DAYS,
        ACCOUNT_RECON_CAGR_TOLERANCE,
        ACCOUNT_RECON_MDD_TOLERANCE,
        ACCOUNT_UNIT_REFERENCE_CAPITAL,
    )

    seen = _install_account(monkeypatch, tmp_path)
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps({"kind": "mhs", "strategy_id": "x", "base_cagr": 0.1}) + "\n",
        encoding="utf-8",
    )
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(
        tmp_path, index, run_dir="runs/old", base_cagr=unit_cagr, base_mdd=0.05,
    )
    with caplog.at_level("INFO", logger="MhsBacktestCli"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert "[EVAL] mhs-frozen-account" in caplog.text
    (run_dir,) = _run_dirs(tmp_path)
    assert run_dir.name.startswith("20250101_20250201_top20_account_growth_2100_")
    payload = json.loads((run_dir / "account.json").read_text(encoding="utf-8"))
    assert "in_sample_moments" not in payload
    assert payload["moment_source"] == "bayesian_causal_unit_ledger"
    assert payload["venue_rules_applied_retroactively"] is True
    assert payload["entry_anchor"] == "submit_bar"
    assert payload["capital"] == 2100.0
    assert payload["policy"]["kind"] == "growth"
    assert payload["policy"]["prior_days"] == ACCOUNT_PRIOR_DAYS
    assert payload["policy"]["min_moment_days"] == ACCOUNT_MIN_MOMENT_DAYS
    assert "unit_daily_mean" not in payload["policy"]
    assert payload["unit_reference"]["capital"] == ACCOUNT_UNIT_REFERENCE_CAPITAL
    assert payload["unit_reference"]["cagr"] == pytest.approx(unit_cagr)
    assert payload["unit_reference"]["mdd"] == pytest.approx(-0.05)
    assert payload["unit_reference"]["daily_mdd"] == pytest.approx(105000.0 / 110000.0 - 1.0)
    assert payload["cagr"] == pytest.approx((2205.0 / 2100.0) ** (365.0 / 3.0) - 1.0)
    assert payload["mdd"] == pytest.approx(-0.05)
    assert payload["daily_mdd"] == pytest.approx(2205.0 / 2310.0 - 1.0)
    assert payload["liquidated_at"] is None
    assert (payload["mean_exposure"], payload["min_exposure"], payload["last_exposure"]) == (1.5, 1.0, 1.5)
    recon = payload["reconciliation"]
    assert recon["status"] == "ok"
    assert recon["reference_canonical"]["base_cagr"] == pytest.approx(unit_cagr)
    assert recon["reference_canonical"]["name_clip"] == pytest.approx(0.05)
    assert recon["reference_canonical"]["exposure_multiplier"] == pytest.approx(1.0)
    assert recon["cagr_gap"] == pytest.approx(0.0)
    assert recon["mdd"] == pytest.approx(0.05)
    assert recon["mdd_convention"] == "magnitude"
    assert recon["mdd_definition"] == "3m_close_path"
    assert recon["mdd_gap"] == pytest.approx(0.0)
    assert recon["cagr_tolerance"] == ACCOUNT_RECON_CAGR_TOLERANCE
    assert recon["mdd_tolerance"] == ACCOUNT_RECON_MDD_TOLERANCE
    daily = pd.read_parquet(run_dir / "account_daily.parquet")
    assert list(daily.columns) == ["equity", "exposure"]
    assert len(daily) == 3
    rows = index.read_text(encoding="utf-8").splitlines()
    last = json.loads(rows[-1])
    assert last["kind"] == "mhs_frozen_account"
    assert last["base_max_drawdown"] == pytest.approx(0.05)

    index.unlink()
    backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    _, second = _run_dirs(tmp_path)
    again = json.loads((second / "account.json").read_text(encoding="utf-8"))
    assert again["reconciliation"]["status"] == "missing_reference"
    assert again["reconciliation"]["reference_canonical"] is None
    assert again["reconciliation"]["cagr_gap"] is None
    assert again["reconciliation"]["mdd_gap"] is None


def test_account_unit_reference_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed unit reference replay exits without persisting an account artifact."""
    _install_account(monkeypatch, tmp_path, unit_fail=True)
    with pytest.raises(SystemExit, match=r"unit reference"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert _run_dirs(tmp_path) == []


def test_account_unit_reference_liquidation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A liquidated unit reference is a failed input and exits."""
    _install_account(monkeypatch, tmp_path, unit_liquidated=True)
    with pytest.raises(SystemExit, match=r"unit reference"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert _run_dirs(tmp_path) == []


def test_account_growth_policy_constructed_via_shared_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The growth replay policy equals the shared account growth factory output."""
    from src.strategy.sizing import account_growth_policy

    seen = _install_account(monkeypatch, tmp_path)
    backtest_mod.run_frozen_account_command(_parse(_account_argv("--impact-y", "0.7")))
    assert seen["replays"][1]["policy"] == account_growth_policy(impact_y=0.7)


def test_account_invalid_execution_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown execution mode exits before loading any input."""
    import argparse

    seen = _install_account(monkeypatch, tmp_path)
    args = argparse.Namespace(
        source_start="2024-01-01", start="2025-01-01", end="2025-02-01",
        policy="growth", fixed_exposure=None, execution="limit",
    )
    with pytest.raises(SystemExit, match=r"execution must be"):
        backtest_mod.run_frozen_account_command(args)
    assert "request" not in seen


def _write_held_parquet(root: Path, symbol: str, start: pd.Timestamp, bars: int, close: float = 100.0) -> pd.DatetimeIndex:
    grid = pd.date_range(start, periods=bars, freq="3min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64"),
        "open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
    })
    target = root / "3m"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / f"{symbol}.parquet")
    return grid


def _assemble_fixture(tmp_path: Path) -> tuple:
    from src.strategy.targets import FROZEN_MHS_TOP20_V2, FrozenMhsCandidate
    from src.engine.strategy_backtest import FrozenSourceContext
    from src.core.resources import MhsMemoryBudget

    entries = pd.date_range("2021-04-01", periods=2, freq="D", tz="UTC")
    weights = pd.DataFrame({"AAA": [0.05, -0.05], "BBB": [0.0, 0.0]}, index=entries, dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex(entries - pd.Timedelta(hours=1)),
        strategy=FROZEN_MHS_TOP20_V2,
    )
    grid = _write_held_parquet(tmp_path, "AAA", entries[0] - pd.Timedelta(hours=1), 1000)
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
    import src.engine.account_sources as sources_mod

    candidate, context, entries, grid = _assemble_fixture(tmp_path)
    seen: dict = {}
    real_assert = sources_mod.assert_mhs_stage_allocation

    def _spy(*, stage: str, **kwargs: object) -> None:
        seen["stage"] = stage
        real_assert(stage=stage, **kwargs)

    monkeypatch.setattr(sources_mod, "assert_mhs_stage_allocation", _spy)
    unit, marks, funding_cum, adv, daily_sigma, anchors = sources_mod.assemble_account_inputs(candidate, context)

    assert list(unit.columns) == ["AAA"]
    assert seen["stage"] == "account_marks"
    assert marks.close.dtypes.iloc[0] == np.dtype("float32")
    assert marks.close.index[0] == entries[0] - pd.Timedelta(hours=1)
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
    import src.engine.account_sources as sources_mod

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


def test_assemble_account_inputs_malformed_source_fails(tmp_path: Path) -> None:
    """A 3m archive without OHLC columns fails closed."""
    import src.engine.account_sources as sources_mod

    from src.common.errors import DataIntegrityError

    candidate, context, _, _ = _assemble_fixture(tmp_path)
    grid = pd.date_range("2021-04-01", periods=8, freq="3min", tz="UTC")
    pd.DataFrame({"timestamp": np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64"), "open": 1.0}).to_parquet(
        tmp_path / "3m" / "AAA.parquet"
    )
    with pytest.raises(DataIntegrityError, match=r"malformed"):
        sources_mod.assemble_account_inputs(candidate, context)


def test_account_source_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A data-integrity failure during source assembly exits instead of replaying garbage."""
    import src.engine.strategy_backtest as run_mod

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
    import src.engine.account_ledger as ledger_mod

    from src.common.errors import DataIntegrityError

    _install_account(monkeypatch, tmp_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("replay boom")

    monkeypatch.setattr(ledger_mod, "replay_account", _boom)
    with pytest.raises(SystemExit, match=r"frozen account failed"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert _run_dirs(tmp_path) == []


def test_assemble_account_inputs_source_outside_window_fails(tmp_path: Path) -> None:
    """A 3m archive with no bars inside the replay grid fails closed."""
    import numpy as np

    import src.engine.account_sources as sources_mod

    from src.common.errors import DataIntegrityError

    candidate, context, entries, _ = _assemble_fixture(tmp_path)
    stale = pd.date_range("2020-01-01", periods=10, freq="3min", tz="UTC")
    pd.DataFrame({
        "timestamp": np.array([int(ts.value // 1_000_000) for ts in stale], dtype="int64"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    }).to_parquet(tmp_path / "3m" / "AAA.parquet")
    with pytest.raises(DataIntegrityError, match=r"AAA"):
        sources_mod.assemble_account_inputs(candidate, context)


def test_backtest_mhs_frozen_account_unit_variant_only_at_breadth_20(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """account_unit selects the registered clip unit book at Top-20 and names its run directory."""
    from src.strategy.targets import FROZEN_MHS_TOP20_ACCOUNT_UNIT_V2

    assert backtest_mod._frozen_strategy(20, "account_unit") is FROZEN_MHS_TOP20_ACCOUNT_UNIT_V2
    with pytest.raises(SystemExit):
        backtest_mod._frozen_strategy(40, "account_unit")
    name = backtest_mod._frozen_run_name(
        pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"),
        20, pd.Timestamp("2025-03-01T00:00:00Z"), variant="account_unit",
    )
    assert "_account_unit" in name
    seen = _install_frozen(monkeypatch)
    backtest_mod.run_frozen_mhs_backtest_command(_parse([
        "backtest", "mhs-frozen",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--variant", "account_unit", "--output", str(tmp_path / "unit.json"),
    ]))
    assert seen["request"].strategy is FROZEN_MHS_TOP20_ACCOUNT_UNIT_V2




def test_account_service_failure_maps_to_verbatim_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A FrozenAccountError from the service surfaces as a verbatim non-zero exit."""
    import src.application.mhs_frozen_account as app_mod
    from src.application.mhs_frozen_account import FrozenAccountError

    _install_account(monkeypatch, tmp_path)

    def _boom(request: object) -> object:
        raise FrozenAccountError("frozen account failed: x")

    monkeypatch.setattr(app_mod, "run_frozen_account", _boom)
    with pytest.raises(SystemExit) as exc_info:
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))
    assert exc_info.value.code == "frozen account failed: x"
    assert isinstance(exc_info.value.__cause__, FrozenAccountError)


def test_account_request_validation_maps_to_invalid_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ValueError from the service maps to an invalid-request exit."""
    import src.application.mhs_frozen_account as app_mod

    _install_account(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "run_frozen_account", lambda request: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(SystemExit, match=r"invalid frozen account request: bad"):
        backtest_mod.run_frozen_account_command(_parse(_account_argv()))


def test_account_cli_passes_path_roots_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched run/venue roots and flags reach the service request verbatim."""
    import src.application.mhs_frozen_account as app_mod

    _install_account(monkeypatch, tmp_path)
    monkeypatch.setattr(backtest_mod, "FROZEN_BACKTESTS_DIR", tmp_path / "custom" / "runs")
    monkeypatch.setattr(backtest_mod, "VENUE_RULES_DIR", tmp_path / "custom" / "venue")
    captured: dict = {}
    real = app_mod.run_frozen_account

    def _capture(request: object) -> object:
        captured["request"] = request
        return real(request)

    monkeypatch.setattr(app_mod, "run_frozen_account", _capture)
    backtest_mod.run_frozen_account_command(_parse(_account_argv("--no-order-filters", "--export-unit-returns", str(tmp_path / "u.parquet"))))
    request = captured["request"]
    assert request.runs_root == tmp_path / "custom" / "runs"
    assert request.venue_rules_root == tmp_path / "custom" / "venue"
    assert request.apply_order_filters is False
    assert request.export_unit_returns == tmp_path / "u.parquet"


def test_exposure_cli_maps_service_error_and_prints_only_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """An exposure service error maps to SystemExit with no stdout; success prints once."""
    import src.application.mhs_frozen_account as app_mod
    from src.application.mhs_frozen_account import FrozenAccountError

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure(monkeypatch)
    monkeypatch.setattr(app_mod, "run_frozen_exposure", lambda request: (_ for _ in ()).throw(FrozenAccountError("frozen exposure failed: y")))
    with pytest.raises(SystemExit, match=r"frozen exposure failed: y"):
        backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))
    assert capsys.readouterr().out == ""


def test_exposure_cli_prints_path_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """A successful exposure run prints the finalized path once."""
    run_dir = _fake_run_dir(tmp_path)
    _install_exposure(monkeypatch)
    backtest_mod.run_frozen_exposure_command(_parse(["backtest", "mhs-frozen-exposure", "--run-dir", str(run_dir)]))
    out = capsys.readouterr().out.strip()
    assert out == str(run_dir / "exposure.json")
