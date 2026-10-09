"""Invariant scenarios for the deployable strategy backtest leaves."""

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
from src.cli.main import build_root_parser
from tests.unit.cli.commands._backtest_helpers import (
    _install_strategy,
    _parse,
    _patch_releases_root,
    _strategy_argv,
)
from tests.unit.application._strategy_account_helpers import (
    _fake_run_dir,
    _install_exposure_fakes as _install_exposure,
)


@pytest.fixture(autouse=True)
def _isolated_release_ledgers(tmp_path, monkeypatch):
    import shutil
    import src.strategy.release as release_mod

    source = release_mod.release_path("flow_mom_top20")
    root = tmp_path / "default_releases"
    root.mkdir()
    shutil.copy(source, root / "flow_mom_top20.json")
    monkeypatch.setattr(release_mod, "releases_dir", lambda root_arg=None: root)

def test_lab_process_backtest_command_discovery(monkeypatch) -> None:
    """Independent command discovery: loading help never runs evaluation or needs retired tools."""
    import src.lab.mhs.app.supervisor as supmod

    def _boom(*args, **kwargs):
        raise AssertionError("evaluation must not run during discovery")

    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _boom)
    parser = build_root_parser(["lab", "process-backtest", "--help"])
    group = next(action for action in parser._actions if action.dest == "group")
    assert "lab" in group.choices
    assert "lab" in (build_root_parser.__doc__ or "")
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["lab", "process-backtest", "--help"])
    assert excinfo.value.code == 0

def test_lab_process_backtest_source_owned_execution() -> None:
    """Source-owned execution: help stays available and worker avoids CLI/tools imports."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "lab", "process-backtest", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "--total-tree-pss-bytes" in proc.stdout
    import src.lab.mhs.app.supervisor as supmod
    import src.lab.mhs.app.worker as workermod

    supervisor_source = Path(supmod.__file__).read_text(encoding="utf-8")
    assert "src.cli.main" not in supervisor_source
    assert "tools." not in supervisor_source
    worker_source = Path(workermod.__file__).read_text(encoding="utf-8")
    assert "src.cli" not in worker_source
    assert "tools." not in worker_source

def test_backtest_strategy_distinct_from_legacy() -> None:
    """Strategy registration uses its own handler; the legacy mhs leaf is gone from backtest."""
    from src.cli.commands.backtest import run_strategy_backtest_command

    parsed = _parse(_strategy_argv(Path("strategy.json")))
    assert parsed.handler is run_strategy_backtest_command
    with pytest.raises(SystemExit):
        _parse(["backtest", "mhs"])

def test_backtest_old_frozen_names_rejected() -> None:
    """Pre-rename CLI names exit with an argparse error; only the new names parse."""
    import pytest

    assert _parse(["backtest", "strategy", "--source-start", "2024-01-01"]).command == "strategy"
    with pytest.raises(SystemExit):
        _parse(["backtest", "mhs-frozen", "--help"])
    with pytest.raises(SystemExit):
        _parse(["backtest", "mhs-frozen-account", "--help"])
    with pytest.raises(SystemExit):
        _parse(["backtest", "mhs-frozen-exposure", "--help"])

@pytest.mark.parametrize(("strategy_id", "breadth", "multiplier"), [
    ("flow_mom_top20", 20, 1.0),
    ("flow_mom_top40_control", 40, 1.0),
    ("flow_mom_top20_growth", 20, 2.5),
])
def test_registered_strategy_choice_wires_request(tmp_path, monkeypatch, strategy_id, breadth, multiplier) -> None:
    """The canonical CLI choice reaches the replay request and its exact target policy."""
    seen = _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(_parse([*_strategy_argv(tmp_path), "--strategy", strategy_id]))
    strategy = seen["request"].strategy
    assert (strategy.strategy_id, strategy.breadth, strategy.exposure_multiplier) == (strategy_id, breadth, multiplier)

def test_registered_strategy_rejects_conflicting_policy(tmp_path) -> None:
    """Explicit policy conflicts cannot silently replay a different strategy."""
    with pytest.raises(SystemExit, match="strategy conflicts"):
        backtest_mod.run_strategy_backtest_command(_parse([
            *_strategy_argv(tmp_path), "--strategy", "flow_mom_top40_control", "--variant", "growth",
        ]))

def test_backtest_strategy_breadth_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitted breadth selects primary Top-20; breadth 40 names an explicit control variant."""
    seen = _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(_parse(_strategy_argv(tmp_path)))
    assert seen["request"].strategy.strategy_id == "flow_mom_top20"
    assert seen["request"].strategy.breadth == 20
    assert seen["output"] == tmp_path / "strategy.json"
    out40 = tmp_path / "strategy40.json"
    backtest_mod.run_strategy_backtest_command(_parse([
        "backtest", "strategy",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--breadth", "40", "--output", str(out40),
    ]))
    assert seen["request"].strategy.breadth == 40
    assert "40" in seen["request"].strategy.strategy_id
    assert seen["request"].strategy.strategy_id != "flow_mom_top20"

def test_backtest_strategy_growth_variant_selects_registered_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--variant growth selects the registered clip + exposure policy at Top-20 breadth."""
    from src.core.params import GROWTH_EXPOSURE_MULTIPLIER, STRATEGY_NAME_CLIP

    seen = _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(_parse([*_strategy_argv(tmp_path)[:8], "--variant", "growth", "--output", str(tmp_path / "growth.json")]))
    assert seen["request"].strategy.strategy_id == "flow_mom_top20_growth"
    assert seen["request"].strategy.exposure_multiplier == GROWTH_EXPOSURE_MULTIPLIER
    assert seen["request"].strategy.name_clip == STRATEGY_NAME_CLIP
    with pytest.raises(SystemExit):
        backtest_mod._strategy_policy(20, "turbo")

def test_backtest_strategy_growth_rejects_non20_breadth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Growth with a non-20 breadth exits before any workload launch."""
    seen = _install_strategy(monkeypatch)
    with pytest.raises(SystemExit):
        backtest_mod.run_strategy_backtest_command(_parse([*_strategy_argv(tmp_path)[:8], "--variant", "growth", "--breadth", "40", "--output", str(tmp_path / "g40.json")]))
    assert "request" not in seen

def test_backtest_strategy_growth_run_directory_labelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A growth run without --output resolves into a top20_growth_ directory."""
    import argparse

    strategy_root = tmp_path / "strategy"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    output = backtest_mod._resolve_strategy_destination(
        argparse.Namespace(output=None, variant="growth"),
        start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"), breadth=20, variant="growth",
    )
    assert "top20_growth_" in output.parent.name
    assert output.name == "result.json"

def test_backtest_strategy_required_dates_and_fresh_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing source start, invalid order, or an occupied output exits before runner invocation."""
    seen = _install_strategy(monkeypatch)
    with pytest.raises(SystemExit, match=r"source-start"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "a.json")])
        )
    with pytest.raises(SystemExit, match=r"source-start < start"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2025-03-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "b.json")])
        )
    occupied = tmp_path / "occupied.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"fresh"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(occupied)])
        )
    assert "request" not in seen

def test_backtest_strategy_variant_breadth_and_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Arbitrary breadth names a control variant; bad controls and replay failures exit nonzero."""
    seen = _install_strategy(monkeypatch)
    out12 = tmp_path / "strategy12.json"
    backtest_mod.run_strategy_backtest_command(_parse([
        "backtest", "strategy",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--breadth", "12", "--output", str(out12),
    ]))
    assert seen["request"].strategy.breadth == 12
    assert "12" in seen["request"].strategy.strategy_id
    with pytest.raises(SystemExit, match=r"breadth"):
        backtest_mod.run_strategy_backtest_command(_parse([
            "backtest", "strategy",
            "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
            "--breadth", "0", "--output", str(tmp_path / "zero.json"),
        ]))
    with pytest.raises(SystemExit, match=r"start is required"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "c.json")])
        )
    with pytest.raises(SystemExit, match=r"end is required"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--output", str(tmp_path / "d.json")])
        )
    auto = tmp_path / "auto"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", auto)
    backtest_mod.run_strategy_backtest_command(
        _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    resolved = seen["output"]
    assert resolved.name == "result.json"
    assert resolved.parent.parent == auto
    assert resolved.parent.is_dir()
    assert (resolved.parent / "manifest.json").is_file()
    with pytest.raises(SystemExit, match=r"JSON path"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "e.txt")])
        )
    with pytest.raises(SystemExit, match=r"invalid strategy backtest request"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01T00:00:00+00:00", "--end", "2025-01-01T12:00:00+00:00", "--output", str(tmp_path / "f.json")])
        )
    import src.engine.strategy_backtest as run_mod

    monkeypatch.setattr(run_mod, "run_strategy_backtest", lambda request: (_ for _ in ()).throw(ValueError("boom")))
    failed = tmp_path / "failed.json"
    with pytest.raises(SystemExit, match=r"strategy backtest failed"):
        backtest_mod.run_strategy_backtest_command(
            _parse([*_strategy_argv(tmp_path)[:8], "--output", str(failed)])
        )
    assert not failed.exists()

def test_strategy_omitted_output_resolves_to_fresh_run_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitted output resolves into a fresh UUID run directory that already exists."""
    import re

    strategy_root = tmp_path / "strategy"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    seen = _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(
        _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    resolved = seen["output"]
    assert resolved.name == "result.json"
    assert resolved.parent.parent == strategy_root
    assert re.fullmatch(r"20250101_20250201_top20_\d{8}T\d{6}Z", resolved.parent.name) is not None
    assert resolved.parent.is_dir()

def test_strategy_manifest_matches_request_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifest beside the result records the request identity in JSON primitives."""
    strategy_root = tmp_path / "strategy"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    seen = _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(
        _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
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

def test_strategy_explicit_output_rejects_invalid_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit strategy output keeps the legacy suffix and freshness guards."""
    _install_strategy(monkeypatch)
    with pytest.raises(SystemExit, match=r"JSON path"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(tmp_path / "bad.txt")])
        )
    occupied = tmp_path / "occupied.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"fresh"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--output", str(occupied)])
        )

def test_strategy_index_and_prune_keep_only_recent_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each run appends one index line forever; only the newest `keep` run directories survive on disk."""
    strategy_root = tmp_path / "backtests" / "strategy" / "runs"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    monkeypatch.setattr(backtest_mod, "DEFAULT_DETAIL_RETENTION_MAX_RUNS", 2)
    seen = _install_strategy(monkeypatch)
    for day in ("01", "02", "03"):
        backtest_mod.run_strategy_backtest_command(
            _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", f"2025-01-{day}", "--end", "2025-02-01"])
        )
    index_lines = (tmp_path / "backtests" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 3
    for line in index_lines:
        row = json.loads(line)
        assert row["kind"] == "mhs_frozen"
        assert row["strategy_id"] == "flow_mom_top20"
    remaining = sorted(d for d in strategy_root.iterdir() if d.is_dir())
    assert len(remaining) == 2
    assert seen["output"].parent.exists()

def test_strategy_destination_dedupes_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run-name collision (same dates/breadth/second) appends a numeric suffix instead of clobbering."""
    strategy_root = tmp_path / "strategy"
    strategy_root.mkdir(parents=True)
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-02-01", tz="UTC")
    strategy_now = pd.Timestamp("2026-06-01T00:00:00Z")
    name = backtest_mod._strategy_run_name(start, end, 20, strategy_now)
    (strategy_root / name).mkdir()
    orig_now = pd.Timestamp.now
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, tz=None: strategy_now))
    try:
        output = backtest_mod._resolve_strategy_destination(
            argparse.Namespace(output=None), start=start, end=end, breadth=20,
        )
    finally:
        pd.Timestamp.now = orig_now
    assert output.parent.name == f"{name}-2"

def test_prune_strategy_runs_noop_when_directory_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pruning before any strategy run has ever been created is a safe no-op."""
    strategy_root = tmp_path / "never_created"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    backtest_mod._prune_strategy_runs(keep=5)
    assert not strategy_root.exists()

def test_strategy_execution_flag_maps_to_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--execution maker selects the strict-proxy bound and labels the run directory."""
    strategy_root = tmp_path / "strategy"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    seen = _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(
        _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01", "--execution", "maker"])
    )
    assert seen["request"].execution_bound == "OHLCV_STRICT_PROXY"
    assert "_maker_" in seen["output"].parent.name

def test_strategy_default_execution_is_taker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No flag keeps the immediate-taker bound and today's run directory format."""
    import re

    strategy_root = tmp_path / "strategy"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    seen = _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(
        _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    assert seen["request"].execution_bound == "OHLCV_IMMEDIATE_TAKER"
    assert re.fullmatch(r"20250101_20250201_top20_\d{8}T\d{6}Z", seen["output"].parent.name) is not None

def test_strategy_execution_rejects_unknown_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An execution value outside taker/maker exits before any workload launch."""
    seen = _install_strategy(monkeypatch)
    args = _parse(_strategy_argv(tmp_path))
    args.execution = "peg"
    with pytest.raises(SystemExit, match=r"execution"):
        backtest_mod.run_strategy_backtest_command(args)
    assert "request" not in seen

def test_strategy_index_records_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A strategy index row carries execution while a lab process row has no such key."""
    strategy_root = tmp_path / "backtests" / "strategy" / "runs"
    monkeypatch.setattr(backtest_mod, "STRATEGY_BACKTESTS_DIR", strategy_root)
    _install_strategy(monkeypatch)
    backtest_mod.run_strategy_backtest_command(
        _parse(["backtest", "strategy", "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01"])
    )
    strategy_row = json.loads((tmp_path / "backtests" / "index.jsonl").read_text(encoding="utf-8").strip())
    assert strategy_row["execution"] == "taker"

    def _fake(**kwargs):
        Path(kwargs["result_output"]).write_text(
            json.dumps({"financial": {"base": {"cagr": 0.1, "max_drawdown": -0.05}}}), encoding="utf-8",
        )
        return types.SimpleNamespace(status="completed")

    import src.lab.mhs.app.supervisor as supmod
    from src.cli.commands.lab import run_process_backtest

    monkeypatch.setattr("src.common.paths.BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(supmod, "run_mhs_process_backtest", _fake)
    run_process_backtest(_parse(["lab", "process-backtest", "--start", "2022-01-01", "--end", "2022-01-04"]))
    rows = (tmp_path / "backtests" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    canonical_row = json.loads(rows[-1])
    assert canonical_row["kind"] == "mhs"
    assert "execution" not in canonical_row

def test_strategy_exposure_artifact_fresh_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing exposure.json exits before any panel load."""
    import src.core.panel as panel_mod

    run_dir = _fake_run_dir(tmp_path)
    (run_dir / "exposure.json").write_text("{}", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("panel must not load")

    monkeypatch.setattr(panel_mod, "load_base_panel", _boom)
    with pytest.raises(SystemExit, match=r"fresh"):
        backtest_mod.run_exposure_scan_command(_parse(["backtest", "exposure", "--run-dir", str(run_dir)]))

def test_strategy_exposure_rejects_missing_run_dir(tmp_path: Path) -> None:
    """A missing run-dir flag or a non-directory path exits before any workload."""
    with pytest.raises(SystemExit, match=r"run-dir is required"):
        backtest_mod.run_exposure_scan_command(_parse(["backtest", "exposure"]))
    with pytest.raises(SystemExit, match=r"existing strategy run directory"):
        backtest_mod.run_exposure_scan_command(
            _parse(["backtest", "exposure", "--run-dir", str(tmp_path / "missing")])
        )

def test_strategy_exposure_rejects_invalid_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run directory without result.json exits instead of solving garbage."""
    _install_exposure(monkeypatch)
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match=r"invalid strategy run artifacts"):
        backtest_mod.run_exposure_scan_command(_parse(["backtest", "exposure", "--run-dir", str(run_dir)]))

def test_strategy_exposure_solver_rejection_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A solver rejection exits instead of persisting an exposure artifact."""
    import src.evaluation.exposure as growth_mod

    run_dir = _fake_run_dir(tmp_path)
    _install_exposure(monkeypatch)

    def _reject(*args, **kwargs):
        raise ValueError("no admissible rung")

    monkeypatch.setattr(growth_mod, "solve_log_growth_exposure", _reject)
    with pytest.raises(SystemExit, match=r"exposure scan failed"):
        backtest_mod.run_exposure_scan_command(_parse(["backtest", "exposure", "--run-dir", str(run_dir)]))
    assert not (run_dir / "exposure.json").exists()

def test_holdout_gate_refuses_consumed_window(tmp_path, monkeypatch) -> None:
    from src.evaluation.holdout import consume_holdout_look

    root = _patch_releases_root(monkeypatch, tmp_path)
    window = (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
    with pytest.raises(SystemExit, match="post-design"):
        backtest_mod._holdout_gate("flow_mom_top20", *window, False)
    assert (root / "flow_mom.holdout.jsonl").exists() is False
    backtest_mod._holdout_gate("flow_mom_top20", *window, True)
    assert (root / "flow_mom.holdout.jsonl").exists() is True
    with pytest.raises(SystemExit, match="post-design"):
        backtest_mod._holdout_gate("flow_mom_top20", *window, False)
    with pytest.raises(SystemExit):
        backtest_mod._holdout_gate("flow_mom_top20", *window, True)
    with pytest.raises(SystemExit, match="design cutoff"):
        backtest_mod._holdout_gate("flow_mom_top20", pd.Timestamp("2026-06-01", tz="UTC"),
                                  pd.Timestamp("2026-06-20", tz="UTC"), True)
    consume_holdout_look("flow_mom_top20", "legacy", (pd.Timestamp("2026-06-01", tz="UTC"),
                                                    pd.Timestamp("2026-06-20", tz="UTC")),
                         path=root / "flow_mom.holdout.jsonl")
    with pytest.raises(SystemExit, match="consumed holdout"):
        backtest_mod._holdout_gate("flow_mom_top20", pd.Timestamp("2026-06-01", tz="UTC"),
                                  pd.Timestamp("2026-06-20", tz="UTC"), False)
    _ = consume_holdout_look

def test_neighbor_specs_cover_drops_and_breadths() -> None:
    from src.strategy.targets import FLOW_MOM_TOP20

    neighbors = backtest_mod._neighbor_specs(FLOW_MOM_TOP20)
    assert len(neighbors) == 7
    assert sorted(n.breadth for n in neighbors if len(n.members) == 5) == [15, 25]
    assert sorted(len(n.members) for n in neighbors) == [4, 4, 4, 4, 4, 5, 5]

def test_holdout_gate_converts_journal_race_to_exit(tmp_path, monkeypatch) -> None:
    """A journal that appears between check and record maps to SystemExit, not a traceback."""
    import src.evaluation.holdout as holdout_mod
    from src.common.errors import DataIntegrityError

    _patch_releases_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        holdout_mod, "consume_holdout_look",
        lambda *args, **kwargs: (_ for _ in ()).throw(DataIntegrityError("race")),
    )
    window = (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
    with pytest.raises(SystemExit, match="race"):
        backtest_mod._holdout_gate("flow_mom_top20", *window, True)

def test_strategy_command_records_trial_and_neighbors(tmp_path, monkeypatch) -> None:
    """A successful strategy run appends a trial and fans out the neighbor set."""
    import src.engine.backtest_persist as persist_mod
    import src.engine.strategy_backtest as run_mod

    root = _patch_releases_root(monkeypatch, tmp_path)
    index = pd.date_range("2025-01-01", periods=31, freq="D", tz="UTC")
    returns = pd.Series(np.linspace(0.001, 0.002, 31), index=index, dtype="float64")
    run = types.SimpleNamespace(
        evidence=types.SimpleNamespace(base_daily=types.SimpleNamespace(returns=returns)),
        source_gap_excluded_symbols=(),
    )
    monkeypatch.setattr(run_mod, "run_strategy_backtest", lambda request: run)
    monkeypatch.setattr(backtest_mod, "_strategy_run_statistics", lambda run: {})
    monkeypatch.setattr(
        persist_mod, "persist_strategy_backtest",
        lambda run, output, **kwargs: Path(output).write_text("{}", encoding="utf-8"),
    )
    argv = [
        "backtest", "strategy",
        "--source-start", "2024-01-01", "--start", "2025-01-01", "--end", "2025-02-01",
        "--output", str(tmp_path / "strategy.json"), "--neighbors",
    ]
    backtest_mod.run_strategy_backtest_command(_parse(argv))
    rows = (root / "flow_mom.trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 7
    run_output = tmp_path / "strategy.json"
    expected = {
        run_output.parent.parent / f"{run_output.parent.name}_neighbor{i}" / "result.json"
        for i in range(7)
    }
    assert all(path.exists() for path in expected)
    import json as _json

    assert all(_json.loads(line)["source"] == "cli" for line in rows)

def test_run_neighbor_set_suppresses_evidence_failures(tmp_path, monkeypatch) -> None:
    """A neighbor without readable evidence still persists; its trial is skipped quietly."""
    import src.engine.backtest_persist as persist_mod
    import src.engine.strategy_backtest as run_mod
    from src.strategy.targets import FLOW_MOM_TOP20

    root = _patch_releases_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_mod, "run_strategy_backtest",
        lambda request: types.SimpleNamespace(request=request, evidence=None),
    )
    monkeypatch.setattr(
        run_mod, "StrategyBacktestRequest",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        persist_mod, "persist_strategy_backtest",
        lambda run, output, **kwargs: Path(output).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(backtest_mod, "_strategy_run_statistics", lambda run: {})
    output = tmp_path / "runs" / "unit" / "result.json"
    output.parent.mkdir(parents=True)
    backtest_mod._run_neighbor_set(
        FLOW_MOM_TOP20, source_start=pd.Timestamp("2024-01-01", tz="UTC"),
        start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"),
        base_spec=None, stress_spec=None, report_periods=(), data_root=None,
        budget=None, execution_bound="OHLCV_IMMEDIATE_TAKER", output=output,
    )
    assert not (root / "flow_mom.trials.jsonl").exists()
