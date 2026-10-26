"""Offline integration tests for the research mirror CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.application.ops import research_mirror
from src.cli.commands import ops
from src.cli.main import build_root_parser


def test_inventory_cli_scans_real_files_without_rclone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    day = tmp_path / "live_capture/book_ticker/20261020"
    day.mkdir(parents=True)
    (day / "00.parquet").write_bytes(b"not a parquet payload")
    parser = build_root_parser()
    args = parser.parse_args(["ops", "data-inventory", "--root", str(tmp_path)])
    args.handler(args)
    payload = json.loads((tmp_path / "inventory.json").read_text())
    book = next(row for row in payload if row["name"] == "book_ticker")
    assert book["present_days"] == 1
    assert book["incomplete_days"] == ["2026-10-20"]
    assert "inventory name=book_ticker first=2026-10-20 last=2026-10-20 present=1" in capsys.readouterr().out


def test_execute_reports_inventory_even_after_partial_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"check": False}
        assert argv[1] == "copy"
        assert "--dry-run" not in argv
        if argv[2].endswith("book_ticker"):
            day = Path(argv[3]) / "20261020"
            day.mkdir(parents=True)
            (day / "00.parquet").write_bytes(b"x")
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.CompletedProcess(argv, 1)

    original = research_mirror.run_mirror

    def run(datasets: list[str], *, dest_root: Path, dry_run: bool, rclone: str) -> dict[str, int]:
        return original(datasets, dest_root=dest_root, dry_run=dry_run, rclone=rclone, runner=runner)

    monkeypatch.setattr(ops.shutil, "which", lambda binary: binary)
    monkeypatch.setattr(research_mirror, "run_mirror", run)
    args = build_root_parser().parse_args([
        "ops", "research-mirror", "--execute", "--dest", str(tmp_path), "--rclone", "fake-rclone",
        "--datasets", "book_ticker", "exec_depth",
    ])
    with pytest.raises(SystemExit) as exc:
        args.handler(args)
    assert exc.value.code == 3
    payload = json.loads((tmp_path / "inventory.json").read_text())
    assert next(row for row in payload if row["name"] == "book_ticker")["bytes"] == 1
    assert next(row for row in payload if row["name"] == "exec_depth")["present_days"] == 0
