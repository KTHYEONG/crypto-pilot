"""Invariants for the read-only Drive-to-local research mirror."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from src.application.ops import research_mirror
from src.application.ops.research_mirror import (
    MIRROR_DATASETS,
    MIRROR_REMOTE_ROOT,
    build_mirror_commands,
    run_mirror,
)


def _completed(argv: list[str], rc: int) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, rc, "", "")


def test_mirror_commands_only_read_from_the_remote(tmp_path: Path) -> None:
    commands = build_mirror_commands(list(MIRROR_DATASETS), rclone="rclone", dest_root=tmp_path, dry_run=False)
    assert len(commands) == len(MIRROR_DATASETS)
    for argv in commands:
        assert argv[1] == "copy"
        assert argv[2].startswith(MIRROR_REMOTE_ROOT + "/")
        assert Path(argv[3]).resolve().is_relative_to(tmp_path.resolve())
        assert not {"sync", "move", "delete", "purge", "--backup-dir"} & set(argv)


def test_dry_run_flag_is_propagated(tmp_path: Path) -> None:
    dry = build_mirror_commands(list(MIRROR_DATASETS), rclone="rclone", dest_root=tmp_path, dry_run=True)
    real = build_mirror_commands(list(MIRROR_DATASETS), rclone="rclone", dest_root=tmp_path, dry_run=False)
    assert all("--dry-run" in argv for argv in dry)
    assert not any("--dry-run" in argv for argv in real)


def test_unknown_dataset_and_traversal_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        build_mirror_commands(["book_ticker", "nope"], rclone="rclone", dest_root=tmp_path, dry_run=False)
    monkeypatch.setattr(research_mirror, "MIRROR_DATASETS", {"escape": "../escape"})
    with pytest.raises(ValueError, match="outside destination root"):
        build_mirror_commands(["escape"], rclone="rclone", dest_root=tmp_path, dry_run=False)


def test_version_and_hot_paths_are_never_mirrored() -> None:
    for rel in MIRROR_DATASETS.values():
        assert "_versions" not in rel
        assert "raw/hot" not in rel
        assert "quarantine" not in rel


def test_one_failure_does_not_stop_the_rest(tmp_path: Path) -> None:
    attempts: list[str] = []

    def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        dataset = Path(argv[3]).name
        attempts.append(dataset)
        return _completed(argv, 1 if dataset == "exec_depth" else 0)

    codes = run_mirror(
        ["book_ticker", "exec_depth", "liquidations"],
        dest_root=tmp_path,
        dry_run=True,
        runner=runner,
    )
    assert attempts == ["book_ticker", "exec_depth", "liquidations"]
    assert codes == {"book_ticker": 0, "exec_depth": 1, "liquidations": 0}


def test_remote_root_matches_the_backup_script() -> None:
    script = Path(__file__).resolve().parents[4] / "deploy" / "backup" / "crypto-pilot-backup.sh"
    match = re.search(r'REMOTE_ROOT="\$\{REMOTE_ROOT:-([^}]*)\}"', script.read_text(encoding="utf-8"))
    assert match is not None
    assert f"{match.group(1)}/data" == MIRROR_REMOTE_ROOT


def test_ops_cli_mirror_and_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.application.ops import research_mirror as mirror_module
    from src.cli.main import build_root_parser
    from src.cli.commands import ops
    from src.market_data.services import data_inventory as inventory_module
    from src.market_data.services.data_inventory import DatasetInventory

    seen: dict[str, object] = {}
    row = DatasetInventory(
        name="book_ticker",
        replaceable=False,
        first_day=None,
        last_day=None,
        present_days=0,
        missing_days=(),
        incomplete_days=(),
        bytes=0,
        symbols=None,
    )

    def fake_run(datasets: object, **kwargs: object) -> dict[str, int]:
        seen["datasets"] = list(datasets)  # type: ignore[call-overload]
        seen["dry_run"] = kwargs["dry_run"]
        return {"book_ticker": 0, "exec_depth": int(seen.get("rc", 0))}

    monkeypatch.setattr(mirror_module, "run_mirror", fake_run)
    monkeypatch.setattr(ops.shutil, "which", lambda binary: binary)
    monkeypatch.setattr(inventory_module, "build_inventory", lambda root, *, today: [row])
    monkeypatch.setattr(inventory_module, "write_inventory", lambda rows, path: seen.setdefault("written", path))

    parser = build_root_parser()
    args = parser.parse_args(["ops", "research-mirror", "--dest", str(tmp_path), "--rclone", "rclone"])
    args.handler(args)
    assert seen["datasets"] == list(MIRROR_DATASETS)
    assert seen["dry_run"] is True
    assert "written" not in seen

    execute = parser.parse_args(
        ["ops", "research-mirror", "--dest", str(tmp_path), "--rclone", "rclone", "--execute", "--datasets", "book_ticker"]
    )
    execute.handler(execute)
    assert seen["datasets"] == ["book_ticker"]
    assert seen["dry_run"] is False
    assert seen["written"] == tmp_path / "inventory.json"
    out = capsys.readouterr().out
    assert "inventory name=book_ticker first=None last=None present=0 missing=0 incomplete=0 bytes=0" in out

    seen["rc"] = 1
    with pytest.raises(SystemExit) as exc:
        execute.handler(execute)
    assert exc.value.code == 3

    missing = parser.parse_args(["ops", "research-mirror", "--dest", str(tmp_path), "--rclone", "rclone"])
    missing.rclone = None
    with pytest.raises(SystemExit) as exc2:
        missing.handler(missing)
    assert exc2.value.code == 2

    inventory = parser.parse_args(["ops", "data-inventory", "--root", str(tmp_path)])
    inventory.handler(inventory)
    assert seen["written"] == tmp_path / "inventory.json"


@pytest.mark.parametrize(("error", "expected"), [(FileNotFoundError(), 127), (PermissionError(), 126)])
def test_spawn_failure_does_not_stop_remaining_datasets(
    tmp_path: Path, error: OSError, expected: int, caplog: pytest.LogCaptureFixture
) -> None:
    attempts: list[list[str]] = []

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"check": False}
        attempts.append(argv)
        if len(attempts) == 1:
            raise error
        return _completed(argv, 0)

    with caplog.at_level("INFO"):
        codes = run_mirror(["book_ticker", "exec_depth"], dest_root=tmp_path, dry_run=True, runner=runner)
    assert codes == {"book_ticker": expected, "exec_depth": 0}
    assert len(attempts) == 2
    assert "status=start rc=0 elapsed_s=0.0" in caplog.text
    assert f"status=done rc={expected}" in caplog.text


def test_duplicate_dataset_is_copied_once_and_symlink_escape_is_rejected(tmp_path: Path) -> None:
    commands = build_mirror_commands(["book_ticker", "exec_depth", "book_ticker"], rclone="rclone", dest_root=tmp_path, dry_run=True)
    assert len(commands) == 2
    (tmp_path / "live_capture").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="outside destination root"):
        build_mirror_commands(["book_ticker"], rclone="rclone", dest_root=tmp_path, dry_run=True)


def test_cli_rejects_unknown_dataset_and_missing_explicit_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.cli.commands import ops
    from src.cli.main import build_root_parser

    monkeypatch.setattr(ops.shutil, "which", lambda binary: binary)
    parser = build_root_parser()
    args = parser.parse_args(["ops", "research-mirror", "--dest", str(tmp_path), "--rclone", "rclone", "--datasets", "nope"])
    with pytest.raises(SystemExit) as exc:
        args.handler(args)
    assert exc.value.code == 2
    assert "valid datasets:" in capsys.readouterr().err
    monkeypatch.setattr(ops.shutil, "which", lambda binary: None)
    args = parser.parse_args(["ops", "research-mirror", "--rclone", "/missing/rclone"])
    with pytest.raises(SystemExit) as exc:
        args.handler(args)
    assert exc.value.code == 2
    assert "rclone-not-found" in capsys.readouterr().err
