"""CLI lab command dispatch for the part-4 surface.

The ``research`` group is removed; exploratory commands live under ``lab``.
``backtest`` keeps only ``strategy|account|exposure``.
"""

from __future__ import annotations

import pytest

from src.cli.main import build_root_parser


def _parse(argv: list[str]):
    return build_root_parser(argv).parse_args(argv)


def test_lab_horizon_diagnostic_parses() -> None:
    args = _parse(["lab", "horizon-diagnostic"])
    assert args.group == "lab"
    assert args.lab_command == "horizon-diagnostic"
    assert callable(args.handler)


def test_lab_horizon_diagnostic_parses_with_window() -> None:
    args = _parse(["lab", "horizon-diagnostic", "--start", "2021-01-01", "--end", "2025-12-31"])
    assert args.lab_command == "horizon-diagnostic"


def test_lab_process_backtest_help_parses() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse(["lab", "process-backtest", "--help"])
    assert excinfo.value.code == 0


def test_lab_group_requires_command() -> None:
    with pytest.raises(SystemExit):
        _parse(["lab"])


def test_removed_research_group_raises_system_exit() -> None:
    for argv in (
        ["research", "run", "portfolio", "mhs-horizon-diagnostic"],
        ["research", "run", "portfolio", "multi"],
        ["research", "run", "portfolio", "blend"],
        ["research", "run", "portfolio", "growth"],
        ["research", "run", "single", "baseline"],
        ["research", "run", "expert", "eval"],
    ):
        with pytest.raises(SystemExit):
            _parse(argv)


def test_removed_backtest_mhs_and_ops_migrated_raise_system_exit() -> None:
    for argv in (
        ["backtest", "mhs"],
        ["ops", "backtests-migrate", "--registry-path", "x"],
        ["ops", "backtests-verify-history-migration", "--registry-path", "x", "--history-directory", "y"],
        ["ops", "procedure-registry-migrate", "--legacy-path", "a", "--target-path", "b"],
    ):
        with pytest.raises(SystemExit):
            _parse(argv)


def test_lab_migrated_commands_parse() -> None:
    args = _parse(["lab", "procedure-registry-migrate", "--legacy-path", "a", "--target-path", "b"])
    assert args.lab_command == "procedure-registry-migrate"
    args = _parse(["lab", "backtests-migrate", "--registry-path", "x"])
    assert args.lab_command == "backtests-migrate"
    args = _parse(
        ["lab", "backtests-verify-history-migration", "--registry-path", "x", "--history-directory", "y"]
    )
    assert args.lab_command == "backtests-verify-history-migration"


def test_live_cli_startup_does_not_load_lab() -> None:
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "from src.cli.main import build_root_parser\n"
        "try:\n"
        "    build_root_parser().parse_args(['live', 'status', '--help'])\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
        "print([m for m in sorted(sys.modules) if m == 'src.lab' or m.startswith('src.lab.')])\n"
    )
    proc = subprocess.run(  # noqa: S603 -- fixed interpreter running an inline probe
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=120, cwd=".",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "[]", proc.stdout


def test_cli_contract_groups() -> None:
    parser = build_root_parser()
    groups = parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    assert set(groups) == {"backtest", "data", "live", "ops", "lab"}


def test_lab_procedure_registry_migrate_requires_explicit_paths() -> None:
    from src.cli.main import build_root_parser

    parser = build_root_parser(["lab", "procedure-registry-migrate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["lab", "procedure-registry-migrate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["lab", "procedure-registry-migrate", "--legacy-path", "a"])


def test_lab_procedure_registry_migrate_moves_once(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.cli.main import build_root_parser

    legacy = tmp_path / "legacy.jsonl"
    target = tmp_path / "target.jsonl"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"event": "registration"}\n', encoding="utf-8")
    payload = legacy.read_bytes()

    parser = build_root_parser(["lab", "procedure-registry-migrate"])
    args = parser.parse_args(
        ["lab", "procedure-registry-migrate", "--legacy-path", str(legacy), "--target-path", str(target)]
    )
    with caplog.at_level(logging.INFO):
        args.handler(args)
    assert target.is_file()
    assert target.read_bytes() == payload
    assert not legacy.exists()
    assert any("moved=True" in r.message for r in caplog.records)

    caplog.clear()
    args = parser.parse_args(
        ["lab", "procedure-registry-migrate", "--legacy-path", str(legacy), "--target-path", str(target)]
    )
    with caplog.at_level(logging.INFO):
        args.handler(args)
    assert any("moved=False" in r.message for r in caplog.records)


def test_lab_procedure_registry_migrate_fails_closed_on_ambiguity(tmp_path) -> None:
    from src.cli.main import build_root_parser

    legacy = tmp_path / "legacy.jsonl"
    target = tmp_path / "target.jsonl"
    legacy.write_text('{"a": 1}\n', encoding="utf-8")
    target.write_text('{"b": 2}\n', encoding="utf-8")
    before_legacy = legacy.read_bytes()
    before_target = target.read_bytes()

    parser = build_root_parser(["lab", "procedure-registry-migrate"])
    args = parser.parse_args(
        ["lab", "procedure-registry-migrate", "--legacy-path", str(legacy), "--target-path", str(target)]
    )
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1
    assert legacy.read_bytes() == before_legacy
    assert target.read_bytes() == before_target
